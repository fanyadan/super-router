from __future__ import annotations

import argparse
import contextlib
import contextvars
import copy
import datetime
import json
import operator
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Annotated, Any, Callable, Dict, Iterator, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

try:
    import langsmith as _langsmith
except Exception:
    _langsmith = None

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from .config import *  # noqa: F401,F403
from .state_types import *  # noqa: F401,F403
from .resolvers import *  # noqa: F401,F403
from .text_utils import *  # noqa: F401,F403
from .model_meta import *  # noqa: F401,F403



def build_text_generation_result(
    text: str,
    usage_metadata: Dict[str, int] | None,
    provider: str,
    model_name: str,
    usage_source: str = "unavailable",
) -> TextGenerationResult:
    return {
        "text": text,
        "usage_metadata": usage_metadata or {},
        "provider": provider,
        "model_name": model_name,
        "usage_source": usage_source if usage_metadata else "unavailable",
    }


def normalize_usage_metadata(
    *,
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
    cached_tokens: Any = None,
    thought_tokens: Any = None,
    tool_tokens: Any = None,
    candidates_tokens: Any = None,
) -> Dict[str, int]:
    input_count = coerce_non_negative_int(input_tokens)
    output_count = coerce_non_negative_int(output_tokens)
    candidates_count = coerce_non_negative_int(candidates_tokens)
    thought_count = coerce_non_negative_int(thought_tokens)
    total_count = coerce_non_negative_int(total_tokens)
    cached_count = coerce_non_negative_int(cached_tokens)
    tool_count = coerce_non_negative_int(tool_tokens)
    if output_count is None and (candidates_count is not None or thought_count is not None):
        output_count = (candidates_count or 0) + (thought_count or 0)
    if total_count is None and input_count is not None and output_count is not None:
        total_count = input_count + output_count + (tool_count or 0)
    if output_count is None and total_count is not None and input_count is not None:
        output_count = max(0, total_count - input_count)
    if input_count is None and total_count is not None and output_count is not None:
        input_count = max(0, total_count - output_count)

    usage: Dict[str, int] = {}
    if input_count is not None:
        usage["input_tokens"] = input_count
    if output_count is not None:
        usage["output_tokens"] = output_count
    if total_count is not None:
        usage["total_tokens"] = total_count
    if cached_count is not None:
        usage["cached_tokens"] = cached_count
    if thought_count is not None:
        usage["thought_tokens"] = thought_count
    if tool_count is not None:
        usage["tool_tokens"] = tool_count
    if candidates_count is not None:
        usage["candidate_tokens"] = candidates_count
    return usage


def extract_ollama_usage_metadata(data: Dict[str, Any]) -> Dict[str, int]:
    return normalize_usage_metadata(
        input_tokens=data.get("prompt_eval_count"),
        output_tokens=data.get("eval_count"),
    )


def extract_gemini_cli_stats_usage_metadata(payload: Any) -> Dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        return {}
    models = stats.get("models")
    if not isinstance(models, dict):
        return {}

    totals = {
        "prompt": 0,
        "candidates": 0,
        "total": 0,
        "cached": 0,
        "thoughts": 0,
        "tool": 0,
    }
    saw_tokens = False
    for model_stats in models.values():
        if not isinstance(model_stats, dict):
            continue
        tokens = model_stats.get("tokens")
        if not isinstance(tokens, dict):
            continue
        token_values: Dict[str, int] = {}
        for key in totals:
            value = coerce_non_negative_int(tokens.get(key))
            if value is not None:
                token_values[key] = value
        if not token_values:
            continue
        saw_tokens = True
        for key, value in token_values.items():
            totals[key] += value

    if not saw_tokens:
        return {}
    return normalize_usage_metadata(
        input_tokens=totals["prompt"],
        candidates_tokens=totals["candidates"],
        total_tokens=totals["total"],
        cached_tokens=totals["cached"],
        thought_tokens=totals["thoughts"],
        tool_tokens=totals["tool"],
    )


def first_present_value(data: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def extract_usage_metadata_from_mapping(data: Dict[str, Any]) -> Dict[str, int]:
    return normalize_usage_metadata(
        input_tokens=first_present_value(
            data,
            (
                "input_tokens",
                "prompt_tokens",
                "prompt_token_count",
                "promptTokenCount",
            ),
        ),
        output_tokens=first_present_value(
            data,
            (
                "output_tokens",
                "completion_tokens",
                "completion_token_count",
                "completionTokenCount",
            ),
        ),
        candidates_tokens=first_present_value(
            data,
            (
                "candidate_tokens",
                "candidates_tokens",
                "candidates_token_count",
                "candidatesTokenCount",
            ),
        ),
        total_tokens=first_present_value(
            data,
            (
                "total_tokens",
                "total_token_count",
                "totalTokenCount",
            ),
        ),
        cached_tokens=first_present_value(
            data,
            (
                "cached_tokens",
                "cached_token_count",
                "cachedContentTokenCount",
                "cached_content_token_count",
            ),
        ),
        thought_tokens=first_present_value(
            data,
            (
                "thought_tokens",
                "thoughts_token_count",
                "thoughtsTokenCount",
                "thinking_tokens",
            ),
        ),
        tool_tokens=first_present_value(
            data,
            (
                "tool_tokens",
                "tool_token_count",
                "toolUsePromptTokenCount",
                "tool_use_prompt_token_count",
            ),
        ),
    )


def extract_nested_gemini_usage_metadata(payload: Any) -> Dict[str, int]:
    candidates: List[Dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for usage_key in ("usage_metadata", "usageMetadata", "usage"):
                usage_value = value.get(usage_key)
                if isinstance(usage_value, dict):
                    candidates.append(usage_value)
            if extract_usage_metadata_from_mapping(value):
                candidates.append(value)
            for nested_value in value.values():
                visit(nested_value)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    for candidate in candidates:
        usage = extract_usage_metadata_from_mapping(candidate)
        if usage:
            return usage
    return {}


def extract_gemini_usage_metadata_with_source(payload: Any) -> tuple[Dict[str, int], str]:
    cli_stats_usage = extract_gemini_cli_stats_usage_metadata(payload)
    if cli_stats_usage:
        return cli_stats_usage, "gemini_cli_stats"
    nested_usage = extract_nested_gemini_usage_metadata(payload)
    if nested_usage:
        return nested_usage, "usage_metadata"
    return {}, "unavailable"


def extract_gemini_usage_metadata(payload: Any) -> Dict[str, int]:
    usage, _ = extract_gemini_usage_metadata_with_source(payload)
    return usage


def sum_optional_ints(values: List[Any]) -> int | None:
    total = 0
    found = False
    for value in values:
        count = coerce_non_negative_int(value)
        if count is None:
            continue
        total += count
        found = True
    return total if found else None
