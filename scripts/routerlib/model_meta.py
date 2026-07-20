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



def normalize_model_name(model: str) -> str:
    normalized = model.strip()
    for prefix in ("google-gemini-cli/", "codex/", "ollama/", "claude/"):
        if normalized.startswith(prefix):
            normalized = normalized.split("/", 1)[1]
            break
    return normalized


def is_large_model(model: str) -> bool:
    lowered = normalize_model_name(model).lower()
    return any(hint in lowered for hint in (":20b", ":26b", ":31b", ":32b", ":70b", ":72b"))


def normalize_route(value: Any, default: Literal["PRO", "FLASH"] = PRO) -> Literal["PRO", "FLASH"]:
    candidate = str(value or "").strip().upper()
    if candidate == FLASH:
        return FLASH
    if candidate == PRO:
        return PRO
    return default


def dedupe_model_sequence(primary_model: str, fallback_models: List[str]) -> List[str]:
    sequence: List[str] = []
    for model in [primary_model, *fallback_models]:
        candidate = str(model).strip()
        if candidate and candidate not in sequence:
            sequence.append(candidate)
    return sequence


def route_fallback_models(state: RouterState, route: Literal["PRO", "FLASH"]) -> List[str]:
    return state["pro_fallback_models"] if route == PRO else state["flash_fallback_models"]


def score_complexity(scores: ComplexityScores) -> int:
    return (
        scores["reasoning_depth"]
        + scores["code_change_scope"]
        + scores["ambiguity"]
        + scores["risk"]
    )


def decide_route(
    task: str,
    description: str,
    scores: ComplexityScores,
    suggested_route: Literal["PRO", "FLASH"],
    confidence: float,
) -> Literal["PRO", "FLASH"]:
    complexity_score = score_complexity(scores)
    summary_like = is_summary_like_subtask(description)
    synthesis_like = is_synthesis_like_subtask(description)
    deep_work_hint = has_deep_work_hint(description)
    data_gathering_hint = has_data_gathering_hint(description)
    high_risk_core_step = is_high_risk_core_step(task, description)

    if synthesis_like:
        return PRO
    if (
        summary_like
        and not deep_work_hint
        and not data_gathering_hint
        and complexity_score < PRO_COMPLEXITY_THRESHOLD
    ):
        return FLASH
    if high_risk_core_step:
        return PRO
    if deep_work_hint and not summary_like:
        return PRO

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return PRO
    if complexity_score >= PRO_COMPLEXITY_THRESHOLD:
        return PRO
    if (
        scores["reasoning_depth"] >= 2
        or scores["code_change_scope"] >= 2
        or scores["risk"] >= 2
    ):
        return PRO
    if complexity_score <= FLASH_COMPLEXITY_THRESHOLD and scores["io_heaviness"] >= 1:
        return FLASH
    if complexity_score <= 3 and scores["io_heaviness"] == 2 and confidence >= 0.7:
        return FLASH
    if suggested_route == FLASH and complexity_score <= 3 and confidence >= 0.8:
        return FLASH
    if suggested_route == PRO and confidence >= 0.6:
        return PRO
    return PRO


def is_gemini_model(model: str) -> bool:
    raw = model.strip().lower()
    if raw.startswith(("codex/", "ollama/", "claude/")):
        return False
    if raw.startswith("google-gemini-cli/"):
        return True
    normalized = normalize_model_name(model)
    return normalized in {"auto", "pro", "flash", "flash-lite"} or normalized.startswith("gemini-")


def is_codex_model(model: str) -> bool:
    raw = model.strip().lower()
    if raw.startswith(("google-gemini-cli/", "ollama/", "claude/")):
        return False
    normalized = normalize_model_name(model).lower()
    if raw.startswith("codex/"):
        return True
    if ":" in normalized:
        return False
    return (
        normalized.startswith("gpt-")
        or normalized.startswith("chatgpt-")
        or re.match(r"^o\d(?:[-.].*)?$", normalized) is not None
    )


def is_claude_model(model: str) -> bool:
    raw = model.strip().lower()
    if raw.startswith(("google-gemini-cli/", "codex/", "ollama/")):
        return False
    if raw.startswith("claude/"):
        return True
    normalized = normalize_model_name(model).lower()
    return normalized.startswith("claude-")


def model_transport_name(model: str) -> str:
    if is_claude_model(model):
        return "claude_cli"
    if is_gemini_model(model):
        return "gemini_cli"
    if is_codex_model(model):
        return "codex_cli"
    return "ollama_http"


def langsmith_provider_name(model: str) -> str:
    if is_claude_model(model):
        return "anthropic"
    if is_gemini_model(model):
        return "google_genai"
    if is_codex_model(model):
        return "codex"
    return "ollama"
