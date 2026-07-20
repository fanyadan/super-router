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
from .usage import *  # noqa: F401,F403
from .context_packs import *  # noqa: F401,F403
from .langsmith_integration import *  # noqa: F401,F403
from .token_usage import *  # noqa: F401,F403
from .provider_process import *  # noqa: F401,F403
from .provider_ollama import *  # noqa: F401,F403
from .provider_gemini import *  # noqa: F401,F403
from .provider_codex import *  # noqa: F401,F403
from .provider_claude import *  # noqa: F401,F403
from .generation import *  # noqa: F401,F403
from . import generation  # noqa: F401



def empty_flash_review() -> FlashReviewResult:
    return {
        "decision": "record",
        "failure_type": "none",
        "reason": "",
    }


def empty_finalizer_outcome() -> FinalizerOutcome:
    return {
        "route": "DETERMINISTIC",
        "model_name": "",
        "status": "not_started",
        "used_provider_fallback": False,
        "reason": "",
        "attempt_log": [],
    }


def empty_model_invocation_result(primary_model: str = "") -> ModelInvocationResult:
    return {
        "success": False,
        "output": "",
        "model_name": primary_model,
        "used_provider_fallback": False,
        "failure_type": "unknown",
        "error_text": "",
        "attempt_log": [],
    }


def classify_failure_type(error_text: str) -> Literal["infra_transient", "capability_quality", "unknown"]:
    lowered = error_text.lower()
    if contains_any(lowered, INFRA_FAILURE_KEYWORDS):
        return "infra_transient"
    if contains_any(lowered, CAPABILITY_FAILURE_KEYWORDS):
        return "capability_quality"
    return "unknown"


def classify_flash_execution_failure(error_text: str) -> FlashReviewResult:
    detail = compact_text(error_text, 220)
    failure_type = classify_failure_type(error_text)
    if failure_type == "infra_transient":
        return {
            "decision": "retry",
            "failure_type": "infra_transient",
            "reason": f"Transient infrastructure/provider failure during FLASH execution: {detail}",
        }
    if failure_type == "capability_quality":
        return {
            "decision": "escalate",
            "failure_type": "capability_quality",
            "reason": f"FLASH execution indicated the step likely needs a stronger model: {detail}",
        }
    return {
        "decision": "retry",
        "failure_type": "unknown",
        "reason": f"Unknown FLASH execution failure; retry within budget before recording it: {detail}",
    }


def verify_flash_output(
    description: str,
    output: str,
    prior_review: FlashReviewResult,
    retry_count: int,
) -> FlashReviewResult:
    stripped = output.strip()
    if not stripped:
        return {
            "decision": "escalate",
            "failure_type": "capability_quality",
            "reason": "FLASH returned empty output after a nominally successful call.",
        }

    normalized_output = " ".join(stripped.split())
    lowered = normalized_output.lower()
    normalized_desc = " ".join(description.strip().split()).lower()

    if contains_any(lowered, LOW_QUALITY_OUTPUT_MARKERS):
        return {
            "decision": "escalate",
            "failure_type": "capability_quality",
            "reason": "FLASH output explicitly signaled insufficient context or inability to finish the step.",
        }

    if not is_summary_like_subtask(description):
        if lowered == normalized_desc:
            return {
                "decision": "escalate",
                "failure_type": "capability_quality",
                "reason": "FLASH mostly repeated the subtask description instead of executing it.",
            }
        if len(normalized_output) < MIN_NON_SUMMARY_OUTPUT_CHARS:
            return {
                "decision": "escalate",
                "failure_type": "capability_quality",
                "reason": "FLASH output was too short for a non-summary step and likely lacks enough substance.",
            }

    prior_failure_type = prior_review["failure_type"]
    if prior_failure_type != "none" and retry_count > 0:
        return {
            "decision": "record",
            "failure_type": prior_failure_type,
            "reason": (
                f"FLASH succeeded after {retry_count} retr"
                f"{'y' if retry_count == 1 else 'ies'} following {prior_failure_type} issues."
            ),
        }

    return {
        "decision": "record",
        "failure_type": "none",
        "reason": "FLASH output passed heuristic verification.",
    }


MODEL_INVOCATION_GRAPH: Any | None = None


def model_attempt_prepare_node(state: ModelInvocationState) -> Dict[str, Any]:
    candidates = state["candidates"]
    index = state["candidate_index"]
    log = list(state["log"])
    if not candidates or index >= len(candidates):
        result = empty_model_invocation_result(state["primary_model"])
        result["error_text"] = "No provider model candidates were available."
        result["attempt_log"] = log
        return {"result": result, "status": "provider_finished"}

    model_name = candidates[index]
    if index == 0:
        log.append(f"{state['label']} primary model attempt: {model_name}")
    else:
        log.append(f"{state['label']} provider fallback attempt {index}: {model_name}")
    return {
        "current_model": model_name,
        "log": log,
        "status": "provider_attempt_ready",
    }


def model_invoke_node(state: ModelInvocationState) -> Dict[str, Any]:
    if state["status"] == "provider_finished":
        return {}

    log = list(state["log"])
    errors = list(state["errors"])
    model_name = state["current_model"]
    index = state["candidate_index"]
    try:
        output = generation.generate_text(
            model_name,
            state["prompt"],
            timeout=state["timeout"],
            num_predict=state["num_predict"],
            temperature=state["temperature"],
            usage_label=state["label"],
        )
        log.append(f"{state['label']} succeeded with model {model_name}.")
        return {
            "result": {
                "success": True,
                "output": output.strip(),
                "model_name": model_name,
                "used_provider_fallback": index > 0,
                "failure_type": "none",
                "error_text": "",
                "attempt_log": log,
            },
            "log": log,
            "status": "provider_finished",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 220)
        failure_type = classify_failure_type(error_text)
        errors.append(f"{model_name}: {error_text}")
        log.append(
            f"{state['label']} failed with model {model_name}: {error_text} ({failure_type})"
        )
        is_last_candidate = index == len(state["candidates"]) - 1
        if is_last_candidate or failure_type == "capability_quality":
            if failure_type == "capability_quality" and not is_last_candidate:
                log.append(
                    f"{state['label']} stopped before provider fallback because the failure looked like capability/quality, not infrastructure."
                )
            return {
                "result": {
                    "success": False,
                    "output": "",
                    "model_name": state["primary_model"],
                    "used_provider_fallback": False,
                    "failure_type": failure_type,
                    "error_text": "; ".join(errors),
                    "attempt_log": log,
                },
                "log": log,
                "errors": errors,
                "status": "provider_finished",
            }
        return {
            "candidate_index": index + 1,
            "log": log,
            "errors": errors,
            "status": "provider_retry",
        }


def route_after_model_invoke(state: ModelInvocationState) -> str:
    if state["status"] == "provider_finished":
        return END
    return "model_attempt_prepare"


def build_model_invocation_graph():
    workflow = StateGraph(ModelInvocationState)
    workflow.add_node("model_attempt_prepare", model_attempt_prepare_node)
    workflow.add_node("model_invoke", model_invoke_node)
    workflow.add_edge(START, "model_attempt_prepare")
    workflow.add_edge("model_attempt_prepare", "model_invoke")
    workflow.add_conditional_edges("model_invoke", route_after_model_invoke)
    return workflow.compile()


def get_model_invocation_graph():
    global MODEL_INVOCATION_GRAPH
    if MODEL_INVOCATION_GRAPH is None:
        MODEL_INVOCATION_GRAPH = build_model_invocation_graph()
    return MODEL_INVOCATION_GRAPH


def invoke_with_provider_fallback(
    primary_model: str,
    fallback_models: List[str],
    prompt: str,
    *,
    timeout: int,
    num_predict: int,
    temperature: float,
    label: str,
    attempt_log: List[str] | None = None,
) -> ModelInvocationResult:
    candidates = dedupe_model_sequence(primary_model, fallback_models)
    initial_log = list(attempt_log or [])
    max_attempts = resolve_positive_int(
        None,
        ROUTER_MAX_PROVIDER_ATTEMPTS_ENV_VAR,
        DEFAULT_MAX_PROVIDER_ATTEMPTS,
    )
    if len(candidates) > max_attempts:
        initial_log.append(
            f"{label} provider candidates limited to {max_attempts}/"
            f"{len(candidates)} by {ROUTER_MAX_PROVIDER_ATTEMPTS_ENV_VAR}."
        )
        candidates = candidates[:max_attempts]
    initial_state: ModelInvocationState = {
        "primary_model": primary_model,
        "candidates": candidates,
        "candidate_index": 0,
        "current_model": "",
        "prompt": prompt,
        "timeout": timeout,
        "num_predict": num_predict,
        "temperature": temperature,
        "label": label,
        "log": initial_log,
        "errors": [],
        "result": empty_model_invocation_result(primary_model),
        "status": "provider_created",
    }
    final_state = get_model_invocation_graph().invoke(
        initial_state,
        config={"recursion_limit": max(8, len(candidates) * 3 + 4)},
    )
    return final_state["result"]
