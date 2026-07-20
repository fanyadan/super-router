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



def annotate_langsmith_model_run(
    *,
    model: str,
    provider: str,
    num_predict: int,
    temperature: float,
) -> None:
    if _langsmith is None or not langsmith_tracing_configured():
        return
    get_current_run_tree = getattr(_langsmith, "get_current_run_tree", None)
    if get_current_run_tree is None:
        return
    try:
        run_tree = get_current_run_tree()
    except Exception:
        return
    if run_tree is None:
        return
    try:
        run_tree.add_metadata(
            {
                "ls_provider": provider,
                "ls_model_name": normalize_model_name(model),
                "ls_temperature": temperature,
                "ls_max_tokens": num_predict,
                "ls_invocation_params": {
                    "model": normalize_model_name(model),
                    "raw_model": model,
                    "provider": provider,
                },
            }
        )
    except Exception:
        return


def resolve_bool_value(value: str | None, fallback: bool = False) -> bool:
    if value is None or not str(value).strip():
        return fallback
    return str(value).strip().lower() in {"1", "true", "yes", "on", "debug"}


def langsmith_tracing_requested() -> bool:
    router_value = os.environ.get(ROUTER_LANGSMITH_ENABLED_ENV_VAR)
    if router_value is not None and router_value.strip():
        return resolve_bool_value(router_value)
    for env_name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value.strip():
            return resolve_bool_value(env_value)
    return False


def langsmith_tracing_forced_disabled() -> bool:
    router_value = os.environ.get(ROUTER_LANGSMITH_ENABLED_ENV_VAR)
    return router_value is not None and router_value.strip() and not resolve_bool_value(router_value)


def langsmith_api_key_configured() -> bool:
    return bool(
        os.environ.get("LANGSMITH_API_KEY", "").strip()
        or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    )


def langsmith_tracing_configured() -> bool:
    return _langsmith is not None and langsmith_tracing_requested() and langsmith_api_key_configured()


def langsmith_project_name() -> str:
    return (
        os.environ.get(ROUTER_LANGSMITH_PROJECT_ENV_VAR, "").strip()
        or os.environ.get("LANGSMITH_PROJECT", "").strip()
        or "super-router"
    )


def parse_langsmith_tags() -> List[str]:
    tags = ["super-router", "langgraph"]
    raw_tags = os.environ.get(ROUTER_LANGSMITH_TAGS_ENV_VAR, "")
    for raw_tag in raw_tags.split(","):
        tag = raw_tag.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def build_langsmith_metadata(state: RouterState) -> Dict[str, Any]:
    return {
        "component": "super-router",
        "run_id": state["run_id"],
        "planner_model": state["planner_model"],
        "judge_model": state["judge_model"],
        "pro_model": state["pro_model"],
        "flash_model": state["flash_model"],
        "pro_fallback_count": len(state["pro_fallback_models"]),
        "flash_fallback_count": len(state["flash_fallback_models"]),
        "flash_retry_budget": state["flash_retry_budget"],
        "task_chars": len(state["task"]),
    }


def process_langsmith_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if resolve_bool("ROUTER_LANGSMITH_HIDE_INPUTS"):
        return {}
    args = inputs.get("args")
    if not isinstance(args, (list, tuple)):
        args = []
    model = str(inputs.get("model") or (args[0] if len(args) > 0 else ""))
    prompt = str(inputs.get("prompt") or (args[1] if len(args) > 1 else ""))
    processed: Dict[str, Any] = {
        "model": model,
        "provider": langsmith_provider_name(model),
        "transport": model_transport_name(model),
        "timeout": inputs.get("timeout"),
        "num_predict": inputs.get("num_predict"),
        "temperature": inputs.get("temperature"),
        "usage_label": inputs.get("usage_label", ""),
        "prompt_chars": len(prompt),
    }
    if resolve_bool("ROUTER_LANGSMITH_TRACE_PROMPTS"):
        processed["prompt_preview"] = compact_text(prompt, 1000)
    return processed


def process_langsmith_model_outputs(output: Any) -> Dict[str, Any]:
    output_text = str(output.get("text", "")) if isinstance(output, dict) else str(output or "")
    usage_metadata = (
        output.get("usage_metadata", {})
        if isinstance(output, dict) and isinstance(output.get("usage_metadata"), dict)
        else {}
    )
    usage_source = str(output.get("usage_source", "")) if isinstance(output, dict) else ""
    if resolve_bool("ROUTER_LANGSMITH_HIDE_OUTPUTS"):
        processed_hidden: Dict[str, Any] = {}
        if usage_metadata:
            processed_hidden["usage_metadata"] = usage_metadata
        if usage_source:
            processed_hidden["usage_source"] = usage_source
        return processed_hidden
    processed: Dict[str, Any] = {"output_chars": len(output_text)}
    if usage_metadata:
        processed["usage_metadata"] = usage_metadata
    if usage_source:
        processed["usage_source"] = usage_source
    if resolve_bool("ROUTER_LANGSMITH_TRACE_OUTPUTS"):
        processed["output_preview"] = compact_text(output_text, 1000)
    return processed


def create_langsmith_client() -> Any | None:
    if _langsmith is None:
        return None
    client_cls = getattr(_langsmith, "Client", None)
    if client_cls is None:
        return None
    kwargs: Dict[str, Any] = {}
    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip() or os.environ.get("LANGCHAIN_API_KEY", "").strip()
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "").strip() or os.environ.get("LANGCHAIN_ENDPOINT", "").strip()
    workspace_id = os.environ.get("LANGSMITH_WORKSPACE_ID", "").strip()
    if api_key:
        kwargs["api_key"] = api_key
    if endpoint:
        kwargs["api_url"] = endpoint
    if workspace_id:
        kwargs["workspace_id"] = workspace_id
    if resolve_bool("ROUTER_LANGSMITH_HIDE_INPUTS"):
        kwargs["hide_inputs"] = True
    if resolve_bool("ROUTER_LANGSMITH_HIDE_OUTPUTS"):
        kwargs["hide_outputs"] = True
    try:
        return client_cls(**kwargs)
    except Exception as exc:
        print(f"[LangSmith] Failed to initialize client; continuing without telemetry: {compact_text(str(exc), 220)}")
        return None


@contextlib.contextmanager
def langsmith_tracing_context(state: RouterState) -> Iterator[None]:
    if langsmith_tracing_forced_disabled():
        tracing_context = getattr(_langsmith, "tracing_context", None) if _langsmith is not None else None
        if tracing_context is None:
            yield
        else:
            with tracing_context(enabled=False):
                yield
        return
    if not langsmith_tracing_requested():
        yield
        return
    if _langsmith is None:
        print("[LangSmith] Telemetry requested but the langsmith package is unavailable; continuing without tracing.")
        yield
        return
    tracing_context = getattr(_langsmith, "tracing_context", None)
    if tracing_context is None:
        print("[LangSmith] Telemetry requested but tracing_context is unavailable; continuing without tracing.")
        yield
        return
    if not langsmith_api_key_configured():
        print("[LangSmith] Telemetry requested but LANGSMITH_API_KEY is not set; continuing without tracing.")
        with tracing_context(enabled=False):
            yield
        return

    client = create_langsmith_client()
    context_kwargs: Dict[str, Any] = {
        "enabled": True,
        "project_name": langsmith_project_name(),
        "tags": parse_langsmith_tags(),
        "metadata": build_langsmith_metadata(state),
    }
    if client is not None:
        context_kwargs["client"] = client
    try:
        with tracing_context(**context_kwargs):
            yield
    finally:
        if client is not None and resolve_bool("ROUTER_LANGSMITH_FLUSH", True):
            try:
                client.flush()
            except Exception as exc:
                print(f"[LangSmith] Failed to flush traces: {compact_text(str(exc), 220)}")
