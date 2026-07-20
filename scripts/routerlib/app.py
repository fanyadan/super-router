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
from .model_invocation import *  # noqa: F401,F403
from .planning import *  # noqa: F401,F403
from .nodes_planner import *  # noqa: F401,F403
from .nodes_executor import *  # noqa: F401,F403
from .nodes_finalizer import *  # noqa: F401,F403
from .graph import *  # noqa: F401,F403



def resolve_graph_max_concurrency(explicit_value: int | None, state: RouterState) -> int | None:
    resolved = resolve_optional_positive_int(explicit_value, "ROUTER_MAX_CONCURRENCY")
    if resolved is not None:
        return resolved
    if is_large_model(state["judge_model"]):
        return 1
    return None


def build_graph_config(
    recursion_limit: int,
    max_concurrency: int | None = None,
    state: RouterState | None = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {"recursion_limit": recursion_limit}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if state is not None:
        config["run_name"] = "super-router"
        config["tags"] = parse_langsmith_tags()
        config["metadata"] = build_langsmith_metadata(state)
    return config


def prepare_router_run(
    user_task: str,
    *,
    planner_model: str | None = None,
    judge_model: str | None = None,
    pro_model: str | None = None,
    flash_model: str | None = None,
    pro_fallback_models: List[str] | None = None,
    flash_fallback_models: List[str] | None = None,
    flash_retry_budget: int | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
):
    graph = build_router_graph()
    initial_state = create_initial_state(
        user_task,
        planner_model=planner_model,
        judge_model=judge_model,
        pro_model=pro_model,
        flash_model=flash_model,
        pro_fallback_models=pro_fallback_models,
        flash_fallback_models=flash_fallback_models,
        flash_retry_budget=flash_retry_budget,
    )
    resolved_recursion_limit = resolve_positive_int(
        recursion_limit,
        "ROUTER_RECURSION_LIMIT",
        DEFAULT_ROUTER_RECURSION_LIMIT,
    )
    resolved_max_concurrency = resolve_graph_max_concurrency(max_concurrency, initial_state)
    return graph, initial_state, resolved_recursion_limit, resolved_max_concurrency


def create_initial_state(
    task: str,
    *,
    planner_model: str | None = None,
    judge_model: str | None = None,
    pro_model: str | None = None,
    flash_model: str | None = None,
    pro_fallback_models: List[str] | None = None,
    flash_fallback_models: List[str] | None = None,
    flash_retry_budget: int | None = None,
) -> RouterState:
    resolved_planner = resolve_model(planner_model, "ROUTER_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
    resolved_judge = resolve_model(judge_model, "ROUTER_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    resolved_pro = resolve_execution_model(pro_model, "ROUTER_PRO_MODEL", DEFAULT_PRO_MODEL)
    resolved_flash = resolve_execution_model(flash_model, "ROUTER_FLASH_MODEL", DEFAULT_FLASH_MODEL)
    resolved_pro_fallback_models = resolve_model_list(
        pro_fallback_models,
        "ROUTER_PRO_FALLBACK_MODELS",
    )
    resolved_flash_fallback_models = resolve_model_list(
        flash_fallback_models,
        "ROUTER_FLASH_FALLBACK_MODELS",
    )
    resolved_flash_retry_budget = resolve_non_negative_int(
        flash_retry_budget,
        "ROUTER_FLASH_RETRY_BUDGET",
        DEFAULT_FLASH_RETRY_BUDGET,
    )
    return {
        "run_id": str(uuid.uuid4()),
        "task": task,
        "planner_model": resolved_planner,
        "judge_model": resolved_judge,
        "pro_model": resolved_pro,
        "flash_model": resolved_flash,
        "pro_fallback_models": resolved_pro_fallback_models,
        "flash_fallback_models": resolved_flash_fallback_models,
        "flash_retry_budget": resolved_flash_retry_budget,
        "planned_subtasks": [],
        "planner_raw_text": "",
        "planner_error": "",
        "dependency_raw_text": "",
        "dependency_error": "",
        "dependency_issues": [],
        "dependency_confidence": 0.0,
        "planner_warmup_attempt": 0,
        "judge_warmup_done": False,
        "subtasks": [],
        "judge_index": 0,
        "judge_desc": "",
        "judge_results": {},
        "execution_index": 0,
        "execution_subtask": {},
        "execution_results": {},
        "execution_context_results": [],
        "current_step": 0,
        "active_subtask": {},
        "active_route": "",
        "active_model_name": "",
        "active_output": "",
        "active_last_error": "",
        "active_attempt_count": 0,
        "active_retry_count": 0,
        "active_escalated_from_flash": False,
        "active_used_provider_fallback": False,
        "active_flash_review": empty_flash_review(),
        "active_attempt_log": [],
        "active_invocation_result": empty_model_invocation_result(),
        "results": [],
        "history": [],
        "errors": [],
        "status": "created",
        "final_report": "",
        "finalizer_outcome": empty_finalizer_outcome(),
        "finalizer_attempt_log": [],
        "finalizer_error": "",
        "finalizer_flash_reason": "",
        "finalizer_invocation_result": empty_model_invocation_result(),
        "token_usage": [],
        "token_usage_summary": empty_token_usage_summary(),
    }


def unpack_stream_event(event: Any) -> tuple[str | None, Any]:
    if isinstance(event, tuple):
        if len(event) == 3:
            _, mode, payload = event
            return str(mode), payload
        if len(event) == 2:
            mode, payload = event
            return str(mode), payload
    return None, event


def summarize_stream_update(node_name: str, update: Any) -> str:
    if not isinstance(update, dict):
        return f"[LangGraph Stream] {node_name} completed | payload={compact_text(str(update), 120)}"

    details: List[str] = []
    if "status" in update:
        details.append(f"status={update['status']}")
    if "planned_subtasks" in update:
        details.append(f"planned_subtasks={len(update['planned_subtasks'])}")
    if "subtasks" in update:
        details.append(f"subtasks={len(update['subtasks'])}")
    if "execution_results" in update:
        details.append(f"execution_results={len(update['execution_results'])}")
    if "execution_context_results" in update:
        details.append(f"context_results={len(update['execution_context_results'])}")
    if "current_step" in update:
        details.append(f"current_step={update['current_step']}")
    if "active_route" in update and update["active_route"]:
        details.append(f"route={update['active_route']}")
    if "active_model_name" in update and update["active_model_name"]:
        details.append(f"model={update['active_model_name']}")
    active_subtask = update.get("active_subtask")
    if isinstance(active_subtask, dict) and active_subtask.get("desc"):
        details.append(f"subtask={compact_text(str(active_subtask['desc']), 80)}")
    flash_review = update.get("active_flash_review")
    if isinstance(flash_review, dict) and flash_review.get("decision"):
        details.append(
            "flash_review="
            f"{flash_review['decision']}/{flash_review.get('failure_type', 'none')}"
        )
    if "results" in update:
        details.append(f"results={len(update['results'])}")
    if "errors" in update and isinstance(update["errors"], list) and update["errors"]:
        details.append(f"errors={len(update['errors'])}")
    if "final_report" in update and str(update["final_report"]).strip():
        details.append("final_report=ready")
    if not details:
        details.append(f"updated={', '.join(update.keys()) or 'no_fields'}")
    return f"[LangGraph Stream] {node_name} completed | " + " | ".join(details)


def emit_stream_updates(event: Any) -> None:
    if not isinstance(event, dict):
        print(f"[LangGraph Stream] update={compact_text(str(event), 120)}")
        return
    for node_name, update in event.items():
        print(summarize_stream_update(node_name, update))


def observe_stream_event(
    event: Any,
    *,
    final_state: RouterState,
    on_update: Callable[[Any], None] = emit_stream_updates,
) -> RouterState:
    mode, payload = unpack_stream_event(event)
    if mode == "updates":
        on_update(payload)
        return final_state
    if mode == "values" and isinstance(payload, dict):
        return payload
    return final_state


def run_router_app(
    user_task: str,
    *,
    planner_model: str | None = None,
    judge_model: str | None = None,
    pro_model: str | None = None,
    flash_model: str | None = None,
    pro_fallback_models: List[str] | None = None,
    flash_fallback_models: List[str] | None = None,
    flash_retry_budget: int | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    stream: bool = False,
) -> RouterState:
    graph, initial_state, resolved_recursion_limit, resolved_max_concurrency = prepare_router_run(
        user_task,
        planner_model=planner_model,
        judge_model=judge_model,
        pro_model=pro_model,
        flash_model=flash_model,
        pro_fallback_models=pro_fallback_models,
        flash_fallback_models=flash_fallback_models,
        flash_retry_budget=flash_retry_budget,
        recursion_limit=recursion_limit,
        max_concurrency=max_concurrency,
    )
    graph_config = build_graph_config(
        resolved_recursion_limit,
        resolved_max_concurrency,
        initial_state,
    )
    if resolved_max_concurrency == 1 and is_large_model(initial_state["judge_model"]):
        print("[LangGraph Config] Large Judge model detected; max_concurrency=1 to avoid local model contention.")
    run_timeout = resolve_non_negative_int(
        None,
        ROUTER_RUN_TIMEOUT_ENV_VAR,
        DEFAULT_ROUTER_RUN_TIMEOUT,
    )
    with router_run_deadline_context(run_timeout):
        with token_usage_tracking_context(initial_state["run_id"]):
            with langsmith_tracing_context(initial_state):
                if not stream:
                    return graph.invoke(
                        initial_state,
                        config=graph_config,
                    )

                print("\n[LangGraph Stream] 🔄 节点级流式输出已启用。")
                final_state: RouterState = initial_state
                for event in graph.stream(
                    initial_state,
                    config=graph_config,
                    stream_mode=["updates", "values"],
                ):
                    final_state = observe_stream_event(event, final_state=final_state)
                    check_run_deadline()
                return final_state


def parse_cli_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph-based super-router."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream node-level LangGraph updates during execution.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help=f"Task description to route. If omitted, {ROUTER_TASK_ENV_VAR} is used when present.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    raw_args = argv if argv is not None else sys.argv[1:]
    parsed_args = parse_cli_args(raw_args)
    task = (
        " ".join(parsed_args.task).strip()
        or os.environ.get(ROUTER_TASK_ENV_VAR, "").strip()
    )
    if not task:
        raise SystemExit(
            f"Task description required. Provide positional args or set {ROUTER_TASK_ENV_VAR}."
        )
    run_router_app(task, stream=parsed_args.stream)
    return 0
