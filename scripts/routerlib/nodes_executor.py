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
from . import generation  # noqa: F401
from . import model_invocation  # noqa: F401



def extract_technical_metadata_for_result(state: RouterState, result: StepResult) -> List[str]:
    step_number = result["step"]
    step_status = result["status"]
    active_output = result["output"].strip()

    print(f"\n[Node: Metadata Extractor] 🔍 Extracting technical gold from Step {step_number}")

    if (
        not active_output
        or "failed" in step_status
        or step_status == "executor_fallback"
        or "exhausted" in step_status
    ):
        return [f"Step {step_number} metadata extraction skipped (no output or failure)."]

    metadata_context = build_metadata_context_pack_json(state, result, active_output)
    prompt = (
        f"Metadata context JSON:\n{metadata_context}\n\n"
        "Instruction: Extract the 'technical gold' from this output. "
        "Identify: 1. Key architectural decisions, 2. Specific library/tool choices, 3. Critical logic/algorithm details, "
        "4. CAP theorem trade-offs identified, 5. Final outcomes or verified results. "
        "Do not summarize generally; list these as atomic, high-precision facts. "
        "Return as a concise bulleted list."
    )

    invocation = model_invocation.invoke_with_provider_fallback(
        state["pro_model"],
        state["pro_fallback_models"],
        prompt,
        timeout=resolve_positive_int(None, ROUTER_METADATA_TIMEOUT_ENV_VAR, DEFAULT_METADATA_TIMEOUT),
        num_predict=800,
        temperature=0.0,
        label="Metadata Extractor",
        attempt_log=list(state["history"]),
    )

    metadata = invocation["output"] if invocation["success"] else "Metadata extraction failed."
    return [f"--- TECHNICAL METADATA STEP {step_number} ---\n{metadata}\n---"]


def route_to_parallel_executor_subtasks(state: RouterState) -> List[Send] | str:
    sends = [
        Send(
            "parallel_executor",
            {
                **state,
                "execution_index": index,
                "execution_subtask": subtask,
            },
        )
        for index, subtask in enumerate(state["subtasks"], start=1)
        if not is_deferred_execution_subtask(subtask)
    ]
    if not sends:
        return "parallel_execution_join"

    print(
        "\n[Edge: Plan -> Executor Fanout] 🚀 Dispatching "
        f"{len(sends)} independent subtasks with LangGraph concurrency..."
    )
    return sends


def completed_subtask_ids(state: RouterState) -> set[str]:
    return {
        str(result.get("subtask_id") or "")
        for result in state["execution_results"].values()
        if str(result.get("subtask_id") or "")
    }


def dependency_context_results_for_subtask(
    state: RouterState,
    subtask: Subtask,
) -> List[StepResult]:
    dependency_ids = set(subtask["depends_on"])
    if not dependency_ids:
        return []
    return [
        result
        for result in sorted(state["execution_results"].values(), key=lambda item: item["step"])
        if result.get("subtask_id") in dependency_ids
    ]


def dependency_scheduler_node(state: RouterState) -> Dict[str, Any]:
    completed_ids = completed_subtask_ids(state)
    remaining_count = sum(1 for subtask in state["subtasks"] if subtask["id"] not in completed_ids)
    return {
        "current_step": len(completed_ids),
        "history": [
            f"Dependency scheduler sees {len(completed_ids)} completed and {remaining_count} remaining subtasks."
        ],
        "status": "dependency_scheduling",
    }


def route_to_ready_executor_subtasks(state: RouterState) -> List[Send] | str:
    completed_ids = completed_subtask_ids(state)
    ready: List[tuple[int, Subtask]] = []
    remaining: List[Subtask] = []

    for index, subtask in enumerate(state["subtasks"], start=1):
        if subtask["id"] in completed_ids:
            continue
        remaining.append(subtask)
        if all(dependency_id in completed_ids for dependency_id in subtask["depends_on"]):
            ready.append((index, subtask))

    if not remaining:
        return "execution_finalize_join"
    if not ready:
        return "dependency_deadlock"

    print(
        "\n[Edge: Dependency Scheduler -> Executor Fanout] 🚀 Dispatching "
        f"{len(ready)} ready subtasks; completed={len(completed_ids)}, remaining={len(remaining)}."
    )
    return [
        Send(
            "parallel_executor",
            {
                **state,
                "execution_index": index,
                "execution_subtask": subtask,
            },
        )
        for index, subtask in ready
    ]


def route_to_deferred_executor_subtasks(state: RouterState) -> List[Send] | str:
    sends = [
        Send(
            "deferred_executor",
            {
                **state,
                "execution_index": index,
                "execution_subtask": subtask,
            },
        )
        for index, subtask in enumerate(state["subtasks"], start=1)
        if is_deferred_execution_subtask(subtask)
    ]
    if not sends:
        return "execution_finalize_join"

    print(
        "\n[Edge: Context Join -> Deferred Executor Fanout] 🧾 Dispatching "
        f"{len(sends)} synthesis/reporting subtasks after parallel context is ready..."
    )
    return sends


def invoke_parallel_executor_attempt(
    state: RouterState,
    index: int,
    subtask: Subtask,
    route: Literal["PRO", "FLASH"],
    attempt_count: int,
    retry_count: int,
    escalated_from_flash: bool,
    flash_review: FlashReviewResult,
    attempt_log: List[str],
) -> tuple[ModelInvocationResult, int]:
    model_name = state["pro_model"] if route == PRO else state["flash_model"]
    dependency_results = dependency_context_results_for_subtask(state, subtask)
    prompt_state = dict(state)
    prompt_state.update(
        {
            "results": dependency_results,
            "current_step": len(state["execution_results"]),
            "active_subtask": subtask,
            "active_route": route,
            "active_model_name": model_name,
            "active_output": "",
            "active_last_error": "",
            "active_attempt_count": attempt_count,
            "active_retry_count": retry_count,
            "active_escalated_from_flash": escalated_from_flash,
            "active_used_provider_fallback": False,
            "active_flash_review": flash_review,
            "active_attempt_log": list(attempt_log),
            "active_invocation_result": empty_model_invocation_result(),
        }
    )

    next_attempt_count = attempt_count + 1
    next_attempt_log = list(attempt_log)
    next_attempt_log.append(f"Attempt {next_attempt_count}: route={route} model={model_name}")
    invocation = model_invocation.invoke_with_provider_fallback(
        model_name,
        route_fallback_models(state, route),
        build_execution_prompt(prompt_state, route),
        timeout=resolve_executor_timeout(route),
        num_predict=450 if route == PRO else 240,
        temperature=0.0,
        label=f"{route} executor step {index}",
        attempt_log=next_attempt_log,
    )
    return invocation, next_attempt_count


def build_parallel_step_result(
    *,
    index: int,
    subtask: Subtask,
    planned_route: Literal["PRO", "FLASH"],
    final_route: Literal["PRO", "FLASH"],
    model_name: str,
    output: str,
    status: str,
    attempt_count: int,
    retry_count: int,
    escalated_from_flash: bool,
    used_provider_fallback: bool,
    flash_review: FlashReviewResult,
    attempt_log: List[str],
) -> StepResult:
    return {
        "step": index,
        "subtask_id": subtask["id"],
        "depends_on": list(subtask["depends_on"]),
        "planned_route": planned_route,
        "route": final_route,
        "model_name": model_name,
        "desc": subtask["desc"],
        "output": output,
        "status": status,
        "attempt_count": attempt_count,
        "retry_count": retry_count,
        "escalated_from_flash": escalated_from_flash,
        "used_provider_fallback": used_provider_fallback,
        "flash_review": flash_review,
        "attempt_log": attempt_log,
    }


def execute_subtask_in_parallel_branch(
    state: RouterState,
    index: int,
    subtask: Subtask,
) -> tuple[StepResult, List[str]]:
    planned_route = normalize_route(subtask.get("model"), default=PRO)
    route = planned_route
    retry_count = 0
    attempt_count = 0
    escalated_from_flash = False
    flash_review = empty_flash_review()
    attempt_log: List[str] = []
    errors: List[str] = []
    output = ""
    status = "executor_failed"
    model_name = state["pro_model"] if route == PRO else state["flash_model"]
    used_provider_fallback = False
    total_steps = len(state["subtasks"])

    icon = "🧠 [PRO]" if route == PRO else "⚡ [FLASH]"
    print(
        f"\n[Node: Parallel Executor] {icon} Step {index}/{total_steps} -> {model_name}"
    )
    print(
        f"  描述: {subtask['desc']} | score={subtask['assessment']['complexity_score']} "
        f"| conf={subtask['assessment']['confidence']:.2f}"
    )

    while True:
        invocation, attempt_count = invoke_parallel_executor_attempt(
            state,
            index,
            subtask,
            route,
            attempt_count,
            retry_count,
            escalated_from_flash,
            flash_review,
            attempt_log,
        )
        attempt_log = list(invocation["attempt_log"])
        model_name = invocation["model_name"] or model_name
        used_provider_fallback = invocation["used_provider_fallback"] if invocation["success"] else False

        if invocation["success"]:
            output = invocation["output"].strip()
            status = "executed_via_provider_fallback" if used_provider_fallback else "executed"
            last_error = ""
        else:
            output = ""
            status = "executor_failed"
            last_error = invocation["error_text"] or "Unknown execution failure"

        if route == FLASH:
            review = (
                verify_flash_output(subtask["desc"], output, flash_review, retry_count)
                if invocation["success"]
                else classify_flash_execution_failure(last_error)
            )
            attempt_log.append(
                f"FLASH review => decision={review['decision']} failure_type={review['failure_type']} reason={review['reason']}"
            )
            flash_review = review

            if review["decision"] == "record":
                break

            if review["decision"] == "retry":
                if retry_count < state["flash_retry_budget"]:
                    retry_count += 1
                    message = (
                        f"Retrying FLASH for step {index} "
                        f"({retry_count}/{state['flash_retry_budget']}) after {review['failure_type']} failure."
                    )
                    print(f"\n[Node: Parallel Retry Guard] 🔁 {message}")
                    attempt_log.append(message)
                    continue

                exhausted_action = (
                    "escalating to PRO"
                    if not escalated_from_flash
                    else "recording exhausted FLASH result"
                )
                exhausted_reason = (
                    f"{review['reason']} Retry budget exhausted after {retry_count} retr"
                    f"{'y' if retry_count == 1 else 'ies'}; {exhausted_action}."
                )
                flash_review = {
                    "decision": "escalate" if not escalated_from_flash else "record",
                    "failure_type": review["failure_type"],
                    "reason": exhausted_reason,
                }
                if not escalated_from_flash:
                    escalated_from_flash = True
                    route = PRO
                    model_name = state["pro_model"]
                    status = "escalated_to_pro"
                    message = (
                        f"FLASH retry budget exhausted for step {index} "
                        f"after {retry_count} retr{'y' if retry_count == 1 else 'ies'}; "
                        f"escalating to PRO because {review['failure_type']}: {review['reason']}"
                    )
                    print(f"\n[Node: Parallel Retry Guard] 🧠 {message}")
                    attempt_log.append(message)
                    continue

                if not output or output.startswith("FLASH executor fallback output:"):
                    output = (
                        f"FLASH execution failed after {retry_count} retr"
                        f"{'y' if retry_count == 1 else 'ies'} "
                        f"({review['failure_type']}): {exhausted_reason}"
                    )
                status = "flash_retry_exhausted"
                break

            escalated_from_flash = True
            route = PRO
            model_name = state["pro_model"]
            message = (
                f"Escalated step {index} from FLASH to PRO "
                f"because {review['failure_type']}: {review['reason']}"
            )
            print(f"\n[Node: Parallel Escalation] 🧠 {message}")
            attempt_log.append(message)
            continue

        if invocation["success"]:
            break

        error = f"PRO executor fallback on step {index}: {last_error}"
        errors.append(error)
        output = f"PRO executor fallback output: {subtask['desc']}"
        status = "executor_fallback"
        print(f"\n[Node: Parallel Executor Fallback] 🧯 {error}")
        break

    result = build_parallel_step_result(
        index=index,
        subtask=subtask,
        planned_route=planned_route,
        final_route=route,
        model_name=model_name,
        output=output,
        status=status,
        attempt_count=attempt_count,
        retry_count=retry_count,
        escalated_from_flash=escalated_from_flash,
        used_provider_fallback=used_provider_fallback,
        flash_review=flash_review,
        attempt_log=attempt_log,
    )
    print(f"[Node: Parallel Recorder] 已记录步骤 {index} -> {result['route']} ({result['model_name']})")
    return result, errors


def parallel_executor_node(state: RouterState) -> Dict[str, Any]:
    index = state["execution_index"]
    subtask = state["execution_subtask"]
    try:
        result, errors = execute_subtask_in_parallel_branch(state, index, subtask)
    except Exception as exc:
        error_text = compact_text(str(exc), 260)
        subtask = state["execution_subtask"]
        planned_route = normalize_route(subtask.get("model"), default=PRO)
        result = build_parallel_step_result(
            index=index,
            subtask=subtask,
            planned_route=planned_route,
            final_route=planned_route,
            model_name=state["pro_model"] if planned_route == PRO else state["flash_model"],
            output=f"{planned_route} executor fallback output: {subtask.get('desc', 'N/A')}",
            status="executor_fallback",
            attempt_count=0,
            retry_count=0,
            escalated_from_flash=False,
            used_provider_fallback=False,
            flash_review=empty_flash_review(),
            attempt_log=[f"Unhandled parallel executor exception: {error_text}"],
        )
        errors = [f"Parallel executor fallback on step {index}: {error_text}"]

    history = [f"Recorded parallel step {index}: {result['desc']}"]
    history.extend(extract_technical_metadata_for_result(state, result))
    return {
        "execution_results": {index: result},
        "history": history,
        "errors": errors,
    }


def parallel_execution_join_node(state: RouterState) -> RouterState:
    context_results = sorted(
        [
            result
            for index, result in state["execution_results"].items()
            if not is_deferred_execution_subtask(state["subtasks"][index - 1])
        ],
        key=lambda result: result["step"],
    )
    print(
        "\n[Node: Parallel Execution Join] 🧩 "
        f"Collected {len(context_results)} independent subtask results."
    )
    return {
        "execution_context_results": context_results,
        "current_step": len(context_results),
        "history": [f"Parallel executor completed {len(context_results)} independent subtasks."],
        "status": "parallel_executed",
    }


def dependency_execution_join_node(state: RouterState) -> RouterState:
    ordered_results = sorted(state["execution_results"].values(), key=lambda result: result["step"])
    completed_ids = completed_subtask_ids(state)
    print(
        "\n[Node: Dependency Execution Join] 🧩 "
        f"Collected {len(ordered_results)} completed dependency-aware subtask results."
    )
    return {
        "execution_context_results": ordered_results,
        "current_step": len(completed_ids),
        "history": [f"Dependency executor has completed {len(completed_ids)} subtasks."],
        "status": "dependency_wave_executed",
    }


def dependency_deadlock_node(state: RouterState) -> RouterState:
    completed_ids = completed_subtask_ids(state)
    fallback_results: Dict[int, StepResult] = {}
    errors: List[str] = []
    for index, subtask in enumerate(state["subtasks"], start=1):
        if subtask["id"] in completed_ids:
            continue
        missing = [
            dependency_id
            for dependency_id in subtask["depends_on"]
            if dependency_id not in completed_ids
        ]
        planned_route = normalize_route(subtask.get("model"), default=PRO)
        error = (
            f"Dependency deadlock on step {index} ({subtask['id']}): "
            f"missing dependencies {', '.join(missing) or 'unknown'}"
        )
        errors.append(error)
        fallback_results[index] = build_parallel_step_result(
            index=index,
            subtask=subtask,
            planned_route=planned_route,
            final_route=planned_route,
            model_name=state["pro_model"] if planned_route == PRO else state["flash_model"],
            output=f"Dependency scheduler fallback output: {error}",
            status="dependency_deadlock",
            attempt_count=0,
            retry_count=0,
            escalated_from_flash=False,
            used_provider_fallback=False,
            flash_review=empty_flash_review(),
            attempt_log=[error],
        )
    print(
        "\n[Node: Dependency Deadlock] ⚠️ "
        f"Recorded {len(fallback_results)} fallback results for blocked subtasks."
    )
    return {
        "execution_results": fallback_results,
        "errors": errors,
        "history": ["Dependency scheduler deadlock fallback recorded blocked subtasks."],
        "status": "dependency_deadlock",
    }


def execution_finalize_join_node(state: RouterState) -> RouterState:
    ordered_results = sorted(state["execution_results"].values(), key=lambda result: result["step"])
    print(
        "\n[Node: Execution Final Join] ✅ "
        f"Collected {len(ordered_results)} total subtask results; entering finalizer."
    )
    return {
        "results": ordered_results,
        "current_step": len(ordered_results),
        "execution_index": 0,
        "execution_subtask": {},
        "history": [f"Executor fanout joined {len(ordered_results)} total subtask results."],
        "status": "ready_to_finalize",
    }


def dispatch_node(state: RouterState) -> RouterState:
    total_steps = len(state["subtasks"])
    if state["current_step"] >= total_steps:
        print("\n[Node: Dispatcher] ✅ 所有步骤已完成，进入最终汇总节点。")
        return {
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
            "status": "ready_to_finalize",
        }

    subtask = state["subtasks"][state["current_step"]]
    route = subtask["model"]
    model_name = state["pro_model"] if route == PRO else state["flash_model"]
    icon = "🧠 [PRO]" if route == PRO else "⚡ [FLASH]"
    assessment = subtask["assessment"]

    print(
        f"\n[Node: Dispatcher] {icon} Step {state['current_step'] + 1}/{total_steps} "
        f"-> {model_name}"
    )
    print(
        f"  描述: {subtask['desc']} | score={assessment['complexity_score']} "
        f"| conf={assessment['confidence']:.2f}"
    )

    return {
        "active_subtask": subtask,
        "active_route": route,
        "active_model_name": model_name,
        "active_output": "",
        "active_last_error": "",
        "active_attempt_count": 0,
        "active_retry_count": 0,
        "active_escalated_from_flash": False,
        "active_used_provider_fallback": False,
        "active_flash_review": empty_flash_review(),
        "active_attempt_log": [],
        "active_invocation_result": empty_model_invocation_result(),
        "history": [f"Dispatched step {state['current_step'] + 1} to {route} using {model_name}."],
        "status": "dispatched",
    }


def route_after_dispatch(state: RouterState) -> str:
    if state["status"] == "ready_to_finalize":
        return "flash_finalizer"
    return "pro_executor" if state["active_route"] == PRO else "flash_executor"


def build_execution_prompt(state: RouterState, route: Literal["PRO", "FLASH"]) -> str:
    context_pack = build_executor_context_pack_json(state, route)
    escalation_context = ""
    if route == PRO and state["active_escalated_from_flash"]:
        flash_review = state["active_flash_review"]
        escalation_context = (
            "Escalation context: this step was first attempted on FLASH and then escalated to PRO. "
            f"failure_type={flash_review['failure_type']}, retries={state['active_retry_count']}, "
            f"reason={flash_review['reason']}\n"
        )

    mode_instruction = (
        "Think carefully and provide a high-signal technical result."
        if route == PRO
        else "Respond quickly and concisely with the operational or summary result."
    )
    return (
        f"Role: {route} task executor.\n"
        f"Execution mode: {mode_instruction}\n"
        f"Execution context JSON:\n{context_pack}\n"
        f"{escalation_context}"
        "Return only the result for the current subtask. No markdown fences."
    )


def invoke_executor_with_route(state: RouterState, route: Literal["PRO", "FLASH"]) -> Dict[str, Any]:
    node_label = "PRO Executor" if route == PRO else "FLASH Executor"
    model_name = state["active_model_name"] or (
        state["pro_model"] if route == PRO else state["flash_model"]
    )
    print(f"\n[Node: {node_label} Invoke] 开始执行子任务...")

    attempt_log = list(state["active_attempt_log"])
    attempt_count = state["active_attempt_count"] + 1
    attempt_log.append(f"Attempt {attempt_count}: route={route} model={model_name}")
    invocation = model_invocation.invoke_with_provider_fallback(
        model_name,
        route_fallback_models(state, route),
        build_execution_prompt(state, route),
        timeout=resolve_executor_timeout(route),
        num_predict=450 if route == PRO else 240,
        temperature=0.0,
        label=f"{route} executor step {state['current_step'] + 1}",
        attempt_log=attempt_log,
    )
    return {
        "active_invocation_result": invocation,
        "active_attempt_count": attempt_count,
        "active_attempt_log": invocation["attempt_log"],
        "status": "executor_invoked",
    }


def executor_result_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: Executor Result] Classifying executor invocation result...")
    invocation = state["active_invocation_result"]
    model_name = state["active_model_name"] or (
        state["pro_model"] if state["active_route"] == PRO else state["flash_model"]
    )
    if invocation["success"]:
        output = invocation["output"]
        status = "executed_via_provider_fallback" if invocation["used_provider_fallback"] else "executed"
        print(
            f"  状态: 模型执行成功。"
            f"{' (provider fallback)' if invocation['used_provider_fallback'] else ''}"
        )
    else:
        output = ""
        status = "executor_failed"
        print(f"  状态: 执行异常 ({invocation['error_text']})，交由 LangGraph 回退路径处理。")

    return {
        "active_output": output.strip(),
        "active_last_error": invocation["error_text"],
        "active_model_name": invocation["model_name"] if invocation["success"] else model_name,
        "active_used_provider_fallback": invocation["used_provider_fallback"] if invocation["success"] else False,
        "active_attempt_log": invocation["attempt_log"],
        "history": [
            f"Executed step {state['current_step'] + 1} with route {state['active_route']} using {invocation['model_name'] if invocation['success'] else model_name}."
        ],
        "status": status,
    }


def pro_executor_node(state: RouterState) -> RouterState:
    return invoke_executor_with_route(state, PRO)


def flash_executor_node(state: RouterState) -> RouterState:
    return invoke_executor_with_route(state, FLASH)


def route_after_executor_result(state: RouterState) -> str:
    if state["status"] == "executor_failed":
        # RESILIENCE FIX: If PRO fails, try to la-concisely rescue with FLASH before giving up
        if state["active_route"] == PRO:
            return "flash_review" 
        return "flash_review"
    if state["active_route"] == FLASH:
        return "flash_review"
    return "recorder"


def executor_fallback_node(state: RouterState) -> Dict[str, Any]:
    route = state["active_route"] or PRO
    output = f"{route} executor fallback output: {state['active_subtask'].get('desc', 'N/A')}"
    error = (
        f"{route} executor fallback on step {state['current_step'] + 1}: "
        f"{state['active_last_error'] or 'Unknown execution failure'}"
    )
    print(f"\n[Node: Executor Fallback] 🧯 {error}")
    return {
        "active_output": output,
        "errors": [error],
        "history": [f"Executor fallback produced deterministic output for step {state['current_step'] + 1}."],
        "status": "executor_fallback",
    }


def flash_review_node(state: RouterState) -> RouterState:
    print("\n[Node: Flash Review] 🧪 正在验证 FLASH 结果并分类失败原因...")
    attempt_log = list(state["active_attempt_log"])
    prior_review = state["active_flash_review"]

    if state["status"] in {"executed", "executed_via_provider_fallback"}:
        review = verify_flash_output(
            state["active_subtask"].get("desc", ""),
            state["active_output"],
            prior_review,
            state["active_retry_count"],
        )
        if review["decision"] == "record":
            status = "flash_verified"
            print("  状态: FLASH 输出通过验证。")
        else:
            status = "flash_needs_escalation"
            print(f"  状态: FLASH 输出需要升级到 PRO。原因: {review['reason']}")
    else:
        review = classify_flash_execution_failure(
            state["active_last_error"] or state["active_output"] or "Unknown FLASH failure"
        )
        status = "flash_retry_candidate" if review["decision"] == "retry" else "flash_needs_escalation"
        if review["decision"] == "retry":
            print(f"  状态: FLASH 失败被判定为可重试。原因: {review['reason']}")
        else:
            print(f"  状态: FLASH 失败被判定应升级到 PRO。原因: {review['reason']}")

    attempt_log.append(
        f"FLASH review => decision={review['decision']} failure_type={review['failure_type']} reason={review['reason']}"
    )
    return {
        "active_flash_review": review,
        "active_attempt_log": attempt_log,
        "history": [
            f"Flash review for step {state['current_step'] + 1}: decision={review['decision']} failure_type={review['failure_type']}."
        ],
        "status": status,
    }


def route_after_flash_review(state: RouterState) -> str:
    decision = state["active_flash_review"]["decision"]
    if decision == "retry":
        return "retry_guard"
    if decision == "escalate":
        return "escalation"
    return "recorder"


def retry_guard_node(state: RouterState) -> Dict[str, Any]:
    review = state["active_flash_review"]
    retries_used = state["active_retry_count"]
    retry_budget = state["flash_retry_budget"]
    attempt_log = list(state["active_attempt_log"])

    if retries_used < retry_budget:
        next_retry_count = retries_used + 1
        message = (
            f"Retrying FLASH for step {state['current_step'] + 1} "
            f"({next_retry_count}/{retry_budget}) after {review['failure_type']} failure."
        )
        print(f"\n[Node: Retry Guard] 🔁 {message}")
        attempt_log.append(message)
        return {
            "active_retry_count": next_retry_count,
            "active_attempt_log": attempt_log,
            "history": [message],
            "status": "flash_retrying",
        }

    exhausted_reason = (
        f"{review['reason']} Retry budget exhausted after {retries_used} retr"
        f"{'y' if retries_used == 1 else 'ies'}; escalating to PRO."
    )
    print(
        "\n[Node: Retry Guard] 🧠 FLASH retry budget exhausted，升级到 PRO 执行当前步骤。"
    )
    attempt_log.append(exhausted_reason)
    review: FlashReviewResult = {
        "decision": "escalate",
        "failure_type": review["failure_type"],
        "reason": exhausted_reason,
    }
    return {
        "active_flash_review": review,
        "active_attempt_log": attempt_log,
        "history": [f"FLASH retry budget exhausted on step {state['current_step'] + 1}; escalating to PRO."],
        "status": "flash_needs_escalation",
    }


def route_after_retry_guard(state: RouterState) -> str:
    if state["status"] == "flash_retrying":
        return "retry_flash"
    if state["status"] == "flash_needs_escalation":
        return "escalation"
    return "recorder"


def retry_flash_node(state: RouterState) -> RouterState:
    print("\n[Node: Retry FLASH] ⚡ 准备重新执行 FLASH 子任务...")
    return {
        "active_output": "",
        "active_last_error": "",
        "history": [f"Retrying FLASH executor on step {state['current_step'] + 1} after review classification."],
        "status": "retrying_flash",
    }


def escalation_node(state: RouterState) -> RouterState:
    review = state["active_flash_review"]
    print(
        "\n[Node: Escalation] 🧠 FLASH 结果不足或不适配，升级到 PRO 执行..."
    )
    attempt_log = list(state["active_attempt_log"])
    message = (
        f"Escalated step {state['current_step'] + 1} from FLASH to PRO "
        f"because {review['failure_type']}: {review['reason']}"
    )
    attempt_log.append(message)
    return {
        "active_route": PRO,
        "active_model_name": state["pro_model"],
        "active_escalated_from_flash": True,
        "active_used_provider_fallback": False,
        "active_attempt_log": attempt_log,
        "history": [message],
        "status": "escalated_to_pro",
    }


def record_step_node(state: RouterState) -> RouterState:
    step_number = state["current_step"] + 1
    route = state["active_route"] or PRO
    planned_route = normalize_route(state["active_subtask"].get("model"), default=route)
    model_name = state["active_model_name"] or (
        state["pro_model"] if route == PRO else state["flash_model"]
    )
    desc = state["active_subtask"].get("desc", "N/A")
    result: StepResult = {
        "step": step_number,
        "subtask_id": str(state["active_subtask"].get("id") or f"S{step_number}"),
        "depends_on": list(state["active_subtask"].get("depends_on", [])),
        "planned_route": planned_route,
        "route": PRO if route == PRO else FLASH,
        "model_name": model_name,
        "desc": desc,
        "output": state["active_output"],
        "status": state["status"],
        "attempt_count": state["active_attempt_count"],
        "retry_count": state["active_retry_count"],
        "escalated_from_flash": state["active_escalated_from_flash"],
        "used_provider_fallback": state["active_used_provider_fallback"],
        "flash_review": state["active_flash_review"],
        "attempt_log": list(state["active_attempt_log"]),
    }

    print(
        f"[Node: Recorder] 已记录步骤 {step_number} -> {result['route']} ({result['model_name']})"
    )

    return {
        "results": [result],
        "history": [f"Recorded step {step_number}: {desc}"],
        "current_step": step_number,
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
        "status": "recorded",
    }
