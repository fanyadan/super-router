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
from . import generation  # noqa: F401
from . import model_invocation  # noqa: F401



def build_fallback_report(state: RouterState, finalizer_error: str = "") -> str:
    lines = [
        "路由执行摘要",
        f"- 原始任务: {state['task']}",
        f"- 规划模型: {state['planner_model']}",
        f"- 判定模型: {state['judge_model']}",
        f"- PRO 模型: {state['pro_model']}",
        f"- FLASH 模型: {state['flash_model']}",
        f"- 已完成步骤数: {len(state['results'])}",
    ]
    for result in state["results"]:
        detail_bits: List[str] = []
        if result["planned_route"] != result["route"]:
            detail_bits.append(f"planned={result['planned_route']}")
        if result["retry_count"] > 0:
            detail_bits.append(f"retries={result['retry_count']}")
        if result["escalated_from_flash"]:
            detail_bits.append("escalated")
        if result["used_provider_fallback"]:
            detail_bits.append("provider_fallback")
        flash_review = result["flash_review"]
        if flash_review["failure_type"] != "none":
            detail_bits.append(f"failure={flash_review['failure_type']}")
        detail_suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        lines.append(
            f"  {result['step']}. [{result['route']}:{result['model_name']}] "
            f"{result['desc']}{detail_suffix} -> "
            f"{compact_text(result['output'], 140)}"
        )
        if flash_review["reason"]:
            lines.append(f"     cascade: {compact_text(flash_review['reason'], 140)}")
    if state["errors"] or finalizer_error:
        lines.append("- 回退/异常:")
        for error in state["errors"]:
            lines.append(f"  - {error}")
        if finalizer_error:
            lines.append(f"  - Finalizer fallback: {finalizer_error}")
    finalizer_outcome = state["finalizer_outcome"]
    if finalizer_outcome["status"] not in {"", "not_started"}:
        lines.append("- Finalizer path:")
        lines.append(
            f"  - route={finalizer_outcome['route']} model={finalizer_outcome['model_name'] or 'deterministic'} "
            f"status={finalizer_outcome['status']}"
        )
        if finalizer_outcome["reason"]:
            lines.append(f"  - reason: {finalizer_outcome['reason']}")
    return "\n".join(lines)


def build_routing_table(state: RouterState) -> str:
    """Deterministic per-step routing table.

    Model names are copied verbatim from each step's recorded ``model_name`` so the
    routing summary always reflects the exact model that was invoked (e.g.
    ``claude/claude-opus-4-8``) instead of a name paraphrased by the finalizer LLM.
    """
    results = sorted(state["results"], key=lambda item: item["step"])
    if not results:
        return ""
    lines = [
        "路由明细 / Routing detail (models copied verbatim as invoked)",
        "| Step | Subtask | Route | Model |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['step']} | {result['subtask_id']} | "
            f"{result['route']} | {result['model_name']} |"
        )
    return "\n".join(lines)


def has_distinct_finalizer_model_path(state: RouterState) -> bool:
    flash_candidates = [
        normalize_model_name(model)
        for model in dedupe_model_sequence(
            state["flash_model"], state["flash_fallback_models"]
        )
    ]
    pro_candidates = [
        normalize_model_name(model)
        for model in dedupe_model_sequence(
            state["pro_model"], state["pro_fallback_models"]
        )
    ]
    return any(candidate and candidate not in flash_candidates for candidate in pro_candidates)



def build_finalizer_prompt(state: RouterState, route: Literal["PRO", "FLASH"]) -> str:
    finalizer_context = build_finalizer_context_pack_json(state, route)
    role = "FLASH summarizer" if route == FLASH else "PRO summarizer"
    route_instruction = (
        "Write a concise final report with three sections: Routing Summary, Step Outcomes, Next Action. Use the Technical Metadata blocks for precise facts."
        if route == FLASH
        else "Write a concise but higher-signal final report with three sections: Routing Summary, Step Outcomes, Next Action. Use the compact execution results and Technical Metadata blocks for precision."
    )
    return (
        f"Role: {role}.\n"
        f"Finalizer context JSON:\n{finalizer_context}\n"
        f"{route_instruction}"
    )



def verify_finalizer_output(report: str) -> tuple[bool, str]:
    stripped = report.strip()
    if not stripped:
        return False, "Finalizer returned empty output."
    lowered = " ".join(stripped.split()).lower()
    if contains_any(lowered, LOW_QUALITY_OUTPUT_MARKERS):
        return False, "Finalizer output explicitly signaled inability to finish."
    if len(stripped) < 80:
        return False, "Finalizer output was too short for a useful final report."
    return True, "Finalizer output passed heuristic verification."


def flash_finalizer_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: FLASH Finalizer Invoke] 🧾 正在生成最终路由报告...")
    attempt_log: List[str] = []
    flash_invocation = model_invocation.invoke_with_provider_fallback(
        state["flash_model"],
        state["flash_fallback_models"],
        build_finalizer_prompt(state, FLASH),
        timeout=resolve_positive_int(
            None,
            "ROUTER_FINALIZER_TIMEOUT",
            DEFAULT_FLASH_FINALIZER_TIMEOUT,
        ),
        num_predict=320,
        temperature=0.0,
        label="Finalizer FLASH",
        attempt_log=attempt_log,
    )
    return {
        "finalizer_invocation_result": flash_invocation,
        "finalizer_attempt_log": flash_invocation["attempt_log"],
        "status": "flash_finalizer_invoked",
    }


def flash_finalizer_verify_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: FLASH Finalizer Verify] 验证 FLASH 最终报告...")
    flash_invocation = state["finalizer_invocation_result"]
    attempt_log = list(state["finalizer_attempt_log"])
    if flash_invocation["success"]:
        passed, reason = verify_finalizer_output(flash_invocation["output"])
        if passed:
            finalizer_outcome = {
                "route": FLASH,
                "model_name": flash_invocation["model_name"],
                "status": "finished",
                "used_provider_fallback": flash_invocation["used_provider_fallback"],
                "reason": reason,
                "attempt_log": attempt_log,
            }
            print(
                "✅ 最终报告生成成功。"
                f"{' (provider fallback)' if flash_invocation['used_provider_fallback'] else ''}"
            )
            return {
                "final_report": flash_invocation["output"],
                "finalizer_outcome": finalizer_outcome,
                "finalizer_attempt_log": attempt_log,
                "finalizer_error": "",
                "finalizer_flash_reason": reason,
                "status": "finalizer_finished",
            }

        attempt_log.append(f"Finalizer FLASH output rejected: {reason}")
        print(f"⚠️ FLASH finalizer 输出不足，准备升级到 PRO。原因: {reason}")
        return {
            "finalizer_attempt_log": attempt_log,
            "finalizer_error": f"FLASH finalizer rejected: {reason}",
            "finalizer_flash_reason": reason,
            "status": "finalizer_flash_rejected",
        }

    if (
        flash_invocation["failure_type"] != "capability_quality"
        and not has_distinct_finalizer_model_path(state)
    ):
        attempt_log.append(
            "Skipped PRO finalizer escalation because no distinct finalizer model path was available after FLASH failed."
        )
        finalizer_error = (
            f"FLASH finalizer failed: {flash_invocation['error_text']}; "
            "skipped redundant PRO cascade because PRO would reuse the same effective model path"
        )
        print("⚠️ FLASH finalizer 失败且 PRO 无新增模型路径，准备进入确定性回退报告节点。")
        return {
            "finalizer_attempt_log": attempt_log,
            "finalizer_error": finalizer_error,
            "finalizer_flash_reason": flash_invocation["error_text"],
            "status": "finalizer_deterministic_needed",
        }

    attempt_log.append(
        f"Finalizer FLASH failed with {flash_invocation['failure_type']}; escalating to PRO finalizer."
    )
    print(
        f"⚠️ FLASH finalizer 调用失败，准备升级到 PRO。原因: {flash_invocation['error_text']}"
    )
    return {
        "finalizer_attempt_log": attempt_log,
        "finalizer_error": f"FLASH finalizer failed: {flash_invocation['error_text']}",
        "finalizer_flash_reason": flash_invocation["error_text"],
        "status": "finalizer_flash_failed",
    }


def route_after_flash_finalizer_verify(state: RouterState) -> str:
    if state["status"] == "finalizer_finished":
        return "finalizer_complete"
    if state["status"] == "finalizer_deterministic_needed":
        return "deterministic_finalizer"
    return "pro_finalizer"



def extract_technical_metadata_node(state: RouterState) -> Dict[str, Any]:
    results = state.get("results", [])
    if not results:
        print("\n[Node: Metadata Extractor] 🔍 No recorded step available for metadata extraction.")
        return {"history": ["Metadata extraction skipped because no step result was recorded."]}

    latest_result = results[-1]
    return {"history": extract_technical_metadata_for_result(state, latest_result)}


def pro_finalizer_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: PRO Finalizer Invoke] 🧠 正在级联生成最终路由报告...")
    attempt_log = list(state["finalizer_attempt_log"])
    pro_invocation = model_invocation.invoke_with_provider_fallback(
        state["pro_model"],
        state["pro_fallback_models"],
        build_finalizer_prompt(state, PRO),
        timeout=resolve_positive_int(
            None,
            "ROUTER_FINALIZER_TIMEOUT",
            DEFAULT_PRO_FINALIZER_TIMEOUT,
        ),
        num_predict=420,
        temperature=0.0,
        label="Finalizer PRO",
        attempt_log=attempt_log,
    )
    return {
        "finalizer_invocation_result": pro_invocation,
        "finalizer_attempt_log": pro_invocation["attempt_log"],
        "status": "pro_finalizer_invoked",
    }


def pro_finalizer_verify_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: PRO Finalizer Verify] 验证 PRO 最终报告...")
    pro_invocation = state["finalizer_invocation_result"]
    attempt_log = list(state["finalizer_attempt_log"])
    flash_reason = state["finalizer_flash_reason"]

    if pro_invocation["success"]:
        pro_passed, pro_reason = verify_finalizer_output(pro_invocation["output"])
        if pro_passed:
            reason_prefix = (
                "Escalated after FLASH finalizer rejection"
                if "rejected" in state["finalizer_error"]
                else "Escalated after FLASH finalizer failure"
            )
            finalizer_outcome = {
                "route": PRO,
                "model_name": pro_invocation["model_name"],
                "status": "finished",
                "used_provider_fallback": pro_invocation["used_provider_fallback"],
                "reason": f"{reason_prefix}: {flash_reason}",
                "attempt_log": attempt_log,
            }
            print(
                "✅ 最终报告通过 PRO 级联生成成功。"
                f"{' (provider fallback)' if pro_invocation['used_provider_fallback'] else ''}"
            )
            return {
                "final_report": pro_invocation["output"],
                "finalizer_outcome": finalizer_outcome,
                "finalizer_attempt_log": attempt_log,
                "status": "finalizer_finished",
            }

        finalizer_error = f"{state['finalizer_error']}; PRO finalizer rejected: {pro_reason}"
        print("⚠️ PRO finalizer 输出不足，准备进入确定性回退报告节点。")
        return {
            "finalizer_attempt_log": attempt_log,
            "finalizer_error": finalizer_error,
            "status": "finalizer_deterministic_needed",
        }

    finalizer_error = f"{state['finalizer_error']}; PRO finalizer failed: {pro_invocation['error_text']}"
    print("⚠️ PRO finalizer 级联失败，准备进入确定性回退报告节点。")
    return {
        "finalizer_attempt_log": attempt_log,
        "finalizer_error": finalizer_error,
        "status": "finalizer_deterministic_needed",
    }


def route_after_pro_finalizer_verify(state: RouterState) -> str:
    if state["status"] == "finalizer_finished":
        return "finalizer_complete"
    return "deterministic_finalizer"


def deterministic_finalizer_node(state: RouterState) -> RouterState:
    print("\n[Node: Deterministic Finalizer] 🧾 使用确定性模板生成最终路由报告...")
    finalizer_error = state["finalizer_error"]
    final_report = build_fallback_report(state, finalizer_error)
    finalizer_outcome = {
        "route": "DETERMINISTIC",
        "model_name": "",
        "status": "deterministic_fallback",
        "used_provider_fallback": False,
        "reason": finalizer_error,
        "attempt_log": list(state["finalizer_attempt_log"]),
    }
    return {
        "final_report": final_report,
        "finalizer_outcome": finalizer_outcome,
        "status": "finalizer_finished",
    }


def finalizer_complete_node(state: RouterState) -> RouterState:
    token_usage = get_token_usage_records(state["run_id"])
    token_usage_summary = summarize_token_usage_records(token_usage)
    token_usage_history: List[str] = ["Finalizer completed."]
    routing_table = build_routing_table(state)
    final_report = state["final_report"]
    if routing_table:
        final_report = f"{final_report}\n\n{routing_table}"
    print("\n" + "=" * 58)
    print(final_report)
    print("-" * 58)
    print(format_token_usage_summary(token_usage_summary))
    try:
        ledger_path = persist_token_usage_ledger(
            token_usage,
            token_usage_summary,
            state=state,
        )
        if ledger_path:
            print(f"- Ledger: {ledger_path}")
            token_usage_history.append(f"Token usage ledger appended: {ledger_path}")
    except Exception as exc:
        error_text = compact_text(str(exc), 220)
        print(f"- Ledger write failed: {error_text}")
        token_usage_history.append(f"Token usage ledger write failed: {error_text}")
    print("=" * 58)

    return {
        "history": token_usage_history,
        "token_usage": token_usage,
        "token_usage_summary": token_usage_summary,
        "final_report": final_report,
        "status": "finished",
    }
