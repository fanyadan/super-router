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
from . import generation  # noqa: F401
from . import model_invocation  # noqa: F401



def planner_warmup_node(state: RouterState) -> Dict[str, Any]:
    if ROUTER_SKIP_WARMUP:
        print("[Node: Planner Warmup] ⏭️  Skipping warmup (ROUTER_SKIP_WARMUP=1)")
        return {"planner_warmup_attempt": 3, "status": "planner_warmup_skipped"}
    attempt = state["planner_warmup_attempt"] + 1
    if attempt == 1:
        print("\n[Node: Planner Warmup] 🔥 Warming up planner model with a LangGraph loop...")
    try:
        generation.generate_text(
            state["planner_model"],
            "OK",
            timeout=resolve_positive_int(None, ROUTER_WARMUP_TIMEOUT_ENV_VAR, DEFAULT_WARMUP_TIMEOUT),
            num_predict=4,
            usage_label=f"Planner warmup {attempt}",
        )
        print(f"[Planner Warmup] ✅ Ping {attempt}/3 successful")
    except Exception as exc:
        print(f"[Planner Warmup] ⚠️ Ping {attempt}/3 failed: {exc}")
    return {
        "planner_warmup_attempt": attempt,
        "status": f"planner_warmup_{attempt}_done",
    }


def route_after_planner_warmup(state: RouterState) -> str:
    if state["planner_warmup_attempt"] < 3:
        return "planner_warmup"
    return "planner_invoke"


def planner_invoke_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: Planner Invoke] 🧠 调用规划模型拆解任务...")
    try:
        planner_output_tokens = resolve_positive_int(
            None,
            ROUTER_PLANNER_MAX_OUTPUT_TOKENS_ENV_VAR,
            DEFAULT_PLANNER_MAX_OUTPUT_TOKENS,
        )
        raw_text = generation.generate_text(
            state["planner_model"],
            build_planner_prompt(state["task"]),
            timeout=resolve_positive_int(None, ROUTER_PLANNER_TIMEOUT_ENV_VAR, DEFAULT_PLANNER_TIMEOUT),
            num_predict=planner_output_tokens,
            usage_label="Planner invoke",
        )
        return {
            "planner_raw_text": raw_text,
            "planner_error": "",
            "status": "planner_invoked",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 260)
        print(f"⚠️ 规划模型调用异常：{error_text}")
        return {
            "planner_raw_text": "",
            "planner_error": error_text,
            "status": "planner_invoke_failed",
        }


def route_after_planner_invoke(state: RouterState) -> str:
    if state["status"] == "planner_invoke_failed":
        return "planner_fallback"
    return "planner_parse"


def planner_parse_node(state: RouterState) -> Dict[str, Any]:
    raw_text = state["planner_raw_text"]
    if router_debug_enabled():
        print(f"\n[DEBUG Planner Output]")
        print(f"  Model: {state['planner_model']}")
        print(f"  Raw text length: {len(raw_text)} chars")
        print(f"  First 400 chars: {raw_text[:400]}")
        if len(raw_text) > 400:
            print(f"  Last 200 chars: {raw_text[-200:]}")
        print(f"  Contains '[': { '[' in raw_text }, Contains ']': {']' in raw_text }")
        print(f"  ---\n")
    try:
        planned_subtasks = normalize_planned_subtasks(extract_first_json_array(raw_text))
        planned_subtasks = ensure_communication_subtask(state["task"], planned_subtasks)
        planned_subtasks = normalize_planned_subtasks(planned_subtasks)
        print(f"✅ 规划成功，拆解出 {len(planned_subtasks)} 个步骤。")
        return {
            "planned_subtasks": planned_subtasks,
            "planner_error": "",
            "status": "planner_parsed",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 260)
        print(f"⚠️ 规划输出解析异常：{error_text}")
        return {
            "planner_error": error_text,
            "status": "planner_parse_failed",
        }


def route_after_planner_parse(state: RouterState) -> str:
    if state["status"] == "planner_parse_failed":
        return "planner_fallback"
    return "dependency_judge"


def planner_fallback_node(state: RouterState) -> Dict[str, Any]:
    planned_subtasks = build_fallback_subtasks(state["task"])
    planned_subtasks = ensure_communication_subtask(state["task"], planned_subtasks)
    planned_subtasks = normalize_planned_subtasks(planned_subtasks)
    error_text = state["planner_error"] or "Unknown planner failure"
    print(f"⚠️ 规划器异常：{error_text}。已切换到启发式回退规划。")
    return {
        "planned_subtasks": planned_subtasks,
        "errors": [f"Planner fallback: {error_text}"],
        "status": "planner_fallback",
    }


def dependency_judge_node(state: RouterState) -> Dict[str, Any]:
    print("\n[Node: Dependency Judge] 🧭 验证子任务依赖关系...")
    try:
        judged_subtasks, confidence, issues, raw_text = judge_dependencies_with_model(
            state["task"],
            state["planned_subtasks"],
            state["judge_model"],
        )
        print(
            f"  依赖判定完成: subtasks={len(judged_subtasks)} "
            f"| confidence={confidence:.2f} | issues={len(issues)}"
        )
        return {
            "planned_subtasks": judged_subtasks,
            "dependency_raw_text": raw_text,
            "dependency_error": "",
            "dependency_issues": issues,
            "dependency_confidence": confidence,
            "history": [f"Dependency judge reviewed {len(judged_subtasks)} planned subtasks."],
            "status": "dependencies_judged",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 260)
        print(f"  依赖判定异常: {error_text}。保留规划器依赖并进入结构校验。")
        return {
            "dependency_error": error_text,
            "errors": [f"Dependency judge fallback: {error_text}"],
            "status": "dependency_judge_failed",
        }


def dependency_validate_node(state: RouterState) -> Dict[str, Any]:
    try:
        validated_subtasks = validate_dependency_graph(state["planned_subtasks"])
        dependency_edges = sum(len(subtask["depends_on"]) for subtask in validated_subtasks)
        print(
            f"\n[Node: Dependency Validate] ✅ DAG 校验通过: "
            f"subtasks={len(validated_subtasks)}, edges={dependency_edges}"
        )
        return {
            "planned_subtasks": validated_subtasks,
            "history": [
                f"Dependency graph validated with {len(validated_subtasks)} subtasks and {dependency_edges} edges."
            ],
            "status": "dependencies_validated",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 260)
        fallback_subtasks = make_serial_dependency_plan(state["planned_subtasks"])
        print(
            f"\n[Node: Dependency Validate] ⚠️ DAG 校验失败: {error_text}。"
            "已切换到保守串行依赖。"
        )
        return {
            "planned_subtasks": fallback_subtasks,
            "dependency_error": error_text,
            "dependency_issues": state["dependency_issues"] + [error_text],
            "errors": [f"Dependency validation fallback: {error_text}"],
            "history": ["Dependency graph invalid; switched to conservative serial execution order."],
            "status": "dependencies_validated_with_fallback",
        }


def planner_ready_node(state: RouterState) -> Dict[str, Any]:
    return {
        "subtasks": [],
        "current_step": 0,
        "history": [f"Planner produced {len(state['planned_subtasks'])} planned subtasks."],
        "status": "planned",
    }


def judge_warmup_node(state: RouterState) -> Dict[str, Any]:
    if ROUTER_SKIP_WARMUP:
        print("[Node: Judge Warmup] ⏭️  Skipping warmup (ROUTER_SKIP_WARMUP=1)")
        return {"judge_warmup_done": True, "status": "judge_warmup_skipped"}
    print("\n[Node: Judge Warmup] 🔥 Warming up judge model before LangGraph fanout...")
    try:
        generation.generate_text(
            state["judge_model"],
            "OK",
            timeout=resolve_positive_int(None, ROUTER_WARMUP_TIMEOUT_ENV_VAR, DEFAULT_WARMUP_TIMEOUT),
            num_predict=4,
            usage_label="Judge warmup",
        )
        print("[Judge Warmup] ✅ Warmup successful")
        return {
            "judge_warmup_done": True,
            "history": ["Judge warmup completed before fanout."],
            "status": "judge_warmed",
        }
    except Exception as exc:
        error_text = compact_text(str(exc), 220)
        print(f"[Judge Warmup] ⚠️ Warmup failed: {error_text}")
        return {
            "judge_warmup_done": False,
            "history": [f"Judge warmup failed before fanout: {error_text}"],
            "status": "judge_warmup_failed",
        }


def route_to_judge_subtasks(state: RouterState) -> List[Send] | str:
    if not state["planned_subtasks"]:
        return "assemble_plan"
    print("\n[Edge: Planner -> Judge Fanout] 🎯 为每个子任务分发独立 LangGraph 判定节点...")
    return [
        Send(
            "judge_subtask",
            {
                **state,
                "judge_index": index,
                "judge_desc": planned["desc"],
            },
        )
        for index, planned in enumerate(state["planned_subtasks"], start=1)
    ]


def judge_subtask_node(state: RouterState) -> Dict[str, Dict[int, JudgedSubtask]]:
    index = state["judge_index"]
    planned = state["planned_subtasks"][index - 1]
    desc = state["judge_desc"] or planned["desc"]
    print(f"\n[Node: Judge Subtask] 🎯 Step {index} 结构化复杂度评分...")
    error = ""
    try:
        assessment = score_subtask_with_model(state["task"], desc, state["judge_model"])
        print(
            f"  步骤 {index}: ✅ {assessment['final_route']} "
            f"(score={assessment['complexity_score']}, conf={assessment['confidence']:.2f})"
        )
    except Exception as exc:
        assessment = build_fallback_assessment(state["task"], desc)
        error = f"Judge fallback on step {index}: {exc}"
        print(f"  步骤 {index}: ⚠️ 判定异常 ({exc})，已切换到启发式评分。")

    return {
        "judge_results": {
            index: {
                "index": index,
                "subtask": build_subtask(planned, assessment),
                "error": error,
            }
        }
    }


def assemble_plan_node(state: RouterState) -> RouterState:
    print("\n[Node: Assemble Plan] 🧩 汇总 LangGraph 判定结果...")
    ordered_results = sorted(state["judge_results"].values(), key=lambda item: item["index"])
    subtasks = [result["subtask"] for result in ordered_results]
    errors = [result["error"] for result in ordered_results if result["error"]]
    display_plan(subtasks, state["planner_model"], state["judge_model"])
    return {
        "subtasks": subtasks,
        "judge_index": 0,
        "judge_desc": "",
        "history": [f"Judge assigned routes for {len(subtasks)} subtasks via LangGraph fanout."],
        "errors": errors,
        "status": "judged",
    }
