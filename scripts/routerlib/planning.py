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
from . import generation  # noqa: F401



def find_communication_clause(text: str) -> tuple[int, str] | None:
    patterns = (
        r"(?:并|并且|再|然后|最后)\s*((?:把[^。；;]{0,120}?)?(?:整理|准备|撰写|输出|生成|写|发送|同步|汇总|总结|概述|汇报)[^。；;]*)",
        r"(?:and|then|finally)\s+((?:prepare|write|draft|generate|send|summarize|report|share)[^.;;]*)",
        r"((?:整理|准备|撰写|输出|生成|写|发送|同步|汇总|总结|概述|汇报)[^。；;]*?(?:状态更新|影响说明|风险说明|行动摘要|摘要|总结|报告|简报|说明))",
        r"((?:prepare|write|draft|generate|send|summarize|report|share)[^.;;]*?(?:status update|impact note|risk note|action summary|summary|report))",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        clause = match.group(1).strip(" ，,；;。:：")
        if clause and is_summary_like_subtask(clause):
            return match.start(1), clause
    return None


def extract_communication_audience_markers(text: str) -> List[str]:
    lowered = text.lower()
    markers: List[str] = []
    for keyword in COMMUNICATION_AUDIENCE_KEYWORDS:
        if keyword.lower() in lowered:
            markers.append(keyword)
    return markers


def default_communication_subtask(task: str) -> PlannedSubtask:
    lowered = task.lower()
    if "值班" in task:
        return {"desc": "整理一份发给值班同事的简短中文行动摘要。"}
    if "产品经理" in task:
        return {"desc": "整理一段给产品经理的简短中文影响说明。"}
    if "运营" in task:
        return {"desc": "整理一份发给运营同事的简短中文影响说明。"}
    if "项目负责人" in task or "负责人" in task:
        return {"desc": "整理一段给项目负责人的简短中文风险说明。"}
    if "团队" in task:
        return {"desc": "整理一段给团队的简短中文状态更新。"}
    if "on-call" in lowered:
        return {"desc": "Prepare a concise action summary for the on-call engineer."}
    if "manager" in lowered or "pm" in lowered:
        return {"desc": "Prepare a concise impact note for the manager."}
    return {"desc": "整理面向相关方的简短中文总结或状态更新。"}


def split_mixed_planned_subtask(description: str) -> List[PlannedSubtask]:
    match = find_communication_clause(description)
    if not match or not has_non_summary_work_hint(description):
        return [{"desc": description}]

    clause_start, communication_clause = match
    base_desc = description[:clause_start].strip(" ，,；;。")
    if not base_desc or base_desc == communication_clause:
        return [{"desc": description}]
    return [{"desc": base_desc}, {"desc": communication_clause}]


def ensure_communication_subtask(task: str, planned_subtasks: List[PlannedSubtask]) -> List[PlannedSubtask]:
    expanded: List[PlannedSubtask] = []
    for subtask in planned_subtasks:
        expanded.extend(split_mixed_planned_subtask(subtask["desc"]))

    match = find_communication_clause(task)
    summary_indices = [
        index for index, subtask in enumerate(expanded) if is_summary_like_subtask(subtask["desc"])
    ]
    if summary_indices:
        if match:
            task_clause = match[1]
            audience_markers = extract_communication_audience_markers(task_clause)
            if audience_markers and not any(
                any(marker.lower() in expanded[index]["desc"].lower() for marker in audience_markers)
                for index in summary_indices
            ):
                expanded[summary_indices[-1]] = {"desc": task_clause}
        return expanded
    if match:
        expanded.append({"desc": match[1]})
        return expanded

    if is_summary_like_subtask(task):
        expanded.append(default_communication_subtask(task))
    return expanded


def extract_first_json_array(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    raise ValueError("No valid JSON array found in planner output")


def extract_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No valid JSON object found in judge output")


def normalize_dependency_id(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_")
    return normalized or fallback


def normalize_dependency_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        stripped = value.strip()
        raw_items = re.split(r"[\s,]+", stripped) if stripped else []
    else:
        raw_items = [value]

    dependencies: List[str] = []
    for item in raw_items:
        dependency_id = normalize_dependency_id(item, "")
        if dependency_id and dependency_id not in dependencies:
            dependencies.append(dependency_id)
    return dependencies


def dependency_reason_from_raw(raw: Any, default: str) -> str:
    if isinstance(raw, dict):
        reason = (
            raw.get("dependency_reason")
            or raw.get("depends_reason")
            or raw.get("reason")
            or raw.get("dependency")
        )
        if reason:
            return compact_text(str(reason), 220)
    return default


def normalize_planned_subtasks(raw_subtasks: List[Any]) -> List[PlannedSubtask]:
    pending: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    id_aliases: Dict[str, str] = {}

    for item in raw_subtasks:
        if isinstance(item, dict):
            desc = str(item.get("desc") or item.get("description") or item.get("step") or "").strip()
            raw_id = item.get("id") or item.get("step_id") or item.get("name")
            raw_depends_on = item.get("depends_on") or item.get("dependencies") or item.get("requires")
        else:
            desc = str(item).strip()
            raw_id = None
            raw_depends_on = []

        if not desc:
            continue

        fallback_id = f"S{len(pending) + 1}"
        subtask_id = normalize_dependency_id(raw_id, fallback_id)
        if subtask_id in used_ids:
            subtask_id = fallback_id
            suffix = 2
            while subtask_id in used_ids:
                subtask_id = f"{fallback_id}_{suffix}"
                suffix += 1
        used_ids.add(subtask_id)
        if raw_id:
            id_aliases[str(raw_id).strip()] = subtask_id
            id_aliases[normalize_dependency_id(raw_id, subtask_id)] = subtask_id

        pending.append(
            {
                "id": subtask_id,
                "desc": desc,
                "raw_depends_on": raw_depends_on,
                "dependency_reason": dependency_reason_from_raw(
                    item,
                    "Dependency not specified by planner.",
                ),
            }
        )

    normalized: List[PlannedSubtask] = []
    for item in pending:
        dependencies: List[str] = []
        for dependency_id in normalize_dependency_list(item["raw_depends_on"]):
            resolved_dependency_id = id_aliases.get(dependency_id, dependency_id)
            if resolved_dependency_id != item["id"] and resolved_dependency_id not in dependencies:
                dependencies.append(resolved_dependency_id)
        normalized.append(
            {
                "id": item["id"],
                "desc": item["desc"],
                "depends_on": dependencies,
                "dependency_reason": item["dependency_reason"],
            }
        )

    if not normalized:
        raise ValueError("Planner did not return any usable subtasks")
    return normalized


def validate_dependency_graph(subtasks: List[PlannedSubtask]) -> List[PlannedSubtask]:
    ids = [subtask["id"] for subtask in subtasks]
    duplicate_ids = sorted({subtask_id for subtask_id in ids if ids.count(subtask_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate dependency ids: {', '.join(duplicate_ids)}")

    known_ids = set(ids)
    for subtask in subtasks:
        if not subtask["id"]:
            raise ValueError("Dependency graph contains an empty subtask id")
        missing = [dependency_id for dependency_id in subtask["depends_on"] if dependency_id not in known_ids]
        if missing:
            raise ValueError(
                f"Subtask {subtask['id']} references unknown dependencies: {', '.join(missing)}"
            )
        if subtask["id"] in subtask["depends_on"]:
            raise ValueError(f"Subtask {subtask['id']} depends on itself")

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {subtask["id"]: subtask for subtask in subtasks}

    def visit(subtask_id: str, path: List[str]) -> None:
        if subtask_id in visited:
            return
        if subtask_id in visiting:
            cycle = " -> ".join(path + [subtask_id])
            raise ValueError(f"Dependency cycle detected: {cycle}")
        visiting.add(subtask_id)
        for dependency_id in by_id[subtask_id]["depends_on"]:
            visit(dependency_id, path + [subtask_id])
        visiting.remove(subtask_id)
        visited.add(subtask_id)

    for subtask_id in ids:
        visit(subtask_id, [])

    return subtasks


def make_serial_dependency_plan(
    subtasks: List[PlannedSubtask],
    *,
    reason: str = "Conservative serial fallback after invalid dependency judgment.",
) -> List[PlannedSubtask]:
    serial_subtasks: List[PlannedSubtask] = []
    previous_id = ""
    for subtask in subtasks:
        serial_subtasks.append(
            {
                **subtask,
                "depends_on": [previous_id] if previous_id else [],
                "dependency_reason": reason if previous_id else "First serial fallback step has no dependency.",
            }
        )
        previous_id = subtask["id"]
    return serial_subtasks


def normalize_dependency_judgment(
    raw: Dict[str, Any],
    planned_subtasks: List[PlannedSubtask],
) -> tuple[List[PlannedSubtask], float, List[str]]:
    raw_subtasks = raw.get("subtasks") or raw.get("planned_subtasks") or raw.get("plan")
    raw_dependencies = raw.get("dependencies")
    updates: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_subtasks, list):
        for item in raw_subtasks:
            if not isinstance(item, dict):
                continue
            subtask_id = normalize_dependency_id(
                item.get("id") or item.get("step_id") or item.get("name"),
                "",
            )
            if subtask_id:
                updates[subtask_id] = item

    if isinstance(raw_dependencies, dict):
        for raw_id, raw_depends_on in raw_dependencies.items():
            subtask_id = normalize_dependency_id(raw_id, "")
            if not subtask_id:
                continue
            updates.setdefault(subtask_id, {})["depends_on"] = raw_depends_on

    judged: List[PlannedSubtask] = []
    for planned in planned_subtasks:
        update = updates.get(planned["id"], {})
        dependency_value = planned["depends_on"]
        for key in ("depends_on", "dependencies", "requires"):
            if key in update:
                dependency_value = update[key]
                break
        dependencies = normalize_dependency_list(dependency_value)
        dependencies = [
            dependency_id
            for dependency_id in dependencies
            if dependency_id != planned["id"]
        ]
        judged.append(
            {
                **planned,
                "depends_on": dependencies,
                "dependency_reason": dependency_reason_from_raw(
                    update,
                    planned["dependency_reason"],
                ),
            }
        )

    confidence = clamp_float(raw.get("confidence"), 0.0, 1.0, default=0.5)
    raw_issues = raw.get("issues")
    issues = [
        compact_text(str(issue), 220)
        for issue in raw_issues
        if str(issue).strip()
    ] if isinstance(raw_issues, list) else []
    return judged, confidence, issues


def build_fallback_subtasks(task: str) -> List[PlannedSubtask]:
    lowered = task.lower()
    subtasks: List[Any] = []

    if any(
        keyword in lowered
        for keyword in (
            "analy",
            "debug",
            "fix",
            "refactor",
            "rewrite",
            "implement",
            "design",
            "logic",
            "分析",
            "调试",
            "修复",
            "重构",
            "实现",
            "设计",
            "逻辑",
        )
    ):
        subtasks.append({"desc": f"分析任务中的核心逻辑与高风险部分: {task}"})
        subtasks.append({"desc": f"执行核心实现或修复步骤: {task}"})
    else:
        subtasks.append({"desc": task})

    if any(
        keyword in lowered
        for keyword in (
            "summary",
            "summarize",
            "report",
            "pdf",
            "status",
            "save",
            "document",
            "总结",
            "报告",
            "保存",
            "整理",
        )
    ):
        subtasks.append({"desc": "整理执行结果并生成最终总结或输出。"})

    return normalize_planned_subtasks(subtasks)


def build_planner_prompt(task: str) -> str:
    task_limit = resolve_positive_int(
        None,
        ROUTER_PLANNER_TASK_CHAR_LIMIT_ENV_VAR,
        DEFAULT_PLANNER_TASK_CHAR_LIMIT,
    )
    planner_context = build_planner_context_manifest(task, task_limit)
    original_chars = len(" ".join(task.strip().split()))
    compaction_note = ""
    if len(planner_context) < original_chars:
        compaction_note = (
            f"Planner context manifest compacted from {original_chars} to {len(planner_context)} chars. "
            "Plan from the manifest JSON's objective, entities, constraints, deliverables, evidence requirements, "
            "and decomposition hints; do not copy omitted context into every subtask.\n"
        )
    return (
        "Role: Task decomposition planner.\n"
        "Goal: produce the execution plan only; do not solve the task.\n"
        f"{compaction_note}"
        "Planner context manifest JSON:\n"
        f"{planner_context}\n"
        "Rules:\n"
        "- Split by independent entity/component/file/provider/region/backend/technical area.\n"
        "- Separate investigation/implementation/validation/reporting when mixed.\n"
        "- Preserve deliverables, constraints, and needed evidence.\n"
        "- Use stable ids S1, S2, S3.\n"
        "- depends_on only when earlier outputs/evidence/decisions are required; otherwise [].\n"
        "- No model labels, complexity labels, or scores.\n"
        "Return raw JSON only: [{\"id\":\"S1\",\"desc\":\"atomic subtask\",\"depends_on\":[],"
        "\"dependency_reason\":\"why\"}]\n"
        "JSON Output:"
    )


def build_dependency_judge_prompt(task: str, planned_subtasks: List[PlannedSubtask]) -> str:
    planned_json = json.dumps(planned_subtasks, ensure_ascii=False, indent=2)
    return (
        "Role: Dependency judge for a LangGraph task router.\n"
        "Goal: verify and correct only execution dependencies; do not solve the task and do not score complexity.\n"
        f"Original task:\n{task}\n"
        f"Planner subtasks JSON:\n{planned_json}\n"
        "Rules:\n"
        "- Keep every existing id exactly as provided.\n"
        "- Do not add, remove, merge, split, or rewrite subtasks.\n"
        "- Use depends_on only when the current subtask needs prior outputs, evidence, decisions, or generated artifacts.\n"
        "- Independent investigations, file/provider/region checks, and unrelated evidence collection should have empty depends_on.\n"
        "- Synthesis, comparison, validation-after-implementation, reporting, and final answer steps should depend on the producing subtasks.\n"
        "- If uncertain whether a dependency is required for correctness, include the dependency.\n"
        "- Dependencies may reference only ids from the provided list.\n"
        "Return raw JSON only with keys: subtasks, confidence, issues.\n"
        "Each subtask item must contain: id, depends_on, dependency_reason.\n"
        "JSON Output:"
    )


def judge_dependencies_with_model(
    task: str,
    planned_subtasks: List[PlannedSubtask],
    judge_model: str,
) -> tuple[List[PlannedSubtask], float, List[str], str]:
    default_timeout = DEFAULT_LARGE_MODEL_TIMEOUT if is_large_model(judge_model) else 300
    judge_timeout = int(os.environ.get("ROUTER_JUDGE_TIMEOUT", str(default_timeout)))
    raw_text = generation.generate_text(
        judge_model,
        build_dependency_judge_prompt(task, planned_subtasks),
        timeout=judge_timeout,
        num_predict=4096,
        temperature=0.0,
        usage_label="Dependency judge",
    )
    if router_debug_enabled():
        print("\n[DEBUG Dependency Judge Output]")
        print(f"  Model: {judge_model}")
        print(f"  Raw text length: {len(raw_text)} chars")
        print(f"  First 400 chars: {raw_text[:400]}")
        if len(raw_text) > 400:
            print(f"  Last 200 chars: {raw_text[-200:]}")
        print(f"  Contains '{{': { '{' in raw_text }, Contains '}}': {'}' in raw_text }")
        print("  ---\n")
    judged_subtasks, confidence, issues = normalize_dependency_judgment(
        extract_first_json_object(raw_text),
        planned_subtasks,
    )
    return judged_subtasks, confidence, issues, raw_text


def build_judge_prompt(task: str, subtask_desc: str) -> str:
    context_pack = build_judge_context_pack_json(task, subtask_desc)
    return (
        "Role: Complexity judge for model routing.\n"
        f"Task context JSON:\n{context_pack}\n"
        f"Subtask: {subtask_desc}\n"
        "Judge the subtask itself, but use the task context JSON to understand domain risk. Only avoid inheriting the overall-task complexity when the subtask is purely a summary, report, or status update.\n"
        "Score the subtask with these ranges:\n"
        "- reasoning_depth: 0-3 (0 = lookup/formatting only, 1 = straightforward transformation, 2 = debugging or multi-step reasoning, 3 = architecture or open-ended investigation)\n"
        "- code_change_scope: 0-3 (0 = no code changes, 1 = small local edit, 2 = non-trivial or multi-file change, 3 = broad refactor/migration)\n"
        "- ambiguity: 0-2 (0 = clear, 1 = some interpretation needed, 2 = unclear/open-ended)\n"
        "- risk: 0-2 (0 = low risk, 1 = moderate impact, 2 = high-risk or hard-to-reverse)\n"
        "- io_heaviness: 0-2 (0 = little IO/reporting, 1 = some read/write/reporting, 2 = mostly IO/reporting/formatting)\n"
        "If the subtask is primarily preparing a team update, writing a concise stakeholder note, or formatting output, prefer reasoning_depth <= 1, code_change_scope = 0, and suggested_route = FLASH unless the subtask explicitly says to debug, fix, implement, or investigate.\n"
        "If the subtask synthesizes, consolidates, compares, or writes a final report/answer from findings produced by other steps, score reasoning_depth >= 2 and suggested_route = PRO even if it sounds like summarization.\n"
        "Diagnostic evidence gathering such as inspecting logs, checking failing paths, comparing config changes, or isolating a root cause should still lean PRO even if it mostly reads files.\n"
        "If the original task is a billing/payment/finance/security/production incident, then evidence gathering, stop-loss or rollback evaluation, and containment decisions should usually score reasoning_depth >= 2, risk = 2, and suggested_route = PRO even when the step mostly reads logs or data.\n"
        "Also provide:\n"
        "- suggested_route: PRO or FLASH\n"
        "- confidence: 0.0-1.0\n"
        "- reason: one short sentence\n"
        "Guidance example A: {\"scores\":{\"reasoning_depth\":0,\"code_change_scope\":0,\"ambiguity\":0,\"risk\":0,\"io_heaviness\":2},\"suggested_route\":\"FLASH\",\"confidence\":0.92,\"reason\":\"Mostly reporting and formatting work.\"}\n"
        "Guidance example B: {\"scores\":{\"reasoning_depth\":2,\"code_change_scope\":2,\"ambiguity\":1,\"risk\":1,\"io_heaviness\":0},\"suggested_route\":\"PRO\",\"confidence\":0.87,\"reason\":\"Requires debugging and non-trivial code changes.\"}\n"
        "Guidance example C: {\"scores\":{\"reasoning_depth\":1,\"code_change_scope\":0,\"ambiguity\":0,\"risk\":0,\"io_heaviness\":2},\"suggested_route\":\"FLASH\",\"confidence\":0.9,\"reason\":\"This step only packages the findings into a concise update.\"}\n"
        "Guidance example D: {\"scores\":{\"reasoning_depth\":2,\"code_change_scope\":0,\"ambiguity\":1,\"risk\":1,\"io_heaviness\":1},\"suggested_route\":\"PRO\",\"confidence\":0.84,\"reason\":\"Inspecting logs here is part of root-cause diagnosis, not just copying data.\"}\n"
        "Guidance example E: {\"scores\":{\"reasoning_depth\":2,\"code_change_scope\":0,\"ambiguity\":1,\"risk\":2,\"io_heaviness\":1},\"suggested_route\":\"PRO\",\"confidence\":0.86,\"reason\":\"Collecting evidence for a billing incident is part of triage and should stay on the stronger model.\"}\n"
        "Guidance example F: {\"scores\":{\"reasoning_depth\":2,\"code_change_scope\":0,\"ambiguity\":1,\"risk\":2,\"io_heaviness\":0},\"suggested_route\":\"PRO\",\"confidence\":0.88,\"reason\":\"Assessing stop-loss or rollback in a high-risk incident is a consequential decision step.\"}\n"
        "Return raw JSON only with keys: scores, suggested_route, confidence, reason.\n"
        "JSON Output:"
    )


def normalize_complexity_assessment(raw: Dict[str, Any], task: str, desc: str) -> ComplexityAssessment:
    raw_scores = raw.get("scores")
    score_source = raw_scores if isinstance(raw_scores, dict) else {}
    base_scores: ComplexityScores = {
        "reasoning_depth": clamp_int(
            score_source.get("reasoning_depth", raw.get("reasoning_depth")),
            0,
            3,
        ),
        "code_change_scope": clamp_int(
            score_source.get("code_change_scope", raw.get("code_change_scope")),
            0,
            3,
        ),
        "ambiguity": clamp_int(score_source.get("ambiguity", raw.get("ambiguity")), 0, 2),
        "risk": clamp_int(score_source.get("risk", raw.get("risk")), 0, 2),
        "io_heaviness": clamp_int(
            score_source.get("io_heaviness", raw.get("io_heaviness")),
            0,
            2,
        ),
    }
    scores = apply_contextual_score_biases(task, desc, base_scores)
    confidence = clamp_float(raw.get("confidence"), 0.0, 1.0, default=0.5)
    suggested_route = normalize_route(
        raw.get("suggested_route") or raw.get("route") or raw.get("model"),
        default=PRO,
    )
    summary_like = is_summary_like_subtask(desc)
    synthesis_like = is_synthesis_like_subtask(desc)
    deep_work_hint = has_deep_work_hint(desc)
    data_gathering_hint = has_data_gathering_hint(desc)
    high_risk_core_step = is_high_risk_core_step(task, desc)
    reason = compact_text(
        str(raw.get("reason") or raw.get("summary") or f"Structured complexity judgment for: {desc}"),
        220,
    )
    complexity_score = score_complexity(scores)
    final_route = decide_route(task, desc, scores, suggested_route, confidence)
    judge_source = "structured_llm"
    if synthesis_like and final_route == PRO:
        reason = compact_text(
            "Synthesis/comparison subtask kept on PRO because it must combine information from other executor steps.",
            220,
        )
        judge_source = "structured_llm+synthesis_bias"
    elif (
        summary_like
        and not deep_work_hint
        and not data_gathering_hint
        and final_route == FLASH
        and (suggested_route != FLASH or complexity_score > 3)
    ):
        reason = compact_text(
            "Subtask contains summary-like terms but scores as low-complexity data formatting or aggregation; routing to FLASH.",
            220,
        )
        judge_source = "structured_llm+summary_bias"
    elif (
        high_risk_core_step
        and final_route == PRO
    ):
        reason = compact_text(build_high_risk_reason(desc), 220)
        judge_source = "structured_llm+high_risk_bias"
    elif deep_work_hint and not summary_like and final_route == PRO and suggested_route != PRO:
        reason = compact_text(
            "Diagnostic investigation step kept on PRO because log inspection/config comparison here supports root-cause analysis.",
            220,
        )
        judge_source = "structured_llm+diagnostic_bias"
    return {
        "scores": scores,
        "complexity_score": complexity_score,
        "suggested_route": suggested_route,
        "final_route": final_route,
        "confidence": confidence,
        "reason": reason,
        "judge_source": judge_source,
    }


def build_fallback_assessment(task: str, desc: str) -> ComplexityAssessment:
    text = desc.lower()
    summary_like = is_summary_like_subtask(desc)
    synthesis_like = is_synthesis_like_subtask(desc)
    deep_work_hint = has_deep_work_hint(desc)
    data_gathering_hint = has_data_gathering_hint(desc)
    reasoning_depth = 0
    code_change_scope = 0
    ambiguity = 0
    risk = 0
    io_heaviness = 0

    if contains_any(
        text,
        (
            "architecture",
            "architect",
            "migrate",
            "migration",
            "refactor",
            "redesign",
            "架构",
            "迁移",
            "重构",
            "重新设计",
        ),
    ):
        reasoning_depth = 3
        code_change_scope = 3
    elif contains_any(
        text,
        (
            "analy",
            "debug",
            "diagnos",
            "fix",
            "implement",
            "investig",
            "trace",
            "optimiz",
            "logic",
            "分析",
            "调试",
            "诊断",
            "修复",
            "实现",
            "排查",
            "追踪",
            "优化",
            "逻辑",
        ),
    ):
        reasoning_depth = 2
        code_change_scope = max(code_change_scope, 1)
    elif len(desc.strip()) > 40:
        reasoning_depth = 1

    if contains_any(
        text,
        (
            "multi-file",
            "across",
            "rewrite",
            "refactor",
            "migration",
            "重写",
            "重构",
            "迁移",
            "跨文件",
        ),
    ):
        code_change_scope = max(code_change_scope, 2)
    elif contains_any(
        text,
        (
            "edit",
            "update",
            "patch",
            "fix",
            "implement",
            "修改",
            "更新",
            "补丁",
            "修复",
            "实现",
        ),
    ):
        code_change_scope = max(code_change_scope, 1)

    if contains_any(
        text,
        (
            "investig",
            "determine",
            "why",
            "compare",
            "evaluate",
            "explore",
            "排查",
            "确定原因",
            "比较",
            "评估",
            "探索",
        ),
    ):
        ambiguity = 1
    if contains_any(
        text,
        (
            "open-ended",
            "strategy",
            "design direction",
            "方案",
            "策略",
            "方向",
        ),
    ):
        ambiguity = 2

    if contains_any(
        text,
        (
            "prod",
            "production",
            "database",
            "schema",
            "auth",
            "security",
            "payment",
            "delete",
            "生产",
            "数据库",
            "鉴权",
            "安全",
            "支付",
            "删除",
            "模式变更",
        ),
    ):
        risk = 2
    elif contains_any(
        text,
        (
            "config",
            "deploy",
            "k8s",
            "yaml",
            "配置",
            "部署",
            "集群",
            "清单",
        ),
    ):
        risk = 1

    if contains_any(
        text,
        (
            "summary",
            "summarize",
            "report",
            "status",
            "format",
            "document",
            "整理",
            "总结",
            "报告",
            "状态",
            "格式化",
            "文档",
        ),
    ):
        io_heaviness = 2
    elif contains_any(
        text,
        (
            "save",
            "write",
            "read",
            "collect",
            "list",
            "scan",
            "保存",
            "写入",
            "读取",
            "收集",
            "列出",
            "扫描",
        ),
    ):
        io_heaviness = 1

    base_scores: ComplexityScores = {
        "reasoning_depth": reasoning_depth,
        "code_change_scope": code_change_scope,
        "ambiguity": ambiguity,
        "risk": risk,
        "io_heaviness": io_heaviness,
    }
    scores = apply_contextual_score_biases(task, desc, base_scores)
    complexity_score = score_complexity(scores)
    suggested_route = (
        FLASH
        if complexity_score <= FLASH_COMPLEXITY_THRESHOLD and io_heaviness >= 1
        else PRO
    )
    confidence = 0.55 if (complexity_score > 0 or io_heaviness > 0) else 0.45
    final_route = decide_route(task, desc, scores, suggested_route, confidence)
    judge_source = "heuristic_fallback"
    if synthesis_like and final_route == PRO:
        reason = "启发式规则将该子任务视为跨步骤综合/对比步骤，因此优先走 PRO。"
        judge_source = "heuristic_fallback+synthesis_bias"
    elif summary_like and not deep_work_hint and not data_gathering_hint and final_route == FLASH:
        reason = "启发式规则将该子任务视为独立的总结/状态更新步骤，优先走 FLASH。"
        judge_source = "heuristic_fallback+summary_bias"
    elif is_high_risk_core_step(task, desc) and final_route == PRO:
        if is_high_risk_evidence_step(desc):
            reason = "启发式规则将该子任务视为高风险事故里的取证步骤，因此优先走 PRO。"
        elif is_high_risk_decision_step(desc):
            reason = "启发式规则将该子任务视为高风险事故里的止损/回滚等关键决策步骤，因此优先走 PRO。"
        else:
            reason = "启发式规则将该子任务视为高风险事故里的诊断或修复策略步骤，因此优先走 PRO。"
        judge_source = "heuristic_fallback+high_risk_bias"
    elif deep_work_hint and not summary_like and final_route == PRO:
        reason = "启发式规则将该子任务视为诊断/排查步骤，即使主要是读日志或检查配置，也优先走 PRO。"
        judge_source = "heuristic_fallback+diagnostic_bias"
    else:
        reason = (
            "启发式评分判定该步骤以汇总/IO为主。"
            if final_route == FLASH
            else "启发式评分判定该步骤需要更强的推理或实现能力。"
        )
    return {
        "scores": scores,
        "complexity_score": complexity_score,
        "suggested_route": suggested_route,
        "final_route": final_route,
        "confidence": confidence,
        "reason": reason,
        "judge_source": judge_source,
    }


def score_subtask_with_model(task: str, subtask_desc: str, judge_model: str) -> ComplexityAssessment:
    default_timeout = DEFAULT_LARGE_MODEL_TIMEOUT if is_large_model(judge_model) else 300
    judge_timeout = int(os.environ.get("ROUTER_JUDGE_TIMEOUT", str(default_timeout)))
    raw_text = generation.generate_text(
        judge_model,
        build_judge_prompt(task, subtask_desc),
        timeout=judge_timeout,
        num_predict=204800,  # Increased for full JSON object output from large models
        usage_label=f"Judge subtask: {compact_text(subtask_desc, 80)}",
    )
    if router_debug_enabled():
        print(f"\n[DEBUG Judge Output for: {subtask_desc[:60]}...]")
        print(f"  Model: {judge_model}")
        print(f"  Raw text length: {len(raw_text)} chars")
        print(f"  First 400 chars: {raw_text[:400]}")
        if len(raw_text) > 400:
            print(f"  Last 200 chars: {raw_text[-200:]}")
        print(f"  Contains '{{': { '{' in raw_text }, Contains '}}': {'}' in raw_text }")
        print(f"  ---\n")
    return normalize_complexity_assessment(extract_first_json_object(raw_text), task, subtask_desc)


def build_subtask(planned: PlannedSubtask, assessment: ComplexityAssessment) -> Subtask:
    return {
        "id": planned["id"],
        "desc": planned["desc"],
        "depends_on": list(planned["depends_on"]),
        "dependency_reason": planned["dependency_reason"],
        "model": assessment["final_route"],
        "assessment": assessment,
    }


def display_plan(subtasks: List[Subtask], planner_model: str, judge_model: str) -> None:
    print("\n" + "=" * 58)
    print("🤖 LANGGRAPH 路由计划已生成")
    print("=" * 58)
    print(f"规划模型: {planner_model}")
    print(f"判定模型: {judge_model}")
    for index, step in enumerate(subtasks, start=1):
        icon = "🧠 [PRO]  " if step["model"] == PRO else "⚡ [FLASH] "
        assessment = step["assessment"]
        dependencies = ", ".join(step["depends_on"]) if step["depends_on"] else "-"
        print(
            f"步骤 {index} [{step['id']}]: {icon}| score={assessment['complexity_score']} "
            f"| conf={assessment['confidence']:.2f} | deps={dependencies} | {step['desc']}"
        )
        print(
            f"         判定依据: {assessment['reason']} "
            f"({assessment['judge_source']}, suggested={assessment['suggested_route']})"
        )
        print(f"         依赖依据: {step['dependency_reason']}")
    print("=" * 58)
