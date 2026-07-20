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



PLANNER_RELEVANT_TASK_KEYWORDS = (
    "must",
    "should",
    "need",
    "needs",
    "required",
    "requirement",
    "constraint",
    "deliverable",
    "output",
    "format",
    "table",
    "summary",
    "report",
    "brief",
    "compare",
    "analy",
    "debug",
    "diagnos",
    "fix",
    "implement",
    "investig",
    "trace",
    "optimiz",
    "refactor",
    "rewrite",
    "migrate",
    "design",
    "component",
    "provider",
    "model",
    "file",
    "service",
    "region",
    "backend",
    "risk",
    "incident",
    "rollback",
    "validation",
    "test",
    "evidence",
    "citation",
    "version",
    "metric",
    "limit",
    "budget",
)
PLANNER_CONSTRAINT_KEYWORDS = (
    "must",
    "should",
    "required",
    "requirement",
    "constraint",
    "preserve",
    "do not",
    "avoid",
    "without",
    "risk",
    "rollback",
    "security",
    "pci",
    "budget",
    "deadline",
    "slo",
    "sla",
    "必须",
    "需要",
    "要求",
    "约束",
    "保留",
    "不要",
    "避免",
    "风险",
    "回滚",
    "安全",
    "预算",
)
PLANNER_DELIVERABLE_KEYWORDS = (
    "deliverable",
    "output",
    "return",
    "write",
    "summary",
    "report",
    "brief",
    "table",
    "format",
    "final",
    "include",
    "artifact",
    "产出",
    "输出",
    "返回",
    "总结",
    "报告",
    "简报",
    "表格",
    "格式",
    "最终",
    "包含",
)
PLANNER_EVIDENCE_KEYWORDS = (
    "evidence",
    "metric",
    "metrics",
    "version",
    "versions",
    "hard limit",
    "limit",
    "citation",
    "citations",
    "file",
    "files",
    "command",
    "commands",
    "test",
    "tests",
    "validation",
    "verify",
    "logs",
    "traces",
    "数据",
    "证据",
    "指标",
    "版本",
    "限制",
    "引用",
    "文件",
    "命令",
    "测试",
    "验证",
    "日志",
)
PLANNER_DECOMPOSITION_HINT_KEYWORDS = (
    "each",
    "per",
    "independent",
    "component",
    "components",
    "provider",
    "providers",
    "model",
    "models",
    "region",
    "regions",
    "backend",
    "backends",
    "service",
    "services",
    "technical area",
    "technical areas",
    "entity",
    "entities",
    "每个",
    "分别",
    "独立",
    "组件",
    "提供商",
    "模型",
    "区域",
    "后端",
    "服务",
    "技术领域",
)
PLANNER_ENTITY_LIST_PATTERNS = (
    r"\b(?:across|between|among|covering|including)\s+([^.;\n]+)",
    r"\b(?:components?|services?|providers?|models?|regions?|backends?|technical areas?|entities)\s*[:=]\s*([^.;\n]+)",
    r"\bone\s+per\s+([^.;\n]+)",
)
PLANNER_ENTITY_STOPWORDS = {
    "a",
    "an",
    "analyze",
    "and",
    "avoid",
    "brief",
    "do",
    "each",
    "final",
    "goal",
    "if",
    "include",
    "including",
    "investigate",
    "must",
    "one",
    "output",
    "preserve",
    "required",
    "return",
    "role",
    "rules",
    "should",
    "summary",
    "task",
    "technical",
    "the",
    "this",
    "with",
}


def split_planner_task_segments(task: str) -> List[str]:
    segments: List[str] = []
    for line in task.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"(?<=[.!?])\s+", stripped)
        for part in parts:
            normalized = " ".join(part.strip().split())
            if normalized:
                segments.append(normalized)
    return segments


def is_planner_relevant_task_segment(segment: str) -> bool:
    stripped = segment.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if contains_any(lowered, PLANNER_RELEVANT_TASK_KEYWORDS):
        return True
    if re.search(r"^([-*+]|\d+[.)]|[A-Za-z][.)])\s+", stripped):
        return True
    if "`" in stripped:
        return True
    return bool(re.search(r"\b[A-Z][A-Za-z0-9_.:/-]{2,}\b", stripped))


def compact_planner_relevant_segment(segment: str, limit: int) -> str:
    return compact_text_middle(segment, limit)


def compact_planner_task(task: str, limit: int) -> str:
    one_line = " ".join(task.strip().split())
    if len(one_line) <= limit:
        return one_line
    if limit < 240:
        return compact_text_middle(one_line, limit)

    marker = f"[planner input compacted from {len(one_line)} chars]"
    remaining = limit - len(marker) - 2
    if remaining < 160:
        return compact_text_middle(one_line, limit)

    head_budget = min(max(240, remaining // 4), remaining // 2)
    tail_budget = min(max(200, remaining // 5), remaining - head_budget)
    head = one_line[:head_budget].rstrip()
    tail = one_line[-tail_budget:].lstrip()
    middle_budget = remaining - len(head) - len(tail)

    selected: List[str] = []
    used = 0
    if middle_budget > 80:
        for segment in split_planner_task_segments(task):
            if not is_planner_relevant_task_segment(segment):
                continue
            if segment in head or segment in tail:
                continue
            spacing = 1 if selected else 0
            available = middle_budget - used - spacing
            if available < 60:
                break
            compacted_segment = compact_planner_relevant_segment(segment, min(220, available))
            selected.append(compacted_segment)
            used += len(compacted_segment) + spacing

    middle = " ".join(selected)
    compacted = " ".join(part for part in (head, marker, middle, tail) if part)
    if len(compacted) <= limit:
        return compacted
    return compact_text_middle(compacted, limit)


def append_unique_planner_item(items: List[str], value: str, max_items: int, item_limit: int) -> None:
    normalized = " ".join(value.strip().strip("`'\"-:;,.()[]{}").split())
    if not normalized:
        return
    if len(normalized) < 2 or normalized.isdigit():
        return
    if normalized.lower() in PLANNER_ENTITY_STOPWORDS:
        return
    compacted = compact_text_middle(normalized, item_limit)
    existing = {item.lower() for item in items}
    if compacted.lower() in existing:
        return
    items.append(compacted)
    if len(items) > max_items:
        del items[max_items:]


def split_planner_entity_list(text: str) -> List[str]:
    normalized = re.sub(r"\b(?:and|or)\b", ",", text, flags=re.IGNORECASE)
    normalized = normalized.replace("/", ",")
    entities: List[str] = []
    for raw_part in re.split(r"[,;|]", normalized):
        part = " ".join(raw_part.strip().strip("`'\"-:;,.()[]{}").split())
        part = re.sub(r"^(?:the|a|an)\s+", "", part, flags=re.IGNORECASE)
        if not part:
            continue
        if len(part.split()) > 6:
            continue
        entities.append(part)
    return entities


def extract_planner_entities(segments: List[str], max_items: int = 24) -> List[str]:
    entities: List[str] = []
    for segment in segments:
        for pattern in PLANNER_ENTITY_LIST_PATTERNS:
            for match in re.finditer(pattern, segment, flags=re.IGNORECASE):
                for entity in split_planner_entity_list(match.group(1)):
                    append_unique_planner_item(entities, entity, max_items, 80)

        for code_match in re.finditer(r"`([^`]{2,120})`", segment):
            append_unique_planner_item(entities, code_match.group(1), max_items, 80)

        for path_match in re.finditer(
            r"\b(?:[\w.-]+/)+[\w./-]+|\b[\w.-]+\.(?:py|js|ts|tsx|json|ya?ml|md|sql|go|rs|java|sh|tf)\b",
            segment,
        ):
            append_unique_planner_item(entities, path_match.group(0), max_items, 80)

        for token_match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_./:][A-Za-z0-9]+)*\b|\b[a-z]+(?:-[a-z0-9]+)+\b", segment):
            append_unique_planner_item(entities, token_match.group(0), max_items, 80)

        if len(entities) >= max_items:
            break
    return entities


def collect_planner_segments(
    segments: List[str],
    keywords: tuple[str, ...],
    *,
    max_items: int,
    item_limit: int = 180,
) -> List[str]:
    collected: List[str] = []
    for segment in segments:
        lowered = segment.lower()
        if contains_any(lowered, keywords):
            append_unique_planner_item(collected, segment, max_items, item_limit)
        if len(collected) >= max_items:
            break
    return collected


def serialize_planner_manifest(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2)


def build_planner_manifest_payload(
    task: str,
    source_brief_budget: int,
    *,
    max_entities: int = 24,
    max_constraints: int = 10,
    max_deliverables: int = 8,
    max_evidence: int = 8,
    max_decomposition_hints: int = 8,
) -> Dict[str, Any]:
    one_line = " ".join(task.strip().split())
    segments = split_planner_task_segments(task)
    objective = segments[0] if segments else one_line
    return {
        "original_chars": len(one_line),
        "objective": compact_text_middle(objective, 420),
        "entities": extract_planner_entities(segments, max_items=max_entities),
        "constraints": collect_planner_segments(
            segments,
            PLANNER_CONSTRAINT_KEYWORDS,
            max_items=max_constraints,
        ),
        "deliverables": collect_planner_segments(
            segments,
            PLANNER_DELIVERABLE_KEYWORDS,
            max_items=max_deliverables,
        ),
        "evidence_requirements": collect_planner_segments(
            segments,
            PLANNER_EVIDENCE_KEYWORDS,
            max_items=max_evidence,
        ),
        "decomposition_hints": collect_planner_segments(
            segments,
            PLANNER_DECOMPOSITION_HINT_KEYWORDS,
            max_items=max_decomposition_hints,
        ),
        "source_brief": compact_planner_task(task, source_brief_budget),
    }


def build_planner_context_manifest(task: str, limit: int) -> str:
    one_line = " ".join(task.strip().split())
    effective_limit = max(240, limit)
    source_brief_budget = max(80, min(2000, effective_limit // 3))
    manifest = build_planner_manifest_payload(task, source_brief_budget)
    serialized = serialize_planner_manifest(manifest)
    if len(serialized) <= effective_limit:
        return serialized

    for item_count in (8, 4, 2, 1, 0):
        source_budget = max(40, min(source_brief_budget, effective_limit // 5))
        manifest = build_planner_manifest_payload(
            task,
            source_budget,
            max_entities=max(4, item_count * 2),
            max_constraints=item_count,
            max_deliverables=item_count,
            max_evidence=item_count,
            max_decomposition_hints=item_count,
        )
        serialized = serialize_planner_manifest(manifest)
        if len(serialized) <= effective_limit:
            return serialized

    minimal_manifest = {
        "original_chars": len(one_line),
        "objective": compact_text_middle(one_line, max(40, effective_limit // 4)),
        "entities": [],
        "constraints": [],
        "deliverables": [],
        "evidence_requirements": [],
        "decomposition_hints": [],
        "source_brief": compact_text_middle(one_line, max(40, effective_limit // 4)),
    }
    serialized = serialize_planner_manifest(minimal_manifest)
    if len(serialized) <= effective_limit:
        return serialized
    return json.dumps(
        {"source_brief": compact_text_middle(one_line, max(1, effective_limit - 24))},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def resolve_context_char_limit(env_name: str, fallback: int, task: str = "", subtask_desc: str = "") -> int:
    if os.environ.get(env_name, "").strip():
        return resolve_positive_int(None, env_name, fallback)
    if task and is_high_risk_context(task, subtask_desc):
        return fallback * 2
    return fallback


def tokenize_context_text(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{2,}", text)
    }


def prioritize_items_for_context(items: List[str], text: str, max_items: int) -> List[str]:
    if max_items <= 0:
        return []
    text_lower = text.lower()
    text_tokens = tokenize_context_text(text)
    relevant: List[str] = []
    remainder: List[str] = []
    for item in items:
        item_lower = item.lower()
        item_tokens = tokenize_context_text(item)
        if item_lower in text_lower or (item_tokens and item_tokens & text_tokens):
            relevant.append(item)
        else:
            remainder.append(item)
    return (relevant + remainder)[:max_items]


def matched_context_keywords(text: str, keywords: tuple[str, ...], max_items: int = 12) -> List[str]:
    lowered = text.lower()
    matched: List[str] = []
    for keyword in keywords:
        if keyword in lowered and keyword not in matched:
            matched.append(keyword)
        if len(matched) >= max_items:
            break
    return matched


def serialize_context_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_task_context_payload(
    task: str,
    subtask_desc: str = "",
    *,
    source_brief_budget: int,
    max_entities: int = 24,
    max_constraints: int = 10,
    max_deliverables: int = 8,
    max_evidence: int = 8,
    max_decomposition_hints: int = 8,
) -> Dict[str, Any]:
    manifest = build_planner_manifest_payload(
        task,
        source_brief_budget,
        max_entities=max(max_entities, 24),
        max_constraints=max_constraints,
        max_deliverables=max_deliverables,
        max_evidence=max_evidence,
        max_decomposition_hints=max_decomposition_hints,
    )
    context_text = f"{task}\n{subtask_desc}"
    payload: Dict[str, Any] = {
        "original_chars": manifest["original_chars"],
        "objective": manifest["objective"],
    }
    if subtask_desc:
        payload["subtask"] = compact_text_middle(subtask_desc, 420)
    payload.update(
        {
            "entities": prioritize_items_for_context(
                manifest["entities"],
                subtask_desc or task,
                max_entities,
            ),
            "constraints": manifest["constraints"][:max_constraints],
            "deliverables": manifest["deliverables"][:max_deliverables],
            "evidence_requirements": manifest["evidence_requirements"][:max_evidence],
            "decomposition_hints": manifest["decomposition_hints"][:max_decomposition_hints],
            "risk_context": {
                "high_risk": is_high_risk_context(task, subtask_desc),
                "matched_keywords": matched_context_keywords(context_text, HIGH_RISK_CONTEXT_KEYWORDS),
            },
            "source_brief": manifest["source_brief"],
        }
    )
    return payload


def build_task_context_pack_json(task: str, subtask_desc: str, limit: int) -> str:
    effective_limit = max(240, limit)
    source_brief_budget = max(80, min(2000, effective_limit // 3))
    for item_count in (12, 8, 4, 2, 1):
        payload = build_task_context_payload(
            task,
            subtask_desc,
            source_brief_budget=source_brief_budget,
            max_entities=max(4, item_count * 2),
            max_constraints=item_count,
            max_deliverables=item_count,
            max_evidence=item_count,
            max_decomposition_hints=item_count,
        )
        serialized = serialize_context_payload(payload)
        if len(serialized) <= effective_limit:
            return serialized
        source_brief_budget = max(40, source_brief_budget // 2)
    return serialize_context_payload(
        build_task_context_payload(
            task,
            subtask_desc,
            source_brief_budget=max(40, effective_limit // 8),
            max_entities=2,
            max_constraints=1,
            max_deliverables=1,
            max_evidence=1,
            max_decomposition_hints=1,
        )
    )


def build_judge_context_pack_json(task: str, subtask_desc: str) -> str:
    limit = resolve_context_char_limit(
        ROUTER_JUDGE_CONTEXT_CHAR_LIMIT_ENV_VAR,
        DEFAULT_JUDGE_CONTEXT_CHAR_LIMIT,
        task,
        subtask_desc,
    )
    return build_task_context_pack_json(task, subtask_desc, limit)


def compact_step_result_for_context(result: StepResult, output_limit: int) -> Dict[str, Any]:
    return {
        "step": result["step"],
        "subtask_id": result.get("subtask_id", ""),
        "depends_on": list(result.get("depends_on", [])),
        "planned_route": result["planned_route"],
        "actual_route": result["route"],
        "model_name": result["model_name"],
        "status": result["status"],
        "desc": compact_text_middle(result["desc"], 220),
        "output_excerpt": compact_text_middle(result["output"], output_limit) if output_limit > 0 else "",
        "attempt_count": result["attempt_count"],
        "retry_count": result["retry_count"],
        "escalated_from_flash": result["escalated_from_flash"],
        "used_provider_fallback": result["used_provider_fallback"],
    }


def build_executor_context_payload(
    state: RouterState,
    route: Literal["PRO", "FLASH"],
    *,
    source_brief_budget: int,
    prior_output_limit: int,
    max_prior_results: int,
    item_count: int,
) -> Dict[str, Any]:
    subtask_desc = str(state["active_subtask"].get("desc", "N/A"))
    assessment = state["active_subtask"].get("assessment") or {}
    prior_results = state["results"][-max_prior_results:] if max_prior_results > 0 else []
    payload: Dict[str, Any] = {
        "task_context": build_task_context_payload(
            state["task"],
            subtask_desc,
            source_brief_budget=source_brief_budget,
            max_entities=max(4, item_count * 2),
            max_constraints=item_count,
            max_deliverables=item_count,
            max_evidence=item_count,
            max_decomposition_hints=item_count,
        ),
        "routing": {
            "route": route,
            "score": assessment.get("complexity_score", "N/A"),
            "confidence": assessment.get("confidence", "N/A"),
            "reason": str(assessment.get("reason", "N/A")),
        },
        "dependency_context": {
            "subtask_id": state["active_subtask"].get("id", ""),
            "depends_on": list(state["active_subtask"].get("depends_on", [])),
            "dependency_reason": str(state["active_subtask"].get("dependency_reason", "")),
            "prior_results_are_direct_dependencies": True,
        },
        "prior_results": [
            compact_step_result_for_context(result, prior_output_limit)
            for result in prior_results
        ],
        "response_contract": "Return only the result for the current subtask. No markdown fences.",
    }
    if route == PRO and state["active_escalated_from_flash"]:
        flash_review = state["active_flash_review"]
        payload["escalation"] = {
            "from": FLASH,
            "to": PRO,
            "failure_type": flash_review["failure_type"],
            "retries": state["active_retry_count"],
            "reason": flash_review["reason"],
        }
    return payload


def build_executor_context_pack_json(state: RouterState, route: Literal["PRO", "FLASH"]) -> str:
    subtask_desc = str(state["active_subtask"].get("desc", ""))
    limit = resolve_context_char_limit(
        ROUTER_EXECUTOR_CONTEXT_CHAR_LIMIT_ENV_VAR,
        DEFAULT_EXECUTOR_CONTEXT_CHAR_LIMIT,
        state["task"],
        subtask_desc,
    )
    effective_limit = max(480, limit)
    source_brief_budget = max(120, min(3000, effective_limit // 3))
    max_prior = len(state["results"])
    for item_count, prior_limit, prior_count in (
        (12, 320, max_prior),
        (8, 220, min(max_prior, 8)),
        (4, 140, min(max_prior, 4)),
        (2, 80, min(max_prior, 2)),
        (1, 0, 0),
    ):
        payload = build_executor_context_payload(
            state,
            route,
            source_brief_budget=source_brief_budget,
            prior_output_limit=prior_limit,
            max_prior_results=prior_count,
            item_count=item_count,
        )
        serialized = serialize_context_payload(payload)
        if len(serialized) <= effective_limit:
            return serialized
        source_brief_budget = max(60, source_brief_budget // 2)
    return serialize_context_payload(
        build_executor_context_payload(
            state,
            route,
            source_brief_budget=60,
            prior_output_limit=0,
            max_prior_results=0,
            item_count=1,
        )
    )


def build_metadata_context_pack_json(state: RouterState, result: StepResult, output: str) -> str:
    limit = resolve_context_char_limit(
        ROUTER_METADATA_OUTPUT_CHAR_LIMIT_ENV_VAR,
        DEFAULT_METADATA_OUTPUT_CHAR_LIMIT,
        state["task"],
        result["desc"],
    )
    effective_limit = max(480, limit)
    output_budget = max(120, min(effective_limit // 2, DEFAULT_METADATA_OUTPUT_CHAR_LIMIT))
    for item_count in (8, 4, 2, 1):
        payload = {
            "task_context": build_task_context_payload(
                state["task"],
                result["desc"],
                source_brief_budget=max(80, effective_limit // 5),
                max_entities=max(4, item_count * 2),
                max_constraints=item_count,
                max_deliverables=item_count,
                max_evidence=item_count,
                max_decomposition_hints=item_count,
            ),
            "result": compact_step_result_for_context(result, 0),
            "output_excerpt": compact_text_middle(output, output_budget),
            "extraction_schema": [
                "architectural_decisions",
                "library_or_tool_choices",
                "critical_logic_or_algorithm_details",
                "tradeoffs",
                "verified_results",
            ],
        }
        serialized = serialize_context_payload(payload)
        if len(serialized) <= effective_limit:
            return serialized
        output_budget = max(80, output_budget // 2)
    return serialize_context_payload(
        {
            "task_context": build_task_context_payload(
                state["task"],
                result["desc"],
                source_brief_budget=40,
                max_entities=4,
                max_constraints=1,
                max_deliverables=1,
                max_evidence=1,
                max_decomposition_hints=1,
            ),
            "result": compact_step_result_for_context(result, 0),
            "output_excerpt": compact_text_middle(output, max(80, effective_limit // 4)),
        }
    )


def build_finalizer_context_payload(
    state: RouterState,
    route: Literal["PRO", "FLASH"],
    *,
    source_brief_budget: int,
    result_output_limit: int,
    max_results: int,
    metadata_limit: int,
    item_count: int,
) -> Dict[str, Any]:
    metadata_blocks = [line for line in state["history"] if "TECHNICAL METADATA STEP" in line]
    results = state["results"][-max_results:] if max_results > 0 else []
    return {
        "task_context": build_task_context_payload(
            state["task"],
            "",
            source_brief_budget=source_brief_budget,
            max_entities=max(4, item_count * 2),
            max_constraints=item_count,
            max_deliverables=item_count,
            max_evidence=item_count,
            max_decomposition_hints=item_count,
        ),
        "models": {
            "planner": state["planner_model"],
            "judge": state["judge_model"],
            "finalizer_route": route,
        },
        "technical_metadata": [
            compact_text_middle(block, metadata_limit) for block in metadata_blocks
        ] or ["No technical metadata extracted."],
        "execution_results": [
            compact_step_result_for_context(result, result_output_limit)
            for result in results
        ],
        "required_sections": ["Routing Summary", "Step Outcomes", "Next Action"],
    }


def build_finalizer_context_pack_json(state: RouterState, route: Literal["PRO", "FLASH"]) -> str:
    limit = resolve_context_char_limit(
        ROUTER_FINALIZER_CONTEXT_CHAR_LIMIT_ENV_VAR,
        DEFAULT_FINALIZER_CONTEXT_CHAR_LIMIT,
        state["task"],
        "",
    )
    effective_limit = max(800, limit)
    max_results = len(state["results"])
    source_brief_budget = max(160, min(3000, effective_limit // 4))
    for item_count, result_limit, result_count, metadata_limit in (
        (12, 360, max_results, 900),
        (8, 240, min(max_results, 12), 600),
        (4, 160, min(max_results, 8), 420),
        (2, 80, min(max_results, 4), 240),
        (1, 0, min(max_results, 2), 160),
    ):
        payload = build_finalizer_context_payload(
            state,
            route,
            source_brief_budget=source_brief_budget,
            result_output_limit=result_limit,
            max_results=result_count,
            metadata_limit=metadata_limit,
            item_count=item_count,
        )
        serialized = serialize_context_payload(payload)
        if len(serialized) <= effective_limit:
            return serialized
        source_brief_budget = max(80, source_brief_budget // 2)
    return serialize_context_payload(
        build_finalizer_context_payload(
            state,
            route,
            source_brief_budget=80,
            result_output_limit=0,
            max_results=min(max_results, 1),
            metadata_limit=120,
            item_count=1,
        )
    )
