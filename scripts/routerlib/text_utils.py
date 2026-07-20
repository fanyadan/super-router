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



def compact_text(text: str, limit: int = 160) -> str:
    one_line = " ".join(text.strip().split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def compact_text_middle(text: str, limit: int) -> str:
    one_line = " ".join(text.strip().split())
    if len(one_line) <= limit:
        return one_line
    if limit <= 3:
        return one_line[:limit]
    marker = " ... "
    available = limit - len(marker)
    if available <= 0:
        return one_line[:limit]
    head_size = max(1, available // 2)
    tail_size = max(1, available - head_size)
    return f"{one_line[:head_size].rstrip()}{marker}{one_line[-tail_size:].lstrip()}"


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def keyword_matches(text: str, keyword: str) -> bool:
    normalized_keyword = keyword.lower().strip()
    if not normalized_keyword:
        return False
    if contains_cjk(normalized_keyword):
        return normalized_keyword in text

    phrase_pattern = re.escape(normalized_keyword).replace(r"\ ", r"[\s_-]+")
    if normalized_keyword in ENGLISH_PREFIX_MATCH_KEYWORDS:
        phrase_pattern = f"{phrase_pattern}[a-z0-9_-]*"
    return re.search(rf"(?<![a-z0-9]){phrase_pattern}(?![a-z0-9])", text) is not None


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword_matches(lowered, keyword) for keyword in keywords)

def is_summary_like_subtask(description: str) -> bool:
    return contains_any(description.lower(), SUMMARY_ROUTE_KEYWORDS)


def is_synthesis_like_subtask(description: str) -> bool:
    return contains_any(description.lower(), SYNTHESIS_ROUTE_KEYWORDS)


def is_deferred_execution_subtask(subtask: Subtask) -> bool:
    description = subtask["desc"]
    lowered = description.lower()
    if is_summary_like_subtask(description) and not has_non_summary_work_hint(description):
        return True
    return contains_any(lowered, DEFERRED_EXECUTION_KEYWORDS)


def has_deep_work_hint(description: str) -> bool:
    return contains_any(description.lower(), DEEP_WORK_HINT_KEYWORDS)


def has_data_gathering_hint(description: str) -> bool:
    return contains_any(description.lower(), DATA_GATHERING_HINT_KEYWORDS)


def has_non_summary_work_hint(description: str) -> bool:
    return has_deep_work_hint(description) or has_data_gathering_hint(description)


def is_high_risk_context(task: str, description: str) -> bool:
    return contains_any(f"{task.lower()} {description.lower()}", HIGH_RISK_CONTEXT_KEYWORDS)


def is_high_risk_evidence_step(description: str) -> bool:
    return contains_any(description.lower(), HIGH_RISK_EVIDENCE_KEYWORDS)


def is_high_risk_decision_step(description: str) -> bool:
    return contains_any(description.lower(), HIGH_RISK_DECISION_KEYWORDS)


def is_high_risk_core_step(task: str, description: str) -> bool:
    summary_like = is_summary_like_subtask(description)
    deep_work_hint = has_deep_work_hint(description)
    if not is_high_risk_context(task, description):
        return False
    if summary_like and not deep_work_hint:
        return False
    return (
        deep_work_hint
        or is_high_risk_evidence_step(description)
        or is_high_risk_decision_step(description)
    )


def apply_contextual_score_biases(
    task: str,
    description: str,
    scores: ComplexityScores,
) -> ComplexityScores:
    adjusted: ComplexityScores = {
        "reasoning_depth": scores["reasoning_depth"],
        "code_change_scope": scores["code_change_scope"],
        "ambiguity": scores["ambiguity"],
        "risk": scores["risk"],
        "io_heaviness": scores["io_heaviness"],
    }
    if is_high_risk_core_step(task, description):
        adjusted["reasoning_depth"] = max(adjusted["reasoning_depth"], 2)
        adjusted["ambiguity"] = max(adjusted["ambiguity"], 1)
        adjusted["risk"] = max(adjusted["risk"], 2)
        if is_high_risk_evidence_step(description):
            adjusted["io_heaviness"] = min(adjusted["io_heaviness"], 1)
    if is_synthesis_like_subtask(description):
        adjusted["reasoning_depth"] = max(adjusted["reasoning_depth"], 2)
        adjusted["ambiguity"] = max(adjusted["ambiguity"], 1)
    return adjusted


def build_high_risk_reason(description: str) -> str:
    if is_high_risk_evidence_step(description):
        return (
            "High-risk incident evidence gathering stays on PRO because it is part of triage and diagnosis, not mere IO."
        )
    if is_high_risk_decision_step(description):
        return (
            "High-risk incident stop-loss, rollback, or containment evaluation stays on PRO because it is a consequential decision step."
        )
    return (
        "High-risk incident diagnosis or repair strategy stays on PRO because it requires stronger reasoning and safer judgment."
    )
