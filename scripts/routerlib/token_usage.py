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

TOKEN_USAGE_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("TOKEN_USAGE_RUN_ID", default="")
TOKEN_USAGE_LOCK = threading.RLock()
TOKEN_USAGE_RECORDS_BY_RUN: Dict[str, List["TokenUsageRecord"]] = {}
TOKEN_USAGE_ACTIVE_RUN_IDS: set[str] = set()


def empty_token_usage_summary() -> TokenUsageSummary:
    return {
        "calls": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "candidate_tokens": 0,
        "thought_tokens": 0,
        "tool_tokens": 0,
        "by_model": {},
        "by_provider": {},
    }


def current_token_usage_run_id() -> str:
    run_id = TOKEN_USAGE_RUN_ID.get("")
    if run_id:
        return run_id
    with TOKEN_USAGE_LOCK:
        if len(TOKEN_USAGE_ACTIVE_RUN_IDS) == 1:
            return next(iter(TOKEN_USAGE_ACTIVE_RUN_IDS))
    return ""


@contextlib.contextmanager
def token_usage_tracking_context(run_id: str) -> Iterator[None]:
    token = TOKEN_USAGE_RUN_ID.set(run_id)
    with TOKEN_USAGE_LOCK:
        TOKEN_USAGE_RECORDS_BY_RUN[run_id] = []
        TOKEN_USAGE_ACTIVE_RUN_IDS.add(run_id)
    try:
        yield
    finally:
        TOKEN_USAGE_RUN_ID.reset(token)
        with TOKEN_USAGE_LOCK:
            TOKEN_USAGE_ACTIVE_RUN_IDS.discard(run_id)


def get_token_usage_records(run_id: str) -> List[TokenUsageRecord]:
    with TOKEN_USAGE_LOCK:
        return list(TOKEN_USAGE_RECORDS_BY_RUN.get(run_id, []))


def resolve_token_usage_ledger_path() -> str:
    return os.path.expanduser(os.environ.get(ROUTER_TOKEN_USAGE_LEDGER_ENV_VAR, "").strip())


def persist_token_usage_ledger(
    records: List[TokenUsageRecord],
    summary: TokenUsageSummary,
    *,
    state: RouterState,
) -> str:
    ledger_path = resolve_token_usage_ledger_path()
    if not ledger_path or not records:
        return ""

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ledger_dir = os.path.dirname(ledger_path)
    if ledger_dir:
        os.makedirs(ledger_dir, exist_ok=True)

    base_event = {
        "event": "token_usage",
        "schema_version": 1,
        "timestamp": timestamp,
        "run_id": state["run_id"],
        "task_chars": len(state["task"]),
        "status": state["status"],
        "planner_model": state["planner_model"],
        "judge_model": state["judge_model"],
        "pro_model": state["pro_model"],
        "flash_model": state["flash_model"],
        "summary": summary,
    }
    with open(ledger_path, "a", encoding="utf-8") as ledger_file:
        for record in records:
            event = dict(base_event)
            event["record"] = record
            ledger_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return ledger_path


def metric_value(record: TokenUsageRecord, key: str) -> int:
    return int(record.get(key, 0) or 0)


def add_usage_to_bucket(bucket: Dict[str, int], record: TokenUsageRecord) -> None:
    bucket["calls"] = bucket.get("calls", 0) + 1
    if metric_value(record, "total_tokens") > 0:
        bucket["calls_with_usage"] = bucket.get("calls_with_usage", 0) + 1
    else:
        bucket["calls_without_usage"] = bucket.get("calls_without_usage", 0) + 1
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "candidate_tokens",
        "thought_tokens",
        "tool_tokens",
    ):
        bucket[key] = bucket.get(key, 0) + metric_value(record, key)


def summarize_token_usage_records(records: List[TokenUsageRecord]) -> TokenUsageSummary:
    summary = empty_token_usage_summary()
    for record in records:
        add_usage_to_bucket(summary, record)
        model_name = record["model_name"] or "unknown"
        provider = record["provider"] or "unknown"
        model_bucket = summary["by_model"].setdefault(model_name, {})
        provider_bucket = summary["by_provider"].setdefault(provider, {})
        add_usage_to_bucket(model_bucket, record)
        add_usage_to_bucket(provider_bucket, record)
    return summary


def format_int(value: int) -> str:
    return f"{value:,}"


def format_token_usage_summary(summary: TokenUsageSummary) -> str:
    if summary["calls"] == 0:
        return "Token Usage Summary\n- Calls: 0 (no provider token usage captured)."
    lines = [
        "Token Usage Summary",
        (
            f"- Calls: {summary['calls']} "
            f"({summary['calls_with_usage']} with token counts, "
            f"{summary['calls_without_usage']} unavailable)"
        ),
        (
            "- Tokens: "
            f"input={format_int(summary['input_tokens'])}, "
            f"output={format_int(summary['output_tokens'])}, "
            f"total={format_int(summary['total_tokens'])}"
        ),
    ]
    extras: List[str] = []
    if summary["cached_tokens"]:
        extras.append(f"cached={format_int(summary['cached_tokens'])}")
    if summary["candidate_tokens"]:
        extras.append(f"candidates={format_int(summary['candidate_tokens'])}")
    if summary["thought_tokens"]:
        extras.append(f"thoughts={format_int(summary['thought_tokens'])}")
    if summary["tool_tokens"]:
        extras.append(f"tool={format_int(summary['tool_tokens'])}")
    if extras:
        lines.append("- Extra tokens: " + ", ".join(extras))
    if summary["by_model"]:
        lines.append("- By model:")
        for model_name, bucket in sorted(summary["by_model"].items()):
            lines.append(
                f"  - {model_name}: calls={bucket.get('calls', 0)}, "
                f"total={format_int(bucket.get('total_tokens', 0))}, "
                f"input={format_int(bucket.get('input_tokens', 0))}, "
                f"output={format_int(bucket.get('output_tokens', 0))}"
            )
    return "\n".join(lines)


def record_token_usage(
    result: TextGenerationResult,
    *,
    label: str,
    prompt: str,
) -> None:
    run_id = current_token_usage_run_id()
    if not run_id:
        return
    usage = result.get("usage_metadata", {})
    if not isinstance(usage, dict):
        usage = {}
    output_text = str(result.get("text", ""))
    record: TokenUsageRecord = {
        "run_id": run_id,
        "call_index": 0,
        "label": label,
        "provider": str(result.get("provider", "")),
        "model_name": str(result.get("model_name", "")),
        "usage_source": str(result.get("usage_source", "unavailable")),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        "candidate_tokens": int(usage.get("candidate_tokens", 0) or 0),
        "thought_tokens": int(usage.get("thought_tokens", 0) or 0),
        "tool_tokens": int(usage.get("tool_tokens", 0) or 0),
        "prompt_chars": len(prompt),
        "output_chars": len(output_text),
    }
    with TOKEN_USAGE_LOCK:
        records = TOKEN_USAGE_RECORDS_BY_RUN.setdefault(run_id, [])
        record["call_index"] = len(records) + 1
        records.append(record)
