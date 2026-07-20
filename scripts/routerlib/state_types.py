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



class PlannedSubtask(TypedDict):
    id: str
    desc: str
    depends_on: List[str]
    dependency_reason: str


class ComplexityScores(TypedDict):
    reasoning_depth: int
    code_change_scope: int
    ambiguity: int
    risk: int
    io_heaviness: int


class ComplexityAssessment(TypedDict):
    scores: ComplexityScores
    complexity_score: int
    suggested_route: Literal["PRO", "FLASH"]
    final_route: Literal["PRO", "FLASH"]
    confidence: float
    reason: str
    judge_source: str

class FlashReviewResult(TypedDict):
    decision: Literal["record", "retry", "escalate"]
    failure_type: Literal["none", "infra_transient", "capability_quality", "unknown"]
    reason: str
class ModelInvocationResult(TypedDict):
    success: bool
    output: str
    model_name: str
    used_provider_fallback: bool
    failure_type: Literal["none", "infra_transient", "capability_quality", "unknown"]
    error_text: str
    attempt_log: List[str]


class TextGenerationResult(TypedDict):
    text: str
    usage_metadata: Dict[str, int]
    provider: str
    model_name: str
    usage_source: str


class TokenUsageRecord(TypedDict):
    run_id: str
    call_index: int
    label: str
    provider: str
    model_name: str
    usage_source: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    candidate_tokens: int
    thought_tokens: int
    tool_tokens: int
    prompt_chars: int
    output_chars: int


class TokenUsageSummary(TypedDict):
    calls: int
    calls_with_usage: int
    calls_without_usage: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    candidate_tokens: int
    thought_tokens: int
    tool_tokens: int
    by_model: Dict[str, Dict[str, int]]
    by_provider: Dict[str, Dict[str, int]]


class ModelInvocationState(TypedDict):
    primary_model: str
    candidates: List[str]
    candidate_index: int
    current_model: str
    prompt: str
    timeout: int
    num_predict: int
    temperature: float
    label: str
    log: List[str]
    errors: List[str]
    result: ModelInvocationResult
    status: str


class FinalizerOutcome(TypedDict):
    route: Literal["FLASH", "PRO", "DETERMINISTIC"]
    model_name: str
    status: str
    used_provider_fallback: bool
    reason: str
    attempt_log: List[str]


class Subtask(TypedDict):
    id: str
    desc: str
    depends_on: List[str]
    dependency_reason: str
    model: Literal["PRO", "FLASH"]
    assessment: ComplexityAssessment


class JudgedSubtask(TypedDict):
    index: int
    subtask: Subtask
    error: str


class StepResult(TypedDict):
    step: int
    subtask_id: str
    depends_on: List[str]
    planned_route: Literal["PRO", "FLASH"]
    route: Literal["PRO", "FLASH"]
    model_name: str
    desc: str
    output: str
    status: str
    attempt_count: int
    retry_count: int
    escalated_from_flash: bool
    used_provider_fallback: bool
    flash_review: FlashReviewResult
    attempt_log: List[str]


class RouterState(TypedDict):
    run_id: str
    task: str
    planner_model: str
    judge_model: str
    pro_model: str
    flash_model: str
    pro_fallback_models: List[str]
    flash_fallback_models: List[str]
    flash_retry_budget: int
    planned_subtasks: List[PlannedSubtask]
    planner_raw_text: str
    planner_error: str
    dependency_raw_text: str
    dependency_error: str
    dependency_issues: List[str]
    dependency_confidence: float
    planner_warmup_attempt: int
    judge_warmup_done: bool
    subtasks: List[Subtask]
    judge_index: int
    judge_desc: str
    judge_results: Annotated[Dict[int, JudgedSubtask], operator.or_]
    execution_index: int
    execution_subtask: Dict[str, Any]
    execution_results: Annotated[Dict[int, StepResult], operator.or_]
    execution_context_results: List[StepResult]
    current_step: int
    active_subtask: Dict[str, Any]
    active_route: str
    active_model_name: str
    active_output: str
    active_last_error: str
    active_attempt_count: int
    active_retry_count: int
    active_escalated_from_flash: bool
    active_used_provider_fallback: bool
    active_flash_review: FlashReviewResult
    active_attempt_log: List[str]
    active_invocation_result: ModelInvocationResult
    results: Annotated[List[StepResult], operator.add]
    history: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    status: str
    final_report: str
    finalizer_outcome: FinalizerOutcome
    finalizer_attempt_log: List[str]
    finalizer_error: str
    finalizer_flash_reason: str
    finalizer_invocation_result: ModelInvocationResult
    token_usage: List[TokenUsageRecord]
    token_usage_summary: TokenUsageSummary
