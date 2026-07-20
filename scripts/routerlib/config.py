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



OLLAMA_URL = os.environ.get("ROUTER_OLLAMA_URL", "http://localhost:11434/api/generate")
ROUTER_MODEL_ENV_VAR = "ROUTER_MODEL"
ROUTER_TASK_ENV_VAR = "ROUTER_TASK"
CODEX_CLI_PATH = os.environ.get("ROUTER_CODEX_CLI", shutil.which("codex") or "codex")
ROUTER_CODEX_CWD_ENV_VAR = "ROUTER_CODEX_CWD"
ROUTER_CODEX_SANDBOX_ENV_VAR = "ROUTER_CODEX_SANDBOX"
GEMINI_CLI_PATH = os.environ.get("ROUTER_GEMINI_CLI", shutil.which("gemini") or "/opt/homebrew/bin/gemini")
CLAUDE_CLI_PATH = os.environ.get("ROUTER_CLAUDE_CLI", shutil.which("claude") or "claude")
ROUTER_CLAUDE_CWD_ENV_VAR = "ROUTER_CLAUDE_CWD"
ROUTER_CLAUDE_SANDBOX_ENV_VAR = "ROUTER_CLAUDE_SANDBOX"
GEMINI_SYSTEM_SETTINGS_ENV_VAR = "GEMINI_CLI_SYSTEM_SETTINGS_PATH"
ROUTER_SKIP_WARMUP = os.environ.get("ROUTER_SKIP_WARMUP", "0").lower() in ("1", "true", "yes")
ROUTER_LANGSMITH_ENABLED_ENV_VAR = "ROUTER_LANGSMITH_ENABLED"
ROUTER_LANGSMITH_PROJECT_ENV_VAR = "ROUTER_LANGSMITH_PROJECT"
ROUTER_LANGSMITH_TAGS_ENV_VAR = "ROUTER_LANGSMITH_TAGS"
ROUTER_TOKEN_USAGE_LEDGER_ENV_VAR = "ROUTER_TOKEN_USAGE_LEDGER"
#GEMINI_EXTENSION_NAME = os.environ.get("ROUTER_GEMINI_EXTENSION", "superpowers")
PRO = "PRO"
FLASH = "FLASH"
PRO_COMPLEXITY_THRESHOLD = 5
FLASH_COMPLEXITY_THRESHOLD = 2
LOW_CONFIDENCE_THRESHOLD = 0.35
ENGLISH_PREFIX_MATCH_KEYWORDS = frozenset(
    (
        "analy",
        "diagnos",
        "investig",
        "optimiz",
    )
)
SUMMARY_ROUTE_KEYWORDS = (
    "summary",
    "summarize",
    "report",
    "impact note",
    "risk note",
    "action summary",
    "status update",
    "team update",
    "brief",
    "briefing",
    "recap",
    "write-up",
    "conclusion",
    "document",
    "format",
    "整理",
    "总结",
    "摘要",
    "概述",
    "汇报",
    "状态更新",
    "团队状态更新",
    "影响说明",
    "风险说明",
    "行动摘要",
    "简报",
    "结论",
    "要点",
)
DEFERRED_EXECUTION_KEYWORDS = (
    "synthesize",
    "synthesis",
    "consolidate",
    "combine findings",
    "final comparison",
    "comparison table",
    "final table",
    "final output",
    "final answer",
    "综合",
    "汇总",
    "整合",
    "最终",
    "对比表",
    "比较表",
)
SYNTHESIS_ROUTE_KEYWORDS = (
    "synthesize",
    "synthesis",
    "consolidate",
    "combined findings",
    "combine findings",
    "compare findings",
    "final comparison",
    "comparison table",
    "final table",
    "final answer",
    "final report",
    "final summary",
    "final synthesis",
    "overall report",
    "overall summary",
    "all findings",
    "across all",
    "across steps",
    "other steps",
    "previous steps",
    "prior results",
    "综合",
    "整合",
    "最终",
    "对比表",
    "比较表",
)
COMMUNICATION_AUDIENCE_KEYWORDS = (
    "产品经理",
    "项目负责人",
    "负责人",
    "值班同事",
    "值班",
    "团队",
    "运营同事",
    "运营",
    "客户",
    "manager",
    "product manager",
    "pm",
    "team",
    "on-call",
    "operations",
    "operator",
    "customer",
    "stakeholder",
)
DEEP_WORK_HINT_KEYWORDS = (
    "inspect",
    "check",
    "audit",
    "examine",
    "identify",
    "compare",
    "determine",
    "discover",
    "isolate",
    "locate",
    "review",
    "verify",
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
    "logic",
    "排查",
    "检查",
    "分析",
    "确定",
    "定位",
    "比对",
    "核实",
    "确认",
    "审查",
    "调试",
    "诊断",
    "修复",
    "实现",
    "追踪",
    "优化",
    "重构",
    "重写",
    "迁移",
    "设计",
    "逻辑",
)
DATA_GATHERING_HINT_KEYWORDS = (
    # Prevent summary-keyword false positives when a subtask produces summaries
    # as output fields rather than being a summary/status-update step.
    "gather",
    "collect",
    "for each",
    "research",
    "compile",
    "retrieve",
    "fetch",
    "extract",
    "ingest",
    "persist",
    "database",
    "connection method",
    "api endpoint",
    "file-based store",
    "write",
    "insert",
    "upsert",
    "records",
    "search",
    "收集",
    "采集",
    "搜索",
    "检索",
    "提取",
    "汇编",
)
HIGH_RISK_CONTEXT_KEYWORDS = (
    "incident",
    "outage",
    "prod",
    "production",
    "billing",
    "payment",
    "payments",
    "finance",
    "financial",
    "settlement",
    "refund",
    "refunds",
    "charge",
    "charges",
    "duplicate charge",
    "double charge",
    "overcharge",
    "auth",
    "authentication",
    "authorization",
    "privilege",
    "security",
    "breach",
    "rollback",
    "roll back",
    "containment",
    "stop-loss",
    "stop loss",
    "kill switch",
    "fraud",
    "事故",
    "故障",
    "线上",
    "生产",
    "计费",
    "账单",
    "支付",
    "财务",
    "金融",
    "结算",
    "退款",
    "扣费",
    "重复扣费",
    "多扣费",
    "鉴权",
    "认证",
    "授权",
    "权限",
    "越权",
    "安全",
    "漏洞",
    "回滚",
    "止损",
    "止血",
    "遏制",
    "熔断",
    "冻结",
    "欺诈",
    "对账",
    "批处理",
)
HIGH_RISK_EVIDENCE_KEYWORDS = (
    "inspect",
    "check",
    "review",
    "collect",
    "gather",
    "compare",
    "trace",
    "query",
    "audit",
    "reconcile",
    "read",
    "log",
    "logs",
    "evidence",
    "data",
    "record",
    "records",
    "transaction",
    "ledger",
    "sample",
    "检查",
    "排查",
    "核对",
    "核实",
    "审查",
    "收集",
    "读取",
    "查看",
    "比对",
    "追踪",
    "查询",
    "审计",
    "对账",
    "日志",
    "证据",
    "数据",
    "记录",
    "流水",
    "交易",
    "账本",
    "样本",
)
HIGH_RISK_DECISION_KEYWORDS = (
    "evaluate",
    "assess",
    "determine",
    "decide",
    "plan",
    "mitigation",
    "containment",
    "rollback",
    "repair plan",
    "recovery",
    "是否需要",
    "必要性",
    "可行性",
    "评估",
    "判断",
    "决定",
    "方案",
    "止损",
    "止血",
    "回滚",
    "缓解",
    "修复方案",
    "补救",
    "恢复",
    "冻结",
)
INFRA_FAILURE_KEYWORDS = (
    "timed out",
    "timeout",
    "cannot reach required google endpoints",
    "unable to reach",
    "service unavailable",
    "temporarily unavailable",
    "rate limit",
    "too many requests",
    "connection reset",
    "connection refused",
    "network",
    "transport",
    "broken pipe",
    "deadline exceeded",
    "preflight failed",
    "认证",
    "鉴权",
    "超时",
    "限流",
    "网络",
    "连接",
    "不可达",
    "服务不可用",
)
CAPABILITY_FAILURE_KEYWORDS = (
    "unable to complete",
    "cannot complete",
    "can't complete",
    "unable to determine",
    "cannot determine",
    "can't determine",
    "need more context",
    "need more information",
    "not enough context",
    "not enough information",
    "insufficient information",
    "无法完成",
    "无法判断",
    "需要更多信息",
    "信息不足",
)
LOW_QUALITY_OUTPUT_MARKERS = (
    "unable to complete",
    "cannot complete",
    "can't complete",
    "unable to determine",
    "cannot determine",
    "can't determine",
    "need more context",
    "need more information",
    "not enough context",
    "not enough information",
    "insufficient information",
    "无法完成",
    "无法判断",
    "需要更多信息",
    "信息不足",
)
DEFAULT_FLASH_RETRY_BUDGET = 1
MIN_NON_SUMMARY_OUTPUT_CHARS = 48
DEFAULT_ROUTER_RECURSION_LIMIT = 128
DEFAULT_LARGE_MODEL_TIMEOUT = 6000
DEFAULT_WARMUP_TIMEOUT = 60
DEFAULT_PLANNER_TIMEOUT = 300
DEFAULT_PRO_EXECUTION_TIMEOUT = 300
DEFAULT_FLASH_EXECUTION_TIMEOUT = 300
DEFAULT_METADATA_TIMEOUT = 120
DEFAULT_PRO_FINALIZER_TIMEOUT = 300
DEFAULT_FLASH_FINALIZER_TIMEOUT = 300
DEFAULT_ROUTER_RUN_TIMEOUT = 7200
DEFAULT_MAX_PROVIDER_ATTEMPTS = 3
DEFAULT_PROVIDER_TERMINATION_GRACE = 5
DEFAULT_PRO_MODEL = "google-gemini-cli/gemini-3-pro-preview"
DEFAULT_FLASH_MODEL = "google-gemini-cli/gemini-3-flash-preview"
DEFAULT_PLANNER_MODEL = DEFAULT_PRO_MODEL
DEFAULT_JUDGE_MODEL = DEFAULT_FLASH_MODEL
ROUTER_WARMUP_TIMEOUT_ENV_VAR = "ROUTER_WARMUP_TIMEOUT"
ROUTER_PLANNER_TIMEOUT_ENV_VAR = "ROUTER_PLANNER_TIMEOUT"
ROUTER_EXECUTOR_TIMEOUT_ENV_VAR = "ROUTER_EXECUTOR_TIMEOUT"
ROUTER_PRO_EXECUTOR_TIMEOUT_ENV_VAR = "ROUTER_PRO_EXECUTOR_TIMEOUT"
ROUTER_FLASH_EXECUTOR_TIMEOUT_ENV_VAR = "ROUTER_FLASH_EXECUTOR_TIMEOUT"
ROUTER_METADATA_TIMEOUT_ENV_VAR = "ROUTER_METADATA_TIMEOUT"
ROUTER_RUN_TIMEOUT_ENV_VAR = "ROUTER_RUN_TIMEOUT"
ROUTER_MAX_PROVIDER_ATTEMPTS_ENV_VAR = "ROUTER_MAX_PROVIDER_ATTEMPTS"
ROUTER_PROVIDER_TERMINATION_GRACE_ENV_VAR = "ROUTER_PROVIDER_TERMINATION_GRACE"
ROUTER_PLANNER_TASK_CHAR_LIMIT_ENV_VAR = "ROUTER_PLANNER_TASK_CHAR_LIMIT"
ROUTER_PLANNER_MAX_OUTPUT_TOKENS_ENV_VAR = "ROUTER_PLANNER_MAX_OUTPUT_TOKENS"
ROUTER_JUDGE_CONTEXT_CHAR_LIMIT_ENV_VAR = "ROUTER_JUDGE_CONTEXT_CHAR_LIMIT"
ROUTER_EXECUTOR_CONTEXT_CHAR_LIMIT_ENV_VAR = "ROUTER_EXECUTOR_CONTEXT_CHAR_LIMIT"
ROUTER_METADATA_OUTPUT_CHAR_LIMIT_ENV_VAR = "ROUTER_METADATA_OUTPUT_CHAR_LIMIT"
ROUTER_FINALIZER_CONTEXT_CHAR_LIMIT_ENV_VAR = "ROUTER_FINALIZER_CONTEXT_CHAR_LIMIT"
DEFAULT_PLANNER_TASK_CHAR_LIMIT = 6000
DEFAULT_PLANNER_MAX_OUTPUT_TOKENS = 4096
DEFAULT_JUDGE_CONTEXT_CHAR_LIMIT = 3000
DEFAULT_EXECUTOR_CONTEXT_CHAR_LIMIT = 8000
DEFAULT_METADATA_OUTPUT_CHAR_LIMIT = 6000
DEFAULT_FINALIZER_CONTEXT_CHAR_LIMIT = 12000
