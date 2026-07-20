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
from . import provider_process  # noqa: F401
from . import config  # noqa: F401



CLAUDE_PERMISSION_MODE_BY_KEY = {
    "acceptedits": "acceptEdits",
    "auto": "auto",
    "bypasspermissions": "bypassPermissions",
    "manual": "manual",
    "dontask": "dontAsk",
    "plan": "plan",
}

CLAUDE_SANDBOX_PERMISSION_MODE_ALIASES = {
    "read-only": "plan",
    "readonly": "plan",
    "workspace-write": "acceptEdits",
    "workspace": "acceptEdits",
    "danger-full-access": "bypassPermissions",
    "full-access": "bypassPermissions",
}

CLAUDE_CODEX_STYLE_SANDBOX_BY_KEY = {
    "read-only": "read-only",
    "readonly": "read-only",
    "workspace-write": "workspace-write",
    "workspace": "workspace-write",
    "danger-full-access": "danger-full-access",
    "full-access": "danger-full-access",
}


def build_claude_sandbox_settings(codex_sandbox_mode: str) -> Dict[str, Any] | None:
    if codex_sandbox_mode == "workspace-write":
        return {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
            }
        }

    if codex_sandbox_mode == "read-only":
        return {
            "sandbox": {
                "enabled": True,
                "failIfUnavailable": True,
                "autoAllowBashIfSandboxed": False,
                "allowUnsandboxedCommands": False,
                "filesystem": {
                    "denyWrite": ["."],
                },
            }
        }

    return None


def normalize_claude_sandbox_config(value: str) -> tuple[str, Dict[str, Any] | None]:
    raw = value.strip()
    if not raw:
        return "", None

    direct_key = raw.replace("-", "").replace("_", "").lower()
    if direct_key in CLAUDE_PERMISSION_MODE_BY_KEY:
        return CLAUDE_PERMISSION_MODE_BY_KEY[direct_key], None

    alias_key = raw.replace("_", "-").lower()
    if alias_key in CLAUDE_CODEX_STYLE_SANDBOX_BY_KEY:
        codex_sandbox_mode = CLAUDE_CODEX_STYLE_SANDBOX_BY_KEY[alias_key]
        return CLAUDE_SANDBOX_PERMISSION_MODE_ALIASES[alias_key], build_claude_sandbox_settings(codex_sandbox_mode)

    supported = ", ".join(
        [
            "read-only",
            "workspace-write",
            "danger-full-access",
            "acceptEdits",
            "auto",
            "bypassPermissions",
            "manual",
            "dontAsk",
            "plan",
        ]
    )
    raise RuntimeError(
        f"Invalid {ROUTER_CLAUDE_SANDBOX_ENV_VAR}={value!r}. "
        f"Use one of: {supported}."
    )


def normalize_claude_permission_mode(value: str) -> str:
    return normalize_claude_sandbox_config(value)[0]


def extract_claude_usage_metadata(parsed: Dict[str, Any], normalized_model: str) -> Dict[str, int]:
    usage = parsed.get("usage")
    if isinstance(usage, dict):
        cache_creation_tokens = first_present_value(
            usage,
            ("cache_creation_input_tokens", "cacheCreationInputTokens"),
        )
        cache_read_tokens = first_present_value(
            usage,
            ("cache_read_input_tokens", "cacheReadInputTokens"),
        )
        return normalize_usage_metadata(
            input_tokens=first_present_value(usage, ("input_tokens", "inputTokens")),
            output_tokens=first_present_value(usage, ("output_tokens", "outputTokens")),
            total_tokens=first_present_value(usage, ("total_tokens", "totalTokens")),
            cached_tokens=sum_optional_ints([cache_creation_tokens, cache_read_tokens]),
        )

    model_usage = parsed.get("model_usage") or parsed.get("modelUsage")
    if isinstance(model_usage, dict):
        model_entry = model_usage.get(normalized_model)
        entries = (
            [model_entry]
            if isinstance(model_entry, dict)
            else [entry for entry in model_usage.values() if isinstance(entry, dict)]
        )
        if entries:
            input_tokens = sum_optional_ints(
                [first_present_value(entry, ("inputTokens", "input_tokens")) for entry in entries]
            )
            output_tokens = sum_optional_ints(
                [first_present_value(entry, ("outputTokens", "output_tokens")) for entry in entries]
            )
            cache_creation_tokens = sum_optional_ints(
                [
                    first_present_value(
                        entry,
                        ("cacheCreationInputTokens", "cache_creation_input_tokens"),
                    )
                    for entry in entries
                ]
            )
            cache_read_tokens = sum_optional_ints(
                [
                    first_present_value(
                        entry,
                        ("cacheReadInputTokens", "cache_read_input_tokens"),
                    )
                    for entry in entries
                ]
            )
            total_tokens = sum_optional_ints(
                [first_present_value(entry, ("totalTokens", "total_tokens")) for entry in entries]
            )
            return normalize_usage_metadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=sum_optional_ints([cache_creation_tokens, cache_read_tokens]),
            )

    cache_creation_tokens = first_present_value(
        parsed,
        ("cache_creation_input_tokens", "cacheCreationInputTokens"),
    )
    cache_read_tokens = first_present_value(
        parsed,
        ("cache_read_input_tokens", "cacheReadInputTokens"),
    )
    return normalize_usage_metadata(
        input_tokens=first_present_value(parsed, ("total_input_tokens", "input_tokens", "inputTokens")),
        output_tokens=first_present_value(parsed, ("total_output_tokens", "output_tokens", "outputTokens")),
        total_tokens=first_present_value(parsed, ("total_tokens", "totalTokens")),
        cached_tokens=sum_optional_ints([cache_creation_tokens, cache_read_tokens]),
    )


def claude_generate_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 120,
    temperature: float = 0.0,
) -> TextGenerationResult:
    del temperature

    normalized_model = normalize_model_name(model)
    if os.path.sep in config.CLAUDE_CLI_PATH and not os.path.exists(config.CLAUDE_CLI_PATH):
        raise RuntimeError("Claude CLI executable was not found. Set ROUTER_CLAUDE_CLI or install `claude`.")

    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    permission_mode, sandbox_settings = normalize_claude_sandbox_config(
        os.environ.get(ROUTER_CLAUDE_SANDBOX_ENV_VAR, "")
    )
    command = [config.CLAUDE_CLI_PATH, "--model", normalized_model, "--output-format", "json"]
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    if sandbox_settings:
        command.extend(["--settings", json.dumps(sandbox_settings, sort_keys=True, separators=(",", ":"))])
    command.extend(["-p", prompt])
    claude_cwd = os.environ.get(ROUTER_CLAUDE_CWD_ENV_VAR, "").strip()

    result = provider_process.run_provider_cli(
        command,
        timeout=timeout,
        env=env,
        label=f"Claude CLI {normalized_model}",
        cwd=claude_cwd or None,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        error_text = compact_text(stderr or stdout or f"exit code {result.returncode}", 280)
        raise RuntimeError(f"Claude CLI failed for model {normalized_model}: {error_text}")

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        error_text = compact_text(stderr or stdout or "no output", 280)
        raise RuntimeError(f"Claude CLI returned non-JSON output for model {normalized_model}: {error_text}") from exc

    if parsed.get("is_error"):
        error_text = compact_text(parsed.get("result") or stderr or "unknown error", 280)
        raise RuntimeError(f"Claude CLI reported error for model {normalized_model}: {error_text}")

    text = parsed.get("result", "")
    if not text.strip():
        raise RuntimeError(f"Claude CLI returned an empty response for model {normalized_model}")

    usage = extract_claude_usage_metadata(parsed, normalized_model)
    return build_text_generation_result(
        text.strip(),
        usage or {},
        "anthropic",
        normalized_model,
        "claude_cli_json" if usage else "unavailable",
    )
