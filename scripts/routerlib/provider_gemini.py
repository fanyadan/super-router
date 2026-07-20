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
from . import provider_process  # noqa: F401
from . import config  # noqa: F401

GEMINI_PREFLIGHT_RESULTS: Dict[str, str] = {}
GEMINI_NETWORK_PREFLIGHT_RESULT: str | None = None


def ensure_gemini_network_ready(timeout: float = 3.0) -> None:
    global GEMINI_NETWORK_PREFLIGHT_RESULT

    if GEMINI_NETWORK_PREFLIGHT_RESULT is not None:
        if GEMINI_NETWORK_PREFLIGHT_RESULT:
            raise RuntimeError(GEMINI_NETWORK_PREFLIGHT_RESULT)
        return

    if has_proxy_config():
        GEMINI_NETWORK_PREFLIGHT_RESULT = ""
        return

    failures: List[str] = []
    for host in ("oauth2.googleapis.com", "generativelanguage.googleapis.com"):
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                pass
        except OSError as exc:
            reason = "timed out" if isinstance(exc, TimeoutError) else compact_text(str(exc), 120)
            failures.append(f"{host}:443 ({reason})")

    if failures:
        GEMINI_NETWORK_PREFLIGHT_RESULT = (
            "Cannot reach required Google endpoints for Gemini CLI: "
            + ", ".join(failures)
            + ". Gemini cannot authenticate or execute until Google network access works or a proxy is configured."
        )
        raise RuntimeError(GEMINI_NETWORK_PREFLIGHT_RESULT)

    GEMINI_NETWORK_PREFLIGHT_RESULT = ""


def default_gemini_system_settings_path() -> str:
    if sys.platform == "darwin":
        return "/Library/Application Support/GeminiCli/settings.json"
    if os.name == "nt":
        return r"C:\ProgramData\gemini-cli\settings.json"
    return "/etc/gemini-cli/settings.json"


def load_gemini_system_settings() -> Dict[str, Any]:
    settings_path = os.environ.get(GEMINI_SYSTEM_SETTINGS_ENV_VAR, "").strip()
    if not settings_path:
        settings_path = default_gemini_system_settings_path()
    if not settings_path or not os.path.exists(settings_path):
        return {}
    try:
        with open(settings_path, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(settings, dict):
        return {}
    return settings


def build_gemini_temperature_settings(normalized_model: str, temperature: float) -> Dict[str, Any]:
    settings = copy.deepcopy(load_gemini_system_settings())
    model_configs = settings.get("modelConfigs")
    if not isinstance(model_configs, dict):
        model_configs = {}
        settings["modelConfigs"] = model_configs

    custom_overrides = model_configs.get("customOverrides")
    if not isinstance(custom_overrides, list):
        custom_overrides = []
    else:
        custom_overrides = list(custom_overrides)

    custom_overrides.append(
        {
            "match": {"model": normalized_model},
            "modelConfig": {
                "generateContentConfig": {
                    "temperature": temperature,
                },
            },
        }
    )
    model_configs["customOverrides"] = custom_overrides
    return settings


def invoke_gemini_cli_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 120,
    temperature: float = 0.0,
) -> TextGenerationResult:
    normalized_model = normalize_model_name(model)
    if not config.GEMINI_CLI_PATH or not os.path.exists(config.GEMINI_CLI_PATH):
        raise RuntimeError("Gemini CLI executable was not found. Set ROUTER_GEMINI_CLI or install `gemini`.")

    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    env["NO_BROWSER"] = "true"
    # Force inject ripgrep + common tool paths (fixes "Ripgrep is not available" warning)
    extra_paths = ["/opt/homebrew/bin", "/usr/local/bin"]
    current_path = env.get("PATH", "")
    path_parts = [p for p in extra_paths if p not in current_path.split(":")]
    if path_parts:
        env["PATH"] = ":".join(path_parts + [current_path]) if current_path else ":".join(path_parts)
    command = [
        config.GEMINI_CLI_PATH,
        "-m",
        normalized_model,
        "-p",
        prompt,
        "-o",
        "json",
        "-y",
#        "-e",
#        GEMINI_EXTENSION_NAME,
    ]
    with tempfile.TemporaryDirectory(prefix="router-gemini-") as settings_dir:
        settings_path = os.path.join(settings_dir, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as settings_file:
            json.dump(build_gemini_temperature_settings(normalized_model, temperature), settings_file)
        env[GEMINI_SYSTEM_SETTINGS_ENV_VAR] = settings_path

        result = provider_process.run_provider_cli(
            command,
            timeout=timeout,
            env=env,
            label=f"Gemini CLI {normalized_model}",
        )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # Filter known benign Gemini CLI tool warnings (YOLO mode, ripgrep fallback, etc.)
    # These are non-fatal and should not cause the call to fail
    BENIGN_WARNINGS = [
        "YOLO mode is enabled",
        "All tool calls will be automatically approved",
        "Ripgrep is not available",
        "Falling back to GrepTool",
        "ripgrep",
    ]

    def _strip_benign_warnings(text: str) -> str:
        if not text:
            return ""
        lines = text.splitlines()
        filtered = [line for line in lines if not any(w.lower() in line.lower() for w in BENIGN_WARNINGS)]
        return "\n".join(filtered).strip()

    stdout = _strip_benign_warnings(stdout)
    stderr = _strip_benign_warnings(stderr)
    payload_text = stdout or stderr

    parsed_payload: Dict[str, Any] | None = None
    if payload_text:
        try:
            candidate = json.loads(payload_text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed_payload = candidate

    if result.returncode != 0:
        if parsed_payload and isinstance(parsed_payload.get("error"), dict):
            error_block = parsed_payload["error"]
            error_text = compact_text(
                str(error_block.get("message") or error_block.get("type") or payload_text),
                280,
            )
        else:
            error_text = compact_text(stderr or stdout or f"exit code {result.returncode}", 280)
        # Only raise if we still have a real error after filtering warnings
        if error_text and not any(w.lower() in error_text.lower() for w in BENIGN_WARNINGS):
            raise RuntimeError(f"Gemini CLI failed for model {normalized_model}: {error_text}")

    if parsed_payload and isinstance(parsed_payload.get("error"), dict):
        error_block = parsed_payload["error"]
        error_text = compact_text(
            str(error_block.get("message") or error_block.get("type") or payload_text),
            280,
        )
        if error_text and not any(w.lower() in error_text.lower() for w in BENIGN_WARNINGS):
            raise RuntimeError(f"Gemini CLI returned an error for model {normalized_model}: {error_text}")

    usage_metadata, usage_source = extract_gemini_usage_metadata_with_source(parsed_payload or {})
    if parsed_payload and str(parsed_payload.get("response", "")).strip():
        return build_text_generation_result(
            str(parsed_payload["response"]).strip(),
            usage_metadata,
            "google_genai",
            normalized_model,
            usage_source,
        )

    if not payload_text:
        raise RuntimeError(f"Gemini CLI returned an empty response for model {normalized_model}")
    return build_text_generation_result(
        payload_text,
        usage_metadata,
        "google_genai",
        normalized_model,
        usage_source,
    )


def invoke_gemini_cli(
    model: str,
    prompt: str,
    *,
    timeout: int = 120,
    temperature: float = 0.0,
) -> str:
    return invoke_gemini_cli_with_usage(
        model,
        prompt,
        timeout=timeout,
        temperature=temperature,
    )["text"]


def ensure_gemini_cli_ready(model: str) -> None:
    normalized_model = normalize_model_name(model)
    cached_error = GEMINI_PREFLIGHT_RESULTS.get(normalized_model)
    if cached_error is not None:
        if cached_error:
            raise RuntimeError(cached_error)
        return

    try:
        ensure_gemini_network_ready()
    except Exception as exc:
        error_text = f"Gemini network preflight failed: {exc}"
        GEMINI_PREFLIGHT_RESULTS[normalized_model] = error_text
        raise RuntimeError(error_text) from exc

    GEMINI_PREFLIGHT_RESULTS[normalized_model] = ""


def gemini_generate(
    model: str,
    prompt: str,
    *,
    timeout: int = 120,
    temperature: float = 0.0,
) -> str:
    return gemini_generate_with_usage(
        model,
        prompt,
        timeout=timeout,
        temperature=temperature,
    )["text"]


def gemini_generate_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 120,
    temperature: float = 0.0,
) -> TextGenerationResult:
    ensure_gemini_cli_ready(model)
    return invoke_gemini_cli_with_usage(model, prompt, timeout=timeout, temperature=temperature)
