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
from . import provider_process  # noqa: F401
from . import config  # noqa: F401



def codex_generate_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
) -> TextGenerationResult:
    del num_predict, temperature

    normalized_model = normalize_model_name(model)
    if os.path.sep in config.CODEX_CLI_PATH and not os.path.exists(config.CODEX_CLI_PATH):
        raise RuntimeError("Codex CLI executable was not found. Set ROUTER_CODEX_CLI or install `codex`.")

    env = dict(os.environ)
    env["NO_COLOR"] = "1"
    sandbox = os.environ.get(ROUTER_CODEX_SANDBOX_ENV_VAR, "read-only").strip() or "read-only"
    command = [
        config.CODEX_CLI_PATH,
        "exec",
        "-m",
        normalized_model,
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
        "--color",
        "never",
        "--ephemeral",
    ]
    codex_cwd = os.environ.get(ROUTER_CODEX_CWD_ENV_VAR, "").strip()
    if codex_cwd:
        command.extend(["--cd", codex_cwd])

    with tempfile.TemporaryDirectory(prefix="router-codex-") as output_dir:
        output_path = os.path.join(output_dir, "last-message.txt")
        command.extend(["--output-last-message", output_path, "-"])
        result = provider_process.run_provider_cli(
            command,
            input_text=prompt,
            timeout=timeout,
            env=env,
            label=f"Codex CLI {normalized_model}",
        )
        output_text = ""
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as output_file:
                output_text = output_file.read().strip()

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        error_text = compact_text(stderr or stdout or f"exit code {result.returncode}", 280)
        raise RuntimeError(f"Codex CLI failed for model {normalized_model}: {error_text}")

    text = output_text or stdout
    if not text.strip():
        raise RuntimeError(f"Codex CLI returned an empty response for model {normalized_model}")
    return build_text_generation_result(
        text.strip(),
        {},
        "codex",
        normalized_model,
        "unavailable",
    )


def codex_generate(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
) -> str:
    return codex_generate_with_usage(
        model,
        prompt,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
    )["text"]
