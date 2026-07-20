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



def provider_cli_name(command: List[str]) -> str:
    if not command:
        return "provider-cli"
    return os.path.basename(command[0]) or command[0]


def terminate_provider_process(process: subprocess.Popen, label: str, grace_timeout: int) -> None:
    print(f"[Provider CLI] {label}: timeout reached; terminating process group for pid={process.pid}.")
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.terminate()
    else:
        process.terminate()

    try:
        process.communicate(timeout=grace_timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[Provider CLI] {label}: process group ignored SIGTERM; sending SIGKILL.")

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
    else:
        process.kill()
    process.communicate()


def run_provider_cli(
    command: List[str],
    *,
    input_text: str | None = None,
    timeout: int,
    env: Dict[str, str],
    label: str,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    effective_timeout = timeout_with_run_deadline(timeout)
    grace_timeout = resolve_positive_int(
        None,
        ROUTER_PROVIDER_TERMINATION_GRACE_ENV_VAR,
        DEFAULT_PROVIDER_TERMINATION_GRACE,
    )
    stdin = subprocess.PIPE if input_text is not None else subprocess.DEVNULL
    popen_kwargs: Dict[str, Any] = {
        "stdin": stdin,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
    }
    if cwd:
        popen_kwargs["cwd"] = cwd
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    print(
        f"[Provider CLI] {label}: launching {provider_cli_name(command)} "
        f"timeout={effective_timeout}s."
    )
    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=effective_timeout)
    except subprocess.TimeoutExpired as exc:
        terminate_provider_process(process, label, grace_timeout)
        raise RuntimeError(f"{label} timed out after {effective_timeout}s") from exc

    print(f"[Provider CLI] {label}: exit={process.returncode}.")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout or "",
        stderr or "",
    )


def has_proxy_config() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
    )
