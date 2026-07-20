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

RUN_DEADLINE_MONOTONIC: contextvars.ContextVar[float] = contextvars.ContextVar(
    "RUN_DEADLINE_MONOTONIC",
    default=0.0,
)
GLOBAL_RUN_DEADLINE_MONOTONIC = 0.0


def resolve_model(explicit_value: str | None, env_name: str, fallback: str) -> str:
    if explicit_value and explicit_value.strip():
        return explicit_value.strip()
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    global_model = os.environ.get(ROUTER_MODEL_ENV_VAR, "").strip()
    return global_model or fallback


def resolve_execution_model(explicit_value: str | None, env_name: str, fallback: str) -> str:
    selected = resolve_model(explicit_value, env_name, fallback)
    uses_default = (
        not (explicit_value and explicit_value.strip())
        and not os.environ.get(env_name, "").strip()
    )
    if uses_default and any(hint in selected.lower() for hint in (":26b", ":31b", ":70b")):
        print(
            f"[Execution Model] ⚠️ Using {selected} - large model may be slow, "
            f"consider setting {env_name} explicitly"
        )
    return selected


def resolve_non_negative_int(explicit_value: int | None, env_name: str, fallback: int) -> int:
    if explicit_value is not None:
        return max(0, explicit_value)
    env_value = os.environ.get(env_name, "").strip()
    if not env_value:
        return fallback
    try:
        parsed = int(env_value)
    except ValueError:
        return fallback
    return max(0, parsed)


def resolve_positive_int(explicit_value: int | None, env_name: str, fallback: int) -> int:
    if explicit_value is not None:
        return max(1, explicit_value)
    env_value = os.environ.get(env_name, "").strip()
    if not env_value:
        return fallback
    try:
        parsed = int(env_value)
    except ValueError:
        return fallback
    return max(1, parsed)


def resolve_optional_positive_int(explicit_value: int | None, env_name: str) -> int | None:
    if explicit_value is not None:
        return max(1, explicit_value)
    env_value = os.environ.get(env_name, "").strip()
    if not env_value:
        return None
    try:
        parsed = int(env_value)
    except ValueError:
        return None
    return max(1, parsed)


def resolve_bool(env_name: str, fallback: bool = False) -> bool:
    env_value = os.environ.get(env_name, "").strip().lower()
    if not env_value:
        return fallback
    return env_value in {"1", "true", "yes", "on", "debug"}


def router_debug_enabled() -> bool:
    return resolve_bool("ROUTER_DEBUG")


def resolve_executor_timeout(route: Literal["PRO", "FLASH"]) -> int:
    default_timeout = DEFAULT_PRO_EXECUTION_TIMEOUT if route == PRO else DEFAULT_FLASH_EXECUTION_TIMEOUT
    route_env = (
        ROUTER_PRO_EXECUTOR_TIMEOUT_ENV_VAR
        if route == PRO
        else ROUTER_FLASH_EXECUTOR_TIMEOUT_ENV_VAR
    )
    shared_timeout = resolve_positive_int(None, ROUTER_EXECUTOR_TIMEOUT_ENV_VAR, default_timeout)
    return resolve_positive_int(None, route_env, shared_timeout)


def current_run_deadline() -> float:
    return RUN_DEADLINE_MONOTONIC.get() or GLOBAL_RUN_DEADLINE_MONOTONIC


def check_run_deadline() -> None:
    deadline = current_run_deadline()
    if deadline and time.monotonic() >= deadline:
        raise RuntimeError("Router run deadline exceeded.")


def timeout_with_run_deadline(timeout: int) -> int:
    check_run_deadline()
    deadline = current_run_deadline()
    if not deadline:
        return max(1, timeout)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Router run deadline exceeded.")
    return max(1, min(max(1, timeout), int(remaining + 0.999)))


@contextlib.contextmanager
def router_run_deadline_context(timeout: int) -> Iterator[None]:
    global GLOBAL_RUN_DEADLINE_MONOTONIC

    if timeout <= 0:
        yield
        return

    deadline = time.monotonic() + timeout
    token = RUN_DEADLINE_MONOTONIC.set(deadline)
    previous_global_deadline = GLOBAL_RUN_DEADLINE_MONOTONIC
    GLOBAL_RUN_DEADLINE_MONOTONIC = deadline
    print(f"[Router Deadline] Whole-run deadline enabled: {timeout}s.")
    try:
        yield
    finally:
        RUN_DEADLINE_MONOTONIC.reset(token)
        GLOBAL_RUN_DEADLINE_MONOTONIC = previous_global_deadline


def resolve_model_list(explicit_values: List[str] | None, env_name: str) -> List[str]:
    raw_values = explicit_values
    if raw_values is None:
        env_value = os.environ.get(env_name, "").strip()
        raw_values = env_value.split(",") if env_value else []
    resolved: List[str] = []
    for value in raw_values:
        candidate = str(value).strip()
        if candidate and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: Any, minimum: float, maximum: float, default: float = 0.5) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def coerce_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed
