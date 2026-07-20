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
from .provider_claude import *  # noqa: F401,F403
from . import provider_gemini  # noqa: F401
from . import provider_codex  # noqa: F401
from . import provider_claude  # noqa: F401
from . import langsmith_integration  # noqa: F401



def _execute_generate_text_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
    usage_label: str = "",
) -> TextGenerationResult:
    provider = langsmith_provider_name(model)
    annotate_langsmith_model_run(
        model=model,
        provider=provider,
        num_predict=num_predict,
        temperature=temperature,
    )
    if is_claude_model(model):
        return provider_claude.claude_generate_with_usage(model, prompt, timeout=timeout, temperature=temperature)
    if is_gemini_model(model):
        return provider_gemini.gemini_generate_with_usage(model, prompt, timeout=timeout, temperature=temperature)
    if is_codex_model(model):
        return provider_codex.codex_generate_with_usage(
            model,
            prompt,
            timeout=timeout,
            num_predict=num_predict,
            temperature=temperature,
        )
    return ollama_generate_with_usage(
        model,
        prompt,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
    )


if _langsmith is not None and getattr(_langsmith, "traceable", None) is not None:
    _traced_generate_text = _langsmith.traceable(
        name="Super Router Model Call",
        run_type="llm",
        process_inputs=process_langsmith_model_inputs,
        process_outputs=process_langsmith_model_outputs,
    )(_execute_generate_text_with_usage)
else:
    _traced_generate_text = _execute_generate_text_with_usage


def unwrap_text_generation_result(result: Any) -> str:
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def generate_text(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
    usage_label: str = "",
) -> str:
    effective_timeout = timeout_with_run_deadline(timeout)
    if langsmith_integration.langsmith_tracing_configured():
        result = _traced_generate_text(
            model,
            prompt,
            timeout=effective_timeout,
            num_predict=num_predict,
            temperature=temperature,
            usage_label=usage_label,
        )
        if isinstance(result, dict):
            record_token_usage(
                result,
                label=usage_label or normalize_model_name(model),
                prompt=prompt,
            )
        return unwrap_text_generation_result(result)
    result = _execute_generate_text_with_usage(
        model,
        prompt,
        timeout=effective_timeout,
        num_predict=num_predict,
        temperature=temperature,
        usage_label=usage_label,
    )
    record_token_usage(
        result,
        label=usage_label or normalize_model_name(model),
        prompt=prompt,
    )
    return unwrap_text_generation_result(result)
