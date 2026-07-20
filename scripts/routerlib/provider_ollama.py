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



def ollama_api_endpoint(path: str) -> str:
    normalized_url = OLLAMA_URL.rstrip("/")
    base, separator, _tail = normalized_url.partition("/api/")
    if separator:
        return f"{base}/api/{path.lstrip('/')}"
    if normalized_url.endswith("/api"):
        return f"{normalized_url}/{path.lstrip('/')}"
    return f"{normalized_url}/api/{path.lstrip('/')}"


def read_ollama_json(url: str, *, timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama timed out after {timeout}s while reading {url}") from exc


def list_ollama_models(timeout: int = 5) -> List[str]:
    data = read_ollama_json(ollama_api_endpoint("tags"), timeout=timeout)
    models = data.get("models", [])
    if not isinstance(models, list):
        return []

    model_names: List[str] = []
    for model_info in models:
        if not isinstance(model_info, dict):
            continue
        name = model_info.get("name") or model_info.get("model")
        if isinstance(name, str) and name.strip():
            model_names.append(name.strip())
    return model_names


def ollama_model_not_found(exc: urllib.error.HTTPError, body: str) -> bool:
    lowered = body.lower()
    return exc.code == 404 and "model" in lowered and "not found" in lowered


def request_ollama_generate(
    model: str,
    prompt: str,
    *,
    timeout: int,
    num_predict: int,
    temperature: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        ollama_api_endpoint("generate"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_ollama_num_predict(model: str, num_predict: int) -> int:
    # Large models need more tokens for structured output like JSON.
    if "gemma4" in model.lower() and num_predict < 204800:
        print(f"[ollama_generate] Auto-increased num_predict to 204800 for large model {model}")
        return 204800
    return num_predict


def ollama_generate_with_usage(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
) -> TextGenerationResult:
    actual_model = normalize_model_name(model)
    resolved_num_predict = resolve_ollama_num_predict(actual_model, num_predict)

    try:
        data = request_ollama_generate(
            actual_model,
            prompt,
            timeout=timeout,
            num_predict=resolved_num_predict,
            temperature=temperature,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if not ollama_model_not_found(exc, body):
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc

        available_models = list_ollama_models(timeout=max(1, min(int(timeout), 5)))
        if not available_models:
            raise RuntimeError(
                f"Ollama model {actual_model!r} was not found and /api/tags returned no installed models."
            ) from exc

        fallback_model = available_models[0]
        print(
            f"[ollama_generate] Requested model {actual_model!r} was not found; "
            f"retrying with installed model {fallback_model!r}."
        )
        actual_model = fallback_model
        resolved_num_predict = resolve_ollama_num_predict(actual_model, num_predict)
        try:
            data = request_ollama_generate(
                actual_model,
                prompt,
                timeout=timeout,
                num_predict=resolved_num_predict,
                temperature=temperature,
            )
        except urllib.error.HTTPError as fallback_exc:
            fallback_body = fallback_exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {fallback_exc.code}: {fallback_body}") from fallback_exc
        except urllib.error.URLError as fallback_exc:
            raise RuntimeError(f"Unable to reach Ollama at {OLLAMA_URL}: {fallback_exc.reason}") from fallback_exc
        except TimeoutError as fallback_exc:
            raise RuntimeError(f"Ollama timed out after {timeout}s") from fallback_exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {OLLAMA_URL}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Ollama timed out after {timeout}s") from exc

    if router_debug_enabled():
        print(f"\n[DEBUG ollama_generate] Model: {actual_model}, Timeout: {timeout}s, num_predict: {resolved_num_predict}")
        print(f"  Raw data keys: {list(data.keys())}")
        print(f"  'response' field length: {len(str(data.get('response', '')))} chars")
        print(f"  'response' first 500 chars: {str(data.get('response', ''))[:500]}")
        if len(str(data.get('response', ''))) > 500:
            print(f"  'response' last 200 chars: {str(data.get('response', ''))[-200:]}")
        print(f"  Metadata: eval_count={data.get('eval_count')}, prompt_eval_count={data.get('prompt_eval_count')}")
        print(f"  done={data.get('done')}, done_reason={data.get('done_reason')}")
        print(f"  Text after strip: {len(str(data.get('response', '')).strip())} chars")
        print(f"  ---\n")
    
    text = str(data.get("response", "")).strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty response for model {actual_model}")
    return build_text_generation_result(
        text,
        extract_ollama_usage_metadata(data),
        "ollama",
        actual_model,
        "ollama_generate",
    )


def ollama_generate(
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    num_predict: int = 400,
    temperature: float = 0.0,
) -> str:
    return ollama_generate_with_usage(
        model,
        prompt,
        timeout=timeout,
        num_predict=num_predict,
        temperature=temperature,
    )["text"]
