---
name: super-router
description: LangGraph-based intelligent task router that splits work between PRO (heavy reasoning) and FLASH (fast) models using 5-dimension complexity scoring, configurable model defaults, and FLASH→PRO escalation.
version: 1.0.0
author: Yadan Fan
license: MIT
metadata:
  hermes:
    tags: [model-routing, langgraph, ollama, task-decomposition, multi-model, complexity-scoring]
    related_skills: [dspy, subagent-driven-development, llama-cpp]
---

# Super Router (LangGraph Edition)

Intelligent task decomposition and model routing using LangGraph StateGraph. Automatically routes subtasks between PRO (heavy reasoning) and FLASH (fast) models based on structured complexity assessment.

## When to Use This Skill

Use super-router when you need:
- **Intelligent model routing** — automatically choose between heavy (PRO) and fast (FLASH) models per subtask
- **Task decomposition** — break complex tasks into structured subtasks with independent routing
- **Cost optimization** — use fast models for simple work, heavy models only when needed
- **Configurable models** — use deterministic defaults, with environment-variable overrides for each role
- **Failure escalation** — FLASH retry on infra failures, escalate to PRO on capability failures
- **Audit trail** — full logging of planned vs actual routes, retries, and failure classifications

**Not needed for:** Simple single-turn tasks, tasks where you already know which model to use, or when you want manual control over every routing decision.

## Optimization for Parallelism
To achieve true parallel execution (when `ROUTER_MAX_CONCURRENCY > 1`), the Planner must be instructed to use **Atomic Decomposition**. 

- **Atomic Decomposition**: Breaking a task into the smallest possible independent units (e.g., 10 separate research tasks for 10 companies) rather than "phases" (e.g., one giant 'Research' phase encompassing all companies).
- **Pitfall: Planner Grouping**: Even with explicit instructions, the Planner may occasionally group multiple entities into a single subtask, which kills true parallelism.
- **Verification**: Always verify the `planned_subtasks` count matches the entity count. If the planner groups entities, it should be treated as a capability failure and forced to retry with a correction prompt.
- **Benefit**: This prevents 'lost-in-the-middle' failures and allows executor branches to fire multiple requests simultaneously, significantly reducing wall-clock time.
- **Deferred synthesis**: Summary, reporting, and synthesis subtasks are held until independent executor branches finish, so they receive completed context instead of racing ahead without findings.
- **Implementation**: When prompting the router for multi-entity tasks, explicitly demand: *"Decompose this into exactly X independent subtasks—one subtask per entity. Do not group them into a single phase."*

## Core Architecture (LangGraph StateGraph)

| Node | Function |
|------|----------|
| **Planner** | Decomposes original task into a JSON array of atomic, actionable subtasks. Uses Atomic Decomposition to split multi-entity tasks (e.g., 10 providers $\rightarrow$ 10 subtasks) for maximum parallelism. |
| **Judge** | Scores each subtask on 5 dimensions: `reasoning_depth`, `code_change_scope`, `ambiguity`, `risk`, `io_heaviness`; combines with thresholds + confidence to decide PRO/FLASH |
| **Executor Fanout** | Uses LangGraph `Send(...)` to dispatch independent subtasks concurrently, then joins ordered results by original step number |
| **PRO Executor Branch** | Heavy reasoning model (default: Gemini CLI preview model; override via `ROUTER_PRO_MODEL`) |
| **FLASH Executor Branch** | Fast model with review/retry logic (default: Gemini CLI preview model; override via `ROUTER_FLASH_MODEL`) |
| **FLASH Review** | Validates output quality; distinguishes infra failures (timeout, network) from capability failures; retries FLASH or escalates to PRO |
| **Metadata Extractor** | Extracts 'Technical Gold' (atomic high-precision facts) from step output to prevent finalizer timeouts and loss of detail |
| **Recorder/Finalizer** | Logs every step; compiles final report using a hybrid of Technical Gold and full audit trails; supports FLASH→PRO→deterministic fallback chain |

## Installation

```bash
# Required: LangGraph
pip install langgraph

# If you use Ollama-backed roles, ensure Ollama is running locally
ollama serve

# Pull recommended models if you use Ollama-backed roles
ollama pull gemma4:26b     # Planner or PRO executor (high quality, slow)
ollama pull llama3.1:8b    # Judge (fast scoring, recommended)
ollama pull qwen3         # PRO executor
ollama pull qwen2.5:7b    # FLASH executor
```

By default, planner and judge use Gemini CLI-backed models. To opt into local Ollama models, keep the Planner strong and use a smaller Judge:

```bash
export ROUTER_PLANNER_MODEL=gemma4:26b
export ROUTER_JUDGE_MODEL=llama3.1:8b
export ROUTER_PRO_MODEL=gemma4:26b
export ROUTER_FLASH_MODEL=qwen2.5:7b
```

If you intentionally want an all-`gemma4:26b` Planner/Judge/PRO setup, use longer timeouts and serialized graph execution:

```bash
export ROUTER_PLANNER_MODEL=gemma4:26b
export ROUTER_JUDGE_MODEL=gemma4:26b
export ROUTER_PRO_MODEL=gemma4:26b
export ROUTER_FLASH_MODEL=qwen2.5:7b
export ROUTER_JUDGE_TIMEOUT=600
export ROUTER_MAX_CONCURRENCY=1
```

## Usage

### Configuring ROUTER_* Variables

All `ROUTER_*` environment variables are set in `~/.hermes/.env`. Hermes Agent loads this file at startup and injects the values into every `terminal()` child process automatically — no explicit `env={}` passthrough needed. Router.py reads them via `os.environ.get()` with its own defaults for any unset variables.

To see current values: `cat ~/.hermes/.env | grep ROUTER_`

### Basic Usage (via exec)

When the user says "走 super-router", "use super-router", or asks for router analysis, invoke router.py directly — the `.env` values are already in the environment:

```python
terminal(
    command="/opt/homebrew/Caskroom/miniforge/base/bin/python ~/.hermes/skills/mlops/inference/super-router/scripts/router.py 'Analyze K8s YAML errors and rewrite config'"
)
```

### With Streaming (Node-Level Progress)

```python
terminal(command="/opt/homebrew/Caskroom/miniforge/base/bin/python ~/.hermes/skills/mlops/inference/super-router/scripts/router.py --stream 'Your complex task'")
```

### Via Environment Variable (Agent Compatibility)

For agents that struggle with non-ASCII arguments, pass the task via `ROUTER_TASK`:

```python
terminal(command="/opt/homebrew/Caskroom/miniforge/base/bin/python ~/.hermes/skills/mlops/inference/super-router/scripts/router.py",
         env={"ROUTER_TASK": "Your complex task description"})
```

### Handling Long-Running Execution

If `exec` returns "Command still running":

```python
# Continue polling with process tool
process(action="poll", session_id="<session_id_from_exec>")

# Wait for completion
process(action="wait", session_id="<session_id_from_exec>", timeout=300)
```

**Important:** Once process shows completion, your next assistant message MUST start with `Router result:` or `Router failed:` and include at least one real detail from the output (e.g., "Planner fallback", "Ollama timed out", "BTC"). Never reply with just `---`, punctuation, or empty lines.

## Environment Variables

All `ROUTER_*` variables are loaded from `~/.hermes/.env` by the Hermes runtime and injected into every `terminal()` child process. Router.py reads them via `os.environ.get()` with its own defaults for unset variables.

| Variable | Purpose | Default |
|----------|---------|---------|
| `ROUTER_PLANNER_MODEL` | Task decomposition model | `google-gemini-cli/gemini-3-pro-preview` |
| `ROUTER_JUDGE_MODEL` | Complexity scoring model | `google-gemini-cli/gemini-3-flash-preview` |
| `ROUTER_PRO_MODEL` | Heavy reasoning executor | `google-gemini-cli/gemini-3-pro-preview` |
| `ROUTER_FLASH_MODEL` | Fast executor | `google-gemini-cli/gemini-3-flash-preview` |
| `ROUTER_PRO_FALLBACK_MODELS` | Comma-separated PRO fallback list | None |
| `ROUTER_FLASH_FALLBACK_MODELS` | Comma-separated FLASH fallback list | None |
| `ROUTER_FLASH_RETRY_BUDGET` | Max FLASH retries before escalation | 1 |
| `ROUTER_RECURSION_LIMIT` | Python recursion limit | 128 |
| `ROUTER_JUDGE_TIMEOUT` | Timeout for Judge node LLM calls (seconds) | 300 (up to 6000 for extremely complex tasks with large models) |
| `ROUTER_MAX_CONCURRENCY` | Max concurrent LangGraph branches for judge and executor fanout. Essential for multi-entity atomic tasks; set to `1` for local 26B+ Judge models or constrained hardware. | Auto (`1` for large Judge models) |
| `ROUTER_GEMINI_CLI` | Path to Gemini CLI (if using instead of Ollama) | `/opt/homebrew/bin/gemini` |
| `ROUTER_GEMINI_EXTENSION` | Gemini CLI extension name used with `-e`; `superpowers` is the Gemini extension | `superpowers` |
| `ROUTER_OLLAMA_URL` | Ollama API endpoint | `http://localhost:11434/api/generate` |
| `ROUTER_FINALIZER_TIMEOUT` | Timeout for the final reporting synthesis (seconds). Essential to set high (e.g., 600) for complex tasks to avoid timeouts during context assembly. | 600 |
| `ROUTER_DEBUG` | Print raw planner/judge/Ollama diagnostic snippets | Off |

**For optional large local models (20B+ like gemma4:26b):**
- Prefer `ROUTER_PLANNER_MODEL=gemma4:26b` with `ROUTER_JUDGE_MODEL=llama3.1:8b`
- If using `ROUTER_JUDGE_MODEL=gemma4:26b`, set `ROUTER_JUDGE_TIMEOUT=600` and keep `ROUTER_MAX_CONCURRENCY=1`
- Planner timeout is auto-set to 300s for large models
- Expect 2-5 minute wait times per LLM call
- Model warmup adds ~30-60s upfront but prevents timeouts.
- **Crucial:** A 60s terminal timeout can still kill the run even if internal router timeouts are higher. Use `--stream`, process polling via `process(action='poll')`, and a longer terminal/process wait timeout for large Planner/Judge runs.

## Complexity Routing Rules

### 5-Dimension Scoring

The Judge scores each subtask on:

1. **reasoning_depth** (1-10): How much logical inference is needed?
2. **code_change_scope** (1-10): How many files/lines of code to modify?
3. **ambiguity** (1-10): How unclear is the task specification?
4. **risk** (1-10): What's the impact of getting this wrong?
5. **io_heaviness** (1-10): How much reading/writing vs. thinking?

### Routing Thresholds

| Condition | Route |
|-----------|-------|
| `complexity_score >= 5` | PRO |
| `complexity_score <= 2` | FLASH |
| Summary-like task (no deep work) | FLASH |
| High-risk incident diagnosis | PRO |
| High-risk evidence gathering | PRO |
| High-risk decision/rollback evaluation | PRO |
| Boundary case + low confidence | PRO (safe default) |

### Contextual Score Biases

The router applies automatic adjustments:
- **High-risk context** (production, billing, security): boosts `reasoning_depth`, `risk`, `ambiguity`
- **Evidence gathering** in incident: keeps on PRO (not mere IO)
- **Communication/summary** subtasks: routed to FLASH unless deep work is also required

## FLASH Review & Escalation Logic

When FLASH execution fails or produces questionable output:

1. **Classify failure type:**
   - `infra_transient`: timeout, network, rate limit, service unavailable
   - `capability_quality`: "need more info", empty output, too short, repeated task

2. **Decision:**
   - Infra failure -> Retry FLASH (up to `ROUTER_FLASH_RETRY_BUDGET`)
   - Capability failure -> Escalate to PRO immediately
   - Unknown -> Retry once, then escalate

3. **Post-execution verification:**
   - Empty output -> escalate
   - Output < 48 chars (non-summary) -> escalate
   - Output explicitly says "can't complete" -> escalate
   - Output just repeats task description -> escalate

## Finalizer Fallback Chain

Final report generation follows:

```
FLASH finalizer -> (if fails) -> PRO finalizer -> (if fails) -> Deterministic template
```

## Output Structure

- **Output Structure**: The router returns a JSON-serializable state. When summarizing these results in reports or documentation, always use ASCII/Terminal-style arrows (e.g., '-->', '->') rather than mathematical arrows (e.g., '→', '$\\rightarrow$') for all diagrams and flow representations. This is a high-priority stylistic requirement.

```json
{
  "task": "original task string",
  "planner_model": "model name used for planning",
  "judge_model": "model name used for complexity scoring",
  "pro_model": "primary PRO model",
  "flash_model": "primary FLASH model",
  "planned_subtasks": [{"desc": "..."}],
  "subtasks": [
    {
      "desc": "...",
      "model": "PRO|FLASH",
      "assessment": {
        "scores": {"reasoning_depth": 5, "code_change_scope": 3, "ambiguity": 2, "risk": 4, "io_heaviness": 1},
        "complexity_score": 15,
        "suggested_route": "PRO",
        "final_route": "PRO",
        "confidence": 0.85,
        "reason": "...",
        "judge_source": "llm|heuristic"
      }
    }
  ],
  "results": [
    {
      "step": 1,
      "planned_route": "PRO",
      "route": "PRO",
      "model_name": "qwen3",
      "desc": "...",
      "output": "...",
      "status": "success|failed",
      "attempt_count": 1,
      "retry_count": 0,
      "escalated_from_flash": false,
      "used_provider_fallback": false,
      "flash_review": {"decision": "record", "failure_type": "none", "reason": "..."},
      "attempt_log": ["..."]
    }
  ],
  "final_report": "...",
  "finalizer_outcome": {
    "route": "FLASH|PRO|DETERMINISTIC",
    "model_name": "...",
    "status": "...",
    "used_provider_fallback": false,
    "reason": "...",
    "attempt_log": ["..."]
  }
}
```

## Maintenance

| File | Purpose |
|------|---------|
| `scripts/router.py` | Main LangGraph router script |
| `SKILL.md` | This documentation |

## Troubleshooting


### Verifying Model Routing

To audit which model each stage actually used, run with `--stream` and check the output:

- **Planner model**: printed as `规划模型: <model>` in the routing plan summary
- **Judge model**: printed as `判定模型: <model>` in the routing plan summary
- **PRO Executor**: printed as `-> <model>` per step in executor fanout (e.g. `🧠 [PRO] Step 1/5 -> google-gemini-cli/gemini-3-pro-preview`)
- **FLASH Executor**: printed as `-> <model>` per step (e.g. `⚡ [FLASH] Step 4/5 -> google-gemini-cli/gemini-3-flash-preview`)
- **Metadata Extractor**: uses the PRO model (verified via code at `extract_technical_metadata_for_result()`, not printed to stream)
- **Finalizer**: routed to FLASH first, then PRO on failure — check `finalizer_outcome.route` / `finalizer_outcome.model_name` in the returned state

### Non-Determinism with Cloud APIs

Even at `temperature=0.0`, cloud-hosted models (Gemini, OpenRouter) may produce different decompositions across runs due to GPU floating-point non-determinism and backend inference differences. The planner prompt and temperature are identical, but the same task can yield 5 subtasks on one run and 6 on another. This is expected — the router is deterministic in its *routing logic*, not in upstream model sampling. For guaranteed reproducibility, cache planner results by task hash or use a seed parameter if the provider supports it.


### "Router timed out" / "Ollama returned an empty response"
- **Best fix when using a large Ollama Planner:** keep `ROUTER_PLANNER_MODEL=gemma4:26b`, but set `ROUTER_JUDGE_MODEL=llama3.1:8b`.
- **All-gemma mode:** set `ROUTER_JUDGE_MODEL=gemma4:26b`, `ROUTER_JUDGE_TIMEOUT=600`, and `ROUTER_MAX_CONCURRENCY=1`; expect much longer runs.
- Use `--stream` and increase the terminal/process timeout if the Planner itself may take longer than 60s.
- Set `ROUTER_JUDGE_TIMEOUT=300` or higher only when intentionally using a 20B+ Judge.
- If Ollama appears unexpectedly, check whether Hermes injected stale `ROUTER_PLANNER_MODEL` or `ROUTER_JUDGE_MODEL` values.

### "Planner timed out after 30s" (or 90s)
- Model is too large or not loaded. Warmup helps but large models may still timeout.
- Use `--stream` plus a longer terminal/process timeout, or choose a smaller planner model.
- Check Ollama logs: `ollama serve` output for errors

### Gemini CLI Warmup Timeout Issues
The `planner_warmup_node` and `judge_warmup_node` both use a hard-coded `timeout=180` for the "OK" ping. When using Gemini CLI this frequently fails because the CLI takes >180s to respond in non-interactive sessions.

**Fix applied:** Raised warmup timeout from 180s → 300s in both nodes. If Gemini CLI warmup continues to fail, consider skipping warmup entirely for CLI-based models or increasing to 420s+ in future revisions.

### "FLASH kept escalating to PRO"
- Task may genuinely require heavy reasoning
- Check if FLASH model is too small for your tasks
- Try setting `ROUTER_FLASH_MODEL` to a larger model

### "Gemini CLI AbortError or Auth Failures"
- If gemini-cli returns AbortError or authentication errors in non-interactive sessions, this is often an infrastructure/API timeout or session issue.
- Use `--stream` to monitor real-time progress and ensure ROUTER_JUDGE_TIMEOUT and terminal timeouts are sufficiently high to prevent external process termination.

### "Planner produced only one subtask"
- Task may be simple enough to not need decomposition
- Planner model may be too small; try `ROUTER_PLANNER_MODEL=gemma4:31b` (if you have the patience for 90s+ waits)

### "Planner produced different decompositions across identical runs"
- When using Gemini CLI as planner, `temperature` is **not** passed through to the model — it falls back to Gemini's built-in default (typically >0). This causes non-deterministic output: the same task can decompose into 5 steps on one run and 6 on another.
- Root cause: `generate_text()` passes `temperature=0.0` to `ollama_generate()` but drops the parameter entirely when routing to `gemini_generate()`.
- Workaround: for reproducibility-critical tasks, use an Ollama-backed planner (temperature=0.0 IS respected there). Otherwise, accept minor variation across runs.
- See `references/gemini-temperature-gap.md` for full code trace.

## Related Skills

- **dspy** — Declarative LM programming with automatic prompt optimization (Python framework alternative)
- **subagent-driven-development** — Task decomposition with Hermes delegate_task + two-stage review
- **llama-cpp** — Run LLM inference locally (alternative to Ollama backend)

## See Also

- LangGraph documentation: https://langchain-ai.github.io/langgraph/
- Ollama documentation: https://ollama.com/docs
