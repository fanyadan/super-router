import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import scripts.router as r


def run_quietly(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


class RouterHelperTests(unittest.TestCase):
    def test_initial_state_defaults_and_env_parsing(self):
        with mock.patch.dict(
            os.environ,
            {
                "ROUTER_MODEL": "",
                "ROUTER_PLANNER_MODEL": "",
                "ROUTER_JUDGE_MODEL": "",
                "ROUTER_PRO_MODEL": "",
                "ROUTER_FLASH_MODEL": "",
                "ROUTER_DEBUG": "",
                "ROUTER_FLASH_RETRY_BUDGET": "bad",
            },
            clear=False,
        ):
            state = r.create_initial_state("default check")

        self.assertEqual(state["planner_model"], r.DEFAULT_PLANNER_MODEL)
        self.assertEqual(state["judge_model"], r.DEFAULT_JUDGE_MODEL)
        self.assertEqual(state["pro_model"], r.DEFAULT_PRO_MODEL)
        self.assertEqual(state["flash_model"], r.DEFAULT_FLASH_MODEL)
        self.assertEqual(state["flash_retry_budget"], r.DEFAULT_FLASH_RETRY_BUDGET)
        self.assertFalse(r.router_debug_enabled())

        with mock.patch.dict(os.environ, {"ROUTER_DEBUG": "yes"}, clear=False):
            self.assertTrue(r.router_debug_enabled())

    def test_global_router_model_applies_to_all_roles_unless_overridden(self):
        with mock.patch.dict(
            os.environ,
            {
                "ROUTER_MODEL": "gpt-5.5",
                "ROUTER_PLANNER_MODEL": "",
                "ROUTER_JUDGE_MODEL": "",
                "ROUTER_PRO_MODEL": "",
                "ROUTER_FLASH_MODEL": "",
            },
            clear=True,
        ):
            state = r.create_initial_state("global model check")

        self.assertEqual(state["planner_model"], "gpt-5.5")
        self.assertEqual(state["judge_model"], "gpt-5.5")
        self.assertEqual(state["pro_model"], "gpt-5.5")
        self.assertEqual(state["flash_model"], "gpt-5.5")

        with mock.patch.dict(
            os.environ,
            {
                "ROUTER_MODEL": "gpt-5.5",
                "ROUTER_FLASH_MODEL": "google-gemini-cli/gemini-3-flash-preview",
            },
            clear=True,
        ):
            state = r.create_initial_state("role override check")

        self.assertEqual(state["planner_model"], "gpt-5.5")
        self.assertEqual(state["pro_model"], "gpt-5.5")
        self.assertEqual(state["flash_model"], "google-gemini-cli/gemini-3-flash-preview")

    def test_langsmith_config_metadata_and_tags(self):
        state = r.create_initial_state(
            "Trace this router run",
            planner_model="planner",
            judge_model="judge",
            pro_model="pro",
            flash_model="flash",
            pro_fallback_models=["pro2"],
            flash_retry_budget=2,
        )
        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_TAGS": "local,ci"},
            clear=False,
        ):
            config = r.build_graph_config(42, 1, state)

        self.assertEqual(config["recursion_limit"], 42)
        self.assertEqual(config["max_concurrency"], 1)
        self.assertEqual(config["run_name"], "super-router")
        self.assertEqual(config["tags"], ["super-router", "langgraph", "local", "ci"])
        self.assertEqual(config["metadata"]["planner_model"], "planner")
        self.assertEqual(config["metadata"]["judge_model"], "judge")
        self.assertEqual(config["metadata"]["pro_fallback_count"], 1)
        self.assertEqual(config["metadata"]["flash_retry_budget"], 2)
        self.assertEqual(config["metadata"]["task_chars"], len("Trace this router run"))

    def test_langsmith_model_trace_processors_hide_text_by_default(self):
        inputs = {
            "model": "google-gemini-cli/gemini-3-flash-preview",
            "prompt": "sensitive prompt body",
            "timeout": 12,
            "num_predict": 34,
            "temperature": 0.5,
        }
        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_TRACE_PROMPTS": "", "ROUTER_LANGSMITH_TRACE_OUTPUTS": ""},
            clear=False,
        ):
            processed = r.process_langsmith_model_inputs(inputs)

        self.assertEqual(processed["provider"], "google_genai")
        self.assertEqual(processed["transport"], "gemini_cli")
        self.assertEqual(processed["prompt_chars"], len("sensitive prompt body"))
        self.assertNotIn("sensitive prompt body", str(processed))

        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_TRACE_PROMPTS": "true", "ROUTER_LANGSMITH_TRACE_OUTPUTS": ""},
            clear=False,
        ):
            processed_with_preview = r.process_langsmith_model_inputs(inputs)

        self.assertIn("sensitive prompt body", processed_with_preview["prompt_preview"])
        self.assertEqual(
            r.process_langsmith_model_outputs("model output"),
            {"output_chars": len("model output")},
        )
        output_with_usage = {
            "text": "model output",
            "usage_metadata": {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
        }
        self.assertEqual(
            r.process_langsmith_model_outputs(output_with_usage),
            {
                "output_chars": len("model output"),
                "usage_metadata": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                },
            },
        )
        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_HIDE_INPUTS": "1", "ROUTER_LANGSMITH_HIDE_OUTPUTS": "1"},
            clear=False,
        ):
            self.assertEqual(r.process_langsmith_model_inputs(inputs), {})
            self.assertEqual(
                r.process_langsmith_model_outputs(output_with_usage),
                {
                    "usage_metadata": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                    },
                },
            )

    def test_langsmith_request_and_configuration_flags(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(r.langsmith_tracing_requested())
            self.assertFalse(r.langsmith_api_key_configured())

        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_ENABLED": "true", "LANGSMITH_API_KEY": "test-key"},
            clear=True,
        ):
            self.assertTrue(r.langsmith_tracing_requested())
            self.assertTrue(r.langsmith_api_key_configured())

        with mock.patch.dict(
            os.environ,
            {"ROUTER_LANGSMITH_ENABLED": "false", "LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-key"},
            clear=True,
        ):
            self.assertFalse(r.langsmith_tracing_requested())
            self.assertTrue(r.langsmith_tracing_forced_disabled())

    def test_json_extraction_and_planner_normalization(self):
        self.assertEqual(r.extract_first_json_array("prefix [{\"desc\":\"A\"}] suffix"), [{"desc": "A"}])
        self.assertEqual(r.extract_first_json_object("noise {\"confidence\": 0.8}"), {"confidence": 0.8})
        self.assertEqual(
            r.normalize_planned_subtasks([{"step": "Inspect"}, "Summarize", {"desc": ""}]),
            [{"desc": "Inspect"}, {"desc": "Summarize"}],
        )

        with self.assertRaises(ValueError):
            r.extract_first_json_array("no json here")

    def test_communication_subtask_split_and_route_biases(self):
        task = "Debug intermittent API failure and send a concise team update."
        planned = [{"desc": "Debug intermittent API failure and send a concise team update"}]
        expanded = r.ensure_communication_subtask(task, planned)

        self.assertEqual(len(expanded), 2)
        self.assertIn("Debug intermittent API failure", expanded[0]["desc"])
        self.assertTrue(r.is_summary_like_subtask(expanded[1]["desc"]))

        summary_scores = {
            "reasoning_depth": 0,
            "code_change_scope": 0,
            "ambiguity": 0,
            "risk": 0,
            "io_heaviness": 2,
        }
        self.assertEqual(
            r.decide_route(task, "Prepare a concise team update", summary_scores, r.FLASH, 0.9),
            r.FLASH,
        )

        high_risk = r.build_fallback_assessment(
            "production billing incident",
            "Inspect payment logs and reconcile duplicate charges",
        )
        self.assertEqual(high_risk["final_route"], r.PRO)
        self.assertEqual(high_risk["scores"]["risk"], 2)

    def test_generate_text_honors_gemini_timeout_and_temperature(self):
        captured = []

        def fake_gemini(model, prompt, *, timeout, temperature):
            captured.append((model, prompt, timeout, temperature))
            return r.build_text_generation_result("ok", {}, "google_genai", model)

        with mock.patch.object(r, "gemini_generate_with_usage", side_effect=fake_gemini):
            self.assertEqual(
                r.generate_text("google-gemini-cli/flash", "prompt", timeout=30, temperature=0.2),
                "ok",
            )

        self.assertEqual(captured, [("google-gemini-cli/flash", "prompt", 30, 0.2)])

    def test_generate_text_dispatches_codex_models(self):
        captured = []

        def fake_codex(model, prompt, *, timeout, num_predict, temperature):
            captured.append((model, prompt, timeout, num_predict, temperature))
            return r.build_text_generation_result("ok", {}, "codex", r.normalize_model_name(model))

        with mock.patch.object(r, "codex_generate_with_usage", side_effect=fake_codex):
            self.assertEqual(
                r.generate_text("codex/gpt-5.5", "prompt", timeout=30, num_predict=99, temperature=0.2),
                "ok",
            )

        self.assertEqual(captured, [("codex/gpt-5.5", "prompt", 30, 99, 0.2)])

    def test_generate_text_dispatches_claude_models(self):
        captured = []

        def fake_claude(model, prompt, *, timeout, temperature):
            captured.append((model, prompt, timeout, temperature))
            return r.build_text_generation_result("ok", {}, "anthropic", r.normalize_model_name(model))

        with mock.patch.object(r, "claude_generate_with_usage", side_effect=fake_claude):
            self.assertEqual(
                r.generate_text("claude/sonnet", "prompt", timeout=30, temperature=0.2),
                "ok",
            )

        self.assertEqual(captured, [("claude/sonnet", "prompt", 30, 0.2)])

    def test_provider_prefixes_take_precedence_over_bare_model_patterns(self):
        self.assertTrue(r.is_codex_model("gpt-5.5"))
        self.assertTrue(r.is_codex_model("codex/gpt-5.5"))
        self.assertFalse(r.is_codex_model("ollama/gpt-5.5"))
        self.assertTrue(r.is_claude_model("claude/sonnet"))
        self.assertTrue(r.is_claude_model("claude-3-5-sonnet-latest"))
        self.assertFalse(r.is_codex_model("claude/gpt-5.5"))
        self.assertFalse(r.is_gemini_model("codex/gemini-3-pro-preview"))
        self.assertEqual(r.langsmith_provider_name("ollama/gpt-5.5"), "ollama")
        self.assertEqual(r.model_transport_name("ollama/gpt-5.5"), "ollama_http")
        self.assertEqual(r.langsmith_provider_name("claude/sonnet"), "anthropic")
        self.assertEqual(r.model_transport_name("claude/sonnet"), "claude_cli")
        self.assertEqual(r.langsmith_provider_name("google-gemini-cli/gpt-5.5"), "google_genai")

    def test_invoke_gemini_cli_writes_temperature_settings(self):
        captured = {}

        class FakeResult:
            returncode = 0
            stdout = (
                '{"response": "ok", '
                '"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}}'
            )
            stderr = ""

        def fake_run(command, *, capture_output, text, timeout, env, check, stdin):
            captured["command"] = command
            captured["timeout"] = timeout
            captured["stdin"] = stdin
            with open(env[r.GEMINI_SYSTEM_SETTINGS_ENV_VAR], "r", encoding="utf-8") as settings_file:
                captured["settings"] = json.load(settings_file)
            return FakeResult()

        with (
            mock.patch.object(r, "GEMINI_CLI_PATH", "/tmp/gemini"),
            mock.patch.object(r.os.path, "exists", return_value=True),
            mock.patch.dict(os.environ, {r.GEMINI_SYSTEM_SETTINGS_ENV_VAR: ""}, clear=False),
            mock.patch.object(r.subprocess, "run", side_effect=fake_run),
        ):
            result = r.invoke_gemini_cli_with_usage(
                "google-gemini-cli/gemini-3-pro-preview",
                "prompt",
                timeout=30,
                temperature=0.0,
            )

        self.assertEqual(result["text"], "ok")
        self.assertEqual(
            result["usage_metadata"],
            {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7, "candidate_tokens": 4},
        )
        override = captured["settings"]["modelConfigs"]["customOverrides"][-1]
        self.assertEqual(captured["command"][2], "gemini-3-pro-preview")
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(captured["stdin"], r.subprocess.DEVNULL)
        self.assertEqual(override["match"], {"model": "gemini-3-pro-preview"})
        self.assertEqual(override["modelConfig"]["generateContentConfig"]["temperature"], 0.0)

    def test_invoke_claude_cli_uses_json_output_and_usage(self):
        captured = {}

        class FakeResult:
            returncode = 0
            stdout = json.dumps(
                {
                    "result": "ok",
                    "total_input_tokens": 5,
                    "total_output_tokens": 6,
                }
            )
            stderr = ""

        def fake_run(command, *, capture_output, text, timeout, env, check):
            captured["command"] = command
            captured["timeout"] = timeout
            captured["env"] = env
            return FakeResult()

        with (
            mock.patch.object(r, "CLAUDE_CLI_PATH", "/tmp/claude"),
            mock.patch.object(r.os.path, "exists", return_value=True),
            mock.patch.object(r.subprocess, "run", side_effect=fake_run),
        ):
            result = r.claude_generate_with_usage(
                "claude/sonnet",
                "prompt",
                timeout=30,
                temperature=0.0,
            )

        self.assertEqual(result["text"], "ok")
        self.assertEqual(
            result["usage_metadata"],
            {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        )
        self.assertEqual(
            captured["command"],
            ["/tmp/claude", "--model", "sonnet", "--output-format", "json", "-p", "prompt"],
        )
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(captured["env"]["NO_COLOR"], "1")

    def test_provider_usage_metadata_extraction(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "response": "ollama output",
                        "prompt_eval_count": 5,
                        "eval_count": 6,
                    }
                ).encode("utf-8")

        with mock.patch.object(r.urllib.request, "urlopen", return_value=FakeResponse()):
            result = r.ollama_generate_with_usage("llama3.1:8b", "prompt", timeout=1)

        self.assertEqual(result["text"], "ollama output")
        self.assertEqual(
            result["usage_metadata"],
            {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        )
        self.assertEqual(
            r.extract_gemini_usage_metadata(
                {"nested": {"usageMetadata": {"promptTokenCount": 2, "totalTokenCount": 9}}}
            ),
            {"input_tokens": 2, "output_tokens": 7, "total_tokens": 9},
        )
        self.assertEqual(
            r.extract_gemini_usage_metadata(
                {
                    "stats": {
                        "models": {
                            "gemini-3-pro-preview": {
                                "tokens": {
                                    "prompt": 10,
                                    "candidates": 3,
                                    "thoughts": 2,
                                    "cached": 4,
                                    "tool": 1,
                                    "total": 16,
                                }
                            }
                        }
                    }
                }
            ),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 16,
                "cached_tokens": 4,
                "thought_tokens": 2,
                "tool_tokens": 1,
                "candidate_tokens": 3,
            },
        )

    def test_codex_exec_payload_and_output_file(self):
        captured = {}

        class FakeResult:
            returncode = 0
            stdout = "ignored stdout"
            stderr = ""

        def fake_run(command, *, input, capture_output, text, timeout, env, check):
            captured["command"] = command
            captured["input"] = input
            captured["timeout"] = timeout
            captured["env_no_color"] = env["NO_COLOR"]
            output_path = command[command.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write("codex output")
            return FakeResult()

        with (
            mock.patch.dict(
                os.environ,
                {
                    "ROUTER_CODEX_CWD": "/tmp/codex-work",
                    "ROUTER_CODEX_SANDBOX": "read-only",
                },
                clear=False,
            ),
            mock.patch.object(r, "CODEX_CLI_PATH", "/tmp/codex"),
            mock.patch.object(r.os.path, "exists", return_value=True),
            mock.patch.object(r.subprocess, "run", side_effect=fake_run),
        ):
            result = r.codex_generate_with_usage(
                "codex/gpt-5.5",
                "prompt",
                timeout=11,
                num_predict=123,
                temperature=0.1,
            )

        self.assertEqual(captured["command"][:4], ["/tmp/codex", "exec", "-m", "gpt-5.5"])
        self.assertIn("--ephemeral", captured["command"])
        self.assertIn("--skip-git-repo-check", captured["command"])
        self.assertIn("--output-last-message", captured["command"])
        self.assertNotIn("--ask-for-approval", captured["command"])
        self.assertEqual(captured["command"][-1], "-")
        self.assertEqual(captured["command"][captured["command"].index("--cd") + 1], "/tmp/codex-work")
        self.assertEqual(captured["command"][captured["command"].index("--sandbox") + 1], "read-only")
        self.assertEqual(captured["input"], "prompt")
        self.assertEqual(captured["timeout"], 11)
        self.assertEqual(captured["env_no_color"], "1")
        self.assertEqual(result["text"], "codex output")
        self.assertEqual(result["provider"], "codex")
        self.assertEqual(result["model_name"], "gpt-5.5")
        self.assertEqual(result["usage_source"], "unavailable")
        self.assertEqual(result["usage_metadata"], {})

    def test_token_usage_tracking_records_generate_text_calls(self):
        def fake_gemini(model, prompt, *, timeout, temperature):
            return r.build_text_generation_result(
                "tracked output",
                {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
                "google_genai",
                model,
                "gemini_cli_stats",
            )

        with r.token_usage_tracking_context("test-token-run"):
            with mock.patch.object(r, "gemini_generate_with_usage", side_effect=fake_gemini):
                self.assertEqual(
                    r.generate_text(
                        "google-gemini-cli/gemini-3-pro-preview",
                        "tracked prompt",
                        timeout=30,
                        usage_label="Planner invoke",
                    ),
                    "tracked output",
                )

        records = r.get_token_usage_records("test-token-run")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label"], "Planner invoke")
        self.assertEqual(records[0]["usage_source"], "gemini_cli_stats")
        self.assertEqual(records[0]["input_tokens"], 8)
        self.assertEqual(records[0]["output_tokens"], 5)
        self.assertEqual(records[0]["total_tokens"], 13)

        summary = r.summarize_token_usage_records(records)
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["calls_with_usage"], 1)
        self.assertEqual(summary["by_model"]["google-gemini-cli/gemini-3-pro-preview"]["total_tokens"], 13)

    def test_token_usage_ledger_persistence_jsonl(self):
        state = r.create_initial_state(
            "Persist usage",
            planner_model="planner",
            judge_model="judge",
            pro_model="pro",
            flash_model="flash",
        )
        state["run_id"] = "ledger-run"
        records = [
            {
                "run_id": "ledger-run",
                "call_index": 1,
                "label": "Planner invoke",
                "provider": "google_genai",
                "model_name": "gemini-3-pro-preview",
                "usage_source": "gemini_cli_stats",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 3,
                "candidate_tokens": 4,
                "thought_tokens": 1,
                "tool_tokens": 0,
                "prompt_chars": 20,
                "output_chars": 30,
            }
        ]
        summary = r.summarize_token_usage_records(records)

        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = os.path.join(temp_dir, "usage.jsonl")
            with mock.patch.dict(os.environ, {r.ROUTER_TOKEN_USAGE_LEDGER_ENV_VAR: ledger_path}, clear=False):
                written_path = r.persist_token_usage_ledger(records, summary, state=state)

            self.assertEqual(written_path, ledger_path)
            with open(ledger_path, "r", encoding="utf-8") as ledger_file:
                lines = ledger_file.readlines()

        self.assertEqual(len(lines), 1)
        event = json.loads(lines[0])
        self.assertEqual(event["event"], "token_usage")
        self.assertEqual(event["run_id"], "ledger-run")
        self.assertEqual(event["summary"]["total_tokens"], 15)
        self.assertEqual(event["record"]["label"], "Planner invoke")

    def test_stream_event_helpers(self):
        mode, payload = r.unpack_stream_event(("namespace", "updates", {"node": {"status": "done"}}))
        self.assertEqual(mode, "updates")
        self.assertEqual(payload, {"node": {"status": "done"}})

        summary = r.summarize_stream_update(
            "dispatcher",
            {"status": "dispatched", "current_step": 1, "active_route": r.PRO},
        )
        self.assertIn("dispatcher", summary)
        self.assertIn("status=dispatched", summary)
        self.assertIn("route=PRO", summary)


class ProviderFallbackTests(unittest.TestCase):
    def test_provider_fallback_retries_infra_failure_and_succeeds(self):
        calls = []

        def fake_generate(model, prompt, **kwargs):
            calls.append(model)
            if model == "primary":
                raise RuntimeError("connection reset by provider")
            return "fallback success"

        with mock.patch.object(r, "generate_text", side_effect=fake_generate):
            result = r.invoke_with_provider_fallback(
                "primary",
                ["fallback"],
                "prompt",
                timeout=5,
                num_predict=10,
                temperature=0.0,
                label="test",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["model_name"], "fallback")
        self.assertTrue(result["used_provider_fallback"])
        self.assertEqual(calls, ["primary", "fallback"])

    def test_provider_fallback_stops_on_capability_failure(self):
        calls = []

        def fake_generate(model, prompt, **kwargs):
            calls.append(model)
            raise RuntimeError("need more context to complete")

        with mock.patch.object(r, "generate_text", side_effect=fake_generate):
            result = r.invoke_with_provider_fallback(
                "primary",
                ["fallback"],
                "prompt",
                timeout=5,
                num_predict=10,
                temperature=0.0,
                label="test",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_type"], "capability_quality")
        self.assertEqual(calls, ["primary"])


class FlashReviewAndMetadataTests(unittest.TestCase):
    def test_flash_output_review_and_retry_guard(self):
        self.assertEqual(
            r.verify_flash_output("Inspect config", "", r.empty_flash_review(), 0)["decision"],
            "escalate",
        )
        self.assertEqual(
            r.verify_flash_output("Inspect config", "ok", r.empty_flash_review(), 0)["decision"],
            "escalate",
        )
        self.assertEqual(
            r.verify_flash_output("Prepare summary", "ok", r.empty_flash_review(), 0)["decision"],
            "record",
        )

        state = r.create_initial_state("retry", flash_retry_budget=1)
        state.update(
            {
                "current_step": 0,
                "active_flash_review": {
                    "decision": "retry",
                    "failure_type": "infra_transient",
                    "reason": "connection reset",
                },
                "active_attempt_log": [],
            }
        )
        retry_update = r.retry_guard_node(state)
        self.assertEqual(retry_update["status"], "flash_retrying")
        self.assertEqual(retry_update["active_retry_count"], 1)

        state.update(retry_update)
        exhausted_update = r.retry_guard_node(state)
        self.assertEqual(exhausted_update["status"], "flash_retry_exhausted")
        self.assertIn("FLASH execution failed", exhausted_update["active_output"])

    def test_metadata_extraction_uses_recorded_provider_fallback_result(self):
        state = r.create_initial_state("metadata", pro_model="pro")
        state["results"] = [
            {
                "step": 1,
                "planned_route": r.PRO,
                "route": r.PRO,
                "model_name": "pro-fallback",
                "desc": "Inspect service behavior",
                "output": "Detailed technical output from provider fallback.",
                "status": "executed_via_provider_fallback",
                "attempt_count": 1,
                "retry_count": 0,
                "escalated_from_flash": False,
                "used_provider_fallback": True,
                "flash_review": r.empty_flash_review(),
                "attempt_log": [],
            }
        ]

        def fake_invoke(*args, **kwargs):
            return {
                "success": True,
                "output": "- Extracted technical fact.",
                "model_name": "pro",
                "used_provider_fallback": False,
                "failure_type": "none",
                "error_text": "",
                "attempt_log": [],
            }

        with mock.patch.object(r, "invoke_with_provider_fallback", side_effect=fake_invoke):
            update, _ = run_quietly(r.extract_technical_metadata_node, state)

        self.assertIn("TECHNICAL METADATA STEP 1", update["history"][0])
        self.assertIn("Extracted technical fact", update["history"][0])

    def test_metadata_extraction_skips_executor_fallback_result(self):
        state = r.create_initial_state("metadata", pro_model="pro")
        state["results"] = [
            {
                "step": 1,
                "planned_route": r.PRO,
                "route": r.PRO,
                "model_name": "pro",
                "desc": "Inspect service behavior",
                "output": "PRO executor fallback output: Inspect service behavior",
                "status": "executor_fallback",
                "attempt_count": 1,
                "retry_count": 0,
                "escalated_from_flash": False,
                "used_provider_fallback": False,
                "flash_review": r.empty_flash_review(),
                "attempt_log": [],
            }
        ]

        with mock.patch.object(r, "invoke_with_provider_fallback") as invoke_mock:
            update, _ = run_quietly(r.extract_technical_metadata_node, state)

        invoke_mock.assert_not_called()
        self.assertIn("skipped", update["history"][0])


class FinalizerTests(unittest.TestCase):
    def test_finalizer_timeout_env_is_used_for_flash_and_pro(self):
        captured = []

        def fake_invoke(primary_model, fallback_models, prompt, *, timeout, num_predict, temperature, label, attempt_log=None):
            captured.append((label, timeout))
            return {
                "success": True,
                "output": (
                    "Routing Summary\nEnough content for verification.\n"
                    "Step Outcomes\nThe timeout was captured.\n"
                    "Next Action\nKeep env override wired."
                ),
                "model_name": primary_model,
                "used_provider_fallback": False,
                "failure_type": "none",
                "error_text": "",
                "attempt_log": list(attempt_log or []),
            }

        state = r.create_initial_state("timeout", pro_model="pro", flash_model="flash")
        with mock.patch.dict(os.environ, {"ROUTER_FINALIZER_TIMEOUT": "17"}, clear=False):
            with mock.patch.object(r, "invoke_with_provider_fallback", side_effect=fake_invoke):
                r.flash_finalizer_node(state)
                r.pro_finalizer_node(state)

        self.assertEqual(captured, [("Finalizer FLASH", 17), ("Finalizer PRO", 17)])

    def test_finalizer_model_path_distinctness(self):
        state = r.create_initial_state(
            "paths",
            pro_model="google-gemini-cli/gemini-3-pro-preview",
            flash_model="google-gemini-cli/flash",
        )
        self.assertTrue(r.has_distinct_finalizer_model_path(state))

        same_state = r.create_initial_state(
            "paths",
            pro_model="google-gemini-cli/flash",
            flash_model="google-gemini-cli/flash",
        )
        self.assertFalse(r.has_distinct_finalizer_model_path(same_state))


class RouterGraphIntegrationTests(unittest.TestCase):
    def fake_generate_success(self, model, prompt, **kwargs):
        if prompt == "OK":
            return "OK"
        if "Task Decomposer" in prompt or "Role: Expert task decomposer" in prompt:
            return '[{"desc":"Inspect the router state flow"},{"desc":"Prepare a concise summary"}]'
        if "Role: Complexity judge" in prompt:
            if "Prepare a concise summary" in prompt:
                return (
                    '{"scores":{"reasoning_depth":0,"code_change_scope":0,"ambiguity":0,'
                    '"risk":0,"io_heaviness":2},"suggested_route":"FLASH",'
                    '"confidence":0.9,"reason":"summary"}'
                )
            return (
                '{"scores":{"reasoning_depth":2,"code_change_scope":1,"ambiguity":1,'
                '"risk":0,"io_heaviness":0},"suggested_route":"PRO",'
                '"confidence":0.9,"reason":"inspection"}'
            )
        if "Extract the 'technical gold'" in prompt:
            return "- Verified metadata extraction used the recorded step output."
        if "task executor" in prompt:
            return "This step completed with concrete technical findings and enough detail for verification."
        if "summarizer" in prompt:
            self.assertIn("TECHNICAL METADATA STEP", prompt)
            return (
                "Routing Summary\nThe mocked run completed using metadata.\n"
                "Step Outcomes\nBoth steps produced usable output and metadata was included.\n"
                "Next Action\nReview state transitions and finalizer context."
            )
        return "Fallback mocked output with sufficient detail."

    def test_full_graph_success_path_with_metadata_and_no_debug_output(self):
        with mock.patch.dict(os.environ, {"ROUTER_DEBUG": ""}, clear=False):
            with mock.patch.object(r, "generate_text", side_effect=self.fake_generate_success):
                state, output = run_quietly(
                    r.run_router_app,
                    "Inspect router state flow and summarize",
                    planner_model="mock-planner",
                    judge_model="mock-judge",
                    pro_model="mock-pro",
                    flash_model="mock-flash",
                    max_concurrency=1,
                )

        metadata_blocks = [line for line in state["history"] if "TECHNICAL METADATA STEP" in line]
        self.assertEqual(state["status"], "finished")
        self.assertEqual(len(state["results"]), 2)
        self.assertEqual(len(metadata_blocks), 2)
        self.assertNotIn("[DEBUG", output)

    def test_full_graph_flash_quality_escalates_to_pro(self):
        calls = {"flash_executor": 0, "pro_executor": 0}

        def fake_generate(model, prompt, **kwargs):
            if prompt == "OK":
                return "OK"
            if "Task Decomposer" in prompt or "Role: Expert task decomposer" in prompt:
                return '[{"desc":"List deployment manifests"}]'
            if "Role: Complexity judge" in prompt:
                return (
                    '{"scores":{"reasoning_depth":0,"code_change_scope":0,"ambiguity":0,'
                    '"risk":0,"io_heaviness":2},"suggested_route":"FLASH",'
                    '"confidence":0.95,"reason":"mostly listing"}'
                )
            if "Role: FLASH task executor" in prompt:
                calls["flash_executor"] += 1
                return "ok"
            if "Role: PRO task executor" in prompt:
                calls["pro_executor"] += 1
                self.assertIn("Escalation context", prompt)
                return "PRO completed the manifest listing with enough detail after FLASH quality escalation."
            if "Extract the 'technical gold'" in prompt:
                return "- Escalated manifest listing produced a detailed result."
            if "summarizer" in prompt:
                return (
                    "Routing Summary\nFLASH escalated to PRO after quality review.\n"
                    "Step Outcomes\nThe manifest listing completed successfully.\n"
                    "Next Action\nUse the PRO result."
                )
            return "Fallback mocked output with sufficient detail."

        with mock.patch.object(r, "generate_text", side_effect=fake_generate):
            state, _ = run_quietly(
                r.run_router_app,
                "List deployment manifests",
                planner_model="mock-planner",
                judge_model="mock-judge",
                pro_model="mock-pro",
                flash_model="mock-flash",
                max_concurrency=1,
            )

        self.assertEqual(state["status"], "finished")
        self.assertEqual(len(state["results"]), 1)
        result = state["results"][0]
        self.assertEqual(result["planned_route"], r.FLASH)
        self.assertEqual(result["route"], r.PRO)
        self.assertTrue(result["escalated_from_flash"])
        self.assertEqual(calls, {"flash_executor": 1, "pro_executor": 1})

    def test_executor_subtasks_run_in_parallel_when_concurrency_allows(self):
        barrier = threading.Barrier(2, timeout=3)
        broken_barrier = []
        executor_threads = []

        def fake_generate(model, prompt, **kwargs):
            if prompt == "OK":
                return "OK"
            if "Task Decomposer" in prompt:
                return (
                    '[{"desc":"Analyze provider A architecture"},'
                    '{"desc":"Analyze provider B architecture"}]'
                )
            if "Role: Complexity judge" in prompt:
                return (
                    '{"scores":{"reasoning_depth":2,"code_change_scope":0,"ambiguity":1,'
                    '"risk":0,"io_heaviness":0},"suggested_route":"PRO",'
                    '"confidence":0.9,"reason":"parallel analysis"}'
                )
            if "task executor" in prompt:
                executor_threads.append(threading.get_ident())
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    broken_barrier.append("executor branches did not overlap")
                return "Parallel executor completed with detailed technical analysis and enough verification detail."
            if "Extract the 'technical gold'" in prompt:
                return "- Parallel branch metadata was extracted."
            if "summarizer" in prompt:
                return (
                    "Routing Summary\nParallel executor fanout completed.\n"
                    "Step Outcomes\nBoth provider analyses returned usable results.\n"
                    "Next Action\nUse the joined result set."
                )
            return "Fallback mocked output with sufficient detail."

        with mock.patch.object(r, "generate_text", side_effect=fake_generate):
            state, _ = run_quietly(
                r.run_router_app,
                "Analyze provider A and provider B independently",
                planner_model="mock-planner",
                judge_model="mock-judge",
                pro_model="mock-pro",
                flash_model="mock-flash",
                max_concurrency=2,
            )

        self.assertEqual(state["status"], "finished")
        self.assertEqual(len(state["results"]), 2)
        self.assertEqual([result["step"] for result in state["results"]], [1, 2])
        self.assertFalse(broken_barrier)
        self.assertEqual(len(set(executor_threads)), 2)

    def test_streamed_graph_returns_final_state(self):
        with mock.patch.object(r, "generate_text", side_effect=self.fake_generate_success):
            state, _ = run_quietly(
                r.run_router_app,
                "Inspect router state flow and summarize",
                planner_model="mock-planner",
                judge_model="mock-judge",
                pro_model="mock-pro",
                flash_model="mock-flash",
                max_concurrency=1,
                stream=True,
            )

        self.assertEqual(state["status"], "finished")
        self.assertEqual(len(state["results"]), 2)


if __name__ == "__main__":
    unittest.main()
