"""Guardrail: the public API surface of ``scripts.router`` must stay stable.

``scripts/router.py`` was split into the ``scripts/routerlib`` package. This test
locks the flat, re-exported public surface so that no function, constant, or
re-exported stdlib module is accidentally dropped by future refactors. The
expected set below was captured from the pre-split monolith.
"""

import unittest

import scripts.router as r

# Public names exported by the pre-split monolith (scripts/router.py @ 6614 lines).
EXPECTED_PUBLIC_API = {
    "Annotated", "Any", "CAPABILITY_FAILURE_KEYWORDS", "CLAUDE_CLI_PATH",
    "CLAUDE_CODEX_STYLE_SANDBOX_BY_KEY", "CLAUDE_PERMISSION_MODE_BY_KEY",
    "CLAUDE_SANDBOX_PERMISSION_MODE_ALIASES", "CODEX_CLI_PATH",
    "COMMUNICATION_AUDIENCE_KEYWORDS", "Callable", "ComplexityAssessment", "ComplexityScores",
    "DATA_GATHERING_HINT_KEYWORDS", "DEEP_WORK_HINT_KEYWORDS",
    "DEFAULT_EXECUTOR_CONTEXT_CHAR_LIMIT", "DEFAULT_FINALIZER_CONTEXT_CHAR_LIMIT",
    "DEFAULT_FLASH_EXECUTION_TIMEOUT", "DEFAULT_FLASH_FINALIZER_TIMEOUT",
    "DEFAULT_FLASH_MODEL", "DEFAULT_FLASH_RETRY_BUDGET", "DEFAULT_JUDGE_CONTEXT_CHAR_LIMIT",
    "DEFAULT_JUDGE_MODEL", "DEFAULT_LARGE_MODEL_TIMEOUT", "DEFAULT_MAX_PROVIDER_ATTEMPTS",
    "DEFAULT_METADATA_OUTPUT_CHAR_LIMIT", "DEFAULT_METADATA_TIMEOUT",
    "DEFAULT_PLANNER_MAX_OUTPUT_TOKENS", "DEFAULT_PLANNER_MODEL",
    "DEFAULT_PLANNER_TASK_CHAR_LIMIT", "DEFAULT_PLANNER_TIMEOUT",
    "DEFAULT_PROVIDER_TERMINATION_GRACE", "DEFAULT_PRO_EXECUTION_TIMEOUT",
    "DEFAULT_PRO_FINALIZER_TIMEOUT", "DEFAULT_PRO_MODEL", "DEFAULT_ROUTER_RECURSION_LIMIT",
    "DEFAULT_ROUTER_RUN_TIMEOUT", "DEFAULT_WARMUP_TIMEOUT", "DEFERRED_EXECUTION_KEYWORDS",
    "Dict", "END", "ENGLISH_PREFIX_MATCH_KEYWORDS", "FLASH", "FLASH_COMPLEXITY_THRESHOLD",
    "FinalizerOutcome", "FlashReviewResult", "GEMINI_CLI_PATH",
    "GEMINI_NETWORK_PREFLIGHT_RESULT", "GEMINI_PREFLIGHT_RESULTS",
    "GEMINI_SYSTEM_SETTINGS_ENV_VAR", "GLOBAL_RUN_DEADLINE_MONOTONIC",
    "HIGH_RISK_CONTEXT_KEYWORDS", "HIGH_RISK_DECISION_KEYWORDS", "HIGH_RISK_EVIDENCE_KEYWORDS",
    "INFRA_FAILURE_KEYWORDS", "Iterator", "JudgedSubtask", "LOW_CONFIDENCE_THRESHOLD",
    "LOW_QUALITY_OUTPUT_MARKERS", "List", "Literal", "MIN_NON_SUMMARY_OUTPUT_CHARS",
    "MODEL_INVOCATION_GRAPH", "ModelInvocationResult", "ModelInvocationState", "OLLAMA_URL",
    "PLANNER_CONSTRAINT_KEYWORDS", "PLANNER_DECOMPOSITION_HINT_KEYWORDS",
    "PLANNER_DELIVERABLE_KEYWORDS", "PLANNER_ENTITY_LIST_PATTERNS", "PLANNER_ENTITY_STOPWORDS",
    "PLANNER_EVIDENCE_KEYWORDS", "PLANNER_RELEVANT_TASK_KEYWORDS", "PRO",
    "PRO_COMPLEXITY_THRESHOLD", "PlannedSubtask", "ROUTER_CLAUDE_CWD_ENV_VAR",
    "ROUTER_CLAUDE_SANDBOX_ENV_VAR", "ROUTER_CODEX_CWD_ENV_VAR",
    "ROUTER_CODEX_SANDBOX_ENV_VAR", "ROUTER_EXECUTOR_CONTEXT_CHAR_LIMIT_ENV_VAR",
    "ROUTER_EXECUTOR_TIMEOUT_ENV_VAR", "ROUTER_FINALIZER_CONTEXT_CHAR_LIMIT_ENV_VAR",
    "ROUTER_FLASH_EXECUTOR_TIMEOUT_ENV_VAR", "ROUTER_JUDGE_CONTEXT_CHAR_LIMIT_ENV_VAR",
    "ROUTER_LANGSMITH_ENABLED_ENV_VAR", "ROUTER_LANGSMITH_PROJECT_ENV_VAR",
    "ROUTER_LANGSMITH_TAGS_ENV_VAR", "ROUTER_MAX_PROVIDER_ATTEMPTS_ENV_VAR",
    "ROUTER_METADATA_OUTPUT_CHAR_LIMIT_ENV_VAR", "ROUTER_METADATA_TIMEOUT_ENV_VAR",
    "ROUTER_MODEL_ENV_VAR", "ROUTER_PLANNER_MAX_OUTPUT_TOKENS_ENV_VAR",
    "ROUTER_PLANNER_TASK_CHAR_LIMIT_ENV_VAR", "ROUTER_PLANNER_TIMEOUT_ENV_VAR",
    "ROUTER_PROVIDER_TERMINATION_GRACE_ENV_VAR", "ROUTER_PRO_EXECUTOR_TIMEOUT_ENV_VAR",
    "ROUTER_RUN_TIMEOUT_ENV_VAR", "ROUTER_SKIP_WARMUP", "ROUTER_TASK_ENV_VAR",
    "ROUTER_TOKEN_USAGE_LEDGER_ENV_VAR", "ROUTER_WARMUP_TIMEOUT_ENV_VAR",
    "RUN_DEADLINE_MONOTONIC", "RouterState", "START", "SUMMARY_ROUTE_KEYWORDS",
    "SYNTHESIS_ROUTE_KEYWORDS", "Send", "StateGraph", "StepResult", "Subtask",
    "TOKEN_USAGE_ACTIVE_RUN_IDS", "TOKEN_USAGE_LOCK", "TOKEN_USAGE_RECORDS_BY_RUN",
    "TOKEN_USAGE_RUN_ID", "TextGenerationResult", "TokenUsageRecord", "TokenUsageSummary",
    "TypedDict", "add_usage_to_bucket", "annotate_langsmith_model_run", "annotations",
    "append_unique_planner_item", "apply_contextual_score_biases", "argparse",
    "assemble_plan_node", "build_claude_sandbox_settings", "build_dependency_judge_prompt",
    "build_execution_prompt", "build_executor_context_pack_json",
    "build_executor_context_payload", "build_fallback_assessment", "build_fallback_report",
    "build_fallback_subtasks", "build_finalizer_context_pack_json",
    "build_finalizer_context_payload", "build_finalizer_prompt",
    "build_gemini_temperature_settings", "build_graph_config", "build_high_risk_reason",
    "build_judge_context_pack_json", "build_judge_prompt", "build_langsmith_metadata",
    "build_metadata_context_pack_json", "build_model_invocation_graph",
    "build_parallel_step_result", "build_planner_context_manifest",
    "build_planner_manifest_payload", "build_planner_prompt", "build_router_graph",
    "build_subtask", "build_task_context_pack_json", "build_task_context_payload",
    "build_text_generation_result", "check_run_deadline", "clamp_float", "clamp_int",
    "classify_failure_type", "classify_flash_execution_failure", "claude_generate_with_usage",
    "codex_generate", "codex_generate_with_usage", "coerce_non_negative_int",
    "collect_planner_segments", "compact_planner_relevant_segment", "compact_planner_task",
    "compact_step_result_for_context", "compact_text", "compact_text_middle",
    "completed_subtask_ids", "contains_any", "contains_cjk", "contextlib", "contextvars",
    "copy", "create_initial_state", "create_langsmith_client", "current_run_deadline",
    "current_token_usage_run_id", "datetime", "decide_route", "dedupe_model_sequence",
    "default_communication_subtask", "default_gemini_system_settings_path",
    "dependency_context_results_for_subtask", "dependency_deadlock_node",
    "dependency_execution_join_node", "dependency_judge_node", "dependency_reason_from_raw",
    "dependency_scheduler_node", "dependency_validate_node", "deterministic_finalizer_node",
    "dispatch_node", "display_plan", "emit_stream_updates", "empty_finalizer_outcome",
    "empty_flash_review", "empty_model_invocation_result", "empty_token_usage_summary",
    "ensure_communication_subtask", "ensure_gemini_cli_ready", "ensure_gemini_network_ready",
    "escalation_node", "execute_subtask_in_parallel_branch", "execution_finalize_join_node",
    "executor_fallback_node", "executor_result_node", "extract_claude_usage_metadata",
    "extract_communication_audience_markers", "extract_first_json_array",
    "extract_first_json_object", "extract_gemini_cli_stats_usage_metadata",
    "extract_gemini_usage_metadata", "extract_gemini_usage_metadata_with_source",
    "extract_nested_gemini_usage_metadata", "extract_ollama_usage_metadata",
    "extract_planner_entities", "extract_technical_metadata_for_result",
    "extract_technical_metadata_node", "extract_usage_metadata_from_mapping",
    "finalizer_complete_node", "find_communication_clause", "first_present_value",
    "flash_executor_node", "flash_finalizer_node", "flash_finalizer_verify_node",
    "flash_review_node", "format_int", "format_token_usage_summary", "gemini_generate",
    "gemini_generate_with_usage", "generate_text", "get_model_invocation_graph",
    "get_token_usage_records", "has_data_gathering_hint", "has_deep_work_hint",
    "has_distinct_finalizer_model_path", "has_non_summary_work_hint", "has_proxy_config",
    "invoke_executor_with_route", "invoke_gemini_cli", "invoke_gemini_cli_with_usage",
    "invoke_parallel_executor_attempt", "invoke_with_provider_fallback", "is_claude_model",
    "is_codex_model", "is_deferred_execution_subtask", "is_gemini_model",
    "is_high_risk_context", "is_high_risk_core_step", "is_high_risk_decision_step",
    "is_high_risk_evidence_step", "is_large_model", "is_planner_relevant_task_segment",
    "is_summary_like_subtask", "is_synthesis_like_subtask", "json",
    "judge_dependencies_with_model", "judge_subtask_node", "judge_warmup_node",
    "keyword_matches", "langsmith_api_key_configured", "langsmith_project_name",
    "langsmith_provider_name", "langsmith_tracing_configured", "langsmith_tracing_context",
    "langsmith_tracing_forced_disabled", "langsmith_tracing_requested", "list_ollama_models",
    "load_gemini_system_settings", "main", "make_serial_dependency_plan",
    "matched_context_keywords", "metric_value", "model_attempt_prepare_node",
    "model_invoke_node", "model_transport_name", "normalize_claude_permission_mode",
    "normalize_claude_sandbox_config", "normalize_complexity_assessment",
    "normalize_dependency_id", "normalize_dependency_judgment", "normalize_dependency_list",
    "normalize_model_name", "normalize_planned_subtasks", "normalize_route",
    "normalize_usage_metadata", "observe_stream_event", "ollama_api_endpoint",
    "ollama_generate", "ollama_generate_with_usage", "ollama_model_not_found", "operator",
    "os", "parallel_execution_join_node", "parallel_executor_node", "parse_cli_args",
    "parse_langsmith_tags", "persist_token_usage_ledger", "planner_fallback_node",
    "planner_invoke_node", "planner_parse_node", "planner_ready_node", "planner_warmup_node",
    "prepare_router_run", "prioritize_items_for_context", "pro_executor_node",
    "pro_finalizer_node", "pro_finalizer_verify_node", "process_langsmith_model_inputs",
    "process_langsmith_model_outputs", "provider_cli_name", "re", "read_ollama_json",
    "record_step_node", "record_token_usage", "request_ollama_generate", "resolve_bool",
    "resolve_bool_value", "resolve_context_char_limit", "resolve_execution_model",
    "resolve_executor_timeout", "resolve_graph_max_concurrency", "resolve_model",
    "resolve_model_list", "resolve_non_negative_int", "resolve_ollama_num_predict",
    "resolve_optional_positive_int", "resolve_positive_int", "resolve_token_usage_ledger_path",
    "retry_flash_node", "retry_guard_node", "route_after_dispatch",
    "route_after_executor_result", "route_after_flash_finalizer_verify",
    "route_after_flash_review", "route_after_model_invoke", "route_after_planner_invoke",
    "route_after_planner_parse", "route_after_planner_warmup",
    "route_after_pro_finalizer_verify", "route_after_retry_guard", "route_fallback_models",
    "route_to_deferred_executor_subtasks", "route_to_judge_subtasks",
    "route_to_parallel_executor_subtasks", "route_to_ready_executor_subtasks",
    "router_debug_enabled", "router_run_deadline_context", "run_provider_cli",
    "run_router_app", "score_complexity", "score_subtask_with_model",
    "serialize_context_payload", "serialize_planner_manifest", "shutil", "signal", "socket",
    "split_mixed_planned_subtask", "split_planner_entity_list", "split_planner_task_segments",
    "subprocess", "sum_optional_ints", "summarize_stream_update",
    "summarize_token_usage_records", "sys", "tempfile", "terminate_provider_process",
    "threading", "time", "timeout_with_run_deadline", "token_usage_tracking_context",
    "tokenize_context_text", "unpack_stream_event", "unwrap_text_generation_result", "urllib",
    "uuid", "validate_dependency_graph", "verify_finalizer_output", "verify_flash_output",
}

# Stdlib modules the test suite reaches through scripts.router (e.g.
# r.subprocess, r.urllib.error.HTTPError).
EXPECTED_STDLIB_REEXPORTS = ("os", "signal", "subprocess", "urllib")


class PublicApiSurfaceTests(unittest.TestCase):
    def test_public_api_is_superset_of_monolith(self):
        current = {n for n in dir(r) if not n.startswith("_")}
        missing = EXPECTED_PUBLIC_API - current
        self.assertEqual(missing, set(), f"dropped public names: {sorted(missing)}")

    def test_stdlib_reexports_resolve(self):
        for name in EXPECTED_STDLIB_REEXPORTS:
            self.assertTrue(hasattr(r, name), f"missing stdlib re-export: r.{name}")
        # exercised by the suite: r.urllib.error.HTTPError
        self.assertTrue(hasattr(r.urllib, "error"))
        self.assertTrue(hasattr(r.urllib.error, "HTTPError"))


if __name__ == "__main__":
    unittest.main()
