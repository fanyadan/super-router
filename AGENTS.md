# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Python skill/tool. `scripts/router.py` contains the LangGraph router, CLI entry point, provider fallback logic, planner/judge helpers, executor flow, metadata extraction, and finalizer path. `tests/test_router.py` holds the regression suite, with `tests/__init__.py` keeping tests importable. `SKILL.md` defines the Hermes skill contract and operational guidance, while `README.md` is the user-facing overview. Put design notes and supporting research in `references/`.

## Build, Test, and Development Commands

- `pip install langgraph`: install the only declared runtime dependency.
- `python scripts/router.py "Inspect router state flow and summarize"`: run the router locally with a direct task argument.
- `ROUTER_TASK="Analyze K8s YAML errors" python scripts/router.py`: run via environment variable, useful for long or quoted tasks.
- `python scripts/router.py --stream "Analyze a complex incident"`: print node-level progress while the graph runs.
- `python -m unittest tests/test_router.py`: run the full regression suite.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, 4-space indentation, and explicit type hints where they clarify state shape or helper contracts. Follow existing naming: functions and variables use `snake_case`, classes use `PascalCase`, constants use `UPPER_SNAKE_CASE`, and route values remain `PRO` or `FLASH`. Keep imports grouped as standard library, third-party, then local. There is no repo-local formatter or linter configuration, so preserve the current readable style and avoid broad mechanical rewrites.

## Testing Guidelines

Tests use `unittest` and `unittest.mock`. Add focused tests in `tests/test_router.py` for routing heuristics, JSON extraction, provider fallback, FLASH review, metadata extraction, streaming helpers, and finalization behavior. Mock model, network, and subprocess calls; tests should not require Gemini CLI, Ollama, or external network access. Name test methods `test_<behavior_being_verified>`.

## Commit & Pull Request Guidelines

Git history uses short, imperative subjects such as `Improve parallelism mode.`, `Fix issue of env not referenced during agent running...`, and occasional scoped messages like `docs: add installation guide...`. Prefer concise present-tense summaries, with a scope prefix when useful.

Pull requests should describe the behavior change, list test commands run, call out any `ROUTER_*` environment variable impact, and include sample router output for user-visible routing or final-report changes.

## Security & Configuration Tips

Do not commit secrets, API credentials, or private provider settings. Keep local configuration in the shell or `~/.hermes/.env`. Treat provider names, timeouts, and concurrency defaults as operationally sensitive because they affect cost, latency, and failure behavior.
