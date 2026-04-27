<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# src

## Purpose
The Python package root for all runtime application code. Contains the main agent runner, the multi-subagent SQL orchestration logic, database access utilities, schema inspection, integrity validation, and placeholder extension modules.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker (empty) |
| `run_agent.py` | CLI entry point — builds the agent and runs an interactive REPL loop |
| `debug_agent.py` | Scratch file for agent debugging sessions |
| `out.log` | Runtime log output from agent debug runs |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `db/` | Database connection, schema inspection, and Great Expectations integrity managers (see `db/AGENTS.md`) |
| `sql_agent/` | Main orchestrator agent and all subagent definitions (see `sql_agent/AGENTS.md`) |
| `validator/` | Loads and exposes cached schema/integrity JSON to the agent tools (see `validator/AGENTS.md`) |
| `tools/` | Reserved for future LangChain tool definitions (currently empty — see `tools/AGENTS.md`) |
| `utils/` | Reserved for shared utility functions (currently empty — see `utils/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- `src/` is the Python import root — run scripts with `PYTHONPATH=src python src/run_agent.py` or via the project runner.
- All internal imports use bare module paths (e.g. `from db.db_connect import get_db_engine`), not `src.db.*`.
- `debug_agent.py` and `out.log` are development artefacts; do not treat them as canonical.

### Testing Requirements
- `pytest` from the repo root discovers tests; add new tests under a `tests/` directory at the repo root.
- `ruff check src/` for linting.

### Common Patterns
- The graph is built in `build_app()` which returns a compiled `StateGraph`; invoke it with `app.invoke({...})`.
- Environment variables are loaded via `load_dotenv()` at the top of each module that needs them.

## Dependencies

### Internal
- `db/` ← used by `sql_agent/` and `validator/`
- `validator/` ← used by `sql_agent/` tools

### External
- `langchain-google-genai`, `langgraph`, `sqlalchemy`, `pymysql`, `dotenv`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
