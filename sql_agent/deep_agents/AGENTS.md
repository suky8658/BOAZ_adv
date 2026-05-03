<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# SQL_Agent

## Purpose
A conversational SQL agent that translates natural-language questions into MySQL queries and executes them, built on a LangsGraph (`StateGraph`) pipeline with Google Gemini 2.5 Flash. The project also includes a data quality pipeline (Great Expectations) and schema enrichment tooling for the Olist Brazilian e-commerce dataset.

## Key Files

| File | Description |
|------|-------------|
| `main.py` | Placeholder entry point (stub only) |
| `pyproject.toml` | Project metadata and dependency declarations (uv-managed) |
| `uv.lock` | Locked dependency tree for reproducible installs |
| `.env` | Runtime secrets — DB credentials, API keys (not committed) |
| `.python-version` | Pins Python version for uv/pyenv |
| `csv_to_db.py` | One-off loader: reads Olist CSV files and bulk-inserts them into MySQL |
| `test_agent.ipynb` | Jupyter notebook for interactive agent testing and integrity pipeline runs |
| `README.md` | Project overview |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | All runtime application code (see `src/AGENTS.md`) |
| `data/` | Cached JSON artefacts — schema snapshot and integrity results (see `data/AGENTS.md`) |
| `gx/` | Great Expectations project config, checkpoints, and expectations (see `gx/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- **Never commit `.env`** — it contains DB passwords and API keys.
- Dependency management uses `uv`; run `uv sync` after editing `pyproject.toml`.
- `csv_to_db.py` hardcodes a local path (`/Users/a2485/Desktop/adv/data/`) — update before running on a new machine.
- `main.py` is a stub; actual agent entry point is `src/run_agent.py`.

### Testing Requirements
- Run tests with `pytest` from the repo root.
- Lint with `ruff check`.
- Most integration tests require a live MySQL connection — ensure `.env` is populated.

### Common Patterns
- All submodules import relative to `src/` as the Python root (e.g. `from db.db_connect import ...`).
- The agent pipeline is: `load_context` → `plan_question` → `design_mart` → `generate_sql` → `execute_sql` → `validate_sql_and_result` → (`increase_retry` → `generate_sql`)* → `finalize_answer`.
- Schema and integrity data are loaded from cached JSON files in `data/` to avoid repeated DB round-trips.

## Dependencies

### External
- `langgraph>=0.6.11` — LangsGraph StateGraph pipeline framework
- `langchain-google-genai>=2.1.12` — Gemini model integration
- `sqlalchemy>=2.0.48` + `pymysql>=1.1.2` — MySQL access layer
- `pandas>=2.3.3` — data manipulation for integrity checks
- `pydantic>=2.12.5` — data validation
- `dotenv>=0.9.9` — environment variable loading
- `great_expectations` — data quality validation (used in notebooks/integrity modules)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
