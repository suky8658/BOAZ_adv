<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# db

## Purpose
Database access layer. Provides the SQLAlchemy engine factory, schema inspection and LLM-based description enrichment, and two Great Expectations-based integrity validation managers (physical rules and semantic/LLM-driven PK/FK hypothesis testing).

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker (empty) |
| `db_connect.py` | `get_db_engine()` — builds a pymysql SQLAlchemy engine from `.env`, auto-creates the analytics schema; `get_table_samples()` — random 10-row sampler |
| `db_schema.py` | `get_schema_info()` — inspects live DB schema, optionally enriches columns with LLM descriptions and sample data, caches to `data/db_schema.json` |
| `integrity_physical.py` | `PhysicalIntegrityManager` — runs schema-driven GE expectations (PK uniqueness/nullability, column types, FK referential integrity, row count) |
| `integrity_semantic.py` | `SemanticHypothesizer` + `SemanticValidator` — uses an LLM to infer PK/FK candidates for tables lacking explicit constraints, then confirms via GE |

## For AI Agents

### Working In This Directory
- `get_db_engine()` returns `None` on connection failure — always null-check before use.
- `db_schema.py` uses `langchain_openai.ChatOpenAI` (requires `OPENAI_API_KEY` + `SCHEMA_LLM_MODEL` in `.env`) for schema enrichment; the agent itself uses Google Gemini.
- `integrity_physical.py` and `integrity_semantic.py` import from `src.db.*` (absolute style with `src.` prefix) unlike the rest of the codebase which uses bare imports — be aware of this inconsistency.
- Physical integrity results are written to `data/db_integrity_result_physical.json`; semantic results to `data/db_integrity_result_semantic.json`. The merged `data/db_integrity_result.json` consumed by the agent is assembled externally (in the notebook).

### Testing Requirements
- Tests require a live MySQL connection; ensure `.env` is set before running.
- `db_connect.py` can be run directly as `__main__` for a quick connection test.
- `db_schema.py` can be run directly as `__main__` to regenerate the schema cache.

### Common Patterns
- GE validators are always created fresh per table run to avoid suite accumulation.
- The `run_id` format is `run_SK_MMDD_HHMMSS` for traceability across physical and semantic runs.

## Dependencies

### Internal
- `data/db_schema.json` — read by `db_schema.py` (cache) and `integrity_physical.py`

### External
- `sqlalchemy`, `pymysql` — DB connectivity
- `pandas` — DataFrame intermediary for GE validation
- `great_expectations` — expectation suite engine
- `langchain_openai` — LLM for schema enrichment and semantic hypothesis (OpenAI)
- `dotenv` — environment variable loading

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
