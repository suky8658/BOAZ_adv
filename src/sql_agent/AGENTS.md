<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# sql_agent

## Purpose
Core agent module. Defines a LangsGraph `StateGraph` pipeline that translates natural-language questions into validated MySQL queries and executes them. All graph nodes, security guards, SQL utilities, and the `build_app()` entry point live in a single file.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker (empty) |
| `sql_agent.py` | All agent logic: Pydantic state models, graph nodes, security functions, and `build_app()` |

## Graph Architecture (`sql_agent.py`)

### State (`AgentState` TypedDict)

Key fields passed between nodes:

| Field | Type | Purpose |
|-------|------|---------|
| `user_question` | str | Raw user input |
| `schema_text` / `integrity_text` | str | Loaded by `load_context` |
| `plan` | dict | Output of `plan_question` (maps to `QuestionPlan`) |
| `mart_design` | dict | Output of `design_mart` (maps to `MartDesign`) |
| `sql_draft` | dict | Output of `generate_sql` (maps to `SQLDraft`) |
| `sql_result` / `row_count` | Any / int | Output of `execute_sql` |
| `validation` | dict | Output of `validate_sql_and_result` |
| `retry_count` / `max_retries` | int | Retry loop control |
| `feedback` | str | Validator feedback forwarded to `generate_sql` on retry |
| `final_answer` | str | Output of `finalize_answer` |

### Graph Nodes

| Node | Responsibility |
|------|---------------|
| `load_context` | Calls `load_all_metadata()` to populate `schema_text` and `integrity_text` |
| `plan_question` | LLM prompt → `QuestionPlan` JSON; classifies `task_type` as `query_answer` or `data_mart_build` |
| `design_mart` | LLM prompt → `MartDesign` JSON; skipped (returns `{}`) when `task_type != data_mart_build` |
| `generate_sql` | LLM prompt → `SQLDraft` JSON; incorporates `feedback` on retry passes |
| `execute_sql` | Runs SQL via SQLAlchemy; handles SELECT (read-only) and CREATE/INSERT (mart write) paths |
| `validate_sql_and_result` | LLM prompt verifies result satisfies original question; sets `validation.result` to `valid`/`invalid` |
| `increase_retry` | Increments `retry_count` by 1 |
| `finalize_answer` | LLM prompt → Korean natural-language answer; short-circuits to error message on validation failure |

### Graph Edges

```
START → load_context → plan_question → design_mart → generate_sql
      → execute_sql → validate_sql_and_result
           ├── valid OR retry_count >= max_retries  →  finalize_answer → END
           └── invalid + retries remaining          →  increase_retry → generate_sql
```

### Security Functions

| Function | Behaviour |
|----------|-----------|
| `is_safe_query_sql(sql)` | Returns `True` only for SELECT/WITH; blocks INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE |
| `is_safe_mart_sql(sql, target_table)` | Returns `(bool, reason)`; allows CREATE TABLE / CREATE OR REPLACE TABLE / INSERT INTO targeting `ALLOWED_MART_SCHEMA` only; blocked entirely when `ALLOW_MART_WRITE=false` |

### Pydantic Models

| Model | Used For |
|-------|---------|
| `QuestionPlan` | Structured output of `plan_question` |
| `MartDesign` | Structured output of `design_mart` |
| `SQLDraft` | Structured output of `generate_sql`; includes optional `precheck_sql` / `postcheck_sql` |
| `ValidationResult` | Structured output of `validate_sql_and_result` |

## For AI Agents

### Working In This Directory
- The LLM is `gemini-2.5-flash` at temperature 0 — change the model string in `llm = ChatGoogleGenerativeAI(...)` to upgrade.
- `ALLOWED_MART_SCHEMA` defaults to `"analytics"`; all mart write targets must include this schema name.
- `ALLOW_MART_WRITE` env flag gates all DDL/DML execution — set to `"false"` for read-only environments.
- SELECT results are capped at 10 rows in `format_result_rows`; validator sees up to 10 rows, `finalize_answer` up to 20.
- `build_app()` returns a compiled `StateGraph`; callers invoke it with `app.invoke({...initial_state...})`.

### Testing Requirements
- Integration tests require `GOOGLE_API_KEY` and a live MySQL connection.
- Unit-test `is_safe_query_sql` and `is_safe_mart_sql` with pytest — no DB or LLM needed.

### Common Patterns
- `safe_json_parse()` strips markdown code fences before `json.loads()` — LLM responses often wrap JSON in triple backticks.
- `clean_sql()` strips fences and ensures a trailing semicolon.
- All node functions follow the LangsGraph convention: accept `state: AgentState`, return a partial dict to merge into state.

## Dependencies

### Internal
- `db/db_connect.py` — `get_db_engine()` singleton
- `validator/integrity_loader.py` — `load_all_metadata()`

### External
- `langgraph` — `StateGraph`, `START`, `END`
- `langchain_google_genai` — `ChatGoogleGenerativeAI`
- `pydantic` — `BaseModel`, `Field`
- `sqlalchemy` — `text`, engine execution
- `dotenv`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
