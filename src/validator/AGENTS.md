<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# validator

## Purpose
Thin read layer that loads the cached schema and integrity JSON files from `data/` and exposes them as formatted strings to the agent's LangChain tools. Keeps I/O and JSON parsing out of the agent module.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker (empty) |
| `integrity_loader.py` | All loader and formatter functions for schema and integrity data |

## Public API (`integrity_loader.py`)

| Function | Returns | Used By |
|----------|---------|---------|
| `load_schema_json()` | Raw schema dict | internal helpers |
| `load_integrity_json()` | Raw integrity dict | internal helpers |
| `get_schema_summary()` | JSON string — table names + column name lists only (lightweight) | `get_full_schema_summary_tool` in planner |
| `get_subset_schema_text(table_names)` | JSON string — full schema for requested tables | `get_table_details_tool` |
| `get_subset_integrity_text(table_names)` | JSON string — failed checks only for requested tables | `get_table_details_tool` |
| `load_all_metadata()` | Dict with all four views | available for ad-hoc use |

## For AI Agents

### Working In This Directory
- `ROOT_DIR` is resolved as two levels above this file (`SQL_Agent/`), so `data/` is always found correctly regardless of working directory.
- Both JSON files **must exist** before the agent starts — if missing, `FileNotFoundError` is raised immediately.
- `get_subset_integrity_text` filters to **failed checks only** to keep the tool response compact; PASS results are suppressed.

### Common Patterns
- `get_schema_summary()` is used for the broad planner step (low token cost).
- `get_subset_schema_text` + `get_subset_integrity_text` are combined in a single `get_table_details_tool` call for the designer and generator subagents.

## Dependencies

### Internal
- `data/db_schema.json` — schema cache
- `data/db_integrity_result.json` — integrity results cache

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
