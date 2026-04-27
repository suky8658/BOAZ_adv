<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-27 | Updated: 2026-04-27 -->

# tools

## Purpose
Reserved extension directory for future LangChain `@tool`-decorated functions. Currently contains only a package marker.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package marker (empty) |

## For AI Agents

### Working In This Directory
- Add new LangChain tools here as standalone modules when the agent needs additional capabilities.
- Export tools via `__init__.py` so `sql_agent/sql_agent.py` can import them cleanly.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
