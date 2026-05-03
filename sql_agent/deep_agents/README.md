# SQL_Agent

`deepagents` 기반의 conversational SQL agent입니다. 자연어 질문에 대해 source DB를 읽기 전용으로 조회하고, 필요하면 그 결과를 별도 datamart DB로 적재합니다.

## What Changed

- 기존 LangGraph `StateGraph` 수동 파이프라인을 `deepagents.create_deep_agent(...)` 기반 오케스트레이션으로 전환했습니다.
- `source DB`와 `datamart DB`를 분리했습니다.
- datamart 생성은 더 이상 단일 DB 내부의 `CREATE TABLE AS SELECT`에 의존하지 않습니다.
- 대신 source DB에서 `SELECT`를 실행한 뒤, 결과를 datamart DB에 materialize 합니다.

## Environment Variables

기존 환경 변수는 source DB에 대한 legacy fallback으로 계속 읽습니다.

### Source DB

- `SOURCE_DB_USER`
- `SOURCE_DB_PASSWORD`
- `SOURCE_DB_HOST`
- `SOURCE_DB_PORT`
- `SOURCE_DB_NAME`

Legacy fallback:

- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`

### Datamart DB

- `DATAMART_DB_USER`
- `DATAMART_DB_PASSWORD`
- `DATAMART_DB_HOST`
- `DATAMART_DB_PORT`
- `DATAMART_DB_NAME`

설정하지 않으면 접속 정보는 source DB를 상속하고, 데이터베이스 이름만 `analytics`를 기본값으로 사용합니다.

### Agent

- `GOOGLE_API_KEY`
- `AGENT_MODEL` 기본값: `google_genai:gemini-2.5-flash`
- `ALLOW_MART_WRITE` 기본값: `true`

## Run

`deepagents`는 공식 문서 기준 Python `3.11+`가 필요합니다.

```bash
uv sync
PYTHONPATH=src uv run python src/run_agent.py
```

`.env.example`을 기준으로 `.env`를 채운 뒤 실행합니다.

## Smoke Scenarios

사전 정의된 질문으로 조회/적재 경로를 빠르게 검증할 수 있습니다.

```bash
PYTHONPATH=src uv run python src/smoke_agent.py --scenario query
PYTHONPATH=src uv run python src/smoke_agent.py --scenario datamart
```

커스텀 질문으로도 바로 호출할 수 있습니다.

```bash
PYTHONPATH=src uv run python src/smoke_agent.py --scenario datamart --question "주문 일자별 매출 mart를 만들어줘."
```

## Tests

```bash
pytest
ruff check src tests
```
