from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from sqlalchemy import inspect, text

from SQL_Agent.sql_agent.deep_agents.src.db.db_connect import (
    get_database_settings,
    get_datamart_db_engine,
    get_source_db_engine,
)
from SQL_Agent.sql_agent.deep_agents.src.validator.integrity_loader import load_all_metadata

load_dotenv()

MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))
MAX_RESULT_PREVIEW_ROWS = int(os.getenv("MAX_RESULT_PREVIEW_ROWS", "20"))
ALLOW_MART_WRITE = os.getenv("ALLOW_MART_WRITE", "true").lower() == "true"
GOOGLE_MODEL = os.getenv("AGENT_MODEL", "google_genai:gemini-2.5-flash")
CHECKPOINTER = MemorySaver()

class SourceQueryPlan(BaseModel):
    sql: str = Field(description="Source DB against read-only SELECT SQL")
    business_grain: str = Field(description="Result grain")
    reasoning: str = Field(description="Why this query answers the question")


class DatamartBuildPlan(BaseModel):
    target_table: str = Field(description="Datamart table name only, without database prefix")
    source_sql: str = Field(description="Read-only SELECT SQL to run against the source DB")
    grain: str = Field(description="Datamart grain")
    load_strategy: Literal["replace", "append"] = Field(description="How to materialize the mart")
    key_columns: list[str] = Field(default_factory=list, description="Key columns at the mart grain")
    dimension_columns: list[str] = Field(default_factory=list, description="Dimension columns kept in the mart")
    measure_columns: list[str] = Field(default_factory=list, description="Measure columns kept in the mart")
    reasoning: str = Field(description="Why this mart design matches the request")


def format_rows(rows: list[dict[str, Any]], max_rows: int = MAX_RESULT_PREVIEW_ROWS) -> str:
    if not rows:
        return "결과 없음"
    preview = rows[:max_rows]
    return json.dumps(preview, ensure_ascii=False, indent=2, default=str)


def _normalize_sql(sql: str) -> str:
    cleaned = sql.replace("```sql", "").replace("```", "").strip()
    if not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


def is_safe_select_sql(sql: str) -> bool:
    lowered = _normalize_sql(sql).lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False

    banned_patterns = [
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\btruncate\b",
        r"\bcreate\b",
        r"\breplace\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\brename\b",
        r";\s*\S",
    ]
    return not any(re.search(pattern, lowered) for pattern in banned_patterns)


def is_safe_datamart_table_name(table_name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name))


def _run_select(engine, sql: str) -> pd.DataFrame:
    query = _normalize_sql(sql)
    if not is_safe_select_sql(query):
        raise ValueError("읽기 전용 SELECT SQL만 허용됩니다.")
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _get_source_engine():
    return get_source_db_engine()


def _get_datamart_engine():
    return get_datamart_db_engine()


def _get_datamart_metadata() -> dict[str, Any]:
    settings = get_database_settings("datamart")
    payload: dict[str, Any] = {
        "datamart_database": settings.name or "analytics",
    }
    try:
        datamart_engine = _get_datamart_engine()
        payload["datamart_status"] = "available"
        payload["datamart_schema"] = _introspect_tables(datamart_engine)
    except Exception as exc:
        payload["datamart_status"] = "unavailable"
        payload["datamart_schema"] = {}
        payload["datamart_error"] = str(exc)
    return payload


def _introspect_tables(engine) -> dict[str, Any]:
    inspector = inspect(engine)
    tables: dict[str, Any] = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        tables[table_name] = {
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": bool(column["nullable"]),
                }
                for column in columns
            ]
        }
    return tables


@tool
def get_metadata_context() -> str:
    """Load source schema/integrity metadata and current datamart table metadata."""
    metadata = load_all_metadata()
    payload = {
        "source_schema": metadata["schema_json"],
        "source_integrity": metadata["integrity_json"],
    }
    payload.update(_get_datamart_metadata())
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool
def run_source_query(sql: str) -> str:
    """Execute a read-only SELECT query against the source database and return a JSON preview."""
    df = _run_select(_get_source_engine(), sql)
    rows = df.head(MAX_RESULT_ROWS).to_dict(orient="records")
    payload = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": rows,
        "preview": format_rows(rows),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@tool
def run_datamart_query(sql: str) -> str:
    """Execute a read-only SELECT query against the datamart database and return a JSON preview."""
    try:
        datamart_engine = _get_datamart_engine()
    except Exception as exc:
        raise ValueError(f"datamart DB를 사용할 수 없습니다. {exc}") from exc

    df = _run_select(datamart_engine, sql)
    rows = df.head(MAX_RESULT_ROWS).to_dict(orient="records")
    payload = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": rows,
        "preview": format_rows(rows),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@tool
def materialize_datamart(
    target_table: str,
    source_sql: str,
    load_strategy: Literal["replace", "append"] = "replace",
) -> str:
    """Run a read-only source SELECT and materialize its result into the datamart database."""
    if not ALLOW_MART_WRITE:
        raise ValueError("현재 설정상 datamart 쓰기가 비활성화되어 있습니다.")
    if not is_safe_datamart_table_name(target_table):
        raise ValueError("target_table은 영문/숫자/언더스코어만 포함한 단일 테이블명이어야 합니다.")

    source_engine = _get_source_engine()
    try:
        datamart_engine = _get_datamart_engine()
    except Exception as exc:
        raise ValueError(
            "datamart DB를 사용할 수 없습니다. "
            "DATAMART_DB_* 또는 fallback SOURCE_DB_* 설정을 확인하세요. "
            f"원인: {exc}"
        ) from exc

    df = _run_select(source_engine, source_sql)
    df.to_sql(target_table, datamart_engine, if_exists=load_strategy, index=False)

    verification_sql = f"SELECT COUNT(*) AS row_count FROM `{target_table}`;"
    verify_df = _run_select(datamart_engine, verification_sql)
    payload = {
        "target_database": datamart_engine.url.database,
        "target_table": target_table,
        "load_strategy": load_strategy,
        "source_row_count": int(len(df)),
        "datamart_row_count": int(verify_df.iloc[0]["row_count"]),
        "columns": list(df.columns),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _main_agent_prompt() -> str:
    datamart_db_name = get_database_settings("datamart").name or "analytics"
    return f"""
너는 SQL 분석과 datamart 구축을 함께 수행하는 한국어 데이터 에이전트다.

중요한 운영 규칙:
- source DB와 datamart DB는 분리되어 있다.
- source DB는 조회 전용이다. source DB에 쓰기 SQL을 실행하면 안 된다.
- datamart 구축은 반드시 `materialize_datamart` 도구를 사용한다.
- datamart 구축을 위해서는 source DB에서 읽기 전용 SELECT를 만든 뒤, 그 결과를 datamart DB({datamart_db_name})에 적재한다.
- datamart 테이블명은 데이터베이스 접두어 없이 단일 테이블명만 사용한다.
- source metadata가 필요하면 먼저 `get_metadata_context`를 호출한다.
- query_answer 요청은 보통 `source-sql-analyst`에게 위임하고, datamart 구축 요청은 `datamart-engineer`에게 위임한다.
- 답변에는 추측을 넣지 말고, 실제 조회/적재 결과에 근거해서만 설명한다.
""".strip()


def _source_subagent_prompt() -> str:
    return """
너는 source DB 전용 SQL 분석가다.

역할:
- 자연어 질문을 source DB용 읽기 전용 SELECT SQL로 변환한다.
- 답을 만들기 전에 필요하면 `get_metadata_context`와 `run_source_query`를 사용해 검증한다.

제약:
- SELECT 또는 WITH ... SELECT만 허용한다.
- source DB에 쓰기 SQL을 제안하거나 실행하면 안 된다.
- 결과는 반드시 구조화된 JSON 스키마에 맞춰 반환한다.
""".strip()


def _datamart_subagent_prompt() -> str:
    return """
너는 datamart 설계 및 적재 전용 엔지니어다.

역할:
- 사용자의 mart 요구를 source DB 기반 SELECT와 datamart 테이블 설계로 바꾼다.
- 필요하면 `get_metadata_context`, `run_source_query`, `run_datamart_query`를 사용해 grain과 컬럼을 검증한다.

제약:
- datamart 적재는 상위 에이전트가 `materialize_datamart`로 수행하므로, 너는 source SELECT와 테이블 설계를 명확히 제시해야 한다.
- target_table은 데이터베이스 이름 없이 단일 테이블명만 반환한다.
- 결과는 반드시 구조화된 JSON 스키마에 맞춰 반환한다.
""".strip()


def build_app():
    from deepagents import create_deep_agent

    subagents = [
        {
            "name": "source-sql-analyst",
            "description": "Natural-language question to read-only source SQL plan",
            "system_prompt": _source_subagent_prompt(),
            "tools": [get_metadata_context, run_source_query],
            "response_format": SourceQueryPlan,
        },
        {
            "name": "datamart-engineer",
            "description": "Designs a datamart using source SELECTs and separated datamart loading",
            "system_prompt": _datamart_subagent_prompt(),
            "tools": [get_metadata_context, run_source_query, run_datamart_query],
            "response_format": DatamartBuildPlan,
        },
    ]

    return create_deep_agent(
        model=GOOGLE_MODEL,
        name="sql-agent",
        system_prompt=_main_agent_prompt(),
        tools=[get_metadata_context, run_source_query, run_datamart_query, materialize_datamart],
        subagents=subagents,
        interrupt_on={
            "materialize_datamart": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=CHECKPOINTER,
    )
