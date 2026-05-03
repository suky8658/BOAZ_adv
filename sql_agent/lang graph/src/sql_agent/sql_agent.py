import os
import json
from typing import TypedDict, Any, Optional, List, Dict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import text
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from db.db_connect import get_db_engine
from validator.integrity_loader import load_all_metadata

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
ALLOWED_MART_SCHEMA = os.getenv("ALLOWED_MART_SCHEMA", "analytics")
ALLOW_MART_WRITE = os.getenv("ALLOW_MART_WRITE", "true").lower() == "true"


# -----------------------------
# Pydantic Models
# -----------------------------
class QuestionPlan(BaseModel):
    original_question: str = Field(description="사용자 원문 질문")
    question_type: str = Field(description="aggregation/comparison/ranking/filter/detail/trend/identification/mart_build")
    task_type: str = Field(description="query_answer 또는 data_mart_build")
    requested_output: str = Field(description="sql_only / execute_and_answer / create_table")
    target_metric: str = Field(description="핵심 지표")
    dimensions: List[str] = Field(default_factory=list, description="그룹 기준")
    filters: List[str] = Field(default_factory=list, description="필터 조건")
    time_condition: Optional[str] = Field(default=None, description="시간 조건")
    relevant_tables: List[str] = Field(default_factory=list, description="관련 테이블")
    mart_name: Optional[str] = Field(default=None, description="생성 대상 마트명")
    grain: Optional[str] = Field(default=None, description="마트 grain")
    load_strategy: Optional[str] = Field(default=None, description="full_refresh / incremental")
    ambiguity_note: Optional[str] = Field(default=None, description="애매한 표현")


class MartDesign(BaseModel):
    mart_name: str
    target_schema: str
    grain: str
    source_tables: List[str] = Field(default_factory=list)
    key_columns: List[str] = Field(default_factory=list)
    measure_columns: List[str] = Field(default_factory=list)
    dimension_columns: List[str] = Field(default_factory=list)
    incremental_column: Optional[str] = None
    load_strategy: str = "full_refresh"
    design_reasoning: str


class SQLDraft(BaseModel):
    sql: str
    sql_type: str = Field(description="select / create_table_as / insert_select")
    target_table: Optional[str] = None
    source_tables: List[str] = Field(default_factory=list)
    columns_used: List[str] = Field(default_factory=list)
    business_grain: Optional[str] = None
    precheck_sql: Optional[str] = None
    postcheck_sql: Optional[str] = None
    reasoning: str


class ValidationResult(BaseModel):
    result: str
    reason: str
    feedback: str


class AgentState(TypedDict):
    user_question: str
    schema_text: str
    integrity_text: str

    plan: Dict[str, Any]
    mart_design: Dict[str, Any]
    sql_draft: Dict[str, Any]

    sql_result: Any
    row_count: int

    precheck_result: Any
    postcheck_result: Any
    mart_quality_result: Dict[str, Any]

    validation: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    final_answer: str


engine = get_db_engine()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# -----------------------------
# Utils
# -----------------------------
def safe_json_parse(text_value: str, fallback: dict) -> dict:
    cleaned = text_value.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return fallback


def clean_sql(sql: str) -> str:
    sql = sql.replace("```sql", "").replace("```", "").strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql


def format_result_rows(rows: Any, max_rows: int = 10) -> str:
    if not rows:
        return "결과 없음"
    preview = rows[:max_rows]
    return "\n".join([str(tuple(r)) for r in preview])


def is_safe_query_sql(sql: str) -> bool:
    lowered = sql.strip().lower()
    if lowered.startswith("select") or lowered.startswith("with"):
        banned = ["insert ", "update ", "delete ", "drop ", "alter ", "truncate ", "create "]
        return not any(k in lowered for k in banned)
    return False


def is_safe_mart_sql(sql: str, target_table: Optional[str]) -> tuple[bool, str]:
    lowered = sql.strip().lower()

    if not ALLOW_MART_WRITE:
        return False, "현재 설정상 마트 생성 SQL 실행이 비활성화되어 있습니다."

    banned = [
        "drop database", "drop schema", "truncate ", "alter table",
        "grant ", "revoke ", "rename table"
    ]
    if any(k in lowered for k in banned):
        return False, "위험한 DDL/DCL 문이 포함되어 있습니다."

    allowed_prefixes = [
        "create table",
        "create or replace table",
        "insert into"
    ]
    if not any(lowered.startswith(p) for p in allowed_prefixes):
        return False, "허용되지 않은 마트 생성 SQL 형식입니다."

    if target_table:
        target_table_lower = target_table.lower()
        if ALLOWED_MART_SCHEMA.lower() not in target_table_lower:
            return False, f"타겟 테이블은 허용된 스키마({ALLOWED_MART_SCHEMA}) 안에 있어야 합니다."

    return True, ""


def run_sql_fetchall(sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).fetchall()


def run_sql_commit(sql: str):
    with engine.begin() as conn:
        conn.execute(text(sql))


# -----------------------------
# Nodes
# -----------------------------
def load_context(state: AgentState):
    metadata = load_all_metadata()
    return {
        "schema_text": metadata["schema_text"],
        "integrity_text": metadata["integrity_text"],
    }


def plan_question(state: AgentState):
    prompt = f"""
너는 MySQL 기반 SQL/데이터마트 설계 질문 분석기다.

사용자 질문:
{state['user_question']}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

규칙:
- task_type은 반드시 query_answer 또는 data_mart_build 중 하나
- 사용자가 "데이터마트", "마트 생성", "집계 테이블", "요약 테이블", "분석용 테이블 생성" 의도를 가지면 data_mart_build
- requested_output은 sql_only / execute_and_answer / create_table 중 하나
- relevant_tables는 실제 존재하는 테이블만
- 질문에 없는 조건 임의 추가 금지
- 애매한 점은 ambiguity_note에 기록
- 반드시 JSON만 출력

출력 형식:
{{
  "original_question": "...",
  "question_type": "...",
  "task_type": "...",
  "requested_output": "...",
  "target_metric": "...",
  "dimensions": ["..."],
  "filters": ["..."],
  "time_condition": "... 또는 null",
  "relevant_tables": ["..."],
  "mart_name": "... 또는 null",
  "grain": "... 또는 null",
  "load_strategy": "... 또는 null",
  "ambiguity_note": "... 또는 null"
}}
"""
    response = llm.invoke(prompt).content

    fallback = QuestionPlan(
        original_question=state["user_question"],
        question_type="unknown",
        task_type="query_answer",
        requested_output="execute_and_answer",
        target_metric="unknown",
        dimensions=[],
        filters=[],
        time_condition=None,
        relevant_tables=[],
        mart_name=None,
        grain=None,
        load_strategy=None,
        ambiguity_note="질문 분석 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["original_question"] = state["user_question"]

    return {"plan": parsed}


def design_mart(state: AgentState):
    if state["plan"].get("task_type") != "data_mart_build":
        return {"mart_design": {}}

    prompt = f"""
너는 분석용 데이터마트 설계자다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

설계 규칙:
- 분석에 재사용 가능한 데이터마트 기준으로 설계
- grain을 반드시 명확히 정의
- key_columns, dimension_columns, measure_columns를 분리
- target_schema는 "{ALLOWED_MART_SCHEMA}" 로 고정
- incremental이 자연스러우면 incremental_column 제안
- 질문에 없는 정의를 과도하게 추가하지 말고 reasoning에 근거 설명
- 반드시 JSON만 출력

출력 형식:
{{
  "mart_name": "...",
  "target_schema": "{ALLOWED_MART_SCHEMA}",
  "grain": "...",
  "source_tables": ["..."],
  "key_columns": ["..."],
  "measure_columns": ["..."],
  "dimension_columns": ["..."],
  "incremental_column": "... 또는 null",
  "load_strategy": "full_refresh 또는 incremental",
  "design_reasoning": "..."
}}
"""
    response = llm.invoke(prompt).content

    fallback = MartDesign(
        mart_name=state["plan"].get("mart_name") or "mart_unknown",
        target_schema=ALLOWED_MART_SCHEMA,
        grain=state["plan"].get("grain") or "grain 미정",
        source_tables=state["plan"].get("relevant_tables", []),
        key_columns=[],
        measure_columns=[],
        dimension_columns=[],
        incremental_column=None,
        load_strategy=state["plan"].get("load_strategy") or "full_refresh",
        design_reasoning="마트 설계 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    return {"mart_design": parsed}


def generate_sql(state: AgentState):
    feedback = state.get("feedback", "").strip()
    task_type = state["plan"].get("task_type", "query_answer")

    if task_type == "data_mart_build":
        prompt = f"""
너는 MySQL 데이터마트 생성 SQL 작성기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

마트 설계 결과:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

이전 피드백:
{feedback if feedback else "없음"}

규칙:
- CREATE TABLE ... AS SELECT 또는 INSERT INTO ... SELECT 형태만 허용
- 타겟 스키마는 반드시 {ALLOWED_MART_SCHEMA}
- source는 실제 존재 테이블만 사용
- grain이 깨지지 않게 집계
- 모호한 기준은 reasoning에 명시
- precheck_sql에는 원천 데이터 건수/기간 확인용 SELECT
- postcheck_sql에는 생성 후 row_count / 중복 / null 점검용 SELECT
- DROP, ALTER, TRUNCATE 금지
- 반드시 JSON만 출력

출력 형식:
{{
  "sql": "...",
  "sql_type": "create_table_as 또는 insert_select",
  "target_table": "{ALLOWED_MART_SCHEMA}.xxx",
  "source_tables": ["..."],
  "columns_used": ["..."],
  "business_grain": "...",
  "precheck_sql": "SELECT ...",
  "postcheck_sql": "SELECT ...",
  "reasoning": "..."
}}
"""
    else:
        prompt = f"""
너는 MySQL 조회 SQL 작성기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

스키마 JSON:
{state['schema_text']}

정합성 점검 JSON:
{state['integrity_text']}

이전 피드백:
{feedback if feedback else "없음"}

규칙:
- MySQL SELECT SQL만 생성
- WITH 절 허용
- 질문에 없는 조건 임의 추가 금지
- 정합성 문제가 있는 컬럼/테이블 주의
- 반드시 JSON만 출력

출력 형식:
{{
  "sql": "SELECT ...",
  "sql_type": "select",
  "target_table": null,
  "source_tables": ["..."],
  "columns_used": ["..."],
  "business_grain": null,
  "precheck_sql": null,
  "postcheck_sql": null,
  "reasoning": "..."
}}
"""

    response = llm.invoke(prompt).content

    fallback = SQLDraft(
        sql="SELECT 1;",
        sql_type="select",
        target_table=None,
        source_tables=[],
        columns_used=[],
        business_grain=None,
        precheck_sql=None,
        postcheck_sql=None,
        reasoning="SQL 생성 파싱 실패"
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["sql"] = clean_sql(parsed.get("sql", "SELECT 1;"))

    if parsed.get("precheck_sql"):
        parsed["precheck_sql"] = clean_sql(parsed["precheck_sql"])
    if parsed.get("postcheck_sql"):
        parsed["postcheck_sql"] = clean_sql(parsed["postcheck_sql"])

    return {"sql_draft": parsed}


def execute_sql(state: AgentState):
    sql = state["sql_draft"]["sql"].strip()
    sql_type = state["sql_draft"].get("sql_type", "select")
    target_table = state["sql_draft"].get("target_table")

    try:
        pre_rows = None
        if state["sql_draft"].get("precheck_sql"):
            pre_rows = run_sql_fetchall(state["sql_draft"]["precheck_sql"])

        if sql_type == "select":
            if not is_safe_query_sql(sql):
                return {
                    "sql_result": None,
                    "row_count": 0,
                    "precheck_result": pre_rows,
                    "postcheck_result": None,
                    "error": "조회 SQL 안전성 검사 실패"
                }

            rows = run_sql_fetchall(sql)

            return {
                "sql_result": rows,
                "row_count": len(rows),
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": ""
            }

        ok, reason = is_safe_mart_sql(sql, target_table)
        if not ok:
            return {
                "sql_result": None,
                "row_count": 0,
                "precheck_result": pre_rows,
                "postcheck_result": None,
                "error": reason
            }

        run_sql_commit(sql)

        post_rows = None
        if state["sql_draft"].get("postcheck_sql"):
            post_rows = run_sql_fetchall(state["sql_draft"]["postcheck_sql"])

        return {
            "sql_result": [("마트 생성 완료", target_table)],
            "row_count": 1,
            "precheck_result": pre_rows,
            "postcheck_result": post_rows,
            "error": ""
        }

    except Exception as e:
        return {
            "sql_result": None,
            "row_count": 0,
            "precheck_result": None,
            "postcheck_result": None,
            "error": str(e)
        }


def validate_sql_and_result(state: AgentState):
    if state.get("error"):
        parsed = ValidationResult(
            result="invalid",
            reason=f"SQL 실행 오류: {state['error']}",
            feedback="실행 오류를 해결하고 task_type에 맞는 MySQL SQL로 다시 생성하라."
        ).model_dump()
        return {"validation": parsed, "feedback": parsed["feedback"]}

    prompt = f"""
너는 SQL/데이터마트 검증기다.

사용자 질문:
{state['user_question']}

질문 분석 결과:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

마트 설계 결과:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

정합성 점검 JSON:
{state['integrity_text']}

생성된 SQL:
{state['sql_draft']['sql']}

SQL 설명:
{state['sql_draft']['reasoning']}

사전 점검 결과:
{format_result_rows(state.get('precheck_result'))}

실행 결과:
{format_result_rows(state['sql_result'])}

사후 점검 결과:
{format_result_rows(state.get('postcheck_result'))}

행 수:
{state['row_count']}

검증 규칙:
1. task_type=query_answer 이면 질문 조건 충족 여부 검증
2. task_type=data_mart_build 이면 grain 적합성, 타겟 테이블 적절성, 재사용성 검증
3. 질문에 없는 조건 추가면 invalid
4. 정합성 문제 무시하면 invalid
5. 반드시 JSON만 출력

출력 형식:
{{
  "result": "valid" 또는 "invalid",
  "reason": "...",
  "feedback": "..."
}}
"""
    response = llm.invoke(prompt).content

    fallback = ValidationResult(
        result="invalid",
        reason="검증 결과 파싱 실패",
        feedback="질문 조건, grain, 정합성 점검 내용을 반영해 다시 SQL을 생성하라."
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    return {"validation": parsed, "feedback": parsed.get("feedback", "")}


def finalize_answer(state: AgentState):
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "검증 실패\n"
                f"사유: {state['validation'].get('reason')}\n"
                f"마지막 SQL: {state['sql_draft'].get('sql')}"
            )
        }

    if state["plan"].get("task_type") == "data_mart_build":
        prompt = f"""
너는 데이터 엔지니어/분석가용 결과 요약기다.

사용자 질문:
{state['user_question']}

마트 설계:
{json.dumps(state.get('mart_design', {}), ensure_ascii=False, indent=2)}

생성 SQL:
{state['sql_draft']['sql']}

사전 점검 결과:
{format_result_rows(state.get('precheck_result'))}

사후 점검 결과:
{format_result_rows(state.get('postcheck_result'))}

규칙:
- 한국어
- 마트명, grain, 적재 방식, 핵심 컬럼, 검증 결과를 짧게 요약
- 없는 내용 추측 금지
"""
        answer = llm.invoke(prompt).content.strip()
        return {"final_answer": answer}

    prompt = f"""
너는 데이터 분석 답변 작성기다.

사용자 질문:
{state['user_question']}

SQL 결과:
{format_result_rows(state['sql_result'], max_rows=20)}

규칙:
- 한국어
- 핵심 결과 먼저
- 없는 내용 추측 금지
- 결과 범위 안에서만 설명
"""
    answer = llm.invoke(prompt).content.strip()
    return {"final_answer": answer}


def increase_retry(state: AgentState):
    return {"retry_count": state["retry_count"] + 1}


def route_after_validation(state: AgentState):
    if state["validation"].get("result") == "valid":
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("plan_question", plan_question)
    graph.add_node("design_mart", design_mart)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("validate_sql_and_result", validate_sql_and_result)
    graph.add_node("increase_retry", increase_retry)
    graph.add_node("finalize_answer", finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan_question")
    graph.add_edge("plan_question", "design_mart")
    graph.add_edge("design_mart", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "validate_sql_and_result")

    graph.add_conditional_edges(
        "validate_sql_and_result",
        route_after_validation,
        {
            "retry": "increase_retry",
            "finalize": "finalize_answer"
        }
    )

    graph.add_edge("increase_retry", "generate_sql")
    graph.add_edge("finalize_answer", END)

    return graph.compile()