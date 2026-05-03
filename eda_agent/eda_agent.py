import os
import json
from typing import TypedDict, Any, Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

from eda_agent.db.db_connect import get_db_engine, load_mart
from eda_agent.tools.profiling import get_basic_profile
from eda_agent.skills.data_quality_skill import run_data_quality_skill
from eda_agent.skills.distribution_skill import run_distribution_skill
from eda_agent.skills.comparison_skill import run_comparison_skill
from eda_agent.skills.relationship_skill import run_relationship_skill
from eda_agent.skills.time_skill import run_time_skill
from eda_agent.skills.clustering_skill import run_clustering_skill
from eda_agent.skills.chart_selector_skill import run_chart_selector_skill

load_dotenv()

# ─────────────────────────────
# 모듈 레벨 DataFrame / 키 컬럼 / 지표 컬럼 / 시간 컬럼 / 표본 수 컬럼
# ─────────────────────────────
_df: Optional[pd.DataFrame] = None
_key_col: Optional[str] = None
_measure_cols: Optional[List[str]] = None
_time_cols: Optional[List[str]] = None
_count_col: Optional[str] = None
_question_type: str = ""
_priority_metrics: list = []


# ─────────────────────────────
# Tools
# ─────────────────────────────

# inspect
@tool
def profile_data() -> str:
    """데이터 기본 구조(shape, 컬럼 타입, 카디널리티, 시간 컬럼 여부, 기초통계)를 반환한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    result = get_basic_profile(_df)
    cardinality = {col: int(_df[col].nunique()) for col in _df.columns}
    time_cols = [c for c in _df.columns if "datetime" in str(_df[c].dtype) or "date" in c.lower()]
    return json.dumps({
        "shape": result["shape"],
        "dtypes": result["dtypes"],
        "cardinality": cardinality,
        "time_columns": time_cols,
        "describe": {c: {str(k): str(v) for k, v in s.items()} for c, s in result["describe"].items()},
    }, ensure_ascii=False)


# ─────────────────────────────
# Skill-level tools (planner가 skill 단위로 호출)
# ─────────────────────────────

@tool
def run_quality() -> str:
    """data_quality_skill: 결측치 / 이상치 / 중복 / 표본 신뢰도를 한 번에 점검한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(run_data_quality_skill(_df, key_col=_key_col, measure_cols=_measure_cols, count_col=_count_col), ensure_ascii=False)


@tool
def run_distribution() -> str:
    """distribution_skill: 히스토그램 / 박스플롯 / 범주형 빈도 분포를 한 번에 분석한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(run_distribution_skill(_df, measure_cols=_measure_cols, question_type=_question_type, priority_metrics=_priority_metrics), ensure_ascii=False)


@tool
def run_comparison() -> str:
    """comparison_skill: 카테고리별 상위/하위 barplot + 히트맵 비교를 한 번에 수행한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(run_comparison_skill(_df, key_col=_key_col, measure_cols=_measure_cols, question_type=_question_type), ensure_ascii=False)


@tool
def run_relationship() -> str:
    """relationship_skill: 상관관계 히트맵 + scatter plot을 한 번에 분석한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(run_relationship_skill(_df, measure_cols=_measure_cols, question_type=_question_type), ensure_ascii=False)


@tool
def run_time() -> str:
    """time_skill: 시계열 추세 + 시즌성 분석을 한 번에 수행한다."""
    if _df is None:
        return "데이터가 로드되지 않았습니다."
    return json.dumps(run_time_skill(_df, measure_cols=_measure_cols, time_cols=_time_cols), ensure_ascii=False)


# ─────────────────────────────
# 툴 그룹 정의
# ─────────────────────────────
INSPECT_TOOLS      = [profile_data]
QUALITY_TOOLS      = [run_quality]
DISTRIBUTION_TOOLS = [run_distribution]
COMPARISON_TOOLS   = [run_comparison]
RELATIONSHIP_TOOLS = [run_relationship]
TIME_TOOLS         = [run_time]


# ─────────────────────────────
# State
# ─────────────────────────────
class EDAState(TypedDict):
    # SQL Agent → EDA Agent 인터페이스 (eda_agent_input.json 필드와 동일)
    user_question: str          # 사용자 원본 질문
    target_table: str           # "sql_agent.{mart_name}"
    mart_design: Dict[str, Any] # grain, key_columns, measure_columns
    question_type: str          # "comparison" | "distribution" | "relationship" | "time"

    # planner 결정
    analysis_plan: Dict[str, Any]  # 실행할 노드 목록 + 각 노드 집중 전략

    # 각 노드 결과
    inspect_result: str
    quality_result: str
    distribution_result: str
    comparison_result: str
    relationship_result: str
    time_result: str
    clustering_result: Dict[str, Any]

    # 컬럼 의미 분류 (LLM이 로드 직후 판단)
    time_columns: List[str]    # 시간/날짜 컬럼
    count_column: str          # 표본 수/볼륨 컬럼 (없으면 "")

    # 플래그
    has_time_column: bool

    # 최종 출력
    insight_result: str
    hypotheses: str
    final_summary: str
    key_charts: List[str]
    statistical_metadata: Dict[str, Any]  # downstream 에이전트용 raw 수치

    # 에러 로그
    error_log: List[str]


# ─────────────────────────────
# 재시도 헬퍼
# ─────────────────────────────
MAX_NODE_RETRIES = 2  # 노드당 최대 재시도 횟수

def run_node_with_retry(fn, node_name: str, fallback="분석 스킵 (오류로 인해 생략됨)", max_retries: int = MAX_NODE_RETRIES):
    """
    노드 실행 함수를 감싸 에러 시 재시도하고, 모두 실패하면 fallback을 반환한다.
    반환값: (result, error_message or None)
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), None
        except Exception as e:
            last_error = str(e)
    return fallback, f"[{node_name}] {last_error}"


def run_mini_react_with_retry(
    tools_list: list,
    system_prompt: str,
    node_name: str,
    fallback: str = "분석 스킵 (오류로 인해 생략됨)",
    max_retries: int = MAX_NODE_RETRIES,
    max_iter: int = 8,
) -> tuple:
    """
    run_mini_react 실행 중 에러 발생 시 에러 내용을 LLM에게 피드백으로 넘겨 재시도한다.
    반환값: (result, error_message or None)
    """
    llm_feedback = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    last_error = None
    current_prompt = system_prompt

    for attempt in range(max_retries + 1):
        try:
            return run_mini_react(tools_list, current_prompt, max_iter=max_iter), None
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                # 에러 내용을 LLM에게 넘겨 수정된 프롬프트 생성
                fix_prompt = f"""
아래 분석 노드({node_name})에서 아래 에러가 발생했다.
에러: {last_error}

원래 분석 지시:
{current_prompt}

에러 원인을 파악하고, 에러를 피할 수 있도록 분석 방식을 조정한 새로운 지시문을 작성하라.
- 존재하지 않는 컬럼 참조나 타입 오류라면 해당 분석을 생략하도록 지시하라.
- 수정된 지시문만 출력하라. 설명 없이.
"""
                current_prompt = llm_feedback.invoke(fix_prompt).content.strip()

    return fallback, f"[{node_name}] {last_error}"


# ─────────────────────────────
# Mini-ReAct 헬퍼
# ─────────────────────────────
def run_mini_react(tools_list: list, system_prompt: str, max_iter: int = 8) -> str:
    """
    주어진 툴 목록 안에서만 LLM이 선택·호출하는 mini-ReAct 루프.
    툴 호출이 없으면 종료하고 최종 텍스트를 반환한다.
    """
    tools_dict = {t.name: t for t in tools_list}
    node_llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(tools_list)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="분석을 시작하라."),
    ]

    for _ in range(max_iter):
        response = node_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            fn = tools_dict.get(tc["name"])
            result = fn.invoke(tc["args"]) if fn else "툴을 찾을 수 없습니다."
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return messages[-1].content


# ─────────────────────────────
# Nodes
# ─────────────────────────────
def _classify_columns(df: pd.DataFrame, measure_cols: list) -> dict:
    """
    LLM이 컬럼명 + 샘플값을 보고 시간 컬럼 / 표본 수 컬럼을 의미 기반으로 분류.
    dtype 기반 휴리스틱으로 먼저 확인하고, 나머지를 LLM에게 판단 요청.
    """
    # 1차: dtype으로 확실한 시간 컬럼 추출
    obvious_time = [c for c in df.columns if "datetime" in str(df[c].dtype)]

    # 분류가 필요한 후보 컬럼 (measure가 아닌 것들 + 이름에서 의미 불분명한 것)
    candidate_cols = [c for c in df.columns if c not in (measure_cols or [])]

    if not candidate_cols:
        return {"time_columns": obvious_time, "count_column": ""}

    sample = df[candidate_cols].head(3).to_dict(orient="list")

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = f"""
아래 데이터마트의 컬럼 의미를 파악하라.

[전체 컬럼] {list(df.columns)}
[measure 컬럼 (분석 지표, 분류 불필요)] {measure_cols}
[분류 대상 컬럼 + 샘플값]
{json.dumps(sample, ensure_ascii=False, default=str)}

아래 두 가지를 판단하라:
1. time_columns: 날짜/시간/기간을 나타내는 컬럼 목록 (없으면 빈 리스트)
   - 예: created_at, order_time, month, period, dt, ts, 주문일자 등
2. count_column: 표본 수 또는 볼륨을 나타내는 컬럼 1개 (없으면 빈 문자열)
   - 예: volume, qty, n_records, num_transactions, 주문건수 등
   - measure_cols 안에 있어도 됨

반드시 아래 JSON 형식으로만 응답하라. 설명 없이 JSON만 출력하라.
{{
  "time_columns": [...],
  "count_column": "..."
}}
"""
    response = llm.invoke(prompt).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(response)
        time_cols   = list(set(obvious_time + result.get("time_columns", [])))
        count_col   = result.get("count_column", "")
        # 실제 컬럼에 있는 것만 유지
        time_cols   = [c for c in time_cols  if c in df.columns]
        count_col   = count_col if count_col in df.columns else ""
    except Exception:
        time_cols = obvious_time
        count_col = ""

    return {"time_columns": time_cols, "count_column": count_col}


def _is_meaningless_id(df: pd.DataFrame, col: str) -> bool:
    """컬럼이 UUID/해시처럼 시각화에 무의미한 고유 ID인지 판별."""
    if col not in df.columns:
        return False
    n_unique = df[col].nunique()
    n_rows = len(df)
    # 카디널리티가 전체 행의 50% 이상이면 고유 ID성
    if n_unique >= n_rows * 0.5:
        return True
    # 값이 32자 이상 hex 문자열이면 UUID/해시
    sample = df[col].dropna().astype(str).head(10)
    if sample.str.match(r'^[0-9a-f]{32,}$').mean() >= 0.8:
        return True
    return False


def _select_best_key_col(df: pd.DataFrame, key_columns: list, measure_cols: list) -> Optional[str]:
    """key_columns 중 시각화에 의미 있는 컬럼을 자동 선택."""
    candidates = [c for c in key_columns if c in df.columns]
    measure_set = set(measure_cols or [])
    for col in candidates:
        if col in measure_set:
            continue
        if not _is_meaningless_id(df, col):
            return col
    # 모두 무의미한 ID면 카디널리티 가장 낮은 것 선택
    non_measure = [c for c in candidates if c not in measure_set]
    if non_measure:
        return min(non_measure, key=lambda c: df[c].nunique())
    return candidates[0] if candidates else None


def load_mart_node(state: EDAState) -> dict:
    global _df, _key_col, _measure_cols, _time_cols, _count_col, _question_type, _priority_metrics
    engine = get_db_engine()
    table_name = state["target_table"].split(".")[-1]
    _df = load_mart(engine, table_name)
    key_columns      = state["mart_design"].get("key_columns", [])
    dimension_cols   = state["mart_design"].get("dimension_columns", [])
    _measure_cols    = state["mart_design"].get("measure_columns") or None

    if dimension_cols:
        _key_col = dimension_cols[0]
    else:
        _key_col = _select_best_key_col(_df, key_columns, _measure_cols or [])

    col_meta  = _classify_columns(_df, _measure_cols or [])
    _time_cols = col_meta["time_columns"]
    _count_col = col_meta["count_column"]
    _question_type   = state.get("question_type", "")
    _priority_metrics = []  # planner 실행 후 갱신됨

    return {
        "time_columns":    _time_cols,
        "count_column":    _count_col,
        "has_time_column": len(_time_cols) > 0,
        "error_log":       [],
    }


def planner_node(state: EDAState) -> dict:
    # 구조 데이터 직접 계산 (텍스트 파싱 오류 방지)
    if _df is not None:
        if _measure_cols:
            numeric_cols = [c for c in _measure_cols if c in _df.columns]
        else:
            numeric_cols = list(_df.select_dtypes(include=["float64", "int64"]).columns)
        cat_cols_list = list(_df.select_dtypes(include=["object"]).columns)
    else:
        numeric_cols, cat_cols_list = [], []
    time_columns = state.get("time_columns", [])

    row_count = len(_df) if _df is not None else 0

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = f"""
너는 EDA 분석 전략가다.
아래 정보를 바탕으로 이번 분석에서 어떤 단계를 어떻게 수행할지 계획을 세워라.

[사용자 질문] {state['user_question']}
[question_type] {state['question_type']}
[구조 확인값 (직접 계산)]
전체 행 수(grain 단위 수): {row_count}
수치형 컬럼 수: {len(numeric_cols)} → {numeric_cols}
범주형 컬럼 수: {len(cat_cols_list)} → {cat_cols_list}
시간 컬럼: {time_columns}
표본 수 컬럼: {state.get('count_column', '')}
[데이터 구조 요약]
{state['inspect_result']}
[마트 설계]
grain: {state['mart_design'].get('grain')}
measure_columns: {state['mart_design'].get('measure_columns')}
key_columns: {state['mart_design'].get('key_columns')}

아래 JSON 형식으로만 응답하라. 설명 없이 JSON만 출력하라.

{{
  "run_quality": true,
  "run_distribution": true,
  "run_comparison": true,
  "run_relationship": true,
  "distribution_focus": "사용자 질문과 관련된 수치형 컬럼의 분포 편향 여부 확인",
  "comparison_focus": "key_column 기준 그룹별 주요 지표 비교",
  "relationship_focus": "사용자 질문의 핵심 변수 간 상관관계 집중",
  "skip_reason": "해당 없으면 빈 문자열",
  "priority_metrics": [
    {{"metric": "사용자 질문에서 언급된 핵심 변수명", "reason": "사용자 질문과 직접 관련된 이유"}},
    {{"metric": "두 번째 핵심 변수명", "reason": "분석 목표와 연관된 이유"}}
  ]
}}

결정 기준:
- question_type이 comparison이면 comparison을 최우선으로
- question_type이 time이면 시계열 분석이 핵심 — distribution/comparison은 보조
- 수치형 컬럼이 2개 미만이면 run_relationship을 false로 (위 구조 확인값 기준으로 판단)
- 범주형 컬럼이 없으면 run_comparison을 false로 (위 구조 확인값 기준으로 판단)
- 전체 행 수가 30 미만이면 run_distribution을 false로 (분포 분석이 의미 없음)
- 전체 행 수가 10 미만이면 run_relationship도 false로 (상관관계 신뢰도 없음)
- priority_metrics는 사용자 질문과 가장 관련 높은 measure 컬럼 선택
"""
    response = llm.invoke(prompt).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(response)
    except Exception:
        plan = {
            "run_quality": True,
            "run_distribution": True,
            "run_comparison": True,
            "run_relationship": True,
            "distribution_focus": "전체 수치형 컬럼 분포 확인",
            "comparison_focus": "카테고리별 지표 비교",
            "relationship_focus": "수치형 지표 간 상관관계",
            "skip_reason": "",
            "priority_metrics": [
                {"metric": m, "reason": "measure 컬럼 (fallback)"}
                for m in state['mart_design'].get('measure_columns', [])
            ],
        }
    global _priority_metrics
    _priority_metrics = plan.get("priority_metrics", [])
    return {"analysis_plan": plan}


def inspect_node(state: EDAState) -> dict:
    prompt = f"""
너는 데이터 구조 분석 전문가다.

[사용자 쿼리] {state['user_question']}
[grain] {state['mart_design'].get('grain', '미정')}

profile_data를 호출하여:
1. 컬럼 타입과 카디널리티 파악
2. 수치형 / 범주형 / 시간형 컬럼 분류
3. grain 확인
4. 분석 가능한 지표 목록 정리
결과를 한국어로 요약하라.
"""
    result, err = run_mini_react_with_retry(INSPECT_TOOLS, prompt, "inspect")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"inspect_result": result, "error_log": errors}


def quality_node(state: EDAState) -> dict:
    prompt = f"""
너는 데이터 품질 전문가다.

[사용자 쿼리] {state['user_question']}
[구조 파악 결과]
{state['inspect_result']}

아래 툴을 모두 호출하여 데이터 품질을 점검하라:
1. check_missing — 결측치
2. check_outliers — 이상치
3. check_duplicates — 중복
4. check_sample_reliability — 표본 수 신뢰도

결과를 종합하여 한국어로 요약하라. 특히 분석 시 주의해야 할 품질 이슈를 명시하라.
"""
    result, err = run_mini_react_with_retry(QUALITY_TOOLS, prompt, "quality")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"quality_result": result, "error_log": errors}


def distribution_node(state: EDAState) -> dict:
    plan = state.get("analysis_plan", {})
    prompt = f"""
너는 단변량 분포 분석 전문가다.

[사용자 쿼리] {state['user_question']}
[구조 파악 결과]
{state['inspect_result']}
[이번 분석 집중 전략] {plan.get('distribution_focus', '전체 수치형 컬럼 분포 확인')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

데이터 특성에 맞게 아래 툴 중 필요한 것을 선택하여 호출하라:
- draw_distributions: 수치형 컬럼이 있을 때 (히스토그램)
- draw_boxplots: 이상치 시각화가 필요할 때
- draw_category_distribution: 범주형 컬럼이 있을 때

우선 분석 지표를 중심으로 분포 형태, 치우침, 분산 정도를 한국어로 요약하라.
"""
    result, err = run_mini_react_with_retry(DISTRIBUTION_TOOLS, prompt, "distribution")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"distribution_result": result, "error_log": errors}


def comparison_node(state: EDAState) -> dict:
    plan = state.get("analysis_plan", {})
    prompt = f"""
너는 그룹 비교 분석 전문가다.

[사용자 쿼리] {state['user_question']}
[구조 파악 결과]
{state['inspect_result']}
[이번 분석 집중 전략] {plan.get('comparison_focus', '카테고리별 지표 비교')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

아래 툴을 모두 호출하여 그룹 간 비교를 수행하라:
1. draw_top_n_barplot — 지표별 상위/하위 카테고리
2. draw_heatmap_matrix — 전체 카테고리 × 지표 한눈에 비교

우선 분석 지표를 중심으로 어떤 그룹이 강하고 약한지 한국어로 요약하라.
"""
    result, err = run_mini_react_with_retry(COMPARISON_TOOLS, prompt, "comparison")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"comparison_result": result, "error_log": errors}


def relationship_node(state: EDAState) -> dict:
    plan = state.get("analysis_plan", {})
    prompt = f"""
너는 변수 관계 분석 전문가다.

[사용자 쿼리] {state['user_question']}
[구조 파악 결과]
{state['inspect_result']}
[이번 분석 집중 전략] {plan.get('relationship_focus', '수치형 지표 간 상관관계 탐색')}
[우선 분석 지표] {plan.get('priority_metrics', [])}

아래 툴을 모두 호출하여 변수 간 관계를 탐색하라:
1. draw_correlation — 수치형 변수 간 상관관계
2. draw_scatter_pairs — 변수 쌍별 관계 시각화

우선 분석 지표와 관련된 상관관계, trade-off, 주목할 패턴을 한국어로 요약하라.
"""
    result, err = run_mini_react_with_retry(RELATIONSHIP_TOOLS, prompt, "relationship")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"relationship_result": result, "error_log": errors}


def time_node(state: EDAState) -> dict:
    prompt = f"""
너는 시계열 분석 전문가다.

[사용자 쿼리] {state['user_question']}
[구조 파악 결과]
{state['inspect_result']}

아래 툴을 호출하여 시간 패턴을 분석하라:
1. draw_timeseries — 시계열 추세
2. draw_seasonality — 월/요일 시즌성

추세, 계절성, 특이 시점을 한국어로 요약하라.
"""
    result, err = run_mini_react_with_retry(TIME_TOOLS, prompt, "time")
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"time_result": result, "error_log": errors}


def insight_node(state: EDAState) -> dict:
    # ── downstream 에이전트용 raw 수치 수집 (프롬프트 주입용으로 먼저 계산) ──
    statistical_metadata: Dict[str, Any] = {}
    if _df is not None:
        from eda_agent.tools.missing import detect_missing
        from eda_agent.tools.outlier import detect_outliers_iqr
        from eda_agent.tools.quality import check_duplicates_fn

        numeric_cols = [c for c in (_measure_cols or []) if c in _df.columns and pd.api.types.is_numeric_dtype(_df[c])]
        if not numeric_cols:
            numeric_cols = list(_df.select_dtypes(include=["float64", "int64"]).columns)

        dist_stats = {}
        for col in numeric_cols:
            s = _df[col].dropna()
            dist_stats[col] = {
                "mean":     round(float(s.mean()), 4),
                "median":   round(float(s.median()), 4),
                "std":      round(float(s.std()), 4),
                "skewness": round(float(s.skew()), 4),
                "min":      round(float(s.min()), 4),
                "max":      round(float(s.max()), 4),
            }

        corr_pairs = {}
        if len(numeric_cols) >= 2:
            corr = _df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    key = f"corr_{numeric_cols[i]}_vs_{numeric_cols[j]}"
                    corr_pairs[key] = round(float(corr.iloc[i, j]), 3)

        missing_info = detect_missing(_df)
        outlier_info = detect_outliers_iqr(_df, measure_cols=_measure_cols)
        dup_info     = check_duplicates_fn(_df)
        outliers_by_col = {
            col: v.get("outlier_count", 0)
            for col, v in outlier_info.items()
            if isinstance(v, dict)
        }

        # 그룹 비교 통계: key_col 기준 top3 / bottom3
        group_comparison = {}
        if _key_col and _key_col in _df.columns:
            for col in numeric_cols:
                try:
                    grp = _df.groupby(_key_col)[col].mean().dropna()
                    group_comparison[col] = {
                        "top3_groups":  {str(k): round(float(v), 4) for k, v in grp.nlargest(3).items()},
                        "bottom3_groups": {str(k): round(float(v), 4) for k, v in grp.nsmallest(3).items()},
                        "group_max":    round(float(grp.max()), 4),
                        "group_min":    round(float(grp.min()), 4),
                        "group_std":    round(float(grp.std()), 4),
                    }
                except Exception:
                    pass

        clustering = state.get("clustering_result", {})
        statistical_metadata = {
            "row_count":          len(_df),
            "distribution":       dist_stats,
            "group_comparison":   group_comparison,
            "correlation_pairs":  corr_pairs,
            "missing_total":      missing_info.get("total_missing", 0),
            "outliers_by_column": outliers_by_col,
            "duplicate_count":    dup_info.get("duplicate_count", 0),
            "clustering":         {
                "n_clusters":        clustering.get("n_clusters"),
                "silhouette_score":  clustering.get("silhouette_score"),
                "cluster_centroids": clustering.get("cluster_centroids", {}),
            } if not clustering.get("skip") else {"skip": True},
        }

    all_results = f"""
[구조 탐색] {state.get('inspect_result', '해당 없음')}
[품질 점검] {state.get('quality_result', '해당 없음')}
[분포 분석] {state.get('distribution_result', '해당 없음')}
[그룹 비교] {state.get('comparison_result', '해당 없음')}
[관계 탐색] {state.get('relationship_result', '해당 없음')}
[시간 분석] {state.get('time_result', '해당 없음')}
[클러스터링] {json.dumps(state.get('clustering_result', {}), ensure_ascii=False)}
"""
    prompt = f"""
너는 EDA 종합 분석가다. 데이터를 읽는 것을 넘어, 이 데이터가 어떤 시장·비즈니스 구조를 보여주는지 해석하는 것이 네 역할이다.

[사용자 쿼리]
{state['user_question']}

[검증된 수치 (구체적인 숫자를 쓸 때는 이 값을 우선 참고하라)]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2)}

[전체 EDA 결과 (패턴 해석의 주요 근거 — 분포/그룹비교/관계/시간 결과를 모두 활용하라)]
{all_results}

위 결과를 바탕으로 아래 두 섹션을 작성하라.
마크다운 기호(###, **, * 등)는 절대 사용하지 마라. 일반 텍스트로만 작성하라.

[핵심 패턴]
3~5가지를 번호 목록으로 작성하라.
각 항목은 반드시 아래 구조로 작성하라: 수치 근거 → 패턴 해석 → 해석 한계 또는 주의

작성 규칙:
- 검증된 수치의 clustering.skip이 False이고 n_clusters >= 2인 경우, 클러스터 결과를 번호 항목 중 하나로 반드시 포함하라.
  이때 각 클러스터를 centroid 수치 기반으로 직접 명명하고 (예: "배송지연·저만족 그룹", "고볼륨·균형 그룹"),
  각 그룹의 대표 항목을 cluster_labels에서 2~3개 직접 언급하라.
  클러스터 간 차이가 작으면 "해석에 주의가 필요하다"고 명시하고 비중을 줄여라.
- 클러스터 관련 내용을 별도 섹션으로 분리하거나 [핵심 패턴] 밖에 쓰지 마라. 반드시 번호 목록 안에 포함하라.

나쁜 예시 (쓰지 마라):
  "total_orders 평균 1314.43, 최소 2, 최대 9272" → 기술통계 복붙이지 패턴 해석이 아님
  "배송일을 줄이는 것이 중요한 요소다" → 단정적 결론 금지
  "클러스터링 결과, 네 개의 군집이 나뉜다" → 클러스터를 명명하지 않은 추상적 언급 금지

[구조 해석]
핵심 패턴들을 종합해 이 데이터가 어떤 시장·비즈니스 구조를 시사하는지 2~3문장으로 작성하라.
개별 수치를 나열하는 것이 아니라, 패턴들이 모여서 만드는 큰 그림을 해석하는 것이 목적이다.

아래 유형의 구조 언어를 참고하되, 데이터가 실제로 지지하는 경우에만 사용하라:
  - 집중형 구조: "소수 카테고리/고객에게 매출이 집중되는 헤드 집중형 구조를 보인다"
  - 분절형 구조: "성과 지표 간 뚜렷한 군집이 존재해, 수요 혹은 고객층이 분절되어 있을 가능성이 있다"
  - 편차형 구조: "카테고리별 충성도·재구매 편차가 크며, 이는 제품 특성보다 카테고리 고유의 구매 맥락 차이를 반영할 수 있다"
  - 트레이드오프 구조: "볼륨과 단위 수익성 간 음의 관계가 나타나, 규모와 마진 사이의 구조적 트레이드오프가 존재할 수 있다"

반드시 "~를 시사한다", "~일 가능성이 있다", "~로 해석될 수 있다" 같은 헤지 표현을 사용하라.
데이터에서 직접 관찰되지 않은 원인을 단정하지 마라.

[해석 주의사항]
- 결측치, 이상치, 표본 수 신뢰도 등 구체적 수치와 함께 작성
- 표본 수가 적어 신뢰도가 낮은 항목은 반드시 명시
- 한 줄로 간결하게

한국어로 작성하라.
"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    try:
        insight_result = llm.invoke(prompt).content.strip()
        err = None
    except Exception as e:
        from openai import RateLimitError
        if isinstance(e, RateLimitError):
            # TPM 초과 시 all_results 각 필드 300자로 truncate 후 재시도
            truncated_results = "\n".join([
                f"[{label}] {text[:300]}..."
                for label, text in [
                    ("구조 탐색", state.get("inspect_result", "")),
                    ("품질 점검", state.get("quality_result", "")),
                    ("분포 분석", state.get("distribution_result", "")),
                    ("그룹 비교", state.get("comparison_result", "")),
                    ("관계 탐색", state.get("relationship_result", "")),
                    ("시간 분석", state.get("time_result", "")),
                ]
                if text
            ])
            slim_prompt = prompt.replace(all_results, truncated_results)
            insight_result, err = run_node_with_retry(
                lambda: llm.invoke(slim_prompt).content.strip(), "insight", fallback="인사이트 생성 실패"
            )
        else:
            insight_result, err = run_node_with_retry(
                lambda: llm.invoke(prompt).content.strip(), "insight", fallback="인사이트 생성 실패"
            )
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"insight_result": insight_result, "statistical_metadata": statistical_metadata, "error_log": errors}


def hypothesis_node(state: EDAState) -> dict:
    prompt = f"""
너는 데이터 분석 가설 설계자다. 네 가설은 다음 단계의 분석 에이전트(통계 검정, 모델링 수행)가 바로 실행할 수 있는 수준이어야 한다.

[사용자 쿼리]
{state['user_question']}

[핵심 인사이트 및 구조 해석]
{state['insight_result']}

위 인사이트를 바탕으로 아래 형식으로 작성하라.
마크다운 기호(###, **, * 등)는 절대 사용하지 마라. 일반 텍스트로만 작성하라.

[가설 1] ~ [가설 3] 형식으로 3개를 작성하라. 핵심 패턴에서 검증 가능한 것만 골라라.
각 가설은 아래 구조를 그대로 따르라. 섹션 레이블([가설 1] 등, 관찰:, H0:, H1:, 검증방법:, 필요변수:, 현재데이터:)은 반드시 그대로 유지하라.

[가설 1]
관찰: (이 가설의 근거가 된 수치나 패턴을 1문장으로. 인사이트에서 끌어와라.)
H0: (귀무가설 — A와 B 사이에 유의미한 관계가 없다)
H1: (대립가설 — IF [조건] THEN [결과]. 방향을 명시하라.)
검증방법: (구체적인 통계 검정명. 예: 단순선형회귀, 스피어만 상관검정, 일원배치 ANOVA 등)
필요변수: target=[변수명], feature=[변수명]
현재데이터: (현재 마트로 검증 가능 / 추가 필요: [무엇이 필요한지])

[가설 2]
관찰: ...
H0: ...
H1: ...
검증방법: ...
필요변수: ...
현재데이터: ...

[가설 3]
관찰: ...
H0: ...
H1: ...
검증방법: ...
필요변수: ...
현재데이터: ...

작성 규칙:
- 클러스터 레이블(cluster_group, cluster_id 등)을 feature로 사용하지 마라 → 순환논리. 클러스터에서 발견한 패턴을 원래 measure 변수로 재표현하라.
- 특정 항목 이름(카테고리명, 상품명 등)을 H1 조건에 직접 넣지 마라 → 관찰이지 가설이 아님. measure 변수 패턴으로 일반화하라.
- 현재 마트에 없는 변수를 필요변수로 적으면서 검증 가능하다고 쓰지 마라.

[다음 분석 방향]
위 3개 가설은 단순 이변량(A→B) 관계를 다룬다. 교란변수(confounding variable)가 결과를 왜곡할 수 있으므로 아래 2가지를 작성하라.
1. 교란변수 후보: 위 가설들의 결과에 영향을 미칠 수 있는 변수를 현재 마트에서 골라라. 없으면 추가 확보가 필요한 변수를 명시하라.
2. 통제 방법: 교란변수를 통제하기 위한 다음 분석 방법 제안 (예: 다중회귀로 확장, 층화 분석, 그룹별 하위분석 등)

한국어로 작성하라.
"""
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    hypotheses, err1 = run_node_with_retry(
        lambda: llm.invoke(prompt).content.strip(), "hypothesis", fallback="가설 생성 실패"
    )

    # final_summary: 다음 에이전트 핸드오프용 압축 요약 (insight+hypotheses 중복 X)
    summary_prompt = f"""
아래 EDA 인사이트와 가설을 다음 분석 에이전트에게 전달할 핸드오프 요약으로 압축하라.
마크다운 기호 사용 금지. 일반 텍스트로만 작성하라.

[인사이트]
{state['insight_result']}

[가설]
{hypotheses}

작성 규칙:
- 4~6문장으로 압축
- 첫 문장: 데이터 구조의 핵심 특성 1가지 (수치 포함)
- 중간 문장: 현재 데이터로 바로 검증 가능한 가설을 우선 언급 (검증방법 포함)
- 마지막 문장: "다음 에이전트는 [검증방법]으로 [target]~[feature] 관계를 우선 검증하라"로 마무리
- 수치는 인사이트에서 확인된 것만 포함

한국어로 작성하라.
"""
    final_summary, err2 = run_node_with_retry(
        lambda: llm.invoke(summary_prompt).content.strip(), "final_summary", fallback="요약 생성 실패"
    )
    errors = state.get("error_log", [])
    for e in [err1, err2]:
        if e:
            errors = errors + [e]
    return {"hypotheses": hypotheses, "final_summary": final_summary, "error_log": errors}


def clustering_node(state: EDAState) -> dict:
    result, err = run_node_with_retry(
        lambda: run_clustering_skill(
            df=_df,
            measure_cols=_measure_cols,
            key_col=_key_col,
            question_type=_question_type,
        ),
        "clustering",
        fallback={"skip": True, "reason": "클러스터링 오류로 스킵"},
    )
    errors = state.get("error_log", [])
    if err:
        errors = errors + [err]
    return {"clustering_result": result, "error_log": errors}


def chart_selector_node(state: EDAState) -> dict:
    import glob
    import shutil
    from openai import RateLimitError
    from eda_agent.tools.visualize import OUTPUT_DIR, KEY_DIR

    all_charts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
    if not all_charts:
        return {"key_charts": []}

    analysis_results = {
        "inspect":      state.get("inspect_result", ""),
        "quality":      state.get("quality_result", ""),
        "distribution": state.get("distribution_result", ""),
        "comparison":   state.get("comparison_result", ""),
        "relationship": state.get("relationship_result", ""),
        "time":         state.get("time_result", ""),
    }

    stat = state.get("statistical_metadata", {})

    def _run(ar, st):
        return run_chart_selector_skill(
            chart_paths=all_charts,
            user_question=state["user_question"],
            analysis_results=ar,
            question_type=state.get("question_type", ""),
            statistical_metadata=st,
            priority_metrics=state.get("analysis_plan", {}).get("priority_metrics", []),
        )

    # 단일 요청이 TPM 한도 초과 시 자동으로 입력 크기 줄여 재시도
    try:
        key_charts = _run(analysis_results, stat)
    except RateLimitError:
        truncated = {k: (v[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
                     for k, v in analysis_results.items()}
        clustering = stat.get("clustering", {})
        slim_stat = {"clustering": {k: v for k, v in clustering.items() if k != "cluster_labels"}}
        key_charts = _run(truncated, slim_stat)

    # key/ 폴더 초기화 후 선별 차트 복사
    for f in glob.glob(os.path.join(KEY_DIR, "*.png")):
        os.remove(f)
    for src in key_charts:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(KEY_DIR, os.path.basename(src)))

    return {"key_charts": key_charts}


# ─────────────────────────────
# Routing
# ─────────────────────────────
def route_after_planner(state: EDAState):
    """planner 결정에 따라 첫 분석 노드 결정 (quality 또는 distribution)"""
    plan = state.get("analysis_plan", {})
    if plan.get("run_quality", True):
        return "quality"
    return "distribution"


def route_after_quality(state: EDAState):
    plan = state.get("analysis_plan", {})
    if plan.get("run_distribution", True):
        return "distribution"
    if plan.get("run_comparison", True):
        return "comparison"
    return "relationship"


def route_after_distribution(state: EDAState):
    plan = state.get("analysis_plan", {})
    if plan.get("run_comparison", True):
        return "comparison"
    return "relationship"


def route_after_comparison(state: EDAState):
    plan = state.get("analysis_plan", {})
    if plan.get("run_relationship", True):
        return "relationship"
    return "time_or_insight"


def route_time(state: EDAState):
    return "time" if state["has_time_column"] else "clustering"


# ─────────────────────────────
# Graph
# ─────────────────────────────
def build_app():
    graph = StateGraph(EDAState)

    graph.add_node("load_mart",    load_mart_node)
    graph.add_node("inspect",      inspect_node)
    graph.add_node("planner",      planner_node)
    graph.add_node("quality",      quality_node)
    graph.add_node("distribution", distribution_node)
    graph.add_node("comparison",   comparison_node)
    graph.add_node("relationship", relationship_node)
    graph.add_node("time",         time_node)
    graph.add_node("clustering",      clustering_node)
    graph.add_node("insight",         insight_node)
    graph.add_node("hypothesis",      hypothesis_node)
    graph.add_node("chart_selector",  chart_selector_node)

    graph.add_edge(START,       "load_mart")
    graph.add_edge("load_mart", "inspect")
    graph.add_edge("inspect",   "planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"quality": "quality", "distribution": "distribution"},
    )

    graph.add_conditional_edges(
        "quality",
        route_after_quality,
        {"distribution": "distribution", "comparison": "comparison", "relationship": "relationship"},
    )

    graph.add_conditional_edges(
        "distribution",
        route_after_distribution,
        {"comparison": "comparison", "relationship": "relationship"},
    )

    graph.add_conditional_edges(
        "comparison",
        route_after_comparison,
        {"relationship": "relationship", "time_or_insight": "insight"},
    )

    graph.add_conditional_edges(
        "relationship",
        route_time,
        {"time": "time", "clustering": "clustering"},
    )

    graph.add_edge("time",           "clustering")
    graph.add_edge("clustering",     "insight")
    graph.add_edge("insight",        "hypothesis")
    graph.add_edge("hypothesis",     "chart_selector")
    graph.add_edge("chart_selector", END)

    return graph.compile()
