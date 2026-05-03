import io
import json
import os
import re
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").strip().strip('"').strip("'").lower()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "analysis-agent-0416")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
MAX_SAMPLE_ROWS = int(os.getenv("ANALYSIS_AGENT_SAMPLE_ROWS", 5))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT

from langgraph.graph import END, START, StateGraph


class AnalysisPlan(BaseModel):
    original_question: str = Field(description="?ъ슜???먮Ц 吏덈Ц")
    analysis_goal: str = Field(description="遺꾩꽍 紐⑹쟻")
    analysis_type: Literal["hypothesis_test", "predictive_modeling", "correlation_regression", "eda", "segmentation"]
    techniques: List[str] = Field(default_factory=list, description="沅뚯옣 湲곕쾿 紐⑸줉")
    target_variable: Optional[str] = Field(default=None, description="target variable")
    group_variable: Optional[str] = Field(default=None, description="吏묐떒 鍮꾧탳 湲곗?")
    feature_candidates: List[str] = Field(default_factory=list, description="feature candidates")
    time_variable: Optional[str] = Field(default=None, description="?쒓퀎??湲곗? 而щ읆")
    hypothesis: Optional[str] = Field(default=None, description="hypothesis")
    success_criteria: str = Field(description="遺꾩꽍 ?깃났 ?먮떒 湲곗?")
    notes: Optional[str] = Field(default=None, description="?좊ℓ???먮뒗 二쇱쓽?ы빆")


class PythonDraft(BaseModel):
    code: str
    libraries_used: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    reasoning: str


class ValidationResult(BaseModel):
    result: Literal["valid", "invalid"]
    reason: str
    feedback: str


class AgentState(TypedDict):
    user_question: str
    csv_paths: List[str]
    dataset_context: str
    analysis_plan: Dict[str, Any]
    python_draft: Dict[str, Any]
    execution_result: Dict[str, Any]
    validation: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    final_answer: str


llm = None


def get_llm():
    global llm
    if llm is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Running the analysis agent requires langchain-google-genai. "
                "Install it with: pip install langchain-google-genai or uv add langchain-google-genai"
            ) from exc

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=GOOGLE_API_KEY,
        )
    return llm


def safe_json_parse(text_value: str, fallback: dict) -> dict:
    cleaned = text_value.strip()
    cleaned = cleaned.replace("```json", "").replace("```python", "").replace("```", "").strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return fallback


def clean_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()


def make_json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    if pd.isna(value):
        return None
    return value


def infer_analysis_type_from_question(question: str) -> Optional[str]:
    lowered = question.lower()
    predictive_keywords = ["예측", "forecast", "predict", "다음 달", "미래", "향후"]
    hypothesis_keywords = ["유의", "가설", "검정", "차이", "통계적", "t-test", "anova", "카이제곱"]
    relation_keywords = ["상관", "회귀", "영향", "관계", "correlation", "regression"]

    if any(keyword in lowered for keyword in predictive_keywords):
        return "predictive_modeling"
    if any(keyword in lowered for keyword in hypothesis_keywords):
        return "hypothesis_test"
    if any(keyword in lowered for keyword in relation_keywords):
        return "correlation_regression"
    return None


def refine_analysis_plan(plan: Dict[str, Any], question: str) -> Dict[str, Any]:
    refined = dict(plan)
    inferred_type = infer_analysis_type_from_question(question)

    if inferred_type and refined.get("analysis_type") != inferred_type:
        refined["analysis_type"] = inferred_type

    if refined.get("analysis_type") == "predictive_modeling":
        refined["analysis_goal"] = refined.get("analysis_goal") or "怨쇨굅 ?곗씠?곕? 諛뷀깢?쇰줈 誘몃옒 媛믪쓣 ?덉륫"
        techniques = set(refined.get("techniques", []))
        techniques.update(["time_series_forecasting", "regression"])
        refined["techniques"] = sorted(techniques)
        refined["success_criteria"] = refined.get("success_criteria") or "?ㅼ쓬 ?쒖젏 ?덉륫媛믨낵 紐⑤뜽 ?깅뒫 吏?쒕? ?쒖떆"

    if refined.get("analysis_type") == "hypothesis_test":
        techniques = set(refined.get("techniques", []))
        techniques.add("hypothesis_testing")
        refined["techniques"] = sorted(techniques)

    return refined


def summarize_dataframe(path: str, sample_rows: int = MAX_SAMPLE_ROWS) -> str:
    df = pd.read_csv(path)
    sample = make_json_safe(df.head(sample_rows).to_dict(orient="records"))
    null_counts = df.isna().sum()
    dtype_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    summary = {
        "path": path,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": dtype_map,
        "null_counts_top": make_json_safe(dict(null_counts.sort_values(ascending=False).head(10))),
        "numeric_columns": numeric_cols[:30],
        "categorical_columns": categorical_cols[:30],
        "sample_rows": sample,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def build_dataset_context(csv_paths: List[str]) -> str:
    if not csv_paths:
        raise ValueError("理쒖냼 1媛??댁긽??CSV 寃쎈줈媛 ?꾩슂?⑸땲??")

    contexts = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV ?뚯씪??李얠쓣 ???놁뒿?덈떎: {path}")
        contexts.append(summarize_dataframe(path))
    return "\n\n".join(contexts)


def load_context(state: AgentState):
    try:
        return {"dataset_context": build_dataset_context(state["csv_paths"]), "error": ""}
    except Exception as exc:
        return {"dataset_context": "", "error": str(exc)}


def plan_analysis(state: AgentState):
    if state.get("error"):
        fallback = AnalysisPlan(
            original_question=state["user_question"],
            analysis_goal="?곗씠??濡쒕뵫 ?ㅽ뙣濡?遺꾩꽍 怨꾪쉷 ?섎┰ 遺덇?",
            analysis_type="eda",
            techniques=["descriptive_statistics"],
            target_variable=None,
            group_variable=None,
            feature_candidates=[],
            time_variable=None,
            hypothesis=None,
            success_criteria="CSV 寃쎈줈? 而щ읆 ?뺣낫 ?뺤씤",
            notes=state["error"],
        ).model_dump()
        return {"analysis_plan": fallback}

    prompt = f"""
?덈뒗 ?곗씠??遺꾩꽍 由щ뱶??
?ъ슜??吏덈Ц???쎄퀬 媛???곸젅??遺꾩꽍 ?꾨왂??怨좊Ⅸ??

?ъ슜??吏덈Ц:
{state['user_question']}

?곗씠?곗뀑 ?붿빟:
{state['dataset_context']}

洹쒖튃:
- analysis_type? hypothesis_test / predictive_modeling / correlation_regression / eda / segmentation 以??섎굹
- 誘몃옒 ?덉륫 ?붿껌?대㈃ predictive_modeling ?곗꽑
- 吏묐떒 李⑥씠???듦퀎???좎쓽???붿껌?대㈃ hypothesis_test ?곗꽑
- 蹂??愿怨??ㅻ챸?대㈃ correlation_regression ?곗꽑
- target_variable, group_variable, time_variable? ?곗씠??臾몃㎘??留욎쓣 ?뚮쭔 梨꾩슫??- notes?먮뒗 ?좊ℓ?? 異붽? ?꾩쿂由??꾩슂?ы빆, ?곗씠???쒓퀎瑜??곷뒗??- 諛섎뱶??JSON留?異쒕젰

異쒕젰 ?뺤떇:
{{
  "original_question": "...",
  "analysis_goal": "...",
  "analysis_type": "...",
  "techniques": ["..."],
  "target_variable": "... ?먮뒗 null",
  "group_variable": "... ?먮뒗 null",
  "feature_candidates": ["..."],
  "time_variable": "... ?먮뒗 null",
  "hypothesis": "... ?먮뒗 null",
  "success_criteria": "...",
  "notes": "... ?먮뒗 null"
}}
"""
    response = get_llm().invoke(prompt).content

    fallback = AnalysisPlan(
        original_question=state["user_question"],
        analysis_goal="吏덈Ц 遺꾨쪟 ?ㅽ뙣",
        analysis_type="eda",
        techniques=["descriptive_statistics"],
        target_variable=None,
        group_variable=None,
        feature_candidates=[],
        time_variable=None,
        hypothesis=None,
        success_criteria="질문과 데이터 컬럼의 연결이 명확함",
        notes="怨꾪쉷 ?뚯떛 ?ㅽ뙣",
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["original_question"] = state["user_question"]
    parsed = refine_analysis_plan(parsed, state["user_question"])
    return {"analysis_plan": parsed}


def generate_python_code(state: AgentState):
    feedback = state.get("feedback", "").strip() or "?놁쓬"
    prompt = f"""
?덈뒗 Python ?곗씠??遺꾩꽍媛??
Pandas, Scipy, Scikit-learn???쒖슜???ㅽ뻾 媛?ν븳 遺꾩꽍 肄붾뱶瑜??묒꽦?섎씪.

?ъ슜??吏덈Ц:
{state['user_question']}

遺꾩꽍 怨꾪쉷:
{json.dumps(state['analysis_plan'], ensure_ascii=False, indent=2)}

?곗씠?곗뀑 ?붿빟:
{state['dataset_context']}

CSV 寃쎈줈:
{json.dumps(state['csv_paths'], ensure_ascii=False)}

?댁쟾 ?쇰뱶諛?
{feedback}

諛섎뱶??吏??洹쒖튃:
- 肄붾뱶留??앹꽦?섏? 留먭퀬 JSON留?異쒕젰
- code ?꾨뱶?먮뒗 諛붾줈 exec 媛?ν븳 Python 肄붾뱶留??ｋ뒗??- csv_paths 由ъ뒪?몃? ?ъ슜???곗씠?곕? ?쎈뒗??- 泥?踰덉㎏ CSV??dfs[0] ?먮뒗 main_df濡??ъ슜?대룄 ?쒕떎
- 寃곌낵??諛섎뱶??ANALYSIS_RESULT ?뺤뀛?덈━????ν븳??- ANALYSIS_RESULT?먮뒗 summary, method, key_metrics, insights ?ㅻ? ?ы븿?쒕떎
- ?꾩슂?섎㈃ stats, model_performance, assumptions ?ㅻ? 異붽??쒕떎
- print???덉슜?섏?留?理쒖쥌 寃곌낵??ANALYSIS_RESULT 湲곗??쇰줈 ?먮떒?쒕떎
- ?뚯씪 ???湲덉?, ?ㅽ듃?뚰겕 ?몄텧 湲덉?, ?ъ슜???낅젰 湲덉?
- hypothesis_test硫?p-value? 寃?뺣챸 ?ы븿
- predictive_modeling?대㈃ train/test 遺꾨━ ???깅뒫吏???ы븿
- correlation_regression?대㈃ ?곴?怨꾩닔 ?먮뒗 ?뚭?怨꾩닔 ?ы븿
- 而щ읆紐낆씠 遺덊솗?ㅽ븯硫?肄붾뱶?먯꽌 議댁옱 ?щ?瑜??먭??섍퀬 移쒖젅???ㅻ쪟瑜?raise ?쒕떎
- 諛섎뱶??JSON留?異쒕젰

異쒕젰 ?뺤떇:
{{
  "code": "...",
  "libraries_used": ["pandas", "..."],
  "expected_outputs": ["..."],
  "reasoning": "..."
}}
"""
    try:
        draft = llm.with_structured_output(PythonDraft).invoke(prompt)
        parsed = draft.model_dump() if isinstance(draft, BaseModel) else dict(draft)
        parsed["code"] = clean_code(parsed.get("code", ""))
        return {"python_draft": parsed}
    except Exception:
        response = get_llm().invoke(
            prompt
            + "\n\nReturn only one valid JSON object. Escape all newline characters inside the code string."
        ).content

    fallback = PythonDraft(
        code=(
            "dfs = [pd.read_csv(path) for path in csv_paths]\n"
            "main_df = dfs[0]\n"
            "ANALYSIS_RESULT = {\n"
            "    'summary': '遺꾩꽍 肄붾뱶 ?앹꽦 ?ㅽ뙣濡?湲곕낯 ?붿빟留?諛섑솚?⑸땲??',\n"
            "    'method': 'fallback_summary',\n"
            "    'key_metrics': {'rows': int(main_df.shape[0]), 'columns': int(main_df.shape[1])},\n"
            "    'insights': ['吏덈Ц??留욌뒗 遺꾩꽍 肄붾뱶 ?앹꽦???ㅽ뙣?덉뒿?덈떎. 而щ읆紐낆쓣 ?ㅼ떆 ?뺤씤??二쇱꽭??']\n"
            "}\n"
        ),
        libraries_used=["pandas"],
        expected_outputs=["ANALYSIS_RESULT"],
        reasoning="肄붾뱶 ?앹꽦 ?뚯떛 ?ㅽ뙣",
    ).model_dump()

    parsed = safe_json_parse(response, fallback)
    parsed["code"] = clean_code(parsed.get("code", fallback["code"]))
    return {"python_draft": parsed}


def execute_analysis(state: AgentState):
    dfs = [pd.read_csv(path) for path in state["csv_paths"]]
    stdout_buffer = io.StringIO()

    safe_globals = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "stats": stats,
        "train_test_split": train_test_split,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "accuracy_score": accuracy_score,
        "Pipeline": Pipeline,
        "ColumnTransformer": ColumnTransformer,
        "SimpleImputer": SimpleImputer,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "csv_paths": state["csv_paths"],
        "dfs": dfs,
    }
    safe_locals: Dict[str, Any] = {}

    try:
        with redirect_stdout(stdout_buffer):
            exec(state["python_draft"]["code"], safe_globals, safe_locals)

        analysis_result = safe_locals.get("ANALYSIS_RESULT", safe_globals.get("ANALYSIS_RESULT"))
        if not isinstance(analysis_result, dict):
            raise ValueError("ANALYSIS_RESULT ?뺤뀛?덈━媛 ?앹꽦?섏? ?딆븯?듬땲??")
        analysis_result = make_json_safe(analysis_result)

        return {
            "execution_result": {
                "analysis_result": analysis_result,
                "stdout": stdout_buffer.getvalue().strip(),
            },
            "error": "",
        }
    except Exception as exc:
        return {
            "execution_result": {
                "analysis_result": {},
                "stdout": stdout_buffer.getvalue().strip(),
                "traceback": traceback.format_exc(),
            },
            "error": str(exc),
        }


def validate_result(state: AgentState):
    if state.get("error"):
        parsed = ValidationResult(
            result="invalid",
            reason=f"遺꾩꽍 ?ㅽ뻾 ?ㅻ쪟: {state['error']}",
            feedback="?꾨씫 而щ읆 ?뺤씤, 而щ읆紐?議댁옱 寃利? 遺꾩꽍 ?좏삎??留욌뒗 寃곌낵 ??summary/method/key_metrics/insights) ?앹꽦???꾩슂?섎떎.",
        ).model_dump()
        return {"validation": parsed, "feedback": parsed["feedback"]}

    analysis_type = state["analysis_plan"].get("analysis_type")
    analysis_result = state["execution_result"].get("analysis_result", {})
    key_metrics = analysis_result.get("key_metrics", {})

    if analysis_type == "predictive_modeling":
        predictive_metric_keys = {"mean_absolute_error", "root_mean_squared_error", "r_squared", "accuracy"}
        has_predictive_metrics = any(key in key_metrics for key in predictive_metric_keys)
        if has_predictive_metrics:
            parsed = ValidationResult(
                result="valid",
                reason="?덉륫 遺꾩꽍 寃곌낵? 紐⑤뜽 ?깅뒫 吏?쒓? ?ы븿?섏뼱 ?덉뒿?덈떎.",
                feedback="",
            ).model_dump()
            return {"validation": parsed, "feedback": ""}

    if analysis_type == "hypothesis_test":
        stats_block = analysis_result.get("stats", {})
        if isinstance(stats_block, dict) and ("p_value" in stats_block or "p-value" in stats_block):
            parsed = ValidationResult(
                result="valid",
                reason="媛?ㅺ???寃곌낵??p-value媛 ?ы븿?섏뼱 ?덉뒿?덈떎.",
                feedback="",
            ).model_dump()
            return {"validation": parsed, "feedback": ""}

    if analysis_type == "correlation_regression":
        if any(key in key_metrics for key in {"correlation", "correlation_coefficient", "regression_coefficient", "r_squared"}):
            parsed = ValidationResult(
                result="valid",
                reason="愿怨?遺꾩꽍???꾩슂???듭떖 ?섏튂媛 ?ы븿?섏뼱 ?덉뒿?덈떎.",
                feedback="",
            ).model_dump()
            return {"validation": parsed, "feedback": ""}

    prompt = f"""
?덈뒗 ?곗씠??遺꾩꽍 寃곌낵 寃?좎옄??

?ъ슜??吏덈Ц:
{state['user_question']}

遺꾩꽍 怨꾪쉷:
{json.dumps(state['analysis_plan'], ensure_ascii=False, indent=2)}

?앹꽦 肄붾뱶:
{state['python_draft']['code']}

?ㅽ뻾 寃곌낵:
{json.dumps(state['execution_result'].get('analysis_result', {}), ensure_ascii=False, indent=2)}

stdout:
{state['execution_result'].get('stdout', '')}

寃利?洹쒖튃:
- 吏덈Ц ?섎룄? analysis_type??留욌뒗吏 ?뺤씤
- hypothesis_test硫??듦퀎 寃??寃곌낵媛 ?ㅼ젣 ?ы븿?섏뼱????- predictive_modeling?대㈃ 紐⑤뜽 ?깅뒫 吏?쒓? ?ㅼ젣 ?ы븿?섏뼱????- correlation_regression?대㈃ 愿怨??댁꽍???꾩슂???섏튂媛 ?ы븿?섏뼱????- 寃곌낵媛 吏덈Ц怨?臾닿??섍굅??異붿륫?대㈃ invalid
- 諛섎뱶??JSON留?異쒕젰

異쒕젰 ?뺤떇:
{{
  "result": "valid" ?먮뒗 "invalid",
  "reason": "...",
  "feedback": "..."
}}
"""
    response = get_llm().invoke(prompt).content

    fallback = ValidationResult(
        result="invalid",
        reason="寃利?寃곌낵 ?뚯떛 ?ㅽ뙣",
        feedback="吏덈Ц ?섎룄??留욌뒗 遺꾩꽍 湲곕쾿怨??듭떖 ?섏튂瑜???紐낇솗???ы븿???ㅼ떆 ?앹꽦?섎씪.",
    ).model_dump()
    parsed = safe_json_parse(response, fallback)
    return {"validation": parsed, "feedback": parsed.get("feedback", "")}


def finalize_answer(state: AgentState):
    if state["validation"].get("result") != "valid":
        return {
            "final_answer": (
                "遺꾩꽍 寃利앹뿉 ?ㅽ뙣?덉뒿?덈떎.\n"
                f"?ъ쑀: {state['validation'].get('reason')}\n"
                f"留덉?留??ㅻ쪟: {state.get('error', '')}\n"
                "吏덈Ц??留욌뒗 而щ읆紐낃낵 CSV瑜??ㅼ떆 ?뺤씤??二쇱꽭??"
            )
        }

    prompt = f"""
?덈뒗 ?곗씠??遺꾩꽍 寃곌낵瑜??쒓뎅?대줈 ?뺣━?섎뒗 遺꾩꽍媛??

?ъ슜??吏덈Ц:
{state['user_question']}

遺꾩꽍 怨꾪쉷:
{json.dumps(state['analysis_plan'], ensure_ascii=False, indent=2)}

?ㅽ뻾 寃곌낵:
{json.dumps(state['execution_result'].get('analysis_result', {}), ensure_ascii=False, indent=2)}

洹쒖튃:
- ?쒓뎅??- ?듭떖 寃곕줎 癒쇱?
- ?ъ슜??遺꾩꽍 湲곕쾿??1臾몄옣?쇰줈 紐낆떆
- 以묒슂???섏튂媛 ?덉쑝硫??④퍡 ?멸툒
- ?곗씠???쒓퀎???댁꽍 二쇱쓽?먯씠 ?덉쑝硫?吏㏐쾶 ?㏓텤??"""
    answer = get_llm().invoke(prompt).content.strip()
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
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("generate_python_code", generate_python_code)
    graph.add_node("execute_analysis", execute_analysis)
    graph.add_node("validate_result", validate_result)
    graph.add_node("increase_retry", increase_retry)
    graph.add_node("finalize_answer", finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan_analysis")
    graph.add_edge("plan_analysis", "generate_python_code")
    graph.add_edge("generate_python_code", "execute_analysis")
    graph.add_edge("execute_analysis", "validate_result")

    graph.add_conditional_edges(
        "validate_result",
        route_after_validation,
        {
            "retry": "increase_retry",
            "finalize": "finalize_answer",
        },
    )

    graph.add_edge("increase_retry", "generate_python_code")
    graph.add_edge("finalize_answer", END)

    return graph.compile()

