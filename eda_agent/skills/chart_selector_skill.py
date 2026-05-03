import os
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

TOTAL_MAX = 8


def _load_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _call_llm_remove(
    filenames: list,
    user_question: str,
    question_type: str,
    analysis_results: dict,
    statistical_metadata: dict,
    extra_instruction: str = "",
    priority_metrics: list = None,
) -> dict:
    """LLM에게 제거 대상 차트 목록을 반환받는다."""
    priority_info = ""
    if priority_metrics:
        names = ", ".join(m.get("metric", "") for m in priority_metrics if m.get("metric"))
        priority_info = f"\n[우선 지표]\n{names}\n"

    # 클러스터링 품질 평가 — 실루엣 점수 기반 판단 지침 생성
    clustering = statistical_metadata.get("clustering", {})
    cluster_chart_rule = ""
    if clustering.get("skip"):
        cluster_chart_rule = (
            "\n클러스터링 차트 처리 기준:\n"
            "clustering이 실행되지 않았다. cluster_ 관련 차트는 제거하라.\n"
        )
    else:
        sil = clustering.get("silhouette_score", 0)
        n_k = clustering.get("n_clusters", 0)
        if sil >= 0.5:
            quality = f"실루엣 점수 {sil} (0.5 이상 — 클러스터 구분이 뚜렷함)"
            guidance = "cluster_profile, cluster_scatter 차트는 다차원 그룹 구조를 명확히 보여주므로 유지하라."
        elif sil >= 0.25:
            quality = f"실루엣 점수 {sil} (0.25~0.5 — 클러스터 구분이 보통)"
            guidance = "cluster_profile 차트는 유지하되, cluster_scatter는 다른 차트와 중복 여부를 판단해 결정하라."
        else:
            quality = f"실루엣 점수 {sil} (0.25 미만 — 클러스터 구분이 약함)"
            guidance = "cluster_ 차트의 해석 가치가 낮으므로 다른 차트보다 낮은 우선순위로 처리하라."
        cluster_chart_rule = (
            f"\n클러스터링 차트 처리 기준:\n"
            f"clustering 결과: n_clusters={n_k}, {quality}\n"
            f"{guidance}\n"
        )

    prompt = f"""
너는 데이터 분석 보고서용 차트 편집자다.

목표는 좋은 차트를 찾는 것이 아니라,
전체 후보 중 불필요한 차트만 제거하는 것이다.

[사용자 질문]
{user_question}

[question_type]
{question_type}
{priority_info}
[분석 결과 요약]
{json.dumps(analysis_results, ensure_ascii=False, indent=2)}

[통계 메타데이터]
{json.dumps(statistical_metadata, ensure_ascii=False, indent=2)}

[전체 차트 후보]
{json.dumps(filenames, ensure_ascii=False)}

{extra_instruction}

다음 기준으로 제거하라.
1. 사용자 질문과 직접 관련 없는 차트
2. 같은 정보를 반복하는 차트 (예: 동일 지표의 dist/box/violin 중 가장 정보량 적은 것)
3. 해석 가치가 낮은 차트
4. 통계적으로 의미가 약한 차트 (상관관계가 낮은 scatter 등) — 단, 사용자 질문의 핵심 변수 간 scatter는 상관관계가 낮더라도 제거하지 마라. 낮은 상관관계 자체가 분석 결론이기 때문이다.
5. 보고서에서 설명하기 어려운 차트

단, 다음 차트는 가능하면 유지하라.
- 여러 지표를 한 번에 요약하는 차트
- radar, bubble, grouped_bar처럼 발표용 임팩트가 큰 차트
- bubble 차트는 핵심 지표 2개 이상을 동시에 보여주므로, 질문의 핵심 지표를 포함하면 유지하라
- 사용자가 질문에서 언급한 핵심 지표를 직접 보여주는 차트 (우선 지표가 있으면 해당 지표 포함 차트 우선 유지)
- 사용자 질문 맥락상 "낮을수록 좋은 지표"(예: 배송일, 불량률 등)는 bar_bottom이 더 인사이트 있을 수 있으므로 신중하게 판단하라
{cluster_chart_rule}

추가로 다음 중복은 반드시 하나만 남겨라.
- heatmap_matrix와 correlation_heatmap이 동시에 존재하면 둘 중 하나만 유지하라

반드시 아래 JSON만 출력하라.
{{
  "remove": ["파일명1.png", "파일명2.png"],
  "reason": {{
    "파일명1.png": "제거 이유"
  }}
}}
"""
    llm = _load_llm()
    response = llm.invoke(prompt).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except Exception:
        return {"remove": [], "reason": {}}


def run_chart_selector_skill(
    chart_paths: list,
    user_question: str,
    analysis_results: dict,
    question_type: str = "",
    statistical_metadata: dict = None,
    priority_metrics: list = None,
) -> list:
    """
    1단계: 기계적 품질 필터 (파일 존재 여부, 중복 경로 제거)
    2단계: LLM이 불필요한 차트 제거
    3단계: 8개 초과 시 LLM이 추가 제거
    """
    if not chart_paths:
        return []

    stat = statistical_metadata or {}

    # ── 1단계: 기계적 품질 필터 ──
    seen_paths = set()
    valid_paths = []
    for p in chart_paths:
        abs_p = os.path.abspath(p)
        if abs_p in seen_paths:
            continue
        if not os.path.exists(p):
            continue
        seen_paths.add(abs_p)
        valid_paths.append(p)

    if not valid_paths:
        return []

    name_to_path = {os.path.basename(p): p for p in valid_paths}
    filenames = list(name_to_path.keys())

    # ── 2단계: LLM이 불필요 차트 제거 ──
    result = _call_llm_remove(
        filenames=filenames,
        user_question=user_question,
        question_type=question_type,
        analysis_results=analysis_results,
        statistical_metadata=stat,
        priority_metrics=priority_metrics,
    )

    to_remove = set(result.get("remove", []))
    filtered = [p for p in valid_paths if os.path.basename(p) not in to_remove]

    # ── 3단계: 8개 초과 시 LLM이 추가 제거 ──
    if len(filtered) > TOTAL_MAX:
        excess = len(filtered) - TOTAL_MAX
        filtered_names = [os.path.basename(p) for p in filtered]

        result2 = _call_llm_remove(
            filenames=filtered_names,
            user_question=user_question,
            question_type=question_type,
            analysis_results=analysis_results,
            statistical_metadata=stat,
            extra_instruction=f"현재 차트가 {len(filtered)}개로 {TOTAL_MAX}개를 초과한다. "
                              f"가장 중복되거나 임팩트가 낮은 {excess}개를 추가로 제거하라.",
            priority_metrics=priority_metrics,
        )

        to_remove2 = set(result2.get("remove", []))
        filtered = [p for p in filtered if os.path.basename(p) not in to_remove2]

    return filtered[:TOTAL_MAX]
