import pandas as pd
from eda_agent.tools.visualize import (
    plot_correlation,
    plot_scatter_pairs,
)


def run_relationship_skill(
    df: pd.DataFrame,
    measure_cols: list = None,
    question_type: str = "",
) -> dict:
    """
    변수 간 관계 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - relationship : correlation + scatter_pairs (전체)
    - comparison   : correlation만 (지표 간 관계 파악용)
    - distribution : correlation만
    - time         : 생략
    """
    qt = question_type.lower()
    result = {}

    if qt == "time":
        return result  # 생략

    elif qt in ("comparison", "distribution"):
        result["correlation"] = plot_correlation(df, measure_cols=measure_cols)

    else:  # relationship 또는 기본값
        result["correlation"]   = plot_correlation(df, measure_cols=measure_cols)
        result["scatter_pairs"] = plot_scatter_pairs(df, measure_cols=measure_cols)

    return result
