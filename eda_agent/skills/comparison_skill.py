import pandas as pd
from eda_agent.tools.visualize import (
    plot_top_n_barplot,
    plot_heatmap_matrix,
    plot_bubble,
    plot_radar,
    plot_grouped_bar,
)


def run_comparison_skill(
    df: pd.DataFrame,
    key_col: str = None,
    measure_cols: list = None,
    question_type: str = "",
) -> dict:
    """
    그룹 간 비교 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - comparison   : 전체 (bar + heatmap + bubble + radar + grouped_bar)
    - relationship : bubble만 (주요 지표 간 포지셔닝 확인용)
    - time         : bar만 (시간대별 그룹 비교)
    - distribution : 생략 (비교 관점 불필요)
    """
    qt = question_type.lower()
    result = {}

    if qt == "distribution":
        return result  # 생략

    elif qt == "relationship":
        result["bubble"] = plot_bubble(df, key_col=key_col, measure_cols=measure_cols)

    elif qt == "time":
        result["top_n_barplot"] = plot_top_n_barplot(df, key_col=key_col, measure_cols=measure_cols)

    else:  # comparison 또는 기본값
        result["top_n_barplot"]  = plot_top_n_barplot(df, key_col=key_col, measure_cols=measure_cols)
        result["heatmap_matrix"] = plot_heatmap_matrix(df, key_col=key_col, measure_cols=measure_cols)
        result["bubble"]         = plot_bubble(df, key_col=key_col, measure_cols=measure_cols)
        result["radar"]          = plot_radar(df, key_col=key_col, measure_cols=measure_cols)
        result["grouped_bar"]    = plot_grouped_bar(df, key_col=key_col, measure_cols=measure_cols)

    return result
