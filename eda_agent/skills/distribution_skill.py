import pandas as pd
from eda_agent.tools.visualize import (
    plot_distributions,
    plot_boxplots,
    plot_violins,
    plot_category_distribution,
)


def run_distribution_skill(
    df: pd.DataFrame,
    measure_cols: list = None,
    question_type: str = "",
    priority_metrics: list = None,
) -> dict:
    """
    단변량 분포 분석 skill.
    question_type에 따라 생성 차트를 조정한다.

    - distribution : hist + box + violin + catdist (전체)
    - comparison   : box만 (그룹 간 분포 파악용)
    - relationship : priority_metrics 컬럼만 hist + violin
    - time         : hist만
    """
    qt = question_type.lower()
    result = {}

    if qt == "comparison":
        # 그룹 비교 맥락 — 분포 형태보다 이상치/범위 파악이 우선
        result["boxplots"] = plot_boxplots(df, measure_cols=measure_cols)

    elif qt == "relationship":
        # 관계 분석 맥락 — 전체 컬럼 분포 형태 파악 (편향, 이상치 범위) 필요
        result["distributions"] = plot_distributions(df, measure_cols=measure_cols)
        result["violins"]       = plot_violins(df, measure_cols=measure_cols)

    elif qt == "time":
        result["distributions"] = plot_distributions(df, measure_cols=measure_cols)

    else:  # distribution 또는 기본값
        result["distributions"]         = plot_distributions(df, measure_cols=measure_cols)
        result["boxplots"]              = plot_boxplots(df, measure_cols=measure_cols)
        result["violins"]               = plot_violins(df, measure_cols=measure_cols)
        result["category_distribution"] = plot_category_distribution(df)

    return result
