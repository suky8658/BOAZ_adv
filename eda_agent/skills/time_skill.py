import pandas as pd
from eda_agent.tools.visualize import (
    plot_timeseries,
    plot_seasonality,
)


def run_time_skill(df: pd.DataFrame, measure_cols: list = None, time_cols: list = None) -> dict:
    """
    시계열 / 시즌성 분석 skill.
    구성 tool:
        - plot_timeseries  : datetime 기준 추세 + 변화율
        - plot_seasonality : 월/요일 시즌성 bar chart
    """
    return {
        "timeseries":  plot_timeseries(df, measure_cols=measure_cols, time_cols=time_cols),
        "seasonality": plot_seasonality(df, measure_cols=measure_cols, time_cols=time_cols),
    }
