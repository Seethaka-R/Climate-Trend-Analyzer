"""
trend_analysis.py
Computes long-term climate trends using:
- Annual mean aggregation
- Linear regression (slope = trend rate)
- Mann-Kendall monotonic trend test
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pymannkendall as mk


def compute_annual_means(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to annual mean temperature and total rainfall."""
    annual = df.groupby("year").agg(
        mean_temp=("temperature", "mean"),
        total_rain=("rainfall", "sum"),
        mean_humidity=("humidity", "mean"),
    ).reset_index()
    return annual


def fit_temperature_trend(annual: pd.DataFrame) -> dict:
    """
    Fit a linear regression: mean_temp ~ year
    Returns slope (°C/year), intercept, R², and predicted values.
    """
    X = annual[["year"]].values
    y = annual["mean_temp"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    y_pred = model.predict(X)
    
    print(f"[Trend] Temperature trend: {slope:.4f} °C/year | R²={r2:.3f}")
    
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "y_pred": y_pred,
        "model": model,
    }


def run_mann_kendall(annual: pd.DataFrame, column: str = "mean_temp") -> dict:
    """
    Mann-Kendall non-parametric trend test.
    Null hypothesis: no monotonic trend.
    """
    result = mk.original_test(annual[column].values)
    print(f"[Mann-Kendall] Trend: {result.trend} | p-value: {result.p:.4f} | tau: {result.Tau:.4f}")
    return {
        "trend": result.trend,
        "p_value": result.p,
        "tau": result.Tau,
        "slope_sen": result.slope,   # Sen's slope
    }