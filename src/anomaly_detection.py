"""
anomaly_detection.py
Detects climate anomalies using:
- Z-score method (statistical)
- IQR method (robust statistical)
- Isolation Forest (machine learning)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_zscore_anomalies(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag values more than `threshold` standard deviations from the mean.
    """
    mean = df[column].mean()
    std = df[column].std()
    df = df.copy()
    df[f"{column}_zscore"] = (df[column] - mean) / std
    df[f"{column}_anomaly_zscore"] = df[f"{column}_zscore"].abs() > threshold
    n_anomalies = df[f"{column}_anomaly_zscore"].sum()
    print(f"[Z-Score] {column}: {n_anomalies} anomalies detected")
    return df


def detect_iqr_anomalies(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    IQR (interquartile range) method for robust anomaly detection.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df = df.copy()
    df[f"{column}_anomaly_iqr"] = ~df[column].between(lower, upper)
    n_anomalies = df[f"{column}_anomaly_iqr"].sum()
    print(f"[IQR] {column}: {n_anomalies} anomalies (bounds: [{lower:.2f}, {upper:.2f}])")
    return df


def detect_isolation_forest(df: pd.DataFrame, features: list, contamination: float = 0.01) -> pd.DataFrame:
    """
    Isolation Forest ML-based anomaly detection.
    contamination: expected fraction of anomalies (1% default).
    """
    df = df.copy()
    valid = df[features].dropna()
    
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    model.fit(valid)
    
    # -1 = anomaly, 1 = normal
    df.loc[valid.index, "iso_forest_flag"] = model.predict(valid)
    df["iso_forest_anomaly"] = df["iso_forest_flag"] == -1
    n_anomalies = df["iso_forest_anomaly"].sum()
    print(f"[IsoForest] {n_anomalies} anomalies detected across {features}")
    return df, model


def get_anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean table of rows flagged as anomalies by any method.
    """
    mask = (
        df.get("temperature_anomaly_zscore", False) |
        df.get("rainfall_anomaly_zscore", False) |
        df.get("iso_forest_anomaly", False)
    )
    report = df[mask][["date", "temperature", "rainfall", "humidity",
                        "temperature_anomaly_zscore", "rainfall_anomaly_zscore",
                        "iso_forest_anomaly"]].copy()
    report = report.sort_values("date")
    return report