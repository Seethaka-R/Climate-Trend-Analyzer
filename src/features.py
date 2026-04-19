"""
features.py
Feature engineering: adds time features, rolling statistics, lag variables.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns:
    - year, month, day_of_year
    - rolling 30-day and 365-day average temperature
    - temperature anomaly (deviation from 365-day rolling mean)
    - temperature 1-year lag
    - rainfall 30-day rolling total
    - season label
    """
    df = df.copy()
    
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # Rolling averages (min_periods avoids NaN at start)
    df["temp_roll_30d"] = (
        df["temperature"].rolling(window=30, min_periods=15).mean()
    )
    df["temp_roll_365d"] = (
        df["temperature"].rolling(window=365, min_periods=180).mean()
    )
    
    # Temperature anomaly relative to 365-day rolling mean
    df["temp_anomaly"] = df["temperature"] - df["temp_roll_365d"]
    
    # 1-year lag (shift by 365 rows)
    df["temp_lag_1y"] = df["temperature"].shift(365)
    
    # 30-day rolling rainfall total
    df["rain_roll_30d"] = df["rainfall"].rolling(window=30, min_periods=15).sum()
    
    # Season (Northern Hemisphere convention)
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Autumn",  10: "Autumn", 11: "Autumn",
    }
    df["season"] = df["month"].map(season_map)
    
    print(f"[Features] Columns added: {list(df.columns)}")
    return df