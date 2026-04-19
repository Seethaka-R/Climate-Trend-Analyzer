"""
data_loader.py
Loads and validates raw climate CSV data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_data(years: int = 30, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    dates = pd.date_range(start="1993-01-01", periods=years * 365, freq="D")
    n = len(dates)
    t = np.arange(n)
    
    seasonal = 8 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    trend = 0.03 / 365 * t
    noise = np.random.normal(0, 1.5, n)
    temperature = 22 + seasonal + trend + noise
    
    anomaly_idx = np.random.choice(n, size=15, replace=False)
    temperature[anomaly_idx] += np.random.uniform(6, 12, size=15)
    
    month = pd.DatetimeIndex(dates).month
    rain_base = np.where((month >= 6) & (month <= 9), 6.0, 1.5)
    rainfall = np.random.exponential(rain_base)
    
    rain_anomaly_idx = np.random.choice(n, size=10, replace=False)
    rainfall[rain_anomaly_idx] += np.random.uniform(60, 150, size=10)
    
    humidity = np.clip(60 + 15 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 5, n), 20, 100)
    
    df = pd.DataFrame({
        "date": dates,
        "temperature": np.round(temperature, 2),
        "rainfall": np.round(rainfall, 2),
        "humidity": np.round(humidity, 1),
    })
    
    return df


def load_raw_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")