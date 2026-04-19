"""
test_cleaner.py
Unit tests for the data cleaning module.
Run: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cleaner import clean_data


def make_test_df():
    return pd.DataFrame({
        "date": pd.date_range("2000-01-01", periods=10, freq="D"),
        "temperature": [22.0, 100.0, -60.0, 25.0, np.nan, 21.0, 23.0, 19.0, 24.0, 20.0],
        "rainfall": [5.0, -2.0, 3.0, 600.0, 4.0, 2.0, np.nan, 1.0, 3.0, 2.0],
        "humidity": [65.0, 110.0, 60.0, -5.0, 70.0, 65.0, 68.0, 62.0, 66.0, 64.0],
    })


def test_impossible_temperatures_removed():
    df = make_test_df()
    result = clean_data(df)
    assert result["temperature"].between(-50, 60).all(), "Temperatures out of range remain"


def test_no_nulls_after_cleaning():
    df = make_test_df()
    result = clean_data(df)
    assert result.isnull().sum().sum() == 0, "Nulls remain after cleaning"


def test_rainfall_clipped():
    df = make_test_df()
    result = clean_data(df)
    assert result["rainfall"].max() <= 500, "Rainfall not clipped at 500mm"
    assert result["rainfall"].min() >= 0, "Negative rainfall not removed"


def test_humidity_clipped():
    df = make_test_df()
    result = clean_data(df)
    assert result["humidity"].between(0, 100).all(), "Humidity out of 0-100 range"


def test_sorted_by_date():
    df = make_test_df().sample(frac=1, random_state=42)
    result = clean_data(df)
    assert result["date"].is_monotonic_increasing, "Dates not sorted"