"""
cleaner.py
Cleans raw climate data: handles nulls, invalid values, and type coercions.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Drop duplicates
    2. Parse date column
    3. Remove physically impossible values
    4. Fill remaining nulls
    5. Sort by date
    """
    print(f"[Cleaner] Input shape: {df.shape}")
    
    # Step 1: Drop duplicates
    df = df.drop_duplicates(subset=["date"])
    
    # Step 2: Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    
    # Step 3: Remove impossible physical values
    # Temperature: realistic range -50°C to 60°C
    df = df[df["temperature"].between(-50, 60)]
    # Rainfall: non-negative, cap extreme values at 500mm/day
    df["rainfall"] = df["rainfall"].clip(lower=0, upper=500)
    # Humidity: 0-100%
    df["humidity"] = df["humidity"].clip(lower=0, upper=100)
    
    # Step 4: Fill remaining nulls
    for col in ["temperature", "rainfall", "humidity"]:
        df[col] = df[col].fillna(df[col].median())
    
    # Step 5: Sort and reset index
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"[Cleaner] Output shape: {df.shape}")
    print(f"[Cleaner] Nulls remaining:\n{df.isnull().sum()}")
    return df