"""
forecasting.py
Forecasts future temperature using ARIMA (statsmodels) and/or Prophet.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
import joblib

# Prophet (optional — comment out if not installed)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[Forecasting] Prophet not installed; only ARIMA will be used.")


def prepare_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Resample daily data to monthly means for ARIMA."""
    monthly = df.set_index("date")["temperature"].resample("MS").mean()
    return monthly


def fit_arima(series: pd.Series, order: tuple = (2, 1, 2), forecast_months: int = 60) -> dict:
    """
    Fit ARIMA model and generate forecast.
    order: (p, d, q) — autoregressive, differencing, moving average
    forecast_months: number of months to forecast (60 = 5 years)
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    
    # In-sample fitted values
    in_sample = fitted.fittedvalues
    
    # Out-of-sample forecast
    forecast = fitted.get_forecast(steps=forecast_months)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    print(f"[ARIMA] AIC={fitted.aic:.2f} | Forecast horizon: {forecast_months} months")
    
    # Save model
    joblib.dump(fitted, "models/arima_model.pkl")
    
    return {
        "fitted_model": fitted,
        "in_sample": in_sample,
        "forecast_mean": forecast_mean,
        "conf_int_lower": conf_int.iloc[:, 0],
        "conf_int_upper": conf_int.iloc[:, 1],
    }


def fit_prophet(df: pd.DataFrame, forecast_months: int = 60) -> dict:
    """
    Fit Facebook Prophet model.
    Requires columns: ds (datetime), y (value).
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed.")
    
    monthly = df.set_index("date")["temperature"].resample("MS").mean().reset_index()
    monthly.columns = ["ds", "y"]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(monthly)
    
    future = model.make_future_dataframe(periods=forecast_months, freq="MS")
    forecast = model.predict(future)
    
    print(f"[Prophet] Forecast generated for {forecast_months} months")
    
    return {
        "model": model,
        "forecast": forecast,
        "monthly": monthly,
    }