"""
visualizer.py
All chart-generation functions using Matplotlib, Seaborn, and Plotly.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Set global style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Viz] Saved: {path}")
    plt.close(fig)


# ── 1. Raw temperature time series ──────────────────────────────────────────
def plot_temperature_series(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["date"], df["temperature"], color="#4a90d9", lw=0.5, alpha=0.6, label="Daily temp")
    if "temp_roll_365d" in df.columns:
        ax.plot(df["date"], df["temp_roll_365d"], color="#e74c3c", lw=2, label="365-day rolling avg")
    ax.set_title("Daily Temperature with 365-Day Rolling Average", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    save(fig, "01_temperature_series")


# ── 2. Annual mean trend with regression line ────────────────────────────────
def plot_annual_trend(annual: pd.DataFrame, trend_result: dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(annual["year"], annual["mean_temp"], color="#4a90d9", alpha=0.7, label="Annual mean temp")
    ax.plot(annual["year"], trend_result["y_pred"], color="#e74c3c", lw=2.5,
            label=f"Trend: {trend_result['slope']:.4f} °C/yr (R²={trend_result['r2']:.3f})")
    ax.set_title("Annual Mean Temperature with Linear Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend()
    save(fig, "02_annual_temperature_trend")


# ── 3. Monthly seasonal boxplot ──────────────────────────────────────────────
def plot_seasonal_boxplot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    
    sns.boxplot(data=df, x="month", y="temperature", ax=axes[0], palette="RdYlBu_r")
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(month_labels)
    axes[0].set_title("Monthly Temperature Distribution")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Temperature (°C)")
    
    sns.boxplot(data=df, x="month", y="rainfall", ax=axes[1], palette="Blues")
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(month_labels)
    axes[1].set_title("Monthly Rainfall Distribution")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Rainfall (mm)")
    
    fig.suptitle("Seasonal Climate Patterns", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_seasonal_patterns")


# ── 4. Correlation heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["temperature", "rainfall", "humidity", "temp_roll_30d",
            "temp_roll_365d", "temp_anomaly", "rain_roll_30d"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Climate Variable Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_correlation_heatmap")


# ── 5. Anomaly detection chart ───────────────────────────────────────────────
def plot_anomalies(df: pd.DataFrame, column: str = "temperature"):
    anomaly_col = f"{column}_anomaly_zscore"
    if anomaly_col not in df.columns:
        print(f"[Viz] Column {anomaly_col} not found. Skipping anomaly plot.")
        return
    
    normal = df[~df[anomaly_col]]
    anomalies = df[df[anomaly_col]]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(normal["date"], normal[column], color="#4a90d9", lw=0.5, alpha=0.5, label="Normal")
    ax.scatter(anomalies["date"], anomalies[column], color="#e74c3c", s=20, zorder=5,
               label=f"Anomalies ({len(anomalies)})")
    ax.set_title(f"Climate Anomaly Detection — {column.title()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(column.title())
    ax.legend()
    save(fig, f"05_{column}_anomalies")


# ── 6. ARIMA forecast chart ──────────────────────────────────────────────────
def plot_arima_forecast(series: pd.Series, arima_result: dict):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Historical data (last 5 years for clarity)
    recent = series.iloc[-60:]
    ax.plot(recent.index, recent.values, color="#4a90d9", lw=1.5, label="Observed (last 5 yrs)")
    
    # Forecast
    fc = arima_result["forecast_mean"]
    lo = arima_result["conf_int_lower"]
    hi = arima_result["conf_int_upper"]
    
    ax.plot(fc.index, fc.values, color="#e74c3c", lw=2, label="ARIMA forecast")
    ax.fill_between(fc.index, lo.values, hi.values, color="#e74c3c", alpha=0.15, label="95% confidence")
    
    ax.axvline(series.index[-1], color="gray", linestyle="--", lw=1, label="Forecast start")
    ax.set_title("5-Year Temperature Forecast (ARIMA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    save(fig, "06_arima_forecast")


# ── 7. Decade comparison chart ───────────────────────────────────────────────
def plot_decade_comparison(df: pd.DataFrame):
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    decade_stats = df.groupby(["decade", "month"])["temperature"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for decade, group in decade_stats.groupby("decade"):
        ax.plot(group["month"], group["temperature"], marker="o", markersize=3,
                label=f"{decade}s")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_title("Decade-by-Decade Monthly Temperature Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(title="Decade")
    save(fig, "07_decade_comparison")


# ── 8. Plotly interactive dashboard chart (HTML) ─────────────────────────────
def save_interactive_html(df: pd.DataFrame):
    fig = px.scatter(
        df.dropna(subset=["temp_roll_365d"]),
        x="date", y="temperature",
        color="season",
        color_discrete_map={"Summer": "#e74c3c", "Winter": "#4a90d9",
                            "Spring": "#2ecc71", "Autumn": "#e67e22"},
        opacity=0.4,
        title="Daily Temperature by Season (Interactive)",
        labels={"temperature": "Temperature (°C)", "date": "Date"},
    )
    fig.add_scatter(
        x=df["date"], y=df["temp_roll_365d"],
        mode="lines", name="365-day trend",
        line=dict(color="black", width=2)
    )
    out_path = Path("outputs/plots/interactive_temperature.html")
    fig.write_html(str(out_path))
    print(f"[Viz] Interactive chart saved: {out_path}")