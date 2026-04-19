"""
main.py
Full climate trend analysis pipeline runner.
Run: python main.py
"""

from pathlib import Path
from src.data_loader import generate_synthetic_data, save_data, load_raw_data
from src.cleaner import clean_data
from src.features import engineer_features
from src.trend_analysis import compute_annual_means, fit_temperature_trend, run_mann_kendall
from src.anomaly_detection import detect_zscore_anomalies, detect_isolation_forest, get_anomaly_report
from src.forecasting import prepare_monthly_series, fit_arima
from src.visualizer import (
    plot_temperature_series, plot_annual_trend, plot_seasonal_boxplot,
    plot_correlation_heatmap, plot_anomalies, plot_arima_forecast,
    plot_decade_comparison, save_interactive_html,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/climate_data.csv"
CLEAN_PATH = "data/processed/climate_clean.csv"
REPORT_PATH = "outputs/reports/anomaly_report.csv"
SUMMARY_PATH = "outputs/reports/summary.txt"

Path("models").mkdir(exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)

# ── Step 1: Generate or load data ─────────────────────────────────────────────
print("\n=== STEP 1: Data Loading ===")
if not Path(RAW_PATH).exists():
    df_raw = generate_synthetic_data(years=30)
    save_data(df_raw, RAW_PATH)
    print(f"Generated synthetic dataset: {len(df_raw)} rows")
else:
    df_raw = load_raw_data(RAW_PATH)
    print(f"Loaded existing dataset: {len(df_raw)} rows")

print(df_raw.head())

# ── Step 2: Clean ─────────────────────────────────────────────────────────────
print("\n=== STEP 2: Data Cleaning ===")
df_clean = clean_data(df_raw)
save_data(df_clean, CLEAN_PATH)

# ── Step 3: Feature engineering ───────────────────────────────────────────────
print("\n=== STEP 3: Feature Engineering ===")
df = engineer_features(df_clean)

# ── Step 4: Trend analysis ────────────────────────────────────────────────────
print("\n=== STEP 4: Trend Analysis ===")
annual = compute_annual_means(df)
trend = fit_temperature_trend(annual)
mk_result = run_mann_kendall(annual)

# ── Step 5: Anomaly detection ─────────────────────────────────────────────────
print("\n=== STEP 5: Anomaly Detection ===")
df = detect_zscore_anomalies(df, "temperature", threshold=3.0)
df = detect_zscore_anomalies(df, "rainfall", threshold=3.0)
df, iso_model = detect_isolation_forest(
    df, features=["temperature", "rainfall", "humidity"], contamination=0.01
)
anomaly_report = get_anomaly_report(df)
anomaly_report.to_csv(REPORT_PATH, index=False)
print(f"Anomaly report saved: {len(anomaly_report)} events")

# ── Step 6: Forecasting ───────────────────────────────────────────────────────
print("\n=== STEP 6: Forecasting ===")
monthly_series = prepare_monthly_series(df)
arima_result = fit_arima(monthly_series, order=(2, 1, 2), forecast_months=60)

# ── Step 7: Visualizations ────────────────────────────────────────────────────
print("\n=== STEP 7: Visualizations ===")
plot_temperature_series(df)
plot_annual_trend(annual, trend)
plot_seasonal_boxplot(df)
plot_correlation_heatmap(df)
plot_anomalies(df, "temperature")
plot_anomalies(df, "rainfall")
plot_arima_forecast(monthly_series, arima_result)
plot_decade_comparison(df)
save_interactive_html(df)

# ── Step 8: Summary report ────────────────────────────────────────────────────
print("\n=== STEP 8: Summary Report ===")
slope = trend["slope"]
r2 = trend["r2"]
mk_trend = mk_result["trend"]
mk_p = mk_result["p_value"]
n_anomalies = anomaly_report.shape[0]

summary = f"""
=== CLIMATE TREND ANALYSIS SUMMARY ===
Dataset period: {df["date"].min().date()} to {df["date"].max().date()}
Total records : {len(df):,}

--- Temperature Trend ---
Linear slope  : {slope:.4f} °C/year
Over 30 years : {slope * 30:.2f} °C total warming
R² (fit)      : {r2:.3f}
Mann-Kendall  : {mk_trend} trend (p={mk_p:.4f})

--- Anomaly Detection ---
Total flagged events: {n_anomalies}
Report saved: {REPORT_PATH}

--- Forecast ---
ARIMA(2,1,2) 5-year forecast generated.
Model saved: models/arima_model.pkl

--- Outputs ---
Plots: outputs/plots/
Report: {REPORT_PATH}
Summary: {SUMMARY_PATH}
"""

print(summary)
with open(SUMMARY_PATH, "w") as f:
    f.write(summary)
print(f"\n✅ Pipeline complete. All outputs saved.")