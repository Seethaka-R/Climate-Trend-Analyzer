"""
dashboard.py
Streamlit interactive dashboard for Climate Trend Analyzer.
Run: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import generate_synthetic_data, save_data, load_raw_data
from src.cleaner import clean_data
from src.features import engineer_features
from src.trend_analysis import compute_annual_means, fit_temperature_trend, run_mann_kendall
from src.anomaly_detection import detect_zscore_anomalies, detect_isolation_forest, get_anomaly_report
from src.forecasting import prepare_monthly_series, fit_arima

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Climate Trend Analyzer")
st.markdown("**Advanced climate data analysis | Trend detection | Anomaly identification | Forecasting**")
st.divider()

# ── Load / cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    raw_path = "data/raw/climate_data.csv"
    if not Path(raw_path).exists():
        df_raw = generate_synthetic_data(years=30)
        save_data(df_raw, raw_path)
    else:
        df_raw = load_raw_data(raw_path)
    
    df_clean = clean_data(df_raw)
    df = engineer_features(df_clean)
    df = detect_zscore_anomalies(df, "temperature", threshold=3.0)
    df = detect_zscore_anomalies(df, "rainfall", threshold=3.0)
    df, _ = detect_isolation_forest(df, ["temperature", "rainfall", "humidity"])
    annual = compute_annual_means(df)
    trend = fit_temperature_trend(annual)
    mk = run_mann_kendall(annual)
    monthly = prepare_monthly_series(df)
    arima_result = fit_arima(monthly, forecast_months=60)
    anomaly_report = get_anomaly_report(df)
    return df, annual, trend, mk, monthly, arima_result, anomaly_report

with st.spinner("Loading and processing climate data..."):
    df, annual, trend, mk, monthly, arima_result, anomaly_report = load_and_process()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters")
year_range = st.sidebar.slider(
    "Year range",
    int(df["year"].min()), int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)
season_filter = st.sidebar.multiselect(
    "Season",
    ["Summer", "Winter", "Spring", "Autumn"],
    default=["Summer", "Winter", "Spring", "Autumn"]
)

mask = (
    df["year"].between(*year_range) &
    df["season"].isin(season_filter)
)
df_filtered = df[mask]

# ── KPI cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Records", f"{len(df_filtered):,}")
col2.metric("🌡️ Mean Temp", f"{df_filtered['temperature'].mean():.2f} °C")
col3.metric("📈 Trend", f"{trend['slope']:.4f} °C/yr",
            delta=f"{trend['slope']*30:.2f} °C over 30 yrs")
col4.metric("⚠️ Anomalies", f"{len(anomaly_report)}")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Temperature Trend",
    "🌧️ Rainfall",
    "⚠️ Anomalies",
    "🔮 Forecast",
    "📊 Seasonal Patterns",
])

# Tab 1 — Temperature trend
with tab1:
    st.subheader("Daily Temperature & Long-Term Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["date"], y=df_filtered["temperature"],
        mode="lines", name="Daily temperature",
        line=dict(color="#4a90d9", width=0.7), opacity=0.5
    ))
    if "temp_roll_365d" in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered["date"], y=df_filtered["temp_roll_365d"],
            mode="lines", name="365-day rolling avg",
            line=dict(color="#e74c3c", width=2.5)
        ))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Annual Mean Temperature + Regression")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=annual["year"], y=annual["mean_temp"],
                          name="Annual mean", marker_color="#4a90d9"))
    fig2.add_trace(go.Scatter(x=annual["year"], y=trend["y_pred"],
                              mode="lines", name=f"Trend ({trend['slope']:.4f} °C/yr)",
                              line=dict(color="#e74c3c", width=3)))
    fig2.update_layout(height=350, xaxis_title="Year", yaxis_title="Temp (°C)")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info(f"**Mann-Kendall Test**: {mk['trend']} trend detected | "
            f"p-value = {mk['p_value']:.4f} | Sen's slope = {mk['slope_sen']:.4f} °C/yr")

# Tab 2 — Rainfall
with tab2:
    st.subheader("Annual Total Rainfall")
    fig3 = px.bar(annual, x="year", y="total_rain",
                  color="total_rain", color_continuous_scale="Blues",
                  labels={"total_rain": "Total Rainfall (mm)"})
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Monthly Rainfall Distribution (all years)")
    fig4 = px.box(df_filtered, x="month", y="rainfall",
                  color="month", labels={"rainfall": "Rainfall (mm)"})
    st.plotly_chart(fig4, use_container_width=True)

# Tab 3 — Anomalies
with tab3:
    st.subheader("Detected Climate Anomalies")
    anomaly_col = "temperature_anomaly_zscore"
    
    fig5 = go.Figure()
    normal = df_filtered[~df_filtered.get(anomaly_col, pd.Series(False, index=df_filtered.index))]
    anom = df_filtered[df_filtered.get(anomaly_col, pd.Series(False, index=df_filtered.index))]
    
    fig5.add_trace(go.Scatter(x=normal["date"], y=normal["temperature"],
                              mode="lines", name="Normal",
                              line=dict(color="#4a90d9", width=0.7), opacity=0.4))
    fig5.add_trace(go.Scatter(x=anom["date"], y=anom["temperature"],
                              mode="markers", name=f"Anomalies ({len(anom)})",
                              marker=dict(color="#e74c3c", size=6)))
    fig5.update_layout(height=380, xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig5, use_container_width=True)
    
    st.subheader("Anomaly Event Table")
    st.dataframe(anomaly_report.head(50), use_container_width=True)

# Tab 4 — Forecast
with tab4:
    st.subheader("5-Year Temperature Forecast (ARIMA 2,1,2)")
    fc = arima_result["forecast_mean"]
    lo = arima_result["conf_int_lower"]
    hi = arima_result["conf_int_upper"]
    
    fig6 = go.Figure()
    recent = monthly.iloc[-60:]
    fig6.add_trace(go.Scatter(x=recent.index, y=recent.values,
                              mode="lines", name="Observed (last 5 yrs)",
                              line=dict(color="#4a90d9", width=2)))
    fig6.add_trace(go.Scatter(x=fc.index, y=fc.values,
                              mode="lines", name="Forecast",
                              line=dict(color="#e74c3c", width=2.5, dash="dash")))
    fig6.add_trace(go.Scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(hi.values) + list(lo.values[::-1]),
        fill="toself", fillcolor="rgba(231,76,60,0.1)",
        line=dict(color="rgba(255,255,255,0)"), name="95% CI"
    ))
    fig6.add_vline(x=monthly.index[-1], line_dash="dot", line_color="gray")
    fig6.update_layout(height=400, xaxis_title="Date", yaxis_title="Temp (°C)")
    st.plotly_chart(fig6, use_container_width=True)

# Tab 5 — Seasonal
with tab5:
    st.subheader("Temperature by Season")
    fig7 = px.violin(df_filtered, x="season", y="temperature",
                     color="season", box=True, points="outliers",
                     color_discrete_map={"Summer": "#e74c3c", "Winter": "#4a90d9",
                                         "Spring": "#2ecc71", "Autumn": "#e67e22"})
    st.plotly_chart(fig7, use_container_width=True)
    
    st.subheader("Decade-by-Decade Seasonal Comparison")
    df["decade"] = (df["year"] // 10) * 10
    decade_monthly = df.groupby(["decade", "month"])["temperature"].mean().reset_index()
    fig8 = px.line(decade_monthly, x="month", y="temperature", color="decade",
                   labels={"temperature": "Mean Temp (°C)", "month": "Month"})
    fig8.update_xaxes(tickvals=list(range(1, 13)),
                      ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"])
    st.plotly_chart(fig8, use_container_width=True)

st.divider()
st.caption("Climate Trend Analyzer | Built with Python, Scikit-learn, Statsmodels, Plotly, Streamlit")