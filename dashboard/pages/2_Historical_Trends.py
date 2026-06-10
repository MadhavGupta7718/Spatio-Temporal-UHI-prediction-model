"""
Page 2: Historical Trends
===========================
Time series analysis of Delhi temperature data (2015–2025).
Shows seasonal patterns, UHI trends, and climate anomalies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR, NASA_POWER_CSV, NASA_POWER_HEADER_LINES
from src.utils import add_date_column, replace_sentinel_values, get_season

st.set_page_config(page_title="Historical Trends — UHI Delhi", page_icon="📈", layout="wide")

st.markdown("# 📈 Historical Temperature Trends")
st.markdown("10+ years of daily weather data analysis for Delhi (2015–2025)")

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

@st.cache_data
def load_weather_data():
    """Load processed or raw weather data."""
    processed = PROCESSED_DATA_DIR / "weather_processed.csv"
    
    if processed.exists():
        df = pd.read_csv(processed, parse_dates=["date"])
    elif NASA_POWER_CSV.exists():
        df = pd.read_csv(NASA_POWER_CSV, skiprows=NASA_POWER_HEADER_LINES)
        df.columns = df.columns.str.strip()
        df = replace_sentinel_values(df)
        df = add_date_column(df)
    else:
        st.error(
            "⚠️ Weather data not found. Please run the preprocessing pipeline first:\n\n"
            "```python run_full_pipeline.py```"
        )
        st.stop()
    
    return df


df = load_weather_data()

# ──────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📅 Date Range")
    
    if "YEAR" in df.columns:
        year_range = st.slider(
            "Select Years",
            min_value=int(df["YEAR"].min()),
            max_value=int(df["YEAR"].max()),
            value=(int(df["YEAR"].min()), int(df["YEAR"].max()))
        )
        df_filtered = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]
    else:
        df_filtered = df
    
    st.markdown("---")
    st.markdown("### 📊 Parameters")
    
    param = st.selectbox("Primary Parameter", [
        "T2M", "T2M_MAX", "T2M_MIN", "TS", "RH2M", "WS10M",
        "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"
    ])
    
    param_labels = {
        "T2M": "Temperature at 2m (°C)",
        "T2M_MAX": "Max Temperature (°C)",
        "T2M_MIN": "Min Temperature (°C)",
        "TS": "Earth Skin Temperature (°C)",
        "RH2M": "Relative Humidity (%)",
        "WS10M": "Wind Speed (m/s)",
        "ALLSKY_SFC_SW_DWN": "Solar Irradiance (MJ/m²/day)",
        "PRECTOTCORR": "Precipitation (mm/day)",
    }

# ──────────────────────────────────────────────
# SUMMARY METRICS
# ──────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📅 Total Days", f"{len(df_filtered):,}")
with col2:
    if "T2M" in df_filtered.columns:
        st.metric("🌡️ Mean Temp", f"{df_filtered['T2M'].mean():.1f}°C")
with col3:
    if "T2M_MAX" in df_filtered.columns:
        st.metric("🔥 Max Recorded", f"{df_filtered['T2M_MAX'].max():.1f}°C")
with col4:
    if "T2M_MIN" in df_filtered.columns:
        st.metric("❄️ Min Recorded", f"{df_filtered['T2M_MIN'].min():.1f}°C")
with col5:
    if "PRECTOTCORR" in df_filtered.columns:
        st.metric("🌧️ Total Rainfall", f"{df_filtered['PRECTOTCORR'].sum():.0f}mm")

st.markdown("---")

# ──────────────────────────────────────────────
# CHART 1: DAILY TIME SERIES
# ──────────────────────────────────────────────

st.markdown(f"### 📉 Daily {param_labels.get(param, param)}")

if param in df_filtered.columns and "date" in df_filtered.columns:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=df_filtered[param],
        mode="lines",
        name=param,
        line=dict(color="#f093fb", width=0.8),
        opacity=0.7
    ))
    
    # Add rolling average
    rolling = df_filtered[param].rolling(window=30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=rolling,
        mode="lines",
        name="30-day Average",
        line=dict(color="#fda085", width=2)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title=param_labels.get(param, param),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, width='stretch')

# ──────────────────────────────────────────────
# CHART 2: SEASONAL COMPARISON
# ──────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🌦️ Seasonal Temperature Distribution")
    
    if "season" in df_filtered.columns and "T2M" in df_filtered.columns:
        season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        fig_box = px.box(
            df_filtered, x="season", y="T2M",
            color="season",
            category_orders={"season": season_order},
            color_discrete_map={
                "Winter": "#4488cc",
                "Summer": "#ff4444",
                "Monsoon": "#22aa44",
                "Post-Monsoon": "#ffaa00"
            }
        )
        fig_box.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(fig_box, width='stretch')

with col_right:
    st.markdown("### 📅 Monthly Average Temperature")
    
    if "month" in df_filtered.columns and "T2M" in df_filtered.columns:
        monthly = df_filtered.groupby("month")["T2M"].mean().reset_index()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["month_name"] = monthly["month"].map(
            lambda m: month_names[m-1] if 1 <= m <= 12 else str(m)
        )
        
        fig_monthly = px.bar(
            monthly, x="month_name", y="T2M",
            color="T2M",
            color_continuous_scale="RdYlBu_r"
        )
        fig_monthly.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Month",
            yaxis_title="Mean Temperature (°C)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_monthly, width='stretch')

# ──────────────────────────────────────────────
# CHART 3: YEAR-OVER-YEAR TREND
# ──────────────────────────────────────────────

st.markdown("### 📈 Year-over-Year Temperature Trend (UHI Intensification)")

if "YEAR" in df_filtered.columns and "T2M" in df_filtered.columns:
    yearly = df_filtered.groupby("YEAR").agg({
        "T2M": "mean",
        "T2M_MAX": "max",
        "T2M_MIN": "min"
    }).reset_index()
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=yearly["YEAR"], y=yearly["T2M"],
        mode="lines+markers",
        name="Mean Temp",
        line=dict(color="#f093fb", width=3),
        marker=dict(size=8)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=yearly["YEAR"], y=yearly["T2M_MAX"],
        mode="lines+markers",
        name="Peak Max",
        line=dict(color="#ff4444", width=2, dash="dash"),
        marker=dict(size=6)
    ))
    
    # Add trendline
    if len(yearly) > 2:
        z = np.polyfit(yearly["YEAR"], yearly["T2M"], 1)
        trend = np.poly1d(z)(yearly["YEAR"])
        fig_trend.add_trace(go.Scatter(
            x=yearly["YEAR"], y=trend,
            mode="lines",
            name=f"Trend ({z[0]:+.3f}°C/year)",
            line=dict(color="#fda085", width=2, dash="dot")
        ))
    
    fig_trend.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig_trend, width='stretch')

# ──────────────────────────────────────────────
# CHART 4: UHI PROXY
# ──────────────────────────────────────────────

st.markdown("### 🏙️ UHI Proxy (Surface Temp − Air Temp)")
st.markdown("*Positive values indicate urban heat island effect (surface hotter than air)*")

if "TS" in df_filtered.columns and "T2M" in df_filtered.columns:
    df_filtered_copy = df_filtered.copy()
    df_filtered_copy["uhi_proxy"] = df_filtered_copy["TS"] - df_filtered_copy["T2M"]
    
    if "date" in df_filtered_copy.columns:
        monthly_uhi = df_filtered_copy.groupby(df_filtered_copy["date"].dt.to_period("M"))["uhi_proxy"].mean()
        monthly_uhi.index = monthly_uhi.index.to_timestamp()
        
        fig_uhi = go.Figure()
        
        colors = ["#ff4444" if v > 0 else "#4488cc" for v in monthly_uhi.values]
        
        fig_uhi.add_trace(go.Bar(
            x=monthly_uhi.index,
            y=monthly_uhi.values,
            marker_color=colors,
            name="UHI Proxy"
        ))
        
        fig_uhi.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        
        fig_uhi.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Date",
            yaxis_title="UHI Proxy (°C)",
        )
        
        st.plotly_chart(fig_uhi, width='stretch')

# ──────────────────────────────────────────────
# DATA TABLE
# ──────────────────────────────────────────────

with st.expander("📋 View Raw Data"):
    display_cols = ["date", "T2M", "T2M_MAX", "T2M_MIN", "TS", "RH2M", 
                    "WS10M", "PRECTOTCORR", "season"]
    available_cols = [c for c in display_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[available_cols].tail(100), width='stretch')
