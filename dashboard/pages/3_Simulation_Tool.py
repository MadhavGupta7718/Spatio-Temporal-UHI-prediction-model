"""
Page 3: Simulation Tool (Centrepiece)
=======================================
The star feature — lets planners simulate:
  "If I build a concrete mall and remove vegetation,
   how much will temperature rise in 1/3/5 years?"

Controls:
  - Select neighbourhood on map
  - Adjust concrete/vegetation sliders
  - Click "Predict" → see temperature forecast
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DELHI_LAT, DELHI_LON, PROCESSED_DATA_DIR

st.set_page_config(page_title="Simulation Tool — UHI Delhi", page_icon="🔮", layout="wide")

st.markdown("# 🔮 Construction Impact Simulator")
st.markdown("Predict how urban development changes Delhi's local temperature using our trained AI (XGBoost) model.")

# ──────────────────────────────────────────────
# LOAD ML MODELS & DATA
# ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    models_dir = Path(PROCESSED_DATA_DIR).parent / "models"
    xgb_model = joblib.load(models_dir / "xgb_model.joblib")
    feature_scaler = joblib.load(models_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(models_dir / "target_scaler.joblib")
    return xgb_model, feature_scaler, target_scaler

@st.cache_data
def get_baseline_features(neighbourhood: str):
    """Get a baseline feature row for prediction."""
    master_path = Path(PROCESSED_DATA_DIR) / "master_dataset.csv"
    
    # We will pick a baseline weather day
    base_feats = {
        "T2M": 35.0, "T2M_MAX": 40.0, "T2M_MIN": 28.0, "TS": 38.0, 
        "RH2M": 40.0, "WS10M": 3.0, "ALLSKY_SFC_SW_DWN": 25.0, 
        "PRECTOTCORR": 0.0,
        "concrete_pct": 50.0, "vegetation_pct": 20.0, 
        "asphalt_pct": 20.0, "water_pct": 0.0
    }
    
    if master_path.exists():
        df = pd.read_csv(master_path)
        summer_df = df[df["season"] == "Summer"] if "season" in df.columns else df
        if len(summer_df) > 0:
            if "date" in summer_df.columns:
                summer_df = summer_df.sort_values("date", ascending=False)
            row = summer_df.iloc[0]
            for k in base_feats.keys():
                if k in row:
                    base_feats[k] = row[k]
                    
    # Tweak baseline land cover for the dummy neighborhoods for realism
    if "Connaught" in neighbourhood:
        base_feats["concrete_pct"] = 80.0
        base_feats["vegetation_pct"] = 5.0
    elif "Dwarka" in neighbourhood:
        base_feats["concrete_pct"] = 60.0
        base_feats["vegetation_pct"] = 15.0
    elif "Rohini" in neighbourhood:
        base_feats["concrete_pct"] = 70.0
        base_feats["vegetation_pct"] = 10.0
    elif "Noida" in neighbourhood:
        base_feats["concrete_pct"] = 50.0
        base_feats["vegetation_pct"] = 25.0
    elif "Gurgaon" in neighbourhood:
        base_feats["concrete_pct"] = 75.0
        base_feats["vegetation_pct"] = 10.0
    elif "Chandni" in neighbourhood:
        base_feats["concrete_pct"] = 90.0
        base_feats["vegetation_pct"] = 2.0
        
    return base_feats


def simulate_temperature_impact(
    base_features: dict,
    concrete_change: float,
    vegetation_change: float,
    asphalt_change: float,
    water_change: float,
    forecast_years: list = [1, 3, 5]
) -> dict:
    
    xgb_model, feature_scaler, target_scaler = load_models()
    
    feature_names = [
        "T2M", "T2M_MAX", "T2M_MIN", "TS", "RH2M", "WS10M",
        "ALLSKY_SFC_SW_DWN", "PRECTOTCORR",
        "concrete_pct", "vegetation_pct", "asphalt_pct", "water_pct"
    ]
    
    # Predict Baseline (requires 30-day sequence flattened)
    base_arr = np.array([[base_features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    base_seq = np.tile(base_arr, (30, 1))
    base_scaled = feature_scaler.transform(base_seq)
    base_flat = base_scaled.flatten().reshape(1, -1)
    base_delta = target_scaler.inverse_transform(xgb_model.predict(base_flat).reshape(-1, 1))[0][0]
    base_pred = base_features.get("T2M", 35.0) + base_delta
    
    results = {"baseline_temp": base_pred}
    
    # Predict Modified
    mod_features = base_features.copy()
    mod_features["concrete_pct"] = max(0, min(100, mod_features["concrete_pct"] + concrete_change))
    mod_features["vegetation_pct"] = max(0, min(100, mod_features["vegetation_pct"] + vegetation_change))
    mod_features["asphalt_pct"] = max(0, min(100, mod_features["asphalt_pct"] + asphalt_change))
    mod_features["water_pct"] = max(0, min(100, mod_features["water_pct"] + water_change))
    
    mod_arr = np.array([[mod_features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    mod_seq = np.tile(mod_arr, (30, 1))
    mod_scaled = feature_scaler.transform(mod_seq)
    mod_flat = mod_scaled.flatten().reshape(1, -1)
    mod_delta = target_scaler.inverse_transform(xgb_model.predict(mod_flat).reshape(-1, 1))[0][0]
    mod_pred = base_features.get("T2M", 35.0) + mod_delta
    
    immediate_delta = mod_pred - base_pred
    
    for yr in forecast_years:
        # We also need to add the natural annual UHI trend 
        annual_trend = yr * 0.04  # °C per year warming baseline
        total_delta = immediate_delta + annual_trend
        
        results[f"year_{yr}"] = {
            "predicted_temp": round(base_pred + total_delta, 2),
            "temp_change": round(total_delta, 2),
            "land_cover_effect": round(immediate_delta, 2),
            "trend_effect": round(annual_trend, 2),
        }
    
    return results

# ──────────────────────────────────────────────
# SIMULATION CONTROLS
# ──────────────────────────────────────────────

col_controls, col_results = st.columns([1, 2])

with col_controls:
    st.markdown("### 📍 Select Location")
    
    neighbourhood = st.selectbox("Neighbourhood", [
        "Connaught Place (Central)",
        "Dwarka (Southwest)",
        "Rohini (Northwest)",
        "Noida Extension (East)",
        "Gurgaon Sector 15 (South)",
        "Chandni Chowk (Old Delhi)",
        "Custom Location"
    ])
    
    base_features = get_baseline_features(neighbourhood)
    
    st.markdown("---")
    st.markdown("### 🏗️ Planned Changes")
    st.markdown("*Adjust sliders to simulate construction plans*")
    
    concrete_change = st.slider(
        "🏢 Concrete Change (%)",
        min_value=-30, max_value=50, value=0, step=5,
        help="Positive = adding concrete buildings/structures"
    )
    
    vegetation_change = st.slider(
        "🌳 Vegetation Change (%)",
        min_value=-50, max_value=30, value=0, step=5,
        help="Negative = removing trees/green cover"
    )
    
    asphalt_change = st.slider(
        "🛣️ Asphalt/Roads Change (%)",
        min_value=-20, max_value=30, value=0, step=5,
        help="Adding roads/parking lots"
    )
    
    water_change = st.slider(
        "💧 Water Bodies Change (%)",
        min_value=-10, max_value=20, value=0, step=5,
        help="Adding ponds/fountains/canals"
    )
    
    st.markdown("---")
    
    predict_btn = st.button("🔮 Predict Impact (AI Model)", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────

with col_results:
    # Run simulation
    results = simulate_temperature_impact(
        base_features=base_features,
        concrete_change=concrete_change,
        vegetation_change=vegetation_change,
        asphalt_change=asphalt_change,
        water_change=water_change
    )
    local_baseline = results["baseline_temp"]

    # Always show current baseline
    st.markdown(f"### 🌡️ AI Baseline: **{local_baseline:.1f}°C** ({neighbourhood})")
    
    # Forecast cards
    st.markdown("### 📊 Temperature Forecast")
    
    fcol1, fcol2, fcol3 = st.columns(3)
    
    for col, yr in zip([fcol1, fcol2, fcol3], [1, 3, 5]):
        key = f"year_{yr}"
        data = results[key]
        delta = data["temp_change"]
        
        with col:
            color = "#ff4444" if delta > 0 else "#22aa44" if delta < 0 else "#888888"
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: 2px solid {color};
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
            ">
                <div style="font-size: 0.9rem; color: #a0a0b0;">+{yr} Year{'s' if yr > 1 else ''}</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {color};">
                    {data['predicted_temp']:.1f}°C
                </div>
                <div style="font-size: 1.1rem; color: {color};">
                    {arrow} {abs(delta):.2f}°C
                </div>
                <div style="font-size: 0.75rem; color: #888; margin-top: 0.5rem;">
                    Land cover: {data['land_cover_effect']:+.2f}°C<br>
                    Climate trend: {data['trend_effect']:+.2f}°C
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualization
    st.markdown("---")
    st.markdown("### 📈 Temperature Projection")
    
    years = [0, 1, 3, 5]
    temps = [local_baseline]
    for yr in [1, 3, 5]:
        temps.append(results[f"year_{yr}"]["predicted_temp"])
    
    # Also compute a "no change" scenario
    no_change_temps = [local_baseline + yr * 0.04 for yr in years]
    
    fig = go.Figure()
    
    # No change scenario
    fig.add_trace(go.Scatter(
        x=years, y=no_change_temps,
        mode="lines+markers",
        name="No Change (natural trend only)",
        line=dict(color="#888888", width=2, dash="dash"),
        marker=dict(size=8)
    ))
    
    # With planned changes
    fig.add_trace(go.Scatter(
        x=years, y=temps,
        mode="lines+markers",
        name="With Planned Changes",
        line=dict(color="#f093fb", width=3),
        marker=dict(size=12, color="#f093fb", line=dict(width=2, color="white"))
    ))
    
    # Add baseline reference
    fig.add_hline(y=local_baseline, line_dash="dot", line_color="#fda085", 
                  annotation_text="Current Baseline")
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Years from Now",
        yaxis_title="Predicted Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Impact breakdown
    st.markdown("### 📋 ML Impact Breakdown (Non-Linear)")
    
    def get_isolated_effect(c_c, v_c, a_c, w_c):
        res = simulate_temperature_impact(base_features, c_c, v_c, a_c, w_c, [1])
        return res["year_1"]["land_cover_effect"]
        
    concrete_effect = get_isolated_effect(concrete_change, 0, 0, 0)
    vegetation_effect = get_isolated_effect(0, vegetation_change, 0, 0)
    asphalt_effect = get_isolated_effect(0, 0, asphalt_change, 0)
    water_effect = get_isolated_effect(0, 0, 0, water_change)

    breakdown = pd.DataFrame({
        "Factor": ["Concrete", "Vegetation", "Asphalt", "Water", "Net ML Impact (Combined)"],
        "Change": [f"{concrete_change:+d}%", f"{vegetation_change:+d}%", f"{asphalt_change:+d}%", f"{water_change:+d}%", "—"],
        "Temp Effect": [
            f"{concrete_effect:+.2f}°C",
            f"{vegetation_effect:+.2f}°C",
            f"{asphalt_effect:+.2f}°C",
            f"{water_effect:+.2f}°C",
            f"{results['year_1']['land_cover_effect']:+.2f}°C"
        ]
    })
    
    st.dataframe(breakdown, width='stretch', hide_index=True)
    
    # ML Extrapolation Notes (Dynamic Warnings)
    if concrete_change != 0 and abs(concrete_effect) < 0.005:
        if base_features["concrete_pct"] + concrete_change > 74.3:
            st.info("💡 **Concrete Limit Reached**: You increased concrete, but the temperature didn't change! Reason: The AI model has never seen concrete density above **74.3%** in real Delhi training data. It cannot predict heat increases beyond this known ceiling.")
        else:
            st.info("💡 **Concrete Limit Reached**: Your concrete change didn't affect temperature. Reason: The AI model grouped this specific density into an existing decision-tree leaf node.")

    if vegetation_change != 0 and abs(vegetation_effect) < 0.005:
        if base_features["vegetation_pct"] + vegetation_change <= 0:
            st.info("💡 **Vegetation Limit Reached**: You removed vegetation, but the temperature didn't change! Reason: Vegetation was already at or near 0%. You can't remove trees that don't exist!")
        else:
            st.info("💡 **Vegetation Limit Reached**: Your vegetation change didn't affect temperature. Reason: The value exceeded the maximum or minimum boundaries the AI learned from historical data.")

    if asphalt_change != 0 and abs(asphalt_effect) < 0.005:
        st.info("💡 **Asphalt Limit Reached**: Your asphalt change didn't affect temperature. Reason: The value hit the extrapolation limits of the AI's training data.")

    if water_change != 0 and abs(water_effect) < 0.005:
        st.info("💡 **Water Limit Reached**: Your water bodies change didn't affect temperature. Reason: The value hit the extrapolation limits of the AI's training data.")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    total_impact = results["year_5"]["temp_change"]
    
    if total_impact > 1.0:
        st.error(f"""
        ⚠️ **High Impact Warning**: Your plan increases temperature by **{total_impact:.1f}°C** over 5 years.
        
        **Suggestions:**
        - Add green roofs to offset concrete heat absorption
        - Plant shade trees along roads (-0.5 to -1.5°C cooling)
        - Install reflective (cool) roofing materials
        - Create water features for evaporative cooling
        """)
    elif total_impact > 0.3:
        st.warning(f"""
        ⚡ **Moderate Impact**: Temperature increase of **{total_impact:.1f}°C** over 5 years.
        
        Consider adding vegetation buffers around the development zone.
        """)
    elif total_impact < -0.1:
        st.success(f"""
        ✅ **Positive Impact**: Your plan actually **cools** the area by **{abs(total_impact):.1f}°C**.
        Great urban planning! More vegetation and water bodies help significantly.
        """)
    else:
        st.info(f"""
        ℹ️ **Minimal Impact**: Temperature change of **{total_impact:.1f}°C** — within natural variation.
        """)
