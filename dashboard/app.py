"""
UHI Prediction Dashboard — Main App
=====================================
Streamlit multi-page dashboard for Delhi UHI analysis.

Pages:
  1. Current Heat Map — Live land cover + hot zones
  2. Historical Trends — Temperature trends 2015–2025
  3. Simulation Tool — What-if construction impact
  4. Mined Rules Explorer — Association rules table

Run:
  streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DASHBOARD_CONFIG

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_CONFIG["page_title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 50%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-top: 1rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #a0a0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(240, 147, 251, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f093fb;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0b0;
        margin-top: 0.3rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────

st.markdown('<h1 class="main-title">🌡️ Delhi UHI Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Spatio-Temporal Urban Heat Island Analysis & Simulation for Delhi, India</p>', unsafe_allow_html=True)

# Quick Stats Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">10+</div>
        <div class="metric-label">Years of Data</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">18</div>
        <div class="metric-label">Weather Parameters</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">5</div>
        <div class="metric-label">Land Cover Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">ML Models</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Project Overview
st.markdown("## 📋 System Overview")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="info-card">
    <h4>🔬 What This System Does</h4>
    <ul>
        <li><b>Classifies</b> land cover from satellite imagery (CNN / ResNet-50)</li>
        <li><b>Clusters</b> heat hotspots using DBSCAN spatial mining</li>
        <li><b>Discovers</b> UHI patterns via association rule mining</li>
        <li><b>Predicts</b> future temperatures using LSTM + XGBoost</li>
        <li><b>Simulates</b> construction impact on local temperature</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="info-card">
    <h4>🗺️ Navigate the Dashboard</h4>
    <ul>
        <li>📍 <b>Current Heat Map</b> — Visualize Delhi's thermal landscape</li>
        <li>📈 <b>Historical Trends</b> — 10+ years of temperature analysis</li>
        <li>🔮 <b>Simulation Tool</b> — Predict construction impact</li>
        <li>⛏️ <b>Mined Rules</b> — Explore discovered UHI patterns</li>
    </ul>
    <p style="color: #a0a0b0; font-size: 0.85rem; margin-top: 0.5rem;">
        👈 Use the sidebar to navigate between pages
    </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🌡️ UHI Dashboard")
    st.markdown("---")
    st.markdown("**Data Source:** NASA POWER API")
    st.markdown("**Location:** Delhi, India")
    st.markdown("**Coordinates:** 28.80°N, 77.07°E")
    st.markdown("---")
    st.markdown("**Models:**")
    st.markdown("- CNN: ResNet-50")
    st.markdown("- Clustering: DBSCAN")
    st.markdown("- Mining: FP-Growth")
    st.markdown("- Prediction: LSTM + XGBoost")
