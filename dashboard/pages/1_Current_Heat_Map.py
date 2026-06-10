"""
Page 1: Current Heat Map
=========================
Interactive Folium map showing:
  - Delhi land cover classification
  - Heat hotspot clusters (from DBSCAN)
  - NDVI/NDBI overlay
  - Temperature gradient
"""

import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DELHI_BBOX, DELHI_LAT, DELHI_LON, PROCESSED_DATA_DIR, CNN_CONFIG
from src.utils import generate_delhi_grid

st.set_page_config(page_title="Heat Map — UHI Delhi", page_icon="📍", layout="wide")

st.markdown("# 📍 Current Heat Map")
st.markdown("Interactive visualization of Delhi's thermal landscape and land cover classification")

# ──────────────────────────────────────────────
# LOAD OR GENERATE DATA
# ──────────────────────────────────────────────

@st.cache_data
def load_spatial_data():
    """Load real satellite-derived spatial data."""
    spatial_path = PROCESSED_DATA_DIR / "spatial_features.csv"
    
    if not spatial_path.exists():
        st.error(
            "⚠️ Spatial data not found. Please run the preprocessing pipeline first:\n\n"
            "```python run_full_pipeline.py```"
        )
        st.stop()
    
    df = pd.read_csv(spatial_path)
    
    # Add dominant land cover class
    cover_cols = ["concrete_pct", "vegetation_pct", "water_pct", "asphalt_pct"]
    available = [c for c in cover_cols if c in df.columns]
    if available:
        df["dominant_cover"] = df[available].idxmax(axis=1).str.replace("_pct", "")
    
    return df


@st.cache_data
def load_cluster_data():
    """Load clustered data if available."""
    cluster_path = PROCESSED_DATA_DIR / "clustered_spatial.csv"
    if cluster_path.exists():
        return pd.read_csv(cluster_path)
    return None


spatial_df = load_spatial_data()
cluster_df = load_cluster_data()

# ──────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🗺️ Map Controls")
    
    overlay = st.selectbox("Overlay Layer", [
        "Temperature Gradient",
        "Land Cover Classification",
        "NDVI (Vegetation Index)",
        "NDBI (Built-up Index)",
        "Heat Hotspot Clusters"
    ])
    
    opacity = st.slider("Layer Opacity", 0.3, 1.0, 0.7)
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    if spatial_df is not None:
        st.metric("Grid Points", f"{len(spatial_df):,}")
        
        if "concrete_pct" in spatial_df.columns:
            st.metric("Avg Concrete %", f"{spatial_df['concrete_pct'].mean():.1f}%")
        if "vegetation_pct" in spatial_df.columns:
            st.metric("Avg Vegetation %", f"{spatial_df['vegetation_pct'].mean():.1f}%")
        if "ndvi" in spatial_df.columns:
            st.metric("Avg NDVI", f"{spatial_df['ndvi'].mean():.3f}")

# ──────────────────────────────────────────────
# BUILD MAP
# ──────────────────────────────────────────────

# Create base map centered on Delhi
m = folium.Map(
    location=[DELHI_LAT, DELHI_LON],
    zoom_start=11,
    tiles="CartoDB dark_matter"
)

# Add tile layers
folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(m)

# Sample points for visualization (full grid can be too many)
sample_size = min(500, len(spatial_df))
display_df = spatial_df.sample(sample_size, random_state=42)

# Color mapping based on overlay selection
def get_color(row, overlay_type):
    if overlay_type == "Temperature Gradient":
        # Use UHI proxy or concrete percentage
        val = row.get("concrete_pct", 50) / 100
        if val > 0.7:
            return "#ff0000"
        elif val > 0.5:
            return "#ff6600"
        elif val > 0.3:
            return "#ffaa00"
        else:
            return "#00cc44"
    
    elif overlay_type == "Land Cover Classification":
        cover = row.get("dominant_cover", "concrete")
        color_map = {
            "concrete": "#cc3333",
            "asphalt": "#666666",
            "vegetation": "#22aa22",
            "water": "#2244cc",
        }
        return color_map.get(cover, "#888888")
    
    elif overlay_type == "NDVI (Vegetation Index)":
        val = row.get("ndvi", 0)
        if val > 0.5:
            return "#006600"
        elif val > 0.3:
            return "#22aa22"
        elif val > 0.1:
            return "#88cc44"
        else:
            return "#cc8833"
    
    elif overlay_type == "NDBI (Built-up Index)":
        val = row.get("ndbi", 0)
        if val > 0.3:
            return "#cc0000"
        elif val > 0.15:
            return "#ff6600"
        elif val > 0:
            return "#ffcc00"
        else:
            return "#4488cc"
    
    elif overlay_type == "Heat Hotspot Clusters":
        cluster = row.get("cluster", -1)
        if cluster == -1:
            return "#444444"
        colors = ["#ff0000", "#ff6600", "#ffcc00", "#00cc44", "#0066cc", "#9933cc"]
        return colors[int(cluster) % len(colors)]
    
    return "#888888"


# Add markers/circles to map
for _, row in display_df.iterrows():
    color = get_color(row, overlay)
    
    popup_html = f"""
    <div style="font-family: Arial; font-size: 12px;">
        <b>Location:</b> {row['lat']:.4f}°N, {row['lon']:.4f}°E<br>
        <b>Concrete:</b> {row.get('concrete_pct', 'N/A'):.1f}%<br>
        <b>Vegetation:</b> {row.get('vegetation_pct', 'N/A'):.1f}%<br>
        <b>NDVI:</b> {row.get('ndvi', 'N/A'):.3f}<br>
        <b>NDBI:</b> {row.get('ndbi', 'N/A'):.3f}<br>
    </div>
    """
    
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=opacity,
        popup=folium.Popup(popup_html, max_width=200),
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
st.markdown(f"**Viewing:** {overlay}")
st_folium(m, width=None, height=600)

# ──────────────────────────────────────────────
# LEGEND & STATS
# ──────────────────────────────────────────────

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏷️ Land Cover Distribution")
    if "dominant_cover" in spatial_df.columns:
        cover_dist = spatial_df["dominant_cover"].value_counts()
        st.bar_chart(cover_dist)

with col2:
    st.markdown("### 📊 NDVI Distribution")
    if "ndvi" in spatial_df.columns:
        st.bar_chart(spatial_df["ndvi"].value_counts(bins=10).sort_index())

with col3:
    st.markdown("### 🏗️ Concrete Coverage")
    if "concrete_pct" in spatial_df.columns:
        st.bar_chart(spatial_df["concrete_pct"].value_counts(bins=10).sort_index())
