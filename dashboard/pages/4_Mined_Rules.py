"""
Page 4: Mined Rules Explorer
===============================
Browse discovered association rules that reveal UHI patterns.
Filter by confidence, lift, and keyword search.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RULES_DIR, PROCESSED_DATA_DIR

st.set_page_config(page_title="Mined Rules — UHI Delhi", page_icon="⛏️", layout="wide")

st.markdown("# ⛏️ Association Rules Explorer")
st.markdown("Discovered patterns linking weather, land cover, and UHI effects")

# ──────────────────────────────────────────────
# LOAD RULES
# ──────────────────────────────────────────────

@st.cache_data
def load_rules():
    """Load mined association rules."""
    all_rules_path = RULES_DIR / "all_association_rules.csv"
    uhi_rules_path = RULES_DIR / "uhi_association_rules.csv"
    
    all_rules = None
    uhi_rules = None
    
    if all_rules_path.exists():
        all_rules = pd.read_csv(all_rules_path)
    
    if uhi_rules_path.exists():
        uhi_rules = pd.read_csv(uhi_rules_path)
    
    # Generate demo rules if none exist
    if all_rules is None:
        all_rules = generate_demo_rules()
    
    if uhi_rules is None:
        uhi_rules = all_rules[
            all_rules["consequents_str"].str.contains("temp=|uhi=|anomaly=", na=False)
        ]
    
    return all_rules, uhi_rules


def generate_demo_rules():
    """Generate demonstration rules for showcase."""
    rules = [
        {"antecedents_str": "concrete=VeryHigh, wind=Calm", 
         "consequents_str": "temp=Extreme",
         "support": 0.23, "confidence": 0.85, "lift": 2.8},
        {"antecedents_str": "season=Summer, humidity=Dry", 
         "consequents_str": "temp=Extreme",
         "support": 0.31, "confidence": 0.78, "lift": 2.5},
        {"antecedents_str": "concrete=High, solar=VeryHigh, wind=Calm", 
         "consequents_str": "uhi=Strong",
         "support": 0.18, "confidence": 0.82, "lift": 3.1},
        {"antecedents_str": "veg=Bare, season=Summer", 
         "consequents_str": "temp=High",
         "support": 0.25, "confidence": 0.71, "lift": 1.9},
        {"antecedents_str": "precip=None, wind=Calm, season=Summer", 
         "consequents_str": "anomaly=Spike",
         "support": 0.15, "confidence": 0.76, "lift": 3.4},
        {"antecedents_str": "veg=High, wind=Moderate", 
         "consequents_str": "temp=Medium",
         "support": 0.28, "confidence": 0.68, "lift": 1.6},
        {"antecedents_str": "season=Monsoon, humidity=VeryHumid", 
         "consequents_str": "temp=Medium",
         "support": 0.35, "confidence": 0.72, "lift": 1.7},
        {"antecedents_str": "concrete=VeryHigh, solar=High", 
         "consequents_str": "uhi=Mild",
         "support": 0.20, "confidence": 0.65, "lift": 2.0},
        {"antecedents_str": "ndbi=VeryHigh, wind=Calm", 
         "consequents_str": "temp=Extreme",
         "support": 0.12, "confidence": 0.88, "lift": 3.6},
        {"antecedents_str": "precip=Heavy, veg=High", 
         "consequents_str": "uhi=Cooling",
         "support": 0.10, "confidence": 0.70, "lift": 2.2},
        {"antecedents_str": "season=Winter, wind=Strong", 
         "consequents_str": "temp=Low",
         "support": 0.22, "confidence": 0.80, "lift": 2.1},
        {"antecedents_str": "ndvi=High, precip=Moderate", 
         "consequents_str": "anomaly=BelowNormal",
         "support": 0.14, "confidence": 0.62, "lift": 1.8},
        {"antecedents_str": "concrete=High, asphalt=High, veg=Bare", 
         "consequents_str": "temp=Extreme",
         "support": 0.09, "confidence": 0.91, "lift": 4.1},
        {"antecedents_str": "solar=Low, humidity=Humid", 
         "consequents_str": "uhi=Neutral",
         "support": 0.30, "confidence": 0.58, "lift": 1.3},
        {"antecedents_str": "season=Post-Monsoon, concrete=Medium", 
         "consequents_str": "temp=Medium",
         "support": 0.19, "confidence": 0.64, "lift": 1.5},
    ]
    
    df = pd.DataFrame(rules)
    df["rule"] = df.apply(
        lambda r: f"{{{r['antecedents_str']}}} → {{{r['consequents_str']}}}", axis=1
    )
    return df


all_rules, uhi_rules = load_rules()

# ──────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔍 Filter Rules")
    
    rule_type = st.radio("Rule Set", ["All Rules", "UHI-Related Only"])
    
    display_rules = uhi_rules if rule_type == "UHI-Related Only" else all_rules
    
    if display_rules is not None and len(display_rules) > 0:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        min_lift = st.slider("Min Lift", 0.5, 5.0, 1.0, 0.1)
        
        keyword = st.text_input("Search keyword", placeholder="e.g. concrete, Summer")
        
        # Apply filters
        mask = (display_rules["confidence"] >= min_conf) & (display_rules["lift"] >= min_lift)
        if keyword:
            mask &= display_rules["rule"].str.contains(keyword, case=False, na=False)
        
        display_rules = display_rules[mask]
    
    st.markdown("---")
    st.markdown(f"**Showing:** {len(display_rules) if display_rules is not None else 0} rules")

# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

if display_rules is not None and len(display_rules) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rules", len(display_rules))
    with col2:
        st.metric("Avg Confidence", f"{display_rules['confidence'].mean():.2f}")
    with col3:
        st.metric("Avg Lift", f"{display_rules['lift'].mean():.2f}")
    with col4:
        st.metric("Max Lift", f"{display_rules['lift'].max():.2f}")
    
    st.markdown("---")
    
    # ──────────────────────────────────────────
    # RULES TABLE
    # ──────────────────────────────────────────
    
    st.markdown("### 📋 Association Rules")
    
    # Format for display
    display_df = display_rules[["rule", "support", "confidence", "lift"]].copy()
    display_df = display_df.sort_values("lift", ascending=False)
    display_df["support"] = display_df["support"].round(3)
    display_df["confidence"] = display_df["confidence"].round(3)
    display_df["lift"] = display_df["lift"].round(2)
    
    display_df.columns = ["Rule", "Support", "Confidence", "Lift"]
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Rule": st.column_config.TextColumn("Rule", width="large"),
            "Support": st.column_config.ProgressColumn("Support", min_value=0, max_value=0.5),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1.0),
            "Lift": st.column_config.NumberColumn("Lift", format="%.2f"),
        }
    )
    
    # ──────────────────────────────────────────
    # VISUALIZATIONS
    # ──────────────────────────────────────────
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Confidence vs Lift")
        
        fig_scatter = px.scatter(
            display_rules,
            x="confidence", y="lift",
            size="support",
            color="lift",
            hover_data=["rule"],
            color_continuous_scale="Viridis",
            size_max=20
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Confidence",
            yaxis_title="Lift",
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col_right:
        st.markdown("### 📊 Support Distribution")
        
        fig_hist = px.histogram(
            display_rules,
            x="support",
            nbins=20,
            color_discrete_sequence=["#f093fb"]
        )
        fig_hist.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Support",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, width='stretch')
    
    # ──────────────────────────────────────────
    # TOP RULES HIGHLIGHT
    # ──────────────────────────────────────────
    
    st.markdown("### 🏆 Top 5 Most Significant Rules")
    
    top5 = display_rules.nlargest(5, "lift")
    
    for i, (_, rule) in enumerate(top5.iterrows(), 1):
        lift_color = "#ff4444" if rule["lift"] > 3 else "#ffaa00" if rule["lift"] > 2 else "#22aa44"
        
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border-left: 4px solid {lift_color};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <div style="font-size: 1rem; font-weight: 600;">
                #{i} — {rule['rule']}
            </div>
            <div style="font-size: 0.85rem; color: #a0a0b0; margin-top: 0.3rem;">
                Support: {rule['support']:.3f} | 
                Confidence: {rule['confidence']:.2f} | 
                Lift: <span style="color: {lift_color}; font-weight: 700;">{rule['lift']:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("No rules found. Run the mining pipeline first: `python -m src.mining`")
    
    st.markdown("""
    ### How to generate rules:
    ```bash
    cd "d:\\Spatio-Temporal UHI prediction model"
    python -m src.mining
    ```
    
    Or run the full pipeline:
    ```bash
    python -m src.preprocessing
    python -m src.mining
    ```
    """)
