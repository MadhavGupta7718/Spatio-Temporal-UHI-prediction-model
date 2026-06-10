"""
Run the full UHI prediction pipeline:
  1. Preprocessing (already done, loads from saved)
  2. DBSCAN Clustering
  3. FP-Growth Association Rule Mining
  4. XGBoost + LSTM Prediction Training
"""
import sys
import os

# Fix Windows console encoding for emojis
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, r"d:\Spatio-Temporal UHI prediction model")

print("=" * 60)
print("PHASE 6: END-TO-END PIPELINE EXECUTION")
print("=" * 60)

# ── Step 1: Preprocessing ──
print("\n[1/4] Running preprocessing pipeline...")
from src.preprocessing import run_preprocessing_pipeline
weather_df, spatial_df, master_df = run_preprocessing_pipeline()
print(f"  Weather: {weather_df.shape}")
print(f"  Spatial: {spatial_df.shape}")
print(f"  Master:  {master_df.shape}")

# ── Step 2: DBSCAN Clustering ──
print("\n[2/4] Running DBSCAN clustering pipeline...")
from src.clustering import run_clustering_pipeline
clustered_df, cluster_stats = run_clustering_pipeline(spatial_df, weather_df)
print(f"  Clustered data shape: {clustered_df.shape}")
if len(cluster_stats) > 0:
    print(f"  Found {len(cluster_stats)} clusters")
    for _, row in cluster_stats.iterrows():
        print(f"    Cluster {row['cluster']}: {row['classification']} | Size={row['size']:.0f} | Concrete={row['mean_concrete_pct']:.1f}%")
else:
    print("  WARNING: No clusters found")

# ── Step 3: Association Rule Mining ──
print("\n[3/4] Running FP-Growth association rule mining...")
from src.mining import run_mining_pipeline
all_rules, uhi_rules = run_mining_pipeline(weather_df, spatial_df)
print(f"  Total rules: {len(all_rules)}")
print(f"  UHI-specific rules: {len(uhi_rules)}")
if len(uhi_rules) > 0:
    print("  Top 5 UHI rules:")
    for _, row in uhi_rules.head(5).iterrows():
        print(f"    Lift={row['lift']:.2f} | Conf={row['confidence']:.2f} | {row['rule']}")

# ── Step 4: XGBoost + LSTM Prediction ──
print("\n[4/4] Running prediction pipeline (XGBoost + LSTM)...")
from src.predictor import run_prediction_pipeline
results = run_prediction_pipeline(master_df)
print("\n  Model Comparison:")
print(results["comparison"].to_string(index=False))
print(f"\n  Simulation Demo:")
print(f"    Baseline temp: {results['simulation_demo']['baseline_temp']}C")
for yr in [1, 3, 5]:
    key = f"year_{yr}"
    if key in results["simulation_demo"]:
        r = results["simulation_demo"][key]
        print(f"    +{yr}yr: {r['predicted_temp']}C (change: {r['temp_change']:+.2f}C)")

print("\n" + "=" * 60)
print("ALL PIPELINES COMPLETE!")
print("=" * 60)
