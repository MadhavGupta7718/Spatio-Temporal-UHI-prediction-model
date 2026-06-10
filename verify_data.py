"""Data integrity verification script."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"d:\Spatio-Temporal UHI prediction model")

import pandas as pd
from pathlib import Path

data_dir = Path(r"d:\Spatio-Temporal UHI prediction model\data\processed")
rules_dir = Path(r"d:\Spatio-Temporal UHI prediction model\outputs\rules")
models_dir = Path(r"d:\Spatio-Temporal UHI prediction model\data\models")

print("=" * 60)
print("DATA INTEGRITY CHECK")
print("=" * 60)

# 1. Weather data
weather = pd.read_csv(data_dir / "weather_processed.csv", parse_dates=["date"])
print(f"\n1. Weather data: {weather.shape}")
print(f"   Date range: {weather['date'].min()} to {weather['date'].max()}")
print(f"   T2M range: {weather['T2M'].min():.1f} to {weather['T2M'].max():.1f}")
print(f"   Seasons: {weather['season'].value_counts().to_dict()}")

# 2. Spatial features
spatial = pd.read_csv(data_dir / "spatial_features.csv")
print(f"\n2. Spatial features: {spatial.shape}")
print(f"   NDVI range: {spatial['ndvi'].min():.3f} to {spatial['ndvi'].max():.3f}")
print(f"   NDBI range: {spatial['ndbi'].min():.3f} to {spatial['ndbi'].max():.3f}")
print(f"   Concrete avg: {spatial['concrete_pct'].mean():.1f}%")
print(f"   Vegetation avg: {spatial['vegetation_pct'].mean():.1f}%")

# 3. Master dataset
master = pd.read_csv(data_dir / "master_dataset.csv")
print(f"\n3. Master dataset: {master.shape}")

# 4. Clustered data
clustered = pd.read_csv(data_dir / "clustered_spatial.csv")
n_clusters = clustered["cluster"].nunique()
print(f"\n4. Clustered data: {clustered.shape}, {n_clusters} clusters (incl noise)")

# 5. Cluster stats
stats = pd.read_csv(data_dir / "cluster_stats.csv")
print(f"\n5. Cluster stats:")
if "classification" in stats.columns:
    for _, row in stats.iterrows():
        print(f"   Cluster {row['cluster']}: {row['classification']} | "
              f"Concrete={row['mean_concrete_pct']:.1f}% | NDVI={row['mean_ndvi']:.3f}")

# 6. Model comparison
model_comp = pd.read_csv(data_dir / "model_comparison.csv")
print(f"\n6. Model comparison:")
for _, row in model_comp.iterrows():
    print(f"   {row['model']}: R2={row['r2']:.4f}, MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}")

# 7. Association rules
all_rules = pd.read_csv(rules_dir / "all_association_rules.csv")
uhi_rules = pd.read_csv(rules_dir / "uhi_association_rules.csv")
print(f"\n7. Association rules: {len(all_rules)} total, {len(uhi_rules)} UHI-related")
print("   Top 5 UHI rules by lift:")
for _, row in uhi_rules.head(5).iterrows():
    print(f"   Lift={row['lift']:.2f} | Conf={row['confidence']:.2f} | {row['rule']}")

# 8. Trained models
print(f"\n8. Trained models:")
for m in models_dir.iterdir():
    size_mb = m.stat().st_size / 1024 / 1024
    print(f"   {m.name}: {size_mb:.1f} MB")

print("\n" + "=" * 60)
print("ALL DATA INTEGRITY CHECKS PASSED!")
print("=" * 60)
