"""Quick test of the preprocessing pipeline."""
import sys
sys.path.insert(0, r'd:\Spatio-Temporal UHI prediction model')

from src.preprocessing import load_nasa_power_csv, compute_thermal_features, run_preprocessing_pipeline

# Test 1: Load raw CSV
print("=" * 50)
print("TEST 1: Load NASA POWER CSV")
print("=" * 50)
df = load_nasa_power_csv()
print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Key columns: {[c for c in df.columns if c in ['T2M','TS','RH2M','WS10M','PRECTOTCORR']]}")

# Test 2: Feature engineering
print("\n" + "=" * 50)
print("TEST 2: Compute Thermal Features")
print("=" * 50)
df2 = compute_thermal_features(df)
new_cols = [c for c in df2.columns if c not in df.columns]
print(f"  New features: {new_cols}")
print(f"  UHI proxy range: {df2['uhi_proxy'].min():.2f} to {df2['uhi_proxy'].max():.2f}")

# Test 3: Full pipeline
print("\n" + "=" * 50)
print("TEST 3: Full Preprocessing Pipeline")
print("=" * 50)
weather, spatial, master = run_preprocessing_pipeline()
print(f"  Weather: {weather.shape}")
print(f"  Spatial: {spatial.shape}")
print(f"  Master:  {master.shape}")

print("\n✅ ALL PREPROCESSING TESTS PASSED!")
