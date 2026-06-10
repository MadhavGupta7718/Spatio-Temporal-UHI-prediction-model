"""
Central Configuration for UHI Prediction System
================================================
All paths, hyperparameters, and constants in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = DATA_DIR / "models"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RULES_DIR = OUTPUTS_DIR / "rules"

# ──────────────────────────────────────────────
# STUDY AREA — Delhi, India
# ──────────────────────────────────────────────
DELHI_LAT = 28.6139
DELHI_LON = 77.2090
DELHI_BBOX = {
    "north": 28.88,
    "south": 28.40,
    "east": 77.35,
    "west": 76.84
}

# NASA POWER data location (from downloaded CSV)
NASA_POWER_LAT = 28.7978
NASA_POWER_LON = 77.0668

# ──────────────────────────────────────────────
# NASA POWER CSV SETTINGS
# ──────────────────────────────────────────────
NASA_POWER_CSV = RAW_DATA_DIR / "POWER_Point_Daily_20150101_20251231_028d80N_077d07E_UTC.csv"
NASA_POWER_HEADER_LINES = 25  # Lines to skip (header block ends at line 25)

# ──────────────────────────────────────────────
# LANDSAT SATELLITE IMAGERY
# ──────────────────────────────────────────────
LANDSAT_DIR = RAW_DATA_DIR / "landsat images"
GADM_SHAPEFILE = RAW_DATA_DIR / "gadm41_IND_shp" / "gadm41_IND_1.shp"
DELHI_STATE_NAME = "Delhi"

# Landsat 8 Collection 2 Level-2 scale factors
# Surface Reflectance (B2-B7): DN * 0.0000275 + (-0.2)
LANDSAT_SR_MULT = 0.0000275
LANDSAT_SR_ADD = -0.2

# Surface Temperature (B10): DN * 0.00341802 + 149.0 (result in Kelvin)
LANDSAT_ST_MULT = 0.00341802
LANDSAT_ST_ADD = 149.0

# Band file patterns for Landsat 8 Collection 2
LANDSAT_BANDS = {
    "red": "SR_B4",     # Band 4 — Red (surface reflectance)
    "nir": "SR_B5",     # Band 5 — Near-Infrared (surface reflectance)
    "swir": "SR_B6",    # Band 6 — Short-Wave Infrared (surface reflectance)
    "thermal": "ST_B10", # Band 10 — Thermal (surface temperature)
}

# Grid resolution for spatial sampling (degrees)
SPATIAL_GRID_RESOLUTION = 0.005  # ~550m at Delhi's latitude

# Weather parameters from NASA POWER
WEATHER_PARAMS = [
    "T2M",              # Temperature at 2m (°C)
    "T2M_MAX",          # Max temperature at 2m (°C)
    "T2M_MIN",          # Min temperature at 2m (°C)
    "TS",               # Earth skin temperature (°C)
    "T2MWET",           # Wet bulb temperature at 2m (°C)
    "RH2M",             # Relative humidity at 2m (%)
    "QV2M",             # Specific humidity at 2m (g/kg)
    "T2MDEW",           # Dew/frost point at 2m (°C)
    "GWETTOP",          # Surface soil wetness (0-1)
    "WS10M",            # Wind speed at 10m (m/s)
    "WD10M",            # Wind direction at 10m (degrees)
    "WS10M_MAX",        # Max wind speed at 10m (m/s)
    "ALLSKY_SFC_SW_DWN", # Solar irradiance (MJ/m²/day)
    "ALLSKY_SFC_LW_DWN", # Longwave irradiance (MJ/m²/day)
    "ALLSKY_SFC_UV_INDEX",# UV index
    "PRECTOTCORR",      # Precipitation (mm/day)
    "PS",               # Surface pressure (kPa)
]

# ──────────────────────────────────────────────
# CNN — LAND COVER CLASSIFICATION
# ──────────────────────────────────────────────
CNN_CONFIG = {
    "model_name": "resnet50",
    "num_classes": 5,
    "class_names": [
        "Dense_Concrete",
        "Asphalt_Roads",
        "Dense_Vegetation",
        "Sparse_Vegetation",
        "Water_Bodies"
    ],
    "image_size": 64,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "freeze_layers": True,       # Freeze early ResNet layers
    "unfreeze_after_epoch": 10,  # Unfreeze all layers after this epoch
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "pretrained": True,
}

# EuroSAT mapping to our 5 classes
EUROSAT_CLASS_MAP = {
    # EuroSAT class → Our class index
    "Industrial": 0,           # → Dense_Concrete
    "Residential": 0,          # → Dense_Concrete
    "Highway": 1,              # → Asphalt_Roads
    "AnnualCrop": 3,           # → Sparse_Vegetation
    "Forest": 2,               # → Dense_Vegetation
    "HerbaceousVegetation": 3, # → Sparse_Vegetation
    "PermanentCrop": 2,        # → Dense_Vegetation
    "Pasture": 3,              # → Sparse_Vegetation
    "River": 4,                # → Water_Bodies
    "SeaLake": 4,              # → Water_Bodies
}

# ──────────────────────────────────────────────
# DBSCAN CLUSTERING
# ──────────────────────────────────────────────
DBSCAN_CONFIG = {
    "eps": 0.01,            # ~1.1 km radius at Delhi's latitude
    "min_samples": 5,
    "metric": "haversine",
}

# ──────────────────────────────────────────────
# ASSOCIATION RULE MINING
# ──────────────────────────────────────────────
MINING_CONFIG = {
    "min_support": 0.08,
    "min_confidence": 0.6,
    "min_lift": 1.2,
    "algorithm": "fpgrowth",  # "apriori" or "fpgrowth"
    # Discretization bins
    "bins": {
        "temperature": ["Low", "Medium", "High", "Extreme"],
        "humidity": ["Dry", "Moderate", "Humid", "Very_Humid"],
        "wind_speed": ["Calm", "Light", "Moderate", "Strong"],
        "vegetation": ["Bare", "Low", "Medium", "High"],
        "built_up": ["Low", "Medium", "High", "Very_High"],
    }
}

# ──────────────────────────────────────────────
# PREDICTION MODEL (LSTM + XGBoost)
# ──────────────────────────────────────────────
PREDICTOR_CONFIG = {
    # Time series settings
    "lookback_days": 30,
    "forecast_horizons": [365, 1095, 1825],  # +1yr, +3yr, +5yr
    "train_ratio": 0.8,
    
    # XGBoost
    "xgb_params": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
    },
    
    # LSTM
    "lstm_params": {
        "hidden_size_1": 64,
        "hidden_size_2": 32,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "patience": 15,  # Early stopping patience
    },
}

# ──────────────────────────────────────────────
# RANDOM SEED
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# STREAMLIT DASHBOARD
# ──────────────────────────────────────────────
DASHBOARD_CONFIG = {
    "page_title": "Delhi UHI Prediction System",
    "page_icon": "🌡️",
    "layout": "wide",
    "theme": "dark",
}
