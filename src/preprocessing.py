"""
Data Preprocessing Module for UHI Prediction System
=====================================================
Handles:
  1. NASA POWER CSV loading & cleaning
  2. Real Landsat satellite imagery processing (NDVI, NDBI, LST)
  3. Feature engineering & master DataFrame creation
  4. Spatio-temporal data fusion (weather × satellite)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NASA_POWER_CSV, NASA_POWER_HEADER_LINES, WEATHER_PARAMS,
    DELHI_BBOX, RANDOM_SEED, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    LANDSAT_DIR
)
from src.utils import (
    setup_logger, set_seed, add_date_column, replace_sentinel_values,
    check_missing_values, generate_delhi_grid, save_dataframe, ensure_dir
)

logger = setup_logger("preprocessing")


# ══════════════════════════════════════════════
# 1. NASA POWER DATA LOADING
# ══════════════════════════════════════════════

def load_nasa_power_csv(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load NASA POWER daily weather data from CSV.
    
    The CSV has a multi-line header block (lines starting with '-BEGIN HEADER-')
    followed by actual data. We skip the header lines.
    
    Args:
        csv_path: Path to the CSV file. Defaults to config path.
    
    Returns:
        Cleaned DataFrame with date column added.
    """
    csv_path = csv_path or NASA_POWER_CSV
    
    if not csv_path.exists():
        logger.warning(f"NASA POWER CSV not found at {csv_path}")
        logger.info("Generating synthetic weather data instead...")
        return generate_synthetic_weather_data()
    
    logger.info(f"Loading NASA POWER data from {csv_path}")
    
    # Read CSV, skipping the header block
    df = pd.read_csv(csv_path, skiprows=NASA_POWER_HEADER_LINES)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Replace sentinel values (-999) with NaN
    df = replace_sentinel_values(df)
    
    # Add date column from YEAR + DOY
    df = add_date_column(df)
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Log data summary
    logger.info(f"Loaded {len(df)} daily records from {df['date'].min()} to {df['date'].max()}")
    
    # Check for missing values
    missing = check_missing_values(df, threshold=0.05)
    if missing:
        logger.warning(f"Columns with >5% missing: {missing}")
        # Forward-fill then backward-fill for time series continuity
        df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df


# ══════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════

def compute_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived thermal features from raw weather data.
    
    New features:
        - temp_range: Daily temperature range (T2M_MAX - T2M_MIN)
        - uhi_proxy: TS - T2M (Earth skin temp minus air temp → UHI indicator)
        - heat_stress_index: Combined heat + humidity indicator
        - rolling_7d_temp: 7-day rolling mean temperature
        - rolling_30d_temp: 30-day rolling mean temperature
        - temp_anomaly: Deviation from 30-day rolling mean
    """
    df = df.copy()
    
    # Daily temperature range
    df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]
    
    # UHI proxy: Surface temperature - Air temperature
    # Positive = surface warmer than air (urban heat island effect)
    df["uhi_proxy"] = df["TS"] - df["T2M"]
    
    # Heat stress index (simplified)
    # Combines temperature and humidity
    df["heat_stress_index"] = df["T2M"] + 0.5 * (df["RH2M"] / 100) * df["T2M"]
    
    # Rolling averages
    df["rolling_7d_temp"] = df["T2M"].rolling(window=7, min_periods=1).mean()
    df["rolling_30d_temp"] = df["T2M"].rolling(window=30, min_periods=1).mean()
    
    # Temperature anomaly (deviation from 30-day mean)
    df["temp_anomaly"] = df["T2M"] - df["rolling_30d_temp"]
    
    # Solar radiation features
    df["net_radiation"] = df["ALLSKY_SFC_SW_DWN"] - df["ALLSKY_SFC_LW_DWN"].abs() * 0.3
    
    # Precipitation lag features
    df["precip_7d_sum"] = df["PRECTOTCORR"].rolling(window=7, min_periods=1).sum()
    df["dry_days_streak"] = (df["PRECTOTCORR"] < 0.1).astype(int).groupby(
        (df["PRECTOTCORR"] >= 0.1).cumsum()
    ).cumsum()
    
    logger.info(f"Computed {6} thermal features")
    return df


def compute_seasonal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encoding for day of year (captures seasonality)."""
    df = df.copy()
    
    df["doy_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    
    return df


# ══════════════════════════════════════════════
# 3. SATELLITE INDEX COMPUTATION
# ══════════════════════════════════════════════

def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Values range from -1 to +1:
      - High (+0.6 to +1): Dense vegetation
      - Medium (+0.2 to +0.6): Sparse vegetation
      - Low (-1 to +0.2): Water, bare soil, urban
    
    Args:
        nir_band: Near-infrared band array (Landsat B5)
        red_band: Red band array (Landsat B4)
    
    Returns:
        NDVI array
    """
    # Avoid division by zero
    denominator = nir_band.astype(float) + red_band.astype(float)
    denominator[denominator == 0] = np.nan
    
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / denominator
    
    return np.clip(ndvi, -1, 1)


def compute_ndbi(swir_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Built-up Index.
    
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    
    Positive NDBI → Built-up / urban areas
    Negative NDBI → Vegetation
    
    Args:
        swir_band: Short-wave infrared band (Landsat B6)
        nir_band: Near-infrared band (Landsat B5)
    
    Returns:
        NDBI array
    """
    denominator = swir_band.astype(float) + nir_band.astype(float)
    denominator[denominator == 0] = np.nan
    
    ndbi = (swir_band.astype(float) - nir_band.astype(float)) / denominator
    
    return np.clip(ndbi, -1, 1)


def compute_lst_from_thermal(thermal_band: np.ndarray, 
                               emissivity: float = 0.95) -> np.ndarray:
    """
    Estimate Land Surface Temperature from Landsat thermal band (B10).
    
    Simplified conversion from DN to brightness temperature to LST.
    For Landsat 8 Band 10 (TIRS):
        BT = K2 / ln(K1/Lλ + 1)
        LST = BT / (1 + (λ × BT / ρ) × ln(ε))
    
    Where K1=774.89, K2=1321.08 for Landsat 8 Band 10
    
    Args:
        thermal_band: Thermal band array (calibrated radiance)
        emissivity: Surface emissivity (default 0.95 for urban)
    
    Returns:
        LST array in Celsius
    """
    K1 = 774.89
    K2 = 1321.08
    
    # Avoid log of zero/negative
    thermal_band = np.where(thermal_band > 0, thermal_band, np.nan)
    
    # Brightness temperature in Kelvin
    bt = K2 / np.log(K1 / thermal_band + 1)
    
    # LST correction using emissivity
    wavelength = 10.8e-6  # Band 10 central wavelength in meters
    rho = 1.438e-2        # h × c / σ (Planck's constant × speed of light / Boltzmann)
    
    lst_kelvin = bt / (1 + (wavelength * bt / rho) * np.log(emissivity))
    lst_celsius = lst_kelvin - 273.15
    
    return lst_celsius


# ══════════════════════════════════════════════
# 4. SPATIAL FEATURE GENERATION (REAL LANDSAT)
# ══════════════════════════════════════════════

def generate_spatial_features_from_landsat() -> pd.DataFrame:
    """
    Generate spatial land cover features from real Landsat satellite imagery.
    
    Reads all Landsat scenes in data/raw/landsat images/,
    computes NDVI, NDBI, LST per grid point, and derives land cover
    percentages from spectral index thresholds.
    
    Returns:
        DataFrame with real satellite-derived features per grid point.
    """
    from src.landsat_processor import run_landsat_pipeline
    
    multi_temporal, spatial_features = run_landsat_pipeline(save_outputs=True)
    
    if len(spatial_features) == 0:
        logger.error("Landsat processing returned empty data!")
        raise RuntimeError(
            "Failed to process Landsat data. Check that GeoTIFF files exist in "
            f"{LANDSAT_DIR} and that rasterio is installed."
        )
    
    logger.info(f"Generated REAL spatial features for {len(spatial_features)} grid points")
    logger.info(f"  NDVI range: [{spatial_features['ndvi'].min():.3f}, {spatial_features['ndvi'].max():.3f}]")
    logger.info(f"  NDBI range: [{spatial_features['ndbi'].min():.3f}, {spatial_features['ndbi'].max():.3f}]")
    logger.info(f"  LST  range: [{spatial_features['lst'].min():.1f}, {spatial_features['lst'].max():.1f}]°C")
    
    return spatial_features


# ══════════════════════════════════════════════
# 5. SYNTHETIC DATA FOR DEMO
# ══════════════════════════════════════════════

def generate_synthetic_weather_data(start_year: int = 2015, 
                                      end_year: int = 2023,
                                      seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate realistic synthetic weather data for Delhi.
    
    Mimics NASA POWER data patterns with:
      - Seasonal temperature cycles
      - Monsoon rainfall patterns
      - Realistic humidity correlations
      - Year-over-year warming trend
    """
    np.random.seed(seed)
    
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    n = len(dates)
    
    # Day of year for seasonal patterns
    doy = dates.dayofyear.values
    years = dates.year.values
    
    # Base temperature with seasonal cycle (Delhi pattern)
    # Hot: May-June (40°C+), Cold: Dec-Jan (8°C)
    seasonal_temp = 24 + 14 * np.sin(2 * np.pi * (doy - 120) / 365)
    
    # Year-over-year warming trend (+0.03°C/year — UHI effect)
    warming_trend = (years - start_year) * 0.03
    
    # Base temperature
    t2m = seasonal_temp + warming_trend + np.random.normal(0, 2, n)
    
    # Other parameters
    t2m_max = t2m + np.random.uniform(4, 10, n)
    t2m_min = t2m - np.random.uniform(4, 10, n)
    ts = t2m + np.random.uniform(-2, 4, n)  # Skin temp slightly higher
    
    # Humidity — inversely related to temperature, high in monsoon
    monsoon_mask = (doy >= 150) & (doy <= 270)
    rh2m = np.where(monsoon_mask,
                     np.clip(65 + np.random.normal(0, 10, n), 40, 95),
                     np.clip(40 - (t2m - 25) * 0.8 + np.random.normal(0, 8, n), 15, 85))
    
    # Wind speed
    ws10m = np.clip(np.random.exponential(2.5, n) + 1, 0.5, 12)
    
    # Precipitation — concentrated in monsoon
    precip = np.where(monsoon_mask,
                       np.random.exponential(8, n),
                       np.random.exponential(0.3, n))
    precip = np.clip(precip, 0, 80)
    
    # Solar radiation
    solar = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365) + np.random.normal(0, 3, n)
    solar = np.where(monsoon_mask, solar * 0.6, solar)
    solar = np.clip(solar, 1, 30)
    
    df = pd.DataFrame({
        "YEAR": years,
        "DOY": doy,
        "T2M": np.round(t2m, 2),
        "T2M_MAX": np.round(t2m_max, 2),
        "T2M_MIN": np.round(t2m_min, 2),
        "TS": np.round(ts, 2),
        "T2MWET": np.round(t2m * 0.7 + np.random.normal(0, 1, n), 2),
        "RH2M": np.round(rh2m, 2),
        "QV2M": np.round(rh2m * 0.12, 2),
        "T2MDEW": np.round(t2m - (100 - rh2m) / 5, 2),
        "GWETTOP": np.round(np.clip(rh2m / 200 + np.random.normal(0, 0.05, n), 0.05, 0.9), 2),
        "WS10M": np.round(ws10m, 2),
        "WD10M": np.round(np.random.uniform(0, 360, n), 1),
        "WS10M_MAX": np.round(ws10m * 1.5 + np.random.uniform(0, 2, n), 2),
        "ALLSKY_SFC_SW_DWN": np.round(solar, 2),
        "ALLSKY_SFC_LW_DWN": np.round(30 + 5 * np.sin(2 * np.pi * doy / 365) + np.random.normal(0, 2, n), 2),
        "ALLSKY_SFC_UV_INDEX": np.round(np.clip(solar / 15 + np.random.normal(0, 0.2, n), 0.1, 3), 2),
        "PRECTOTCORR": np.round(precip, 2),
        "PS": np.round(98.5 - 0.5 * np.sin(2 * np.pi * doy / 365) + np.random.normal(0, 0.3, n), 2),
        "date": dates,
    })
    
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(lambda m: 
        "Winter" if m in [12, 1, 2] else
        "Summer" if m in [3, 4, 5] else
        "Monsoon" if m in [6, 7, 8, 9] else
        "Post-Monsoon"
    )
    
    logger.info(f"Generated synthetic weather data: {len(df)} records ({start_year}-{end_year})")
    return df


def create_master_dataset(weather_df: pd.DataFrame, 
                           spatial_df: pd.DataFrame,
                           multi_temporal_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create the master spatio-temporal feature DataFrame.
    
    Uses real multi-temporal satellite data joined with weather records.
    For each Landsat scene date, finds the closest weather record and
    merges satellite indices (NDVI, NDBI, LST) with weather parameters.
    
    Output columns:
        [date, lat, lon, T2M, TS, uhi_proxy, ndvi, ndbi, lst,
         concrete_pct, vegetation_pct, season, ...]
    
    Args:
        weather_df: Processed weather DataFrame (with thermal features)
        spatial_df: Spatial grid DataFrame (with time-averaged land cover features)
        multi_temporal_df: Per-date satellite observations (from Landsat processor)
    
    Returns:
        Master DataFrame ready for modeling
    """
    # --- Strategy: Join multi-temporal satellite data with weather on date ---
    if multi_temporal_df is not None and len(multi_temporal_df) > 0:
        logger.info("Creating master dataset from real multi-temporal satellite data")
        
        # Ensure dates are datetime
        mt = multi_temporal_df.copy()
        mt["date"] = pd.to_datetime(mt["date"])
        weather = weather_df.copy()
        weather["date"] = pd.to_datetime(weather["date"])
        
        # For each satellite observation date, merge with nearest weather record
        # Use merge_asof for nearest-date matching
        mt = mt.sort_values("date")
        weather = weather.sort_values("date")
        
        # Select key weather columns to avoid DataFrame explosion
        weather_cols = ["date", "T2M", "T2M_MAX", "T2M_MIN", "TS", "RH2M", "WS10M",
                        "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "season", "YEAR", "month"]
        weather_cols = [c for c in weather_cols if c in weather.columns]
        weather_subset = weather[weather_cols].copy()
        
        # Add thermal features to weather subset
        if "T2M_MAX" in weather_subset.columns and "T2M_MIN" in weather_subset.columns:
            weather_subset["temp_range"] = weather_subset["T2M_MAX"] - weather_subset["T2M_MIN"]
        if "TS" in weather_subset.columns and "T2M" in weather_subset.columns:
            weather_subset["uhi_proxy"] = weather_subset["TS"] - weather_subset["T2M"]
        
        # Sample spatial points for manageable size
        unique_points = mt.groupby(["lat", "lon"]).size().reset_index(name="count")
        n_sample = min(200, len(unique_points))
        sampled_points = unique_points.sample(n_sample, random_state=RANDOM_SEED)
        
        mt = mt.merge(sampled_points[["lat", "lon"]], on=["lat", "lon"])
        
        # Merge satellite data with weather using nearest date
        master = pd.merge_asof(
            mt, weather_subset,
            on="date", direction="nearest", tolerance=pd.Timedelta("3D")
        )
        
        # Add land cover percentages from the time-averaged spatial features
        # Match by nearest grid point
        if len(spatial_df) > 0:
            spatial_cols = ["lat", "lon", "concrete_pct", "vegetation_pct", 
                           "water_pct", "asphalt_pct"]
            spatial_cols = [c for c in spatial_cols if c in spatial_df.columns]
            
            if len(spatial_cols) > 2:  # lat, lon + at least one feature
                # Round for matching
                master["lat_r"] = master["lat"].round(4)
                master["lon_r"] = master["lon"].round(4)
                sp = spatial_df.copy()
                sp["lat_r"] = sp["lat"].round(4)
                sp["lon_r"] = sp["lon"].round(4)
                
                lc_cols = [c for c in spatial_cols if c not in ["lat", "lon"]]
                master = master.merge(
                    sp[["lat_r", "lon_r"] + lc_cols],
                    on=["lat_r", "lon_r"], how="left"
                )
                master = master.drop(columns=["lat_r", "lon_r"])
        
    else:
        # Fallback: cross-join monthly weather × spatial sample
        logger.info("Creating master dataset from time-averaged spatial features")
        
        n_spatial_samples = min(100, len(spatial_df))
        spatial_sample = spatial_df.sample(n_spatial_samples, random_state=RANDOM_SEED)
        
        # Monthly weather aggregates
        weather_copy = weather_df.copy()
        monthly_weather = weather_copy.groupby([weather_copy["date"].dt.to_period("M")]).agg({
            "T2M": "mean", "T2M_MAX": "max", "T2M_MIN": "min",
            "TS": "mean", "RH2M": "mean", "WS10M": "mean",
            "PRECTOTCORR": "sum", "ALLSKY_SFC_SW_DWN": "mean",
            "season": "first", "YEAR": "first", "month": "first",
        }).reset_index(drop=True)
        
        monthly_weather["temp_range"] = monthly_weather["T2M_MAX"] - monthly_weather["T2M_MIN"]
        monthly_weather["uhi_proxy"] = monthly_weather["TS"] - monthly_weather["T2M"]
        
        master = monthly_weather.merge(spatial_sample, how="cross")
    
    # Add spatial-temporal interaction features
    # Temperature adjusted by real land cover composition
    if "concrete_pct" in master.columns and "T2M" in master.columns:
        master["adjusted_temp"] = (
            master["T2M"] 
            + master["concrete_pct"] / 100 * 2.5  # Concrete adds heat
            - master.get("vegetation_pct", pd.Series(0)) / 100 * 1.5  # Vegetation cools
            + master.get("uhi_proxy", pd.Series(0))
        )
    
    # Drop any NaN-heavy rows
    master = master.dropna(subset=["T2M"], errors="ignore")
    
    logger.info(f"Created master dataset: {master.shape[0]} rows × {master.shape[1]} columns")
    return master


# ══════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_preprocessing_pipeline(use_real_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing pipeline using real data.
    
    Steps:
      1. Load NASA POWER weather CSV (real data)
      2. Compute thermal/seasonal features
      3. Process real Landsat GeoTIFF imagery for spatial features
      4. Create master spatio-temporal dataset
      5. Save all processed data
    
    Returns:
        Tuple of (weather_df, spatial_df, master_df)
    """
    set_seed(RANDOM_SEED)
    ensure_dir(PROCESSED_DATA_DIR)
    
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load REAL weather data from NASA POWER CSV
    if not NASA_POWER_CSV.exists():
        raise FileNotFoundError(
            f"NASA POWER CSV not found at {NASA_POWER_CSV}. "
            "Download from https://power.larc.nasa.gov/data-access-viewer/"
        )
    weather_df = load_nasa_power_csv()
    
    # Step 2: Feature engineering
    weather_df = compute_thermal_features(weather_df)
    weather_df = compute_seasonal_encoding(weather_df)
    
    # Step 3: Process REAL Landsat satellite imagery
    multi_temporal_df = None
    if LANDSAT_DIR.exists() and any(LANDSAT_DIR.iterdir()):
        logger.info("Processing real Landsat satellite imagery...")
        spatial_df = generate_spatial_features_from_landsat()
        
        # Also load multi-temporal data for master dataset
        mt_path = PROCESSED_DATA_DIR / "landsat_multi_temporal.csv"
        if mt_path.exists():
            multi_temporal_df = pd.read_csv(mt_path, parse_dates=["date"])
            logger.info(f"Loaded multi-temporal data: {multi_temporal_df.shape}")
    else:
        raise FileNotFoundError(
            f"Landsat imagery not found at {LANDSAT_DIR}. "
            "Download Landsat 8 scenes from https://earthexplorer.usgs.gov/"
        )
    
    # Step 4: Create master spatio-temporal dataset
    master_df = create_master_dataset(weather_df, spatial_df, multi_temporal_df)
    
    # Step 5: Save processed data
    save_dataframe(weather_df, PROCESSED_DATA_DIR / "weather_processed.csv")
    save_dataframe(spatial_df, PROCESSED_DATA_DIR / "spatial_features.csv")
    save_dataframe(master_df, PROCESSED_DATA_DIR / "master_dataset.csv")
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info(f"  Weather: {weather_df.shape}")
    logger.info(f"  Spatial: {spatial_df.shape}")
    logger.info(f"  Master:  {master_df.shape}")
    logger.info("=" * 60)
    
    return weather_df, spatial_df, master_df


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    weather, spatial, master = run_preprocessing_pipeline()
    print(f"\nWeather data shape: {weather.shape}")
    print(f"Spatial data shape: {spatial.shape}")
    print(f"Master dataset shape: {master.shape}")
    print(f"\nWeather columns: {list(weather.columns)}")
    print(f"\nSample weather data:")
    print(weather.head())
