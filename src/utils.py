"""
Utility Functions for UHI Prediction System
=============================================
Shared helpers used across all modules.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a standardized logger for any module."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ──────────────────────────────────────────────
# DATE HELPERS
# ──────────────────────────────────────────────
def doy_to_date(year: int, doy: int) -> datetime:
    """Convert year + day-of-year to a datetime object."""
    return datetime.strptime(f"{year}-{doy}", "%Y-%j")


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a proper 'date' column from YEAR + DOY columns.
    Expects columns: YEAR, DOY
    """
    df = df.copy()
    df["date"] = df.apply(lambda row: doy_to_date(int(row["YEAR"]), int(row["DOY"])), axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(get_season)
    return df


def get_season(month: int) -> str:
    """Map month number to Indian season name."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


# ──────────────────────────────────────────────
# DATA VALIDATION
# ──────────────────────────────────────────────
def check_missing_values(df: pd.DataFrame, threshold: float = 0.1) -> dict:
    """
    Check for missing values in each column.
    Returns dict of columns exceeding the threshold.
    """
    missing = df.isnull().sum() / len(df)
    problematic = {col: pct for col, pct in missing.items() if pct > threshold}
    return problematic


def replace_sentinel_values(df: pd.DataFrame, sentinel: float = -999.0) -> pd.DataFrame:
    """Replace NASA POWER sentinel values (-999) with NaN."""
    return df.replace(sentinel, np.nan)


# ──────────────────────────────────────────────
# DISCRETIZATION (for association mining)
# ──────────────────────────────────────────────
def discretize_column(series: pd.Series, bins: list, labels: list) -> pd.Series:
    """
    Discretize a continuous column into categorical bins.
    
    Args:
        series: Continuous data column
        bins: List of bin edges (e.g., [0, 20, 30, 40, 50])
        labels: List of labels (e.g., ["Low", "Medium", "High", "Extreme"])
    
    Returns:
        Categorical series
    """
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


# ──────────────────────────────────────────────
# SPATIAL HELPERS
# ──────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points (in km).
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def generate_delhi_grid(lat_range=(28.40, 28.88), lon_range=(76.84, 77.35), 
                         resolution=0.01):
    """
    Generate a regular grid of lat/lon points covering Delhi.
    
    Args:
        lat_range: (south, north) latitude bounds
        lon_range: (west, east) longitude bounds
        resolution: Grid spacing in degrees (~1.1 km at Delhi's latitude)
    
    Returns:
        DataFrame with lat, lon columns
    """
    lats = np.arange(lat_range[0], lat_range[1], resolution)
    lons = np.arange(lon_range[0], lon_range[1], resolution)
    
    grid = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
    
    return pd.DataFrame(grid, columns=["lat", "lon"])


# ──────────────────────────────────────────────
# FILE I/O HELPERS
# ──────────────────────────────────────────────
def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False):
    """Save DataFrame to CSV or Parquet based on extension."""
    ensure_dir(path.parent)
    
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=index)
    elif suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load DataFrame from CSV or Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")
