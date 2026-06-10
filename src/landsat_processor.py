"""
Landsat GeoTIFF Processing Module
===================================
Reads real Landsat 8 Collection 2 Level-2 satellite imagery,
clips to Delhi boundary, and computes spectral indices.

Handles:
  1. Loading multi-band GeoTIFF scenes
  2. Clipping rasters to Delhi NCT boundary (GADM shapefile)
  3. Computing NDVI, NDBI, and LST from real bands
  4. Extracting values at grid points for spatial feature creation
  5. Building multi-temporal spatial DataFrame from all scenes
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LANDSAT_DIR, GADM_SHAPEFILE, DELHI_STATE_NAME,
    LANDSAT_SR_MULT, LANDSAT_SR_ADD, LANDSAT_ST_MULT, LANDSAT_ST_ADD,
    LANDSAT_BANDS, DELHI_BBOX, SPATIAL_GRID_RESOLUTION, RANDOM_SEED,
    PROCESSED_DATA_DIR
)
from src.utils import setup_logger, set_seed, save_dataframe, ensure_dir

logger = setup_logger("landsat_processor")


# ══════════════════════════════════════════════
# 1. DELHI BOUNDARY LOADING
# ══════════════════════════════════════════════

def load_delhi_boundary(shapefile_path: Path = None) -> gpd.GeoDataFrame:
    """
    Load Delhi NCT boundary from GADM Level-1 shapefile.

    Returns:
        GeoDataFrame with Delhi boundary polygon
    """
    shapefile_path = shapefile_path or GADM_SHAPEFILE

    if not shapefile_path.exists():
        logger.error(f"GADM shapefile not found: {shapefile_path}")
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)

    # Filter for Delhi — try NAME_1 column (state/UT name)
    name_col = None
    for col in ["NAME_1", "name_1", "NAME_2", "name_2"]:
        if col in gdf.columns:
            name_col = col
            break

    if name_col is None:
        logger.warning("Could not find name column in shapefile, using bounding box instead")
        return None

    # Search for Delhi (case-insensitive, partial match)
    delhi_mask = gdf[name_col].str.contains("Delhi|NCT", case=False, na=False)

    if not delhi_mask.any():
        logger.warning(f"Delhi not found in {name_col}. Available: {gdf[name_col].unique()[:10]}")
        logger.info("Falling back to bounding box clip")
        return None

    delhi = gdf[delhi_mask].copy()
    logger.info(f"Loaded Delhi boundary: {len(delhi)} polygon(s), CRS={delhi.crs}")
    return delhi


def get_delhi_geometry(delhi_gdf: gpd.GeoDataFrame = None, target_crs=None):
    """
    Get Delhi boundary as a list of geometry dicts for rasterio masking.

    Args:
        delhi_gdf: Pre-loaded Delhi GeoDataFrame (or None to load)
        target_crs: If provided, reproject to this CRS (e.g., the Landsat raster CRS)

    Returns:
        List of GeoJSON-like geometry dicts, or None if unavailable
    """
    if delhi_gdf is None:
        delhi_gdf = load_delhi_boundary()

    if delhi_gdf is None or len(delhi_gdf) == 0:
        return None

    # Reproject to target CRS if provided (e.g., Landsat UTM)
    if target_crs is not None:
        delhi_gdf = delhi_gdf.to_crs(target_crs)
        logger.info(f"Reprojected Delhi boundary to {target_crs}")
    elif delhi_gdf.crs and delhi_gdf.crs.to_epsg() != 4326:
        delhi_gdf = delhi_gdf.to_crs(epsg=4326)

    # Convert to list of geometry dicts
    from shapely.geometry import mapping
    geometries = [mapping(geom) for geom in delhi_gdf.geometry]
    return geometries


# ══════════════════════════════════════════════
# 2. LANDSAT SCENE LOADING
# ══════════════════════════════════════════════

def find_band_file(scene_dir: Path, band_key: str) -> Optional[Path]:
    """
    Find the TIF file matching a band pattern in a scene directory.

    Args:
        scene_dir: Path to scene directory (e.g., data/raw/landsat images/2020-02-10)
        band_key: Band pattern from LANDSAT_BANDS (e.g., "SR_B4")

    Returns:
        Path to the matching TIF file, or None
    """
    pattern = f"*{band_key}*"
    matches = list(scene_dir.glob(f"{pattern}.TIF")) + list(scene_dir.glob(f"{pattern}.tif"))

    if not matches:
        logger.warning(f"No file matching '{pattern}' in {scene_dir}")
        return None

    return matches[0]


def load_band(filepath: Path, scale_mult: float, scale_add: float,
              is_thermal: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Load a single Landsat band from GeoTIFF, applying Collection 2 scale factors.

    Args:
        filepath: Path to .TIF file
        scale_mult: Multiplicative scale factor
        scale_add: Additive scale factor
        is_thermal: If True, applies Kelvin-range clamping

    Returns:
        Tuple of (scaled_array, rasterio_metadata)
    """
    with rasterio.open(filepath) as src:
        band = src.read(1).astype(np.float32)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "nodata": src.nodata,
        }

    # Mark fill/nodata as NaN before scaling
    nodata_val = meta.get("nodata")
    if nodata_val is not None:
        band[band == nodata_val] = np.nan

    # Also mark zero values as nodata (Landsat uses 0 for fill)
    band[band == 0] = np.nan

    # Apply Collection 2 Level-2 scale factors
    scaled = band * scale_mult + scale_add

    # Clamp to valid range based on band type
    if is_thermal:
        scaled[(scaled < 200) | (scaled > 400)] = np.nan
    else:
        scaled[(scaled < -1) | (scaled > 2)] = np.nan

    return scaled, meta


def load_landsat_scene(scene_dir: Path) -> Dict[str, Tuple[np.ndarray, dict]]:
    """
    Load all required bands from a Landsat scene directory.

    Args:
        scene_dir: Path to scene directory

    Returns:
        Dict mapping band name to (array, metadata) tuples
    """
    bands = {}

    for band_name, band_pattern in LANDSAT_BANDS.items():
        filepath = find_band_file(scene_dir, band_pattern)
        if filepath is None:
            logger.warning(f"Missing {band_name} band in {scene_dir.name}")
            continue

        if band_name == "thermal":
            scale_mult, scale_add = LANDSAT_ST_MULT, LANDSAT_ST_ADD
        else:
            scale_mult, scale_add = LANDSAT_SR_MULT, LANDSAT_SR_ADD

        array, meta = load_band(filepath, scale_mult, scale_add,
                                is_thermal=(band_name == "thermal"))
        bands[band_name] = (array, meta)

        logger.info(
            f"  Loaded {band_name} ({band_pattern}): shape={array.shape}, "
            f"valid_pix={np.count_nonzero(~np.isnan(array)):,}, "
            f"range=[{np.nanmin(array):.4f}, {np.nanmax(array):.4f}]"
        )

    return bands


# ══════════════════════════════════════════════
# 3. RASTER CLIPPING
# ══════════════════════════════════════════════

def clip_band_to_delhi(filepath: Path, delhi_gdf: gpd.GeoDataFrame,
                       scale_mult: float, scale_add: float,
                       is_thermal: bool = False) -> Tuple[np.ndarray, rasterio.transform.Affine]:
    """
    Clip a Landsat band to Delhi boundary using rasterio.mask.

    Handles CRS mismatch by reprojecting Delhi boundary to match the raster CRS.

    Args:
        filepath: Path to .TIF file
        delhi_gdf: Delhi boundary GeoDataFrame
        scale_mult: Scale factor multiplier
        scale_add: Scale factor additive
        is_thermal: If True, applies Kelvin-range clamping instead of reflectance clamping

    Returns:
        Tuple of (clipped_array, clipped_transform)
    """
    with rasterio.open(filepath) as src:
        # Reproject Delhi boundary to match raster CRS
        raster_crs = src.crs
        delhi_reprojected = delhi_gdf.to_crs(raster_crs)
        from shapely.geometry import mapping
        geometries = [mapping(geom) for geom in delhi_reprojected.geometry]

        # Clip to Delhi boundary
        clipped, clipped_transform = rasterio_mask(
            src, geometries, crop=True, nodata=0, filled=True
        )

    # clipped shape is (1, H, W) — squeeze to (H, W)
    band = clipped[0].astype(np.float32)

    # Mark nodata
    band[band == 0] = np.nan

    # Apply scale factors
    scaled = band * scale_mult + scale_add

    # Clamp to valid range based on band type
    if is_thermal:
        # Thermal band: values are in Kelvin (expect ~200-350K for Earth surface)
        scaled[(scaled < 200) | (scaled > 400)] = np.nan
    else:
        # Surface reflectance: valid range approximately -0.2 to 1.6
        scaled[(scaled < -1) | (scaled > 2)] = np.nan

    return scaled, clipped_transform


def load_and_clip_scene(scene_dir: Path, delhi_gdf: gpd.GeoDataFrame) -> Dict:
    """
    Load a full Landsat scene and clip all bands to Delhi boundary.

    Handles CRS mismatch between the raster (UTM) and shapefile (WGS84)
    by reprojecting the Delhi boundary to match each raster.

    Returns:
        Dict with 'red', 'nir', 'swir', 'thermal' arrays + 'transform' + 'crs'
    """
    result = {}
    transform = None
    raster_crs = None

    for band_name, band_pattern in LANDSAT_BANDS.items():
        filepath = find_band_file(scene_dir, band_pattern)
        if filepath is None:
            continue

        if band_name == "thermal":
            scale_mult, scale_add = LANDSAT_ST_MULT, LANDSAT_ST_ADD
        else:
            scale_mult, scale_add = LANDSAT_SR_MULT, LANDSAT_SR_ADD

        array, clip_transform = clip_band_to_delhi(filepath, delhi_gdf,
                                                    scale_mult, scale_add,
                                                    is_thermal=(band_name == "thermal"))
        result[band_name] = array
        if transform is None:
            transform = clip_transform
        # Get raster CRS for coordinate transforms later
        if raster_crs is None:
            with rasterio.open(filepath) as src:
                raster_crs = src.crs

    result["transform"] = transform
    result["crs"] = raster_crs
    return result


# ══════════════════════════════════════════════
# 4. SPECTRAL INDEX COMPUTATION
# ══════════════════════════════════════════════

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute NDVI = (NIR - Red) / (NIR + Red).

    Values: -1 to +1
      High (+0.6 to +1): Dense vegetation
      Medium (+0.2 to +0.6): Sparse vegetation
      Low (-1 to +0.2): Water, bare soil, urban
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, np.nan)
    return np.clip(ndvi, -1, 1)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute NDBI = (SWIR - NIR) / (SWIR + NIR).

    Positive NDBI → Built-up / urban areas
    Negative NDBI → Vegetation
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = swir + nir
        ndbi = np.where(denom > 0, (swir - nir) / denom, np.nan)
    return np.clip(ndbi, -1, 1)


def compute_lst(thermal_kelvin: np.ndarray) -> np.ndarray:
    """
    Convert thermal band (already in Kelvin from scale factors) to LST in Celsius.

    For Landsat Collection 2 Level-2 ST product, the scale factors
    already produce brightness temperature in Kelvin.
    """
    lst_celsius = thermal_kelvin - 273.15

    # Clamp to realistic range for Delhi (-5°C to 60°C)
    lst_celsius = np.where((lst_celsius > -5) & (lst_celsius < 60), lst_celsius, np.nan)

    return lst_celsius


def process_scene(scene_dir: Path, delhi_gdf: gpd.GeoDataFrame = None) -> Dict:
    """
    Full processing pipeline for a single Landsat scene.

    Steps:
      1. Load and clip bands to Delhi boundary
      2. Compute NDVI, NDBI, LST
      3. Return all arrays and transform

    Returns:
        Dict with keys: ndvi, ndbi, lst, transform, crs, date, shape
    """
    date_str = scene_dir.name  # e.g., "2020-02-10"
    logger.info(f"Processing scene: {date_str}")

    # Load bands (with or without Delhi clipping)
    if delhi_gdf is not None:
        bands = load_and_clip_scene(scene_dir, delhi_gdf)
    else:
        raw_bands = load_landsat_scene(scene_dir)
        bands = {k: v[0] for k, v in raw_bands.items()}
        if raw_bands:
            first_band = list(raw_bands.values())[0]
            bands["transform"] = first_band[1]["transform"]
            bands["crs"] = first_band[1]["crs"]

    # Check we have all required bands
    required = ["red", "nir", "swir", "thermal"]
    missing = [b for b in required if b not in bands]
    if missing:
        logger.error(f"Scene {date_str} missing bands: {missing}")
        return None

    # Compute indices
    ndvi = compute_ndvi(bands["nir"], bands["red"])
    ndbi = compute_ndbi(bands["swir"], bands["nir"])
    lst = compute_lst(bands["thermal"])

    valid_ndvi = np.count_nonzero(~np.isnan(ndvi))
    valid_lst = np.count_nonzero(~np.isnan(lst))

    logger.info(
        f"  NDVI: range=[{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}], valid={valid_ndvi:,} px"
    )
    logger.info(
        f"  NDBI: range=[{np.nanmin(ndbi):.3f}, {np.nanmax(ndbi):.3f}]"
    )
    logger.info(
        f"  LST:  range=[{np.nanmin(lst):.1f}, {np.nanmax(lst):.1f}]°C, valid={valid_lst:,} px"
    )

    return {
        "date": date_str,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "lst": lst,
        "transform": bands.get("transform"),
        "crs": bands.get("crs"),
        "shape": ndvi.shape,
    }


# ══════════════════════════════════════════════
# 5. GRID POINT EXTRACTION
# ══════════════════════════════════════════════

def generate_delhi_grid_for_extraction(resolution: float = None) -> pd.DataFrame:
    """
    Generate a grid of lat/lon points covering Delhi for raster value extraction.

    Uses finer resolution than the default utils grid for better coverage.
    """
    resolution = resolution or SPATIAL_GRID_RESOLUTION

    lats = np.arange(DELHI_BBOX["south"], DELHI_BBOX["north"], resolution)
    lons = np.arange(DELHI_BBOX["west"], DELHI_BBOX["east"], resolution)

    grid = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
    return pd.DataFrame(grid, columns=["lat", "lon"])


def extract_values_at_points(raster: np.ndarray, transform: rasterio.transform.Affine,
                              points_df: pd.DataFrame,
                              raster_crs=None) -> np.ndarray:
    """
    Extract raster values at lat/lon grid points.

    Handles CRS conversion: grid points are in WGS84 (lat/lon) but
    raster may be in UTM. Reprojects points if needed.

    Args:
        raster: 2D numpy array
        transform: Affine transform mapping pixel coords to geo coords
        points_df: DataFrame with 'lat', 'lon' columns (WGS84)
        raster_crs: CRS of the raster (for coordinate reprojection)

    Returns:
        1D array of extracted values (NaN where outside raster or nodata)
    """
    from pyproj import Transformer

    values = np.full(len(points_df), np.nan)

    lats = points_df["lat"].values
    lons = points_df["lon"].values

    # Reproject lat/lon (WGS84) to raster CRS if needed
    if raster_crs is not None and str(raster_crs) != "EPSG:4326":
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)  # lon=x, lat=y
    else:
        xs, ys = lons, lats

    for i in range(len(xs)):
        try:
            row, col = rasterio.transform.rowcol(transform, xs[i], ys[i])
            if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
                val = raster[row, col]
                if not np.isnan(val):
                    values[i] = val
        except Exception:
            continue

    return values


def extract_scene_to_grid(scene_result: Dict, grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract NDVI, NDBI, LST values from a processed scene at grid points.

    Returns:
        DataFrame with columns: [lat, lon, date, ndvi, ndbi, lst]
    """
    if scene_result is None or scene_result.get("transform") is None:
        return pd.DataFrame()

    transform = scene_result["transform"]
    raster_crs = scene_result.get("crs")

    ndvi_vals = extract_values_at_points(scene_result["ndvi"], transform, grid_df, raster_crs)
    ndbi_vals = extract_values_at_points(scene_result["ndbi"], transform, grid_df, raster_crs)
    lst_vals = extract_values_at_points(scene_result["lst"], transform, grid_df, raster_crs)

    df = grid_df.copy()
    df["date"] = scene_result["date"]
    df["ndvi"] = ndvi_vals
    df["ndbi"] = ndbi_vals
    df["lst"] = lst_vals

    # Drop rows where all indices are NaN (points outside raster/Delhi)
    df = df.dropna(subset=["ndvi", "ndbi", "lst"], how="all")

    return df


# ══════════════════════════════════════════════
# 6. MULTI-TEMPORAL PROCESSING
# ══════════════════════════════════════════════

def discover_scenes(landsat_dir: Path = None) -> List[Path]:
    """
    Discover all Landsat scene directories sorted by date.

    Returns:
        List of scene directory paths sorted chronologically
    """
    landsat_dir = landsat_dir or LANDSAT_DIR

    if not landsat_dir.exists():
        logger.error(f"Landsat directory not found: {landsat_dir}")
        return []

    scenes = sorted([
        d for d in landsat_dir.iterdir()
        if d.is_dir() and len(list(d.glob("*.TIF"))) > 0
    ])

    logger.info(f"Found {len(scenes)} Landsat scenes: {[s.name for s in scenes]}")
    return scenes


def process_all_scenes(landsat_dir: Path = None,
                        grid_resolution: float = None) -> pd.DataFrame:
    """
    Process all Landsat scenes and build multi-temporal spatial DataFrame.

    This is the main entry point — reads every scene, clips to Delhi,
    computes NDVI/NDBI/LST, samples at grid points.

    Returns:
        DataFrame with columns: [lat, lon, date, ndvi, ndbi, lst]
        One row per grid point per scene date.
    """
    scenes = discover_scenes(landsat_dir)

    if not scenes:
        logger.error("No Landsat scenes found!")
        return pd.DataFrame()

    # Load Delhi boundary GeoDataFrame for clipping
    delhi_gdf = load_delhi_boundary()
    if delhi_gdf is not None:
        logger.info("Using Delhi boundary for clipping (CRS will be auto-reprojected)")
    else:
        logger.warning("No Delhi boundary available — processing full scene extent")

    # Generate grid points (in WGS84 lat/lon)
    grid_df = generate_delhi_grid_for_extraction(grid_resolution)
    logger.info(f"Grid: {len(grid_df)} points at {grid_resolution or SPATIAL_GRID_RESOLUTION}° resolution")

    all_data = []

    for scene_dir in scenes:
        try:
            scene_result = process_scene(scene_dir, delhi_gdf)
            if scene_result is not None:
                grid_data = extract_scene_to_grid(scene_result, grid_df)
                if len(grid_data) > 0:
                    all_data.append(grid_data)
                    logger.info(f"  Extracted {len(grid_data)} grid points for {scene_dir.name}")
                else:
                    logger.warning(f"  No valid grid points for {scene_dir.name}")
        except Exception as e:
            logger.error(f"  ERROR processing {scene_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_data:
        logger.error("No data extracted from any scene!")
        return pd.DataFrame()

    # Combine all scenes
    combined = pd.concat(all_data, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    logger.info(f"\nCombined multi-temporal dataset: {combined.shape[0]} rows × {combined.shape[1]} cols")
    logger.info(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    logger.info(f"  Unique dates: {combined['date'].nunique()}")
    logger.info(f"  NDVI range: [{combined['ndvi'].min():.3f}, {combined['ndvi'].max():.3f}]")
    logger.info(f"  NDBI range: [{combined['ndbi'].min():.3f}, {combined['ndbi'].max():.3f}]")
    logger.info(f"  LST range: [{combined['lst'].min():.1f}, {combined['lst'].max():.1f}]°C")

    return combined


def compute_temporal_averages(multi_temporal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-averaged spatial features from multi-temporal data.

    For each grid point, computes:
      - Mean/Std NDVI, NDBI, LST across all dates
      - Seasonal NDVI/LST (summer vs winter averages)
      - Derived land cover percentages from index thresholds

    Returns:
        DataFrame with one row per grid point, columns:
        [lat, lon, ndvi, ndbi, lst, ndvi_std, ndbi_std, lst_std,
         concrete_pct, vegetation_pct, water_pct, asphalt_pct, ...]
    """
    if len(multi_temporal_df) == 0:
        return pd.DataFrame()

    # Add month for seasonal grouping
    df = multi_temporal_df.copy()
    df["month"] = df["date"].dt.month

    # Group by grid point (rounded to avoid floating point issues)
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)

    # Temporal statistics per grid point
    stats = df.groupby(["lat_r", "lon_r"]).agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        ndvi=("ndvi", "mean"),
        ndbi=("ndbi", "mean"),
        lst=("lst", "mean"),
        ndvi_std=("ndvi", "std"),
        ndbi_std=("ndbi", "std"),
        lst_std=("lst", "std"),
        ndvi_max=("ndvi", "max"),
        ndvi_min=("ndvi", "min"),
        lst_max=("lst", "max"),
        lst_min=("lst", "min"),
        n_observations=("ndvi", "count"),
    ).reset_index(drop=True)

    # Fill NaN std with 0 (single observation points)
    stats[["ndvi_std", "ndbi_std", "lst_std"]] = stats[["ndvi_std", "ndbi_std", "lst_std"]].fillna(0)

    # ── Derive land cover percentages from spectral indices ──
    # Based on standard remote sensing thresholds:
    #   - NDBI > 0.1 and NDVI < 0.2 → Built-up (concrete)
    #   - NDBI > 0.0 and NDVI < 0.15 → Asphalt/Roads
    #   - NDVI > 0.4 → Dense vegetation
    #   - NDVI 0.2-0.4 → Sparse vegetation
    #   - NDVI < -0.1 → Water

    # Concrete percentage (correlates with high NDBI, low NDVI)
    stats["concrete_pct"] = np.clip(
        (stats["ndbi"] + 0.3) / 0.8 * 70  # Normalized NDBI → 0-70 range
        - stats["ndvi"] * 30,               # Penalize if there's vegetation
        5, 90
    ).round(1)

    # Vegetation percentage (correlates with high NDVI)
    stats["vegetation_pct"] = np.clip(
        stats["ndvi"] * 80,  # NDVI=1 → 80%, NDVI=0 → 0%
        2, 80
    ).round(1)

    # Water percentage (very low or negative NDVI, low reflectance)
    stats["water_pct"] = np.clip(
        np.where(stats["ndvi"] < 0, (-stats["ndvi"]) * 40, 1),
        0, 50
    ).round(1)

    # Asphalt (remainder adjusted)
    stats["asphalt_pct"] = np.clip(
        100 - stats["concrete_pct"] - stats["vegetation_pct"] - stats["water_pct"],
        2, 40
    ).round(1)

    # Normalize to sum to 100%
    total = stats["concrete_pct"] + stats["vegetation_pct"] + stats["water_pct"] + stats["asphalt_pct"]
    for col in ["concrete_pct", "vegetation_pct", "water_pct", "asphalt_pct"]:
        stats[col] = (stats[col] / total * 100).round(1)

    # Distance from city center (for reference, not for computing features)
    center_lat, center_lon = 28.6315, 77.2167
    stats["dist_from_center"] = np.sqrt(
        (stats["lat"] - center_lat) ** 2 + (stats["lon"] - center_lon) ** 2
    )

    logger.info(f"Computed temporal averages for {len(stats)} grid points")
    logger.info(f"  Land cover: concrete={stats['concrete_pct'].mean():.1f}%, "
                f"vegetation={stats['vegetation_pct'].mean():.1f}%, "
                f"water={stats['water_pct'].mean():.1f}%, "
                f"asphalt={stats['asphalt_pct'].mean():.1f}%")

    return stats


# ══════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_landsat_pipeline(save_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Landsat processing pipeline.

    Steps:
      1. Discover and process all scenes
      2. Extract values at grid points
      3. Compute temporal averages
      4. Save multi-temporal and averaged DataFrames

    Returns:
        Tuple of (multi_temporal_df, spatial_features_df)
    """
    set_seed(RANDOM_SEED)
    ensure_dir(PROCESSED_DATA_DIR)

    logger.info("=" * 60)
    logger.info("STARTING LANDSAT PROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Process all scenes
    multi_temporal = process_all_scenes()

    if len(multi_temporal) == 0:
        logger.error("Landsat processing produced no data!")
        return pd.DataFrame(), pd.DataFrame()

    # Step 2: Compute temporal averages (spatial features)
    spatial_features = compute_temporal_averages(multi_temporal)

    # Step 3: Save outputs
    if save_outputs:
        save_dataframe(multi_temporal, PROCESSED_DATA_DIR / "landsat_multi_temporal.csv")
        logger.info(f"Saved multi-temporal data: {multi_temporal.shape}")

    logger.info("=" * 60)
    logger.info("LANDSAT PROCESSING COMPLETE")
    logger.info(f"  Multi-temporal: {multi_temporal.shape}")
    logger.info(f"  Spatial features: {spatial_features.shape}")
    logger.info("=" * 60)

    return multi_temporal, spatial_features


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    multi_temp, spatial = run_landsat_pipeline()

    if len(multi_temp) > 0:
        print(f"\nMulti-temporal data: {multi_temp.shape}")
        print(f"Spatial features: {spatial.shape}")
        print(f"\nSample spatial features:")
        print(spatial.head(10).to_string())
    else:
        print("\nERROR: No data produced. Check that Landsat scenes exist in data/raw/landsat images/")
