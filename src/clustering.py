"""
Spatial Clustering Module (DBSCAN)
===================================
Identifies heat hotspot clusters across Delhi using DBSCAN
on spatio-temporal temperature + land cover data.

Output:
  - Cluster labels for each grid point
  - Hotspot zone map
  - Cluster statistics (mean temp, dominant land cover)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DBSCAN_CONFIG, PROCESSED_DATA_DIR, RANDOM_SEED
from src.utils import setup_logger, set_seed, save_dataframe, ensure_dir

logger = setup_logger("clustering")


# ══════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════

def prepare_clustering_features(spatial_df: pd.DataFrame,
                                  weather_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Prepare feature matrix for DBSCAN clustering.
    
    Features used:
      - lat, lon (spatial position)
      - ndvi (vegetation index — from real Landsat)
      - ndbi (built-up index — from real Landsat)
      - concrete_pct (impervious surface %)
      - vegetation_pct (green cover %)
      - lst (land surface temperature — from real Landsat B10, if available)
    
    Optionally includes temporal aggregates from weather data.
    """
    df = spatial_df.copy()
    
    # Select clustering features
    feature_cols = ["lat", "lon", "ndvi", "ndbi", "concrete_pct", "vegetation_pct"]
    
    # Use real LST from satellite data if available
    if "lst" in df.columns:
        feature_cols.append("lst")
        logger.info("Using real Land Surface Temperature (LST) from Landsat")
    elif weather_df is not None and "T2M" in weather_df.columns:
        # Derive estimated temperature from weather + land cover (no random noise)
        summer_mask = weather_df["season"] == "Summer"
        if summer_mask.any():
            summer_mean_temp = weather_df.loc[summer_mask, "T2M"].mean()
            df["est_summer_temp"] = (
                summer_mean_temp 
                + df["concrete_pct"] / 100 * 3.0  
                - df["vegetation_pct"] / 100 * 2.0
            )
            feature_cols.append("est_summer_temp")
            logger.info("Using weather-derived temperature estimate (no Landsat LST available)")
    
    # Validate features exist
    available_features = [c for c in feature_cols if c in df.columns]
    
    if len(available_features) < 3:
        logger.warning(f"Only {len(available_features)} features available for clustering")
    
    logger.info(f"Clustering features: {available_features}")
    return df, available_features


# ══════════════════════════════════════════════
# 2. DBSCAN CLUSTERING
# ══════════════════════════════════════════════

def run_dbscan(df: pd.DataFrame, feature_cols: list,
               eps: float = None, min_samples: int = None) -> pd.DataFrame:
    """
    Run DBSCAN clustering on spatial features.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: Column names to use as features
        eps: DBSCAN neighborhood radius
        min_samples: Minimum points to form a cluster
    
    Returns:
        DataFrame with 'cluster' column added
    """
    eps = eps or DBSCAN_CONFIG["eps"]
    min_samples = min_samples or DBSCAN_CONFIG["min_samples"]
    
    # Extract and scale features
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle lat/lon differently — convert to radians for haversine
    lat_idx = feature_cols.index("lat") if "lat" in feature_cols else None
    lon_idx = feature_cols.index("lon") if "lon" in feature_cols else None
    
    # Auto-tune eps using k-nearest neighbors distance
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    
    # Use the "knee" of the k-distance curve as eps
    # Approximate knee: where the rate of change is steepest
    gradient = np.gradient(k_distances)
    knee_idx = min(int(len(k_distances) * 0.70), len(k_distances) - 1)
    auto_eps = float(k_distances[knee_idx])
    
    # Use the larger of config eps (scaled) or auto-detected eps
    # Cap at 0.55 to avoid merging everything into one mega-cluster
    effective_eps = min(max(auto_eps, 0.3), 0.55)
    
    logger.info(f"Running DBSCAN (auto_eps={auto_eps:.4f}, effective_eps={effective_eps:.4f}, min_samples={min_samples}) on {len(df)} points")
    
    # Run DBSCAN
    dbscan = DBSCAN(
        eps=effective_eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=-1
    )
    
    clusters = dbscan.fit_predict(X_scaled)
    
    df = df.copy()
    df["cluster"] = clusters
    
    # Statistics
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(df)*100:.1f}%)")
    
    # Compute silhouette score if we have valid clusters
    if n_clusters >= 2:
        valid_mask = clusters != -1
        if valid_mask.sum() > 1:
            score = silhouette_score(X_scaled[valid_mask], clusters[valid_mask])
            logger.info(f"Silhouette Score: {score:.4f}")
    
    return df


# ══════════════════════════════════════════════
# 3. CLUSTER ANALYSIS
# ══════════════════════════════════════════════

def analyze_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each cluster.
    
    Returns:
        DataFrame with per-cluster stats:
          - size, center_lat, center_lon
          - mean concrete/vegetation/ndvi/ndbi
          - classification (Hotspot / Cool Zone / Mixed)
    """
    if "cluster" not in df.columns:
        logger.error("No 'cluster' column found. Run DBSCAN first.")
        return pd.DataFrame()
    
    # Exclude noise (-1)
    clustered = df[df["cluster"] != -1]
    
    if len(clustered) == 0:
        logger.warning("No valid clusters found")
        return pd.DataFrame()
    
    stats = clustered.groupby("cluster").agg({
        "lat": ["mean", "count"],
        "lon": "mean",
        "concrete_pct": "mean",
        "vegetation_pct": "mean",
        "ndvi": "mean",
        "ndbi": "mean",
    })
    
    # Flatten multi-index columns
    stats.columns = [
        "center_lat", "size", "center_lon",
        "mean_concrete_pct", "mean_vegetation_pct",
        "mean_ndvi", "mean_ndbi"
    ]
    
    # Add estimated temperature if available
    if "est_summer_temp" in df.columns:
        temp_stats = clustered.groupby("cluster")["est_summer_temp"].mean()
        stats["mean_est_temp"] = temp_stats
    
    # Classify clusters
    def classify_cluster(row):
        if row["mean_concrete_pct"] > 60 and row["mean_ndvi"] < 0.2:
            return "🔴 Heat Hotspot"
        elif row["mean_vegetation_pct"] > 40 and row["mean_ndvi"] > 0.4:
            return "🟢 Cool Zone"
        else:
            return "🟡 Mixed Zone"
    
    stats["classification"] = stats.apply(classify_cluster, axis=1)
    stats = stats.sort_values("mean_concrete_pct", ascending=False)
    
    logger.info(f"\nCluster Analysis:")
    for idx, row in stats.iterrows():
        logger.info(
            f"  Cluster {idx}: {row['classification']} | "
            f"Size={row['size']:.0f} | "
            f"Concrete={row['mean_concrete_pct']:.1f}% | "
            f"NDVI={row['mean_ndvi']:.3f}"
        )
    
    return stats.reset_index()


def identify_hotspots(df: pd.DataFrame, 
                       temp_threshold: float = None) -> pd.DataFrame:
    """
    Identify heat hotspot zones based on cluster analysis.
    
    Hotspot criteria:
      - High concrete percentage (>50%)
      - Low NDVI (<0.2)
      - High estimated temperature
    
    Returns:
        DataFrame of hotspot locations
    """
    if "cluster" not in df.columns:
        logger.warning("No clusters found — run DBSCAN first")
        return pd.DataFrame()
    
    # Get cluster stats
    stats = analyze_clusters(df)
    
    # Filter hotspots
    hotspot_clusters = stats[stats["classification"].str.contains("Hotspot")]["cluster"].values
    
    hotspots = df[df["cluster"].isin(hotspot_clusters)].copy()
    
    logger.info(f"Identified {len(hotspots)} hotspot grid points in {len(hotspot_clusters)} clusters")
    
    return hotspots


# ══════════════════════════════════════════════
# 4. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_clustering_pipeline(spatial_df: pd.DataFrame,
                             weather_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full spatial clustering pipeline.
    
    Args:
        spatial_df: Spatial features DataFrame
        weather_df: Optional weather DataFrame for temperature integration
    
    Returns:
        Tuple of (clustered_df, cluster_stats)
    """
    set_seed(RANDOM_SEED)
    ensure_dir(PROCESSED_DATA_DIR)
    
    logger.info("=" * 60)
    logger.info("STARTING SPATIAL CLUSTERING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Prepare features
    df, feature_cols = prepare_clustering_features(spatial_df, weather_df)
    
    # Step 2: Run DBSCAN
    clustered_df = run_dbscan(df, feature_cols)
    
    # Step 3: Analyze clusters
    cluster_stats = analyze_clusters(clustered_df)
    
    # Step 4: Save results
    save_dataframe(clustered_df, PROCESSED_DATA_DIR / "clustered_spatial.csv")
    if len(cluster_stats) > 0:
        save_dataframe(cluster_stats, PROCESSED_DATA_DIR / "cluster_stats.csv")
    
    logger.info("=" * 60)
    logger.info("CLUSTERING COMPLETE")
    logger.info("=" * 60)
    
    return clustered_df, cluster_stats


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    
    weather_df, spatial_df, _ = run_preprocessing_pipeline()
    clustered, stats = run_clustering_pipeline(spatial_df, weather_df)
    
    print(f"\nClustered data shape: {clustered.shape}")
    print(f"Cluster stats:\n{stats}")
