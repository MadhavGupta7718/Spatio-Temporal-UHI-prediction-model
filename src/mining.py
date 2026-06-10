"""
Association Rule Mining Module
================================
Discovers patterns like:
  {asphalt=High, wind=Low, season=Summer} → {temp_spike=Large}

Uses Apriori / FP-Growth from mlxtend library.

Steps:
  1. Discretize continuous features into categorical bins
  2. Create transaction matrix (one-hot encoded)
  3. Find frequent itemsets (min_support)
  4. Generate association rules (min_confidence, min_lift)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MINING_CONFIG, PROCESSED_DATA_DIR, RULES_DIR, RANDOM_SEED
from src.utils import setup_logger, set_seed, save_dataframe, ensure_dir, discretize_column

logger = setup_logger("mining")


# ══════════════════════════════════════════════
# 1. DISCRETIZATION
# ══════════════════════════════════════════════

def discretize_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous weather features into categorical bins
    suitable for association rule mining.
    
    Bins are based on Delhi's climate characteristics:
      - Temperature: [0, 20, 30, 38, 50] → Low/Medium/High/Extreme
      - Humidity: [0, 30, 50, 70, 100] → Dry/Moderate/Humid/Very_Humid
      - Wind speed: [0, 2, 4, 7, 20] → Calm/Light/Moderate/Strong
      - Precipitation: categorized as None/Light/Moderate/Heavy
    """
    disc = pd.DataFrame()
    
    # Temperature categories
    if "T2M" in df.columns:
        disc["temp_cat"] = discretize_column(
            df["T2M"],
            bins=[-np.inf, 20, 30, 38, np.inf],
            labels=["temp=Low", "temp=Medium", "temp=High", "temp=Extreme"]
        )
    
    # UHI proxy categories
    if "uhi_proxy" in df.columns:
        disc["uhi_cat"] = discretize_column(
            df["uhi_proxy"],
            bins=[-np.inf, -1, 0.5, 2, np.inf],
            labels=["uhi=Cooling", "uhi=Neutral", "uhi=Mild", "uhi=Strong"]
        )
    elif "TS" in df.columns and "T2M" in df.columns:
        uhi = df["TS"] - df["T2M"]
        disc["uhi_cat"] = discretize_column(
            uhi,
            bins=[-np.inf, -1, 0.5, 2, np.inf],
            labels=["uhi=Cooling", "uhi=Neutral", "uhi=Mild", "uhi=Strong"]
        )
    
    # Humidity categories
    if "RH2M" in df.columns:
        disc["humidity_cat"] = discretize_column(
            df["RH2M"],
            bins=[-np.inf, 30, 50, 70, np.inf],
            labels=["humidity=Dry", "humidity=Moderate", "humidity=Humid", "humidity=VeryHumid"]
        )
    
    # Wind speed categories
    if "WS10M" in df.columns:
        disc["wind_cat"] = discretize_column(
            df["WS10M"],
            bins=[-np.inf, 2, 4, 7, np.inf],
            labels=["wind=Calm", "wind=Light", "wind=Moderate", "wind=Strong"]
        )
    
    # Precipitation categories
    if "PRECTOTCORR" in df.columns:
        disc["precip_cat"] = discretize_column(
            df["PRECTOTCORR"],
            bins=[-np.inf, 0.1, 5, 20, np.inf],
            labels=["precip=None", "precip=Light", "precip=Moderate", "precip=Heavy"]
        )
    
    # Season (already categorical)
    if "season" in df.columns:
        disc["season_cat"] = "season=" + df["season"].astype(str)
    
    # Solar radiation
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        disc["solar_cat"] = discretize_column(
            df["ALLSKY_SFC_SW_DWN"],
            bins=[-np.inf, 10, 18, 25, np.inf],
            labels=["solar=Low", "solar=Medium", "solar=High", "solar=VeryHigh"]
        )
    
    # Temperature anomaly
    if "temp_anomaly" in df.columns:
        disc["anomaly_cat"] = discretize_column(
            df["temp_anomaly"],
            bins=[-np.inf, -2, 2, 5, np.inf],
            labels=["anomaly=BelowNormal", "anomaly=Normal", "anomaly=AboveNormal", "anomaly=Spike"]
        )
    
    logger.info(f"Discretized {len(disc.columns)} features")
    return disc


def discretize_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize spatial land cover features."""
    disc = pd.DataFrame()
    
    if "concrete_pct" in df.columns:
        disc["concrete_cat"] = discretize_column(
            df["concrete_pct"],
            bins=[-np.inf, 25, 50, 75, np.inf],
            labels=["concrete=Low", "concrete=Medium", "concrete=High", "concrete=VeryHigh"]
        )
    
    if "vegetation_pct" in df.columns:
        disc["vegetation_cat"] = discretize_column(
            df["vegetation_pct"],
            bins=[-np.inf, 15, 30, 50, np.inf],
            labels=["veg=Bare", "veg=Low", "veg=Medium", "veg=High"]
        )
    
    if "ndvi" in df.columns:
        disc["ndvi_cat"] = discretize_column(
            df["ndvi"],
            bins=[-np.inf, 0.1, 0.3, 0.5, np.inf],
            labels=["ndvi=VeryLow", "ndvi=Low", "ndvi=Medium", "ndvi=High"]
        )
    
    if "ndbi" in df.columns:
        disc["ndbi_cat"] = discretize_column(
            df["ndbi"],
            bins=[-np.inf, 0, 0.15, 0.3, np.inf],
            labels=["ndbi=Low", "ndbi=Medium", "ndbi=High", "ndbi=VeryHigh"]
        )
    
    return disc


# ══════════════════════════════════════════════
# 2. TRANSACTION ENCODING
# ══════════════════════════════════════════════

def create_transaction_matrix(disc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert discretized DataFrame to one-hot encoded transaction matrix.
    
    Each row = one transaction
    Each column = one item (e.g., "temp=High")
    Values = True/False
    """
    # Convert to list of transactions
    transactions = []
    for _, row in disc_df.iterrows():
        items = [str(val) for val in row.values if pd.notna(val) and str(val) != "nan"]
        transactions.append(items)
    
    # One-hot encode
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)
    
    # Remove any 'nan' columns
    nan_cols = [c for c in transaction_df.columns if "nan" in str(c).lower()]
    transaction_df = transaction_df.drop(columns=nan_cols, errors="ignore")
    
    logger.info(f"Transaction matrix: {transaction_df.shape[0]} transactions × {transaction_df.shape[1]} items")
    
    return transaction_df


# ══════════════════════════════════════════════
# 3. FREQUENT PATTERN MINING
# ══════════════════════════════════════════════

def find_frequent_itemsets(transaction_df: pd.DataFrame,
                            min_support: float = None,
                            algorithm: str = None) -> pd.DataFrame:
    """
    Find frequent itemsets using Apriori or FP-Growth.
    
    Args:
        transaction_df: One-hot encoded transaction matrix
        min_support: Minimum support threshold (0-1)
        algorithm: "apriori" or "fpgrowth"
    
    Returns:
        DataFrame of frequent itemsets with support values
    """
    min_support = min_support or MINING_CONFIG["min_support"]
    algorithm = algorithm or MINING_CONFIG["algorithm"]
    
    logger.info(f"Mining frequent itemsets (algorithm={algorithm}, min_support={min_support})")
    
    if algorithm == "fpgrowth":
        frequent = fpgrowth(transaction_df, min_support=min_support, use_colnames=True)
    else:
        frequent = apriori(transaction_df, min_support=min_support, use_colnames=True)
    
    # Sort by support
    frequent = frequent.sort_values("support", ascending=False).reset_index(drop=True)
    
    logger.info(f"Found {len(frequent)} frequent itemsets")
    
    # Show top 10
    if len(frequent) > 0:
        logger.info(f"\nTop 10 frequent itemsets:")
        for _, row in frequent.head(10).iterrows():
            items = ", ".join(row["itemsets"])
            logger.info(f"  Support={row['support']:.3f} | {{{items}}}")
    
    return frequent


# ══════════════════════════════════════════════
# 4. ASSOCIATION RULE GENERATION
# ══════════════════════════════════════════════

def generate_rules(frequent_itemsets: pd.DataFrame,
                    min_confidence: float = None,
                    min_lift: float = None) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: Output from find_frequent_itemsets
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
    
    Returns:
        DataFrame of association rules with metrics
    """
    min_confidence = min_confidence or MINING_CONFIG["min_confidence"]
    min_lift = min_lift or MINING_CONFIG["min_lift"]
    
    if len(frequent_itemsets) == 0:
        logger.warning("No frequent itemsets found — cannot generate rules")
        return pd.DataFrame()
    
    logger.info(f"Generating rules (min_confidence={min_confidence}, min_lift={min_lift})")
    
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # Filter by lift
    rules = rules[rules["lift"] >= min_lift]
    
    # Sort by lift (most interesting rules first)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    
    # Format antecedents and consequents for readability
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    
    # Create human-readable rule string
    rules["rule"] = rules.apply(
        lambda r: f"{{{r['antecedents_str']}}} → {{{r['consequents_str']}}}",
        axis=1
    )
    
    logger.info(f"Generated {len(rules)} association rules")
    
    # Show top rules
    if len(rules) > 0:
        logger.info(f"\nTop 10 rules by lift:")
        for _, row in rules.head(10).iterrows():
            logger.info(
                f"  Lift={row['lift']:.2f} | "
                f"Conf={row['confidence']:.2f} | "
                f"Supp={row['support']:.3f} | "
                f"{row['rule']}"
            )
    
    return rules


def filter_uhi_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rules specifically related to UHI / temperature effects.
    
    Focuses on rules where the antecedent OR consequent involves:
      - temp=High or temp=Extreme
      - uhi=Strong or uhi=Mild
      - anomaly=Spike or anomaly=AboveNormal
      - season=Summer (heat-relevant)
    """
    if len(rules) == 0:
        return rules
    
    uhi_keywords = ["temp=High", "temp=Extreme", "uhi=Strong", 
                      "uhi=Mild", "anomaly=Spike", "anomaly=AboveNormal"]
    
    # Check both antecedents and consequents for UHI-related items
    mask_consequent = rules["consequents_str"].apply(
        lambda x: any(kw in x for kw in uhi_keywords)
    )
    mask_antecedent = rules["antecedents_str"].apply(
        lambda x: any(kw in x for kw in uhi_keywords)
    )
    
    uhi_rules = rules[mask_consequent | mask_antecedent].copy()
    logger.info(f"Filtered {len(uhi_rules)} UHI-related rules")
    
    return uhi_rules


# ══════════════════════════════════════════════
# 5. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_mining_pipeline(weather_df: pd.DataFrame,
                         spatial_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full association rule mining pipeline.
    
    Merges weather discretized features WITH spatial discretized features
    to discover cross-domain patterns (e.g., high concrete + summer → temperature spike).
    
    Args:
        weather_df: Processed weather DataFrame
        spatial_df: Spatial features DataFrame (from real Landsat data)
    
    Returns:
        Tuple of (all_rules, uhi_rules)
    """
    set_seed(RANDOM_SEED)
    ensure_dir(RULES_DIR)
    
    logger.info("=" * 60)
    logger.info("STARTING ASSOCIATION RULE MINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Discretize weather features
    weather_disc = discretize_weather_features(weather_df)
    
    # Step 2: Merge spatial features if available
    if spatial_df is not None and len(spatial_df) > 0:
        spatial_disc = discretize_spatial_features(spatial_df)
        
        if len(spatial_disc) > 0:
            logger.info(f"Merging {len(spatial_disc.columns)} spatial features into transactions")
            
            # Sample spatial points and repeat for each weather record
            # to create combined spatio-temporal transactions
            n_spatial_sample = min(50, len(spatial_disc))
            spatial_sample = spatial_disc.sample(n_spatial_sample, random_state=RANDOM_SEED)
            
            # For each spatial sample, assign weather records evenly
            n_weather = len(weather_disc)
            records_per_point = n_weather // n_spatial_sample
            
            combined_records = []
            for i, (_, spatial_row) in enumerate(spatial_sample.iterrows()):
                start_idx = i * records_per_point
                end_idx = start_idx + records_per_point
                weather_chunk = weather_disc.iloc[start_idx:end_idx].copy()
                
                # Add spatial columns to each weather record
                for col in spatial_disc.columns:
                    weather_chunk[col] = spatial_row[col]
                
                combined_records.append(weather_chunk)
            
            weather_disc = pd.concat(combined_records, ignore_index=True)
            logger.info(f"Combined transaction matrix: {weather_disc.shape}")
    
    # Step 3: Create transaction matrix
    transaction_df = create_transaction_matrix(weather_disc)
    
    # Step 4: Find frequent itemsets
    frequent = find_frequent_itemsets(transaction_df)
    
    # Step 5: Generate rules
    all_rules = generate_rules(frequent)
    
    # Step 6: Filter UHI-specific rules
    uhi_rules = filter_uhi_rules(all_rules)
    
    # Step 7: Save results
    if len(all_rules) > 0:
        save_dataframe(
            all_rules[["rule", "support", "confidence", "lift", 
                        "antecedents_str", "consequents_str"]],
            RULES_DIR / "all_association_rules.csv"
        )
    
    if len(uhi_rules) > 0:
        save_dataframe(
            uhi_rules[["rule", "support", "confidence", "lift",
                        "antecedents_str", "consequents_str"]],
            RULES_DIR / "uhi_association_rules.csv"
        )
    
    logger.info("=" * 60)
    logger.info(f"MINING COMPLETE — {len(all_rules)} total rules, {len(uhi_rules)} UHI rules")
    logger.info("=" * 60)
    
    return all_rules, uhi_rules


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    
    weather_df, spatial_df, _ = run_preprocessing_pipeline()
    all_rules, uhi_rules = run_mining_pipeline(weather_df, spatial_df)
    
    print(f"\nTotal rules: {len(all_rules)}")
    print(f"UHI rules: {len(uhi_rules)}")
    if len(uhi_rules) > 0:
        print(f"\nTop UHI Rules:")
        print(uhi_rules[["rule", "lift", "confidence"]].head(10).to_string())
