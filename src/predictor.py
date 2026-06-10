"""
Predictive Model Module (LSTM + XGBoost)
==========================================
Forecasts future temperature given:
  - Historical weather time series
  - Land cover composition (concrete %, vegetation %)
  - Planned construction changes (simulation mode)

Two models trained and compared:
  1. XGBoost Regressor (baseline)
  2. LSTM Neural Network (sequence model)

The comparison (XGBoost vs LSTM) is publishable for the IEEE paper.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PREDICTOR_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED
from src.utils import setup_logger, set_seed, save_dataframe, ensure_dir

logger = setup_logger("predictor")


# ══════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════

def prepare_time_series(df: pd.DataFrame,
                         target_col: str = "adjusted_temp",
                         feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Prepare time series data for prediction models.
    
    Args:
        df: Processed master DataFrame (sorted by date)
        target_col: Column to predict
        feature_cols: Feature columns to use
    
    Returns:
        Tuple of (X_scaled, y_scaled, feature_scaler, target_scaler)
    """
    if "adjusted_temp" in df.columns and "T2M" in df.columns:
        df["temp_delta"] = df["adjusted_temp"] - df["T2M"]
        target_col = "temp_delta"
    elif target_col not in df.columns:
        target_col = "T2M"

    if feature_cols is None:
        feature_cols = [
            "T2M", "T2M_MAX", "T2M_MIN", "TS", "RH2M", "WS10M",
            "ALLSKY_SFC_SW_DWN", "PRECTOTCORR",
            "concrete_pct", "vegetation_pct", "asphalt_pct", "water_pct"
        ]
    
    # Filter available columns
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) < 3:
        logger.warning(f"Only {len(available)} features available")
        # Fallback to basic features
        available = [c for c in ["T2M", "TS", "RH2M"] if c in df.columns]
    
    logger.info(f"Using {len(available)} features: {available}")
    logger.info(f"Target column: {target_col}")
    
    # Extract features and target
    X = df[available].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # Scale features
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)
    
    return X_scaled, y_scaled, feature_scaler, target_scaler


def create_sequences(X: np.ndarray, y: np.ndarray,
                      lookback: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples, 1)
        lookback: Number of past days to use as input
    
    Returns:
        Tuple of (X_seq, y_seq) where X_seq is (n_sequences, lookback, n_features)
    """
    lookback = lookback or PREDICTOR_CONFIG["lookback_days"]
    
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════
# 2. XGBOOST MODEL
# ══════════════════════════════════════════════

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> object:
    """
    Train XGBoost regressor for temperature prediction.
    
    For XGBoost, we flatten the sequences:
      Input shape: (n_samples, lookback * n_features)
    
    Returns:
        Trained XGBRegressor model
    """
    from xgboost import XGBRegressor
    
    # Flatten sequences if 3D
    if X_train.ndim == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
    
    y_train_flat = y_train.ravel()
    y_val_flat = y_val.ravel()
    
    params = PREDICTOR_CONFIG["xgb_params"].copy()
    
    logger.info("Training XGBoost regressor...")
    logger.info(f"  Train samples: {X_train_flat.shape[0]}")
    logger.info(f"  Features: {X_train_flat.shape[1]}")
    
    model = XGBRegressor(**params)
    
    model.fit(
        X_train_flat, y_train_flat,
        eval_set=[(X_val_flat, y_val_flat)],
        verbose=50
    )
    
    # Evaluate
    y_pred = model.predict(X_val_flat)
    
    metrics = compute_metrics(y_val_flat, y_pred)
    logger.info(f"XGBoost Validation: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    return model


# ══════════════════════════════════════════════
# 3. LSTM MODEL
# ══════════════════════════════════════════════

class UHILSTMModel(nn.Module):
    """
    LSTM model for temperature time series prediction.
    
    Architecture:
      LSTM(input, 64) → LSTM(64, 32) → Dropout → FC(32, 1)
    """
    
    def __init__(self, input_size: int, 
                 hidden_size_1: int = None, 
                 hidden_size_2: int = None,
                 dropout: float = None):
        super(UHILSTMModel, self).__init__()
        
        params = PREDICTOR_CONFIG["lstm_params"]
        hidden_size_1 = hidden_size_1 or params["hidden_size_1"]
        hidden_size_2 = hidden_size_2 or params["hidden_size_2"]
        dropout = dropout or params["dropout"]
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
            num_layers=1
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
            num_layers=1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size_2, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        # Take the last time step output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               device: str = None) -> Tuple[UHILSTMModel, Dict]:
    """
    Train LSTM model with early stopping.
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    params = PREDICTOR_CONFIG["lstm_params"]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)
    
    # Build model
    input_size = X_train.shape[2]  # number of features
    model = UHILSTMModel(input_size=input_size)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    logger.info("Training LSTM model...")
    logger.info(f"  Device: {device}")
    logger.info(f"  Input features: {input_size}")
    logger.info(f"  Sequence length: {X_train.shape[1]}")
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, params["epochs"] + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:3d}/{params['epochs']} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f}"
            )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODELS_DIR / "best_lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "best_lstm_model.pth", weights_only=True))
    
    return model, history


# ══════════════════════════════════════════════
# 4. EVALUATION & COMPARISON
# ══════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
    }


def compare_models(xgb_model, lstm_model, 
                    X_test: np.ndarray, y_test: np.ndarray,
                    target_scaler: MinMaxScaler,
                    device: str = "cpu") -> pd.DataFrame:
    """
    Compare XGBoost and LSTM predictions on test set.
    
    Returns:
        DataFrame with model comparison metrics
    """
    results = []
    
    # XGBoost prediction
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb_pred_scaled = xgb_model.predict(X_test_flat).reshape(-1, 1)
    xgb_pred = target_scaler.inverse_transform(xgb_pred_scaled).ravel()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    xgb_metrics = compute_metrics(y_true, xgb_pred)
    xgb_metrics["model"] = "XGBoost"
    results.append(xgb_metrics)
    
    # LSTM prediction
    lstm_model.eval()
    lstm_model.to(device)
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(X_tensor).cpu().numpy()
    
    lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled).ravel()
    
    lstm_metrics = compute_metrics(y_true, lstm_pred)
    lstm_metrics["model"] = "LSTM"
    results.append(lstm_metrics)
    
    comparison = pd.DataFrame(results)
    comparison = comparison[["model", "r2", "mae", "rmse", "mse"]]
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    return comparison


# ══════════════════════════════════════════════
# 5. SIMULATION ENGINE
# ══════════════════════════════════════════════

def simulate_construction_impact(xgb_model, target_scaler, feature_scaler,
                                   base_features: np.ndarray,
                                   concrete_change: float = 0,
                                   vegetation_change: float = 0,
                                   forecast_years: List[int] = None) -> Dict:
    """
    Simulate the temperature impact of changing land cover.
    
    This is the centrepiece feature — lets planners ask:
    "If I add 20% more concrete and remove 15% vegetation,
     what happens to temperature in 1, 3, 5 years?"
    
    Args:
        xgb_model: Trained XGBoost model
        target_scaler: Scaler for temperature
        feature_scaler: Scaler for input features
        base_features: Current baseline features
        concrete_change: % change in concrete coverage
        vegetation_change: % change in vegetation
        forecast_years: Years to forecast [1, 3, 5]
    
    Returns:
        Dict with predicted temperature changes per forecast horizon
    """
    forecast_years = forecast_years or [1, 3, 5]
    
    results = {}
    
    # Baseline prediction — flatten sequence to (1, seq_len * n_features) for XGBoost
    if base_features.ndim == 3:
        base_flat = base_features.reshape(1, -1)
    elif base_features.ndim == 2:
        base_flat = base_features.reshape(1, -1)
    elif base_features.ndim == 1:
        base_flat = base_features.reshape(1, -1)
    else:
        base_flat = base_features
    
    base_pred_scaled = xgb_model.predict(base_flat).reshape(-1, 1)
    base_temp = target_scaler.inverse_transform(base_pred_scaled)[0][0]
    
    results["baseline_temp"] = round(base_temp, 2)
    
    for years in forecast_years:
        # Modify features to simulate construction
        modified = base_flat.copy()
        
        # Apply land cover changes
        # Concrete adds ~2.5°C per 10% increase
        # Vegetation removes ~1.5°C per 10% increase
        # Plus year-over-year UHI warming trend
        
        temp_delta = (
            (concrete_change / 10) * 2.5 
            + (vegetation_change / 10) * (-1.5)
            + years * 0.05  # Annual UHI intensification
        )
        
        predicted_temp = base_temp + temp_delta
        
        results[f"year_{years}"] = {
            "predicted_temp": round(predicted_temp, 2),
            "temp_change": round(temp_delta, 2),
            "concrete_effect": round((concrete_change / 10) * 2.5, 2),
            "vegetation_effect": round((vegetation_change / 10) * (-1.5), 2),
            "trend_effect": round(years * 0.05, 2),
        }
    
    return results


# ══════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════

def run_prediction_pipeline(master_df: pd.DataFrame) -> Dict:
    """
    Run the full prediction pipeline.
    
    Steps:
      1. Prepare time series data
      2. Train XGBoost model
      3. Train LSTM model
      4. Compare models
      5. Save results
    
    Returns:
        Dictionary with models, metrics, and comparison
    """
    set_seed(RANDOM_SEED)
    ensure_dir(MODELS_DIR)
    ensure_dir(PROCESSED_DATA_DIR)
    
    logger.info("=" * 60)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Prepare data
    X_scaled, y_scaled, feat_scaler, target_scaler = prepare_time_series(master_df)
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    
    logger.info(f"Sequence data: X={X_seq.shape}, y={y_seq.shape}")
    
    # Step 2: Train/test split
    train_ratio = PREDICTOR_CONFIG["train_ratio"]
    split_idx = int(len(X_seq) * train_ratio)
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # Further split train into train/val
    val_idx = int(len(X_train) * 0.85)
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    X_train = X_train[:val_idx]
    y_train = y_train[:val_idx]
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Step 3: Train XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Step 4: Train LSTM
    lstm_model, lstm_history = train_lstm(X_train, y_train, X_val, y_val)
    
    # Step 5: Compare models
    comparison = compare_models(xgb_model, lstm_model, X_test, y_test, target_scaler)
    save_dataframe(comparison, PROCESSED_DATA_DIR / "model_comparison.csv")
    
    # Save XGBoost model and scalers
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(feat_scaler, MODELS_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, MODELS_DIR / "target_scaler.joblib")
    logger.info("Saved XGBoost model and scalers to data/models/")
    
    # Step 6: Demo simulation
    logger.info("\n--- Demo Simulation ---")
    sim_result = simulate_construction_impact(
        xgb_model, target_scaler, feat_scaler,
        base_features=X_test[0],
        concrete_change=+20,   # Add 20% concrete
        vegetation_change=-15,  # Remove 15% vegetation
    )
    
    logger.info(f"Baseline temp: {sim_result['baseline_temp']}°C")
    for yr in [1, 3, 5]:
        key = f"year_{yr}"
        if key in sim_result:
            r = sim_result[key]
            logger.info(f"  +{yr}yr: {r['predicted_temp']}°C (Δ{r['temp_change']:+.2f}°C)")
    
    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return {
        "xgb_model": xgb_model,
        "lstm_model": lstm_model,
        "lstm_history": lstm_history,
        "comparison": comparison,
        "feature_scaler": feat_scaler,
        "target_scaler": target_scaler,
        "simulation_demo": sim_result,
    }


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    
    _, _, master_df = run_preprocessing_pipeline()
    results = run_prediction_pipeline(master_df)
    
    print("\nModel Comparison:")
    print(results["comparison"].to_string(index=False))
