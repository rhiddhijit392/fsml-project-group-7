"""
train.py - Train 2 Regression Models to Predict Next Day Closing Price
Models  : Random Forest Regressor, XGBoost Regressor
Target  : Next_Day_Close (tomorrow's closing price)
Dataset : Yahoo Finance - 5 Years Stock Data (600K+ rows)

Why 'Close' is NOT a feature:
    Close          = today's closing price    (input side)
    Next_Day_Close = tomorrow's closing price (output/target side)
    They are almost the same number, so keeping Close as a feature
    causes data leakage -- model just copies today's price and gets
    fake R2=0.999 without learning anything real.
"""

import sys
import os
import pandas as pd
import numpy as np

# -- Path setup so src.* and pipeline.* imports work -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from pipeline.pipeline import run_pipeline
from src.utils import (
    setup_logger,
    load_config,
    save_model,
    ensure_dirs,
    compare_models,
)

# -- Logger & Config ----------------------------------------------------------
logger = setup_logger("logs/app.log")
config = load_config("config.yaml")
ensure_dirs(["models/", "logs/"])


# =============================================================================
# HELPER -- Compute Regression Metrics
# =============================================================================

def compute_regression_metrics(y_true, y_pred, model_name: str) -> dict:
    """
    Computes MAE, RMSE, and R2 for a regression model and logs them.

    Args:
        y_true     : Actual Next_Day_Close values
        y_pred     : Model predicted values
        model_name : Name label for logging

    Returns:
        dict: { MAE, RMSE, R2_Score }

    Metrics explained:
        MAE      - average dollar error (e.g. off by $3.50 on average)
        RMSE     - like MAE but punishes large errors more heavily
        R2_Score - how well model explains price variance (1.0 = perfect)
    """
    mae  = round(mean_absolute_error(y_true, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 4)
    r2   = round(r2_score(y_true, y_pred), 4)

    metrics = {"MAE": mae, "RMSE": rmse, "R2_Score": r2}

    logger.info(f"-- {model_name} Metrics --------------------------")
    logger.info(f"   MAE      : {mae}   (avg error in $ price)")
    logger.info(f"   RMSE     : {rmse}  (penalizes large errors more)")
    logger.info(f"   R2 Score : {r2}   (1.0 = perfect fit)")
    logger.info("-" * 50)

    return metrics


# =============================================================================
# STEP 1 -- Load & Prepare Data
# =============================================================================

logger.info("=" * 55)
logger.info("  STOCK PRICE REGRESSION TRAINING PIPELINE")
logger.info("  Models : Random Forest | XGBoost")
logger.info("  Target : Next Day Closing Price")
logger.info("=" * 55)

logger.info("Step 1: Running pipeline (load -> preprocess -> features)...")
df = run_pipeline()

# -- Add regression target if teammate has not added it yet ------------------
if 'Next_Day_Close' not in df.columns:
    logger.warning("'Next_Day_Close' not in columns. Creating it as fallback.")
    logger.warning("Ask teammate to add this line in features.py:")
    logger.warning("  df['Next_Day_Close'] = df.groupby('Company')['Close'].shift(-1)")
    df['Next_Day_Close'] = df.groupby('Company')['Close'].shift(-1)
    df = df.dropna(subset=['Next_Day_Close']).reset_index(drop=True)

# -- Clean infinity and NaN values -------------------------------------------
# Volume_Change produces inf when previous volume = 0 (division by zero)
logger.info("Step 1b: Cleaning infinity and NaN values...")
rows_before = df.shape[0]
df = df.replace([float('inf'), float('-inf')], float('nan'))
df = df.dropna().reset_index(drop=True)
rows_after = df.shape[0]
logger.info(f"Removed {rows_before - rows_after} bad rows. Clean shape: {df.shape}")


# =============================================================================
# STEP 2 -- Define Features (X) and Target (y)
# =============================================================================

logger.info("Step 2: Defining features and target...")

# NOTE: 'Close' is intentionally excluded to prevent data leakage.
# Next_Day_Close is almost identical to today's Close, so including Close
# gives the model a shortcut that produces fake high R2 scores.
FEATURE_COLS = [
    'Open',          # today's open price
    'High',          # today's highest price
    'Low',           # today's lowest price
    'Volume',        # today's trading volume
    'Daily_Return',  # (Close - Open) / Open  -- momentum signal
    'Price_Range',   # High - Low             -- volatility signal
    'MA_5',          # 5-day moving average   -- trend signal
    'Volume_Change', # % change in volume     -- buying/selling pressure
]

TARGET_COL = 'Next_Day_Close'

# Validate all columns exist before proceeding
missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
if missing:
    logger.error(f"Missing columns: {missing}")
    raise ValueError(f"These columns are missing from the DataFrame: {missing}")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

logger.info(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
logger.info(f"Target              : {TARGET_COL}")
logger.info(f"X shape             : {X.shape}")
logger.info(f"y shape             : {y.shape}")
logger.info(f"y range             : min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")


# =============================================================================
# STEP 3 -- Train / Test Split
# =============================================================================

logger.info("Step 3: Splitting data (80% train / 20% test, no shuffle)...")

# shuffle=False is critical for time-series stock data!
# Past data trains the model, future data tests it.
# Shuffling would let the model see future data during training (cheating).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

logger.info(f"Train size : {X_train.shape[0]} rows")
logger.info(f"Test size  : {X_test.shape[0]} rows")


# =============================================================================
# STEP 4 -- Train Model 1: Random Forest Regressor
# =============================================================================

logger.info("Step 4: Training Model 1 - Random Forest Regressor...")
logger.info("(May take 2-3 minutes on 480K rows...)")

rf_model = RandomForestRegressor(
    n_estimators=100,  # 100 decision trees averaged together
    random_state=42,
    n_jobs=-1          # use all CPU cores for speed
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_metrics = compute_regression_metrics(y_test, rf_preds, "Random Forest Regressor")
logger.info("Random Forest training complete.")


# =============================================================================
# STEP 5 -- Train Model 2: XGBoost Regressor
# =============================================================================

logger.info("Step 5: Training Model 2 - XGBoost Regressor...")

xgb_model = XGBRegressor(
    n_estimators=100,
    random_state=42,
    eval_metric='rmse',  # internal loss function = Root Mean Squared Error
    verbosity=0          # suppress XGBoost internal output
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_metrics = compute_regression_metrics(y_test, xgb_preds, "XGBoost Regressor")
logger.info("XGBoost training complete.")


# =============================================================================
# STEP 6 -- Compare Both Models
# =============================================================================

logger.info("Step 6: Comparing both models...")

all_results = {
    "Random Forest Regressor": rf_metrics,
    "XGBoost Regressor":       xgb_metrics,
}

comparison_df = compare_models(all_results)

print("\n")
print("=" * 55)
print("     MODEL COMPARISON - REGRESSION RESULTS")
print("=" * 55)
print(comparison_df.to_string())
print("=" * 55)
print("MAE      -> Lower is better  (avg price error in $)")
print("RMSE     -> Lower is better  (penalizes big errors)")
print("R2_Score -> Higher is better (1.0 = perfect fit)")
print("=" * 55)


# =============================================================================
# STEP 7 -- Save the Best Model
# =============================================================================

logger.info("Step 7: Saving the best model...")

# Pick winner by R2_Score (highest = best overall fit)
best_model_name = comparison_df['R2_Score'].astype(float).idxmax()

model_map = {
    "Random Forest Regressor": rf_model,
    "XGBoost Regressor":       xgb_model,
}

best_model = model_map[best_model_name]
model_path = "models/model_v1.pkl"

save_model(best_model, model_path)

logger.info(f"Best model : {best_model_name}")
logger.info(f"Saved to   : {model_path}")
logger.info(f"R2 Score   : {comparison_df.loc[best_model_name, 'R2_Score']}")

print(f"\nBest Model : {best_model_name}")
print(f"Saved to   : {model_path}")
print("\nTraining pipeline complete!")