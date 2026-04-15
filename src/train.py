"""
train.py - Train 2 Regression Models to Predict Next Day Closing Price
Models  : Random Forest Regressor, XGBoost Regressor
Target  : Next_Day_Close (tomorrow's closing price)
Dataset : Yahoo Finance - 5 Years Stock Data (600K+ rows)

Run standalone : python src/train.py
Called by      : pipeline/pipeline.py
"""

import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils import setup_logger, load_config, save_model, ensure_dirs, compare_models

logger = setup_logger("logs/app.log")
config = load_config("config.yaml")
ensure_dirs(["models/", "logs/"])

tscv = TimeSeriesSplit(n_splits=5)

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Daily_Return', 'Price_Range', 'MA_5', 'Volume_Change',
]
TARGET_COL = 'Next_Day_Close'
MODEL_PATH  = "models/model_v1.pkl"


def compute_regression_metrics(y_true, y_pred, model_name: str) -> dict:
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


def run_training(df: pd.DataFrame):
    logger.info("=" * 55)
    logger.info("  STOCK PRICE REGRESSION TRAINING PIPELINE")
    logger.info("  Models : Random Forest | XGBoost")
    logger.info("  Target : Next Day Closing Price")
    logger.info("=" * 55)

    # Step 3 -- Define features and target
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    logger.info(f"X shape: {X.shape} | y range: {y.min():.2f} to {y.max():.2f}")

    # Step 4 -- TimeSeriesSplit
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        saved_test_idx  = test_idx
        print(f"Fold {fold+1}: Train={len(train_idx):,} | Test={len(test_idx):,}")

    logger.info(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    # Step 5 -- Train Random Forest Regressor
    logger.info("Training Random Forest Regressor (2-3 mins)...")
    rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
    rf_model.fit(X_train, y_train)
    rf_preds   = rf_model.predict(X_test)
    rf_metrics = compute_regression_metrics(y_test, rf_preds, "Random Forest Regressor")
    logger.info("Random Forest training complete.")

    # Step 6 -- Train XGBoost Regressor
    logger.info("Training XGBoost Regressor...")
    xgb_model = XGBRegressor(
    n_estimators=100,
    random_state=42,
    eval_metric='rmse',
    verbosity=0
)
    xgb_model.fit(X_train, y_train)
    xgb_preds   = xgb_model.predict(X_test)
    xgb_metrics = compute_regression_metrics(y_test, xgb_preds, "XGBoost Regressor")
    logger.info("XGBoost training complete.")

    # Step 7 -- Compare both models
    comparison_df = compare_models({
        "Random Forest Regressor": rf_metrics,
        "XGBoost Regressor":       xgb_metrics,
    })
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

    # Step 8 -- Save the best model
    best_model_name = comparison_df['R2_Score'].astype(float).idxmax()
    model_map = {
        "Random Forest Regressor": rf_model,
        "XGBoost Regressor":       xgb_model,
    }
    best_model = model_map[best_model_name]
    save_model(best_model, MODEL_PATH)

    logger.info(f"Best model : {best_model_name}")
    logger.info(f"Saved to   : {MODEL_PATH}")
    logger.info(f"R2 Score   : {comparison_df.loc[best_model_name, 'R2_Score']}")

    print(f"\nBest Model : {best_model_name}")
    print(f"Saved to   : {MODEL_PATH}")
    print("\nTraining pipeline complete!")

    return best_model, X_test, y_test, saved_test_idx

#standalone run warning
if __name__ == "__main__":
    print("train.py is a module only. Use pipeline.py for execution.")