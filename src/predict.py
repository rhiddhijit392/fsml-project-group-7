"""
predict.py - Predict Next Day Closing Price using the saved best model
Loads  : models/model_v1.pkl  (saved by train.py)
Input  : Either a single stock's latest data OR last N rows from dataset
Output : Predicted next day closing price for each input row

Run this AFTER train.py has been run at least once.
Command: python src/predict.py
"""

import sys
import os
import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.pipeline import run_pipeline
from src.utils import setup_logger, load_model, ensure_dirs

# -- Logger & directories -----------------------------------------------------
logger = setup_logger("logs/app.log")
ensure_dirs(["logs/", "outputs/"])

# -- Must match train.py exactly ----------------------------------------------
FEATURE_COLS = [
    'Open',
    'High',
    'Low',
    'Volume',
    'Daily_Return',
    'Price_Range',
    'MA_5',
    'Volume_Change',
]
MODEL_PATH = "models/model_v1.pkl"


# =============================================================================
# PREDICTION FUNCTION — core logic, reusable by app.py later
# =============================================================================

def predict_next_day_close(input_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predicts next day closing price for each row in input_df.

    Args:
        input_df (pd.DataFrame): Must contain all 8 FEATURE_COLS columns.
        model: Optional pre-loaded model. If None, loads from MODEL_PATH.

    Returns:
        pd.DataFrame: input_df with an added 'Predicted_Next_Close' column.

    Example:
        df = pd.DataFrame([{
            'Open': 150.0, 'High': 153.0, 'Low': 149.0, 'Volume': 1200000,
            'Daily_Return': 0.012, 'Price_Range': 4.0,
            'MA_5': 148.5, 'Volume_Change': 0.05
        }])
        result = predict_next_day_close(df)
        print(result['Predicted_Next_Close'])
    """
    if model is None:
        model = load_model(MODEL_PATH)

    # Validate all required features are present
    missing = [c for c in FEATURE_COLS if c not in input_df.columns]
    if missing:
        logger.error(f"Missing feature columns: {missing}")
        raise ValueError(f"Input is missing these columns: {missing}")

    X = input_df[FEATURE_COLS]
    predictions = model.predict(X)

    result = input_df.copy()
    result['Predicted_Next_Close'] = np.round(predictions, 2)

    return result


# =============================================================================
# STEP 1 -- Load Model
# =============================================================================

logger.info("=" * 55)
logger.info("  STOCK PRICE PREDICTION PIPELINE")
logger.info("  Predicting: Next Day Closing Price")
logger.info("=" * 55)

logger.info(f"Step 1: Loading model from '{MODEL_PATH}'...")
model = load_model(MODEL_PATH)
logger.info(f"Model loaded: {type(model).__name__}")


# =============================================================================
# STEP 2 -- Load Latest Data from Pipeline
# =============================================================================

logger.info("Step 2: Loading data from pipeline...")
df = run_pipeline()

# Clean inf/NaN
df = df.replace([float('inf'), float('-inf')], float('nan'))
df = df.dropna().reset_index(drop=True)
logger.info(f"Clean data shape: {df.shape}")


# =============================================================================
# STEP 3 -- Predict on Last 10 Rows per Company (most recent trading days)
# =============================================================================

logger.info("Step 3: Predicting next day close for latest data per company...")

# Get the last 10 rows per company (most recent 10 trading days)
# These are the rows with real upcoming prediction value
latest_df = (
    df.groupby('Company')
    .tail(10)
    .reset_index(drop=True)
)

logger.info(f"Predicting on {len(latest_df)} rows across {latest_df['Company'].nunique()} companies...")

# Run predictions
results = predict_next_day_close(latest_df, model=model)

# Build clean output table
output_cols = ['Company', 'Date', 'Open', 'High', 'Low', 'Close',
               'MA_5', 'Daily_Return', 'Predicted_Next_Close']
output = results[output_cols].copy()

# Add direction signal: will price go UP or DOWN tomorrow?
output['Direction'] = output.apply(
    lambda row: 'UP' if row['Predicted_Next_Close'] > row['Close'] else 'DOWN',
    axis=1
)

# Add expected change in dollars
output['Expected_Change($)'] = round(output['Predicted_Next_Close'] - output['Close'], 2)


# =============================================================================
# STEP 4 -- Display Results
# =============================================================================

logger.info("Step 4: Displaying prediction results...")

print("\n")
print("=" * 75)
print("  NEXT DAY CLOSE PRICE PREDICTIONS")
print(f"  Model: {type(model).__name__}")
print("=" * 75)

# Show sample — one latest prediction per company (just last row)
latest_one = (
    output.groupby('Company')
    .tail(1)
    .reset_index(drop=True)
)

# Display formatted
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

print(latest_one[['Company', 'Date', 'Close', 'Predicted_Next_Close',
                   'Expected_Change($)', 'Direction']].to_string(index=False))

print("=" * 75)
print("  Direction: UP = predicted price higher than today's Close")
print("             DOWN = predicted price lower than today's Close")
print("=" * 75)


# =============================================================================
# STEP 5 -- Save Predictions to CSV
# =============================================================================

logger.info("Step 5: Saving predictions to CSV...")

save_path = "outputs/predictions.csv"
output.to_csv(save_path, index=False)

logger.info(f"Predictions saved to '{save_path}'")
print(f"\nFull predictions saved to: {save_path}")
print("\nPrediction pipeline complete!")