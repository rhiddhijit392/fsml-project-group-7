"""
predict.py - Predict Next Day Closing Price using the saved best model
Output : Predicted price + Direction (UP/DOWN) + Expected change per company

Run standalone : python src/predict.py
Called by      : pipeline/pipeline.py
"""

import sys
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import setup_logger, load_model, ensure_dirs

logger = setup_logger("logs/app.log")
ensure_dirs(["logs/", "outputs/"])

FEATURE_COLS = [
    'Daily_Return', 'Price_Range', 'MA_5', 'Volume_Change',
]
MODEL_PATH = "models/model_v1.pkl"


def predict_next_day_close(input_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Core prediction function. Imported and used by app.py for FastAPI endpoint.

    Args:
        input_df : DataFrame with all 9 FEATURE_COLS columns
        model    : Pre-loaded model (if None, loads from MODEL_PATH)

    Returns:
        DataFrame with added 'Predicted_Next_Close' column
    """
    if model is None:
        model = load_model(MODEL_PATH)
    missing = [c for c in FEATURE_COLS if c not in input_df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    result = input_df.copy()
    result['Predicted_Next_Close'] = np.round(model.predict(input_df[FEATURE_COLS]), 2)
    return result


def run_prediction(df=None, model=None):
    logger.info("=" * 55)
    logger.info("  STOCK PRICE PREDICTION PIPELINE")
    logger.info("  Predicting: Next Day Closing Price")
    logger.info("=" * 55)

    # Step 1 -- Load model
    if model is None:
        model = load_model(MODEL_PATH)
    logger.info(f"Model loaded: {type(model).__name__}")

    # Step 2 -- Load data
    if df is None:
        raise ValueError("Input dataframe required for prediction")
    logger.info(f"Clean data shape: {df.shape}")

    # Step 3 -- Get last 10 rows per company and predict
    latest_df = df.groupby('Company').tail(10).reset_index(drop=True)
    logger.info(f"Predicting for {latest_df['Company'].nunique()} companies...")

    results = predict_next_day_close(latest_df, model=model)
    output  = results[['Company', 'Date', 'Open', 'High', 'Low', 'Close',
                        'MA_5', 'Daily_Return', 'Predicted_Next_Close']].copy()
    output['Direction'] = np.where(output['Predicted_Next_Close'] > output['Close'], 'UP', 'DOWN')
    output['Expected_Change($)'] = round(output['Predicted_Next_Close'] - output['Close'], 2)

    # Step 4 -- Display results
    latest_one = output.groupby('Company').tail(1).reset_index(drop=True)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)

    print("\n")
    print("=" * 75)
    print("  NEXT DAY CLOSE PRICE PREDICTIONS")
    print(f"  Model: {type(model).__name__}")
    print("=" * 75)
    print(latest_one[['Company', 'Date', 'Close', 'Predicted_Next_Close',
                       'Expected_Change($)', 'Direction']].to_string(index=False))
    print("=" * 75)
    print("  Direction: UP = predicted price higher than today's Close")
    print("             DOWN = predicted price lower than today's Close")
    print("=" * 75)

    # Step 5 -- Save predictions to CSV
    save_path = "outputs/predictions.csv"
    output.to_csv(save_path, index=False)
    logger.info(f"Predictions saved to '{save_path}'")
    print(f"\nFull predictions saved to: {save_path}")


if __name__ == "__main__":
    run_prediction()