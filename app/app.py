"""
app.py - FastAPI Deployment Layer for Stock Price Prediction
Endpoint : POST /predict
Input    : Stock features as JSON (defined in schema.py)
Output   : Predicted next day closing price + direction signal

Run with : uvicorn app.app:app --reload
Docs at  : http://localhost:8000/docs
"""

import sys
import os
import numpy as np
import pandas as pd

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from app.schema import StockInput, PredictionResponse
from src.utils import setup_logger, load_model

# -- Logger -------------------------------------------------------------------
logger = setup_logger("logs/app.log")

# -- Load model once at startup (not on every request) ------------------------
MODEL_PATH = "models/model_v1.pkl"

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded at startup: {type(model).__name__}")
except FileNotFoundError:
    logger.error(f"Model not found at '{MODEL_PATH}'. Run train.py first.")
    model = None

# -- Must match FEATURE_COLS in train.py exactly ------------------------------
FEATURE_COLS = [
    # 'Open',
    # 'High',
    # 'Low',
    # 'Close',
    # 'Volume',
    'Daily_Return',
    'Price_Range',
    'MA_5',
    'Volume_Change',
]

# -- Create FastAPI app -------------------------------------------------------
app = FastAPI(
    title="Stock Price Prediction API",
    description="Predicts next day closing price for a stock using a trained ML model.",
    version="1.0.0"
)


# =============================================================================
# HOME ENDPOINT
# =============================================================================

@app.get("/")
def home():
    """
    Health check endpoint.
    Returns a simple message confirming the API is running.
    """
    return {
        "message": "Stock Price Prediction API is running",
        "model": type(model).__name__ if model else "No model loaded",
        "docs": "Visit /docs to test the API"
    }


# =============================================================================
# PREDICT ENDPOINT
# =============================================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StockInput):
    """
    Predicts the next day closing price for a stock.

    Input  : Stock features for today (Open, High, Low, Volume, etc.)
    Output : Predicted next day closing price + UP/DOWN direction signal

    Example input:
        {
            "Open": 150.0,
            "High": 155.0,
            "Low": 148.0,
            "Volume": 1200000,
            "Daily_Return": 0.012,
            "Price_Range": 7.0,
            "MA_5": 149.5,
            "Volume_Change": 0.05
        }
    """

    # -- Check model is loaded ------------------------------------------------
    if model is None:
        logger.error("Prediction attempted but no model is loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run train.py first to generate model_v1.pkl."
        )

    try:
        # -- Build input DataFrame (must match training format) ---------------
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])[FEATURE_COLS]

        logger.info(f"Received prediction request: {input_dict}")

        # -- Run prediction ---------------------------------------------------
        prediction = model.predict(input_df)[0]
        prediction = round(float(prediction), 2)

        # -- Direction signal -------------------------------------------------
        # Compare predicted next close vs today's Open as proxy for current price
        direction = "UP" if prediction > data.Open else "DOWN"
        expected_change = round(prediction - data.Open, 2)

        logger.info(f"Prediction: ${prediction} | Direction: {direction}")

        # -- Return response --------------------------------------------------
        return PredictionResponse(
            predicted_next_close=prediction,
            current_close_approx=data.Open,
            direction=direction,
            expected_change=expected_change,
            model_used=type(model).__name__
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
