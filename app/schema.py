"""
schema.py - Input/Output data models for the Stock Prediction API
Defines what data the /predict endpoint expects and returns.
Uses Pydantic for automatic data validation.
"""

from pydantic import BaseModel, Field


class StockInput(BaseModel):
    """
    Expected input for a single stock prediction request.
    All fields must match the FEATURE_COLS used in train.py exactly.
    """
    Open: float = Field(..., description="Today's opening price", example=150.0)
    High: float = Field(..., description="Today's highest price", example=155.0)
    Low: float = Field(..., description="Today's lowest price", example=148.0)
    Close: float
    Volume: float = Field(..., description="Today's trading volume", example=1200000.0)
    Daily_Return: float = Field(..., description="(Close - Open) / Open", example=0.012)
    Price_Range: float = Field(..., description="High - Low", example=7.0)
    MA_5: float = Field(..., description="5-day moving average of Close", example=149.5)
    Volume_Change: float = Field(..., description="% change in volume from previous day", example=0.05)


class PredictionResponse(BaseModel):
    """
    Response returned by the /predict endpoint.
    """
    predicted_next_close: float = Field(..., description="Predicted next day closing price in $")
    current_close_approx: float = Field(..., description="Approximate current close (Open used as proxy)")
    direction: str = Field(..., description="UP or DOWN based on prediction vs current open")
    expected_change: float = Field(..., description="Predicted change in $ from current open")
    model_used: str = Field(..., description="Name of the model that made the prediction")
