"""
evaluate.py - Evaluate the saved best model on test data
Loads  : models/model_v1.pkl  (saved by train.py)
Runs   : full pipeline to get the same clean data
Outputs: MAE, RMSE, R2, residual plot, actual vs predicted plot

Run this AFTER train.py has been run at least once.
Command: python src/evaluate.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for all systems

# -- Path setup ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline.pipeline import run_pipeline
from src.utils import setup_logger, load_model, ensure_dirs

# -- Logger & directories -----------------------------------------------------
logger = setup_logger("logs/app.log")
ensure_dirs(["logs/", "outputs/"])


# =============================================================================
# STEP 1 -- Load the Saved Model
# =============================================================================

logger.info("=" * 55)
logger.info("  MODEL EVALUATION PIPELINE")
logger.info("  Loading saved model and evaluating on test data")
logger.info("=" * 55)

MODEL_PATH = "models/model_v1.pkl"
logger.info(f"Step 1: Loading model from '{MODEL_PATH}'...")
model = load_model(MODEL_PATH)
logger.info(f"Model loaded: {type(model).__name__}")


# =============================================================================
# STEP 2 -- Rebuild the Same Data (same as train.py)
# =============================================================================

logger.info("Step 2: Running pipeline to rebuild data...")
df = run_pipeline()

# -- Add target if not present ------------------------------------------------
if 'Next_Day_Close' not in df.columns:
    logger.warning("'Next_Day_Close' not found. Creating it as fallback.")
    df['Next_Day_Close'] = df.groupby('Company')['Close'].shift(-1)
    df = df.dropna(subset=['Next_Day_Close']).reset_index(drop=True)

# -- Clean inf/NaN ------------------------------------------------------------
df = df.replace([float('inf'), float('-inf')], float('nan'))
df = df.dropna().reset_index(drop=True)
logger.info(f"Clean data shape: {df.shape}")


# =============================================================================
# STEP 3 -- Recreate the Same Train/Test Split
# =============================================================================

logger.info("Step 3: Recreating train/test split (same as train.py)...")

# Must use EXACT same feature list and split params as train.py
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
TARGET_COL = 'Next_Day_Close'

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# shuffle=False — must match train.py exactly so we get the same test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

logger.info(f"Test set size: {X_test.shape[0]} rows")


# =============================================================================
# STEP 4 -- Generate Predictions & Compute Metrics
# =============================================================================

logger.info("Step 4: Generating predictions and computing metrics...")

y_pred = model.predict(X_test)

mae  = round(mean_absolute_error(y_test, y_pred), 4)
rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
r2   = round(r2_score(y_test, y_pred), 4)

# Residuals = actual - predicted (how far off each prediction was)
residuals = y_test.values - y_pred

logger.info("-" * 55)
logger.info(f"  Model         : {type(model).__name__}")
logger.info(f"  Test samples  : {len(y_test)}")
logger.info(f"  MAE           : {mae}  (avg $ error per prediction)")
logger.info(f"  RMSE          : {rmse} (error penalizing big mistakes)")
logger.info(f"  R2 Score      : {r2}  (1.0 = perfect)")
logger.info("-" * 55)

print("\n")
print("=" * 55)
print(f"  EVALUATION RESULTS — {type(model).__name__}")
print("=" * 55)
print(f"  Test Samples  : {len(y_test):,}")
print(f"  MAE           : ${mae}  (avg price error)")
print(f"  RMSE          : ${rmse}")
print(f"  R2 Score      : {r2}")
print("=" * 55)

# Human readable interpretation
print("\n  What this means:")
print(f"  - On average the model is ${mae} off per prediction")
print(f"  - R2={r2} means model explains {round(r2*100, 2)}% of price variance")
if r2 >= 0.95:
    print("  - Verdict: Excellent fit for stock price prediction")
elif r2 >= 0.85:
    print("  - Verdict: Good fit, acceptable for stock prediction")
else:
    print("  - Verdict: Moderate fit, consider feature engineering")
print("=" * 55)


# =============================================================================
# STEP 5 -- Plot 1: Actual vs Predicted
# =============================================================================

logger.info("Step 5: Generating Actual vs Predicted plot...")

# Sample 1000 points so plot is readable (not 120K dots)
sample_size = min(1000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Model Evaluation — {type(model).__name__}", fontsize=13)

# -- Plot 1: Actual vs Predicted scatter --------------------------------------
axes[0].scatter(
    y_test.values[idx],
    y_pred[idx],
    alpha=0.3,
    s=10,
    color='steelblue',
    label='Predictions'
)
# Perfect prediction line (if model was perfect, all dots would sit on this)
min_val = min(y_test.values[idx].min(), y_pred[idx].min())
max_val = max(y_test.values[idx].max(), y_pred[idx].max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')

axes[0].set_xlabel("Actual Next Day Close ($)")
axes[0].set_ylabel("Predicted Next Day Close ($)")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# -- Plot 2: Residuals distribution -------------------------------------------
axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
axes[1].axvline(residuals.mean(), color='orange', linestyle='--', lw=1.5,
                label=f'Mean={residuals.mean():.2f}')

axes[1].set_xlabel("Residual (Actual - Predicted) in $")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residuals Distribution")
axes[1].legend()

plt.tight_layout()

# Save plot
plot_path = "outputs/evaluation_plots.png"
plt.savefig(plot_path, dpi=120)
plt.close()

logger.info(f"Plots saved to '{plot_path}'")
print(f"\nPlots saved to: {plot_path}")
print("\nEvaluation complete!")