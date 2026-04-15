"""
evaluate.py - Evaluate the saved best model on test data
Outputs: MAE, RMSE, R2, Confusion Matrix (UP/DOWN), 3 plots

Run standalone : python src/evaluate.py
Called by      : pipeline/pipeline.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, accuracy_score, classification_report,
)
from sklearn.model_selection import TimeSeriesSplit
from src.utils import setup_logger, load_model, ensure_dirs

logger = setup_logger("logs/app.log")
ensure_dirs(["logs/", "outputs/"])

FEATURE_COLS = [
    'Daily_Return', 'Price_Range', 'MA_5', 'Volume_Change',
]
TARGET_COL = 'Next_Day_Close'
MODEL_PATH  = "models/model_v1.pkl"


def run_evaluation(df=None, model=None, X_test=None, y_test=None, test_indices=None):
    logger.info("=" * 55)
    logger.info("  MODEL EVALUATION PIPELINE")
    logger.info("  Regression Metrics + Direction Confusion Matrix")
    logger.info("=" * 55)

    # Step 1 -- Load model
    if model is None:
        model = load_model(MODEL_PATH)
    logger.info(f"Model: {type(model).__name__}")

    # # Step 2 -- Check if dataframe is available
    if df is None or len(df) == 0:
        raise ValueError("Evaluation requires valid dataframe from pipeline")

    # Step 3 -- use available dataframe
    X_test = df[FEATURE_COLS]
    y_test = df[TARGET_COL]

    logger.info(f"Test set size: {len(y_test):,} rows")

    # Step 4 -- Generate predictions and regression metrics
    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred
    mae  = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    r2   = round(r2_score(y_test, y_pred), 4)

    logger.info(f"  MAE: {mae} | RMSE: {rmse} | R2: {r2}")

    print("\n")
    print("=" * 55)
    print(f"  REGRESSION RESULTS -- {type(model).__name__}")
    print("=" * 55)
    print(f"  Test Samples : {len(y_test):,}")
    print(f"  MAE          : ${mae}  (avg price error)")
    print(f"  RMSE         : ${rmse}")
    print(f"  R2 Score     : {r2}")
    print("=" * 55)
    print(f"  - On average the model is ${mae} off per prediction")
    print(f"  - R2={r2} means model explains {round(r2*100,2)}% of price variance")
    if r2 >= 0.95:
        print("  - Verdict: Excellent fit for stock price prediction")
    elif r2 >= 0.85:
        print("  - Verdict: Good fit, acceptable for stock prediction")
    else:
        print("  - Verdict: Moderate fit, consider feature engineering")
    print("=" * 55)

    # Step 5 -- Confusion matrix (UP/DOWN direction)
    close_test       = df.loc[y_test.index, 'Close'].values
    pred_direction   = (y_pred - close_test > 0).astype(int)
    actual_direction = (y_test.values - close_test > 0).astype(int)
    cm               = confusion_matrix(actual_direction, pred_direction)
    dir_accuracy     = round(accuracy_score(actual_direction, pred_direction), 4)
    tn, fp, fn, tp   = cm.ravel()

    logger.info(f"  Direction Accuracy: {dir_accuracy} ({round(dir_accuracy*100,2)}%)")

    print("\n")
    print("=" * 55)
    print("  DIRECTION RESULTS (UP / DOWN Prediction)")
    print("=" * 55)
    print(f"  Direction Accuracy : {round(dir_accuracy*100, 2)}%")
    print(f"  True Positives     : {tp:,}  (correctly predicted UP)")
    print(f"  True Negatives     : {tn:,}  (correctly predicted DOWN)")
    print(f"  False Positives    : {fp:,}  (predicted UP, was DOWN)")
    print(f"  False Negatives    : {fn:,}  (predicted DOWN, was UP)")
    print("=" * 55)
    print(classification_report(actual_direction, pred_direction,
                                 target_names=["DOWN (0)", "UP (1)"]))

    # Step 6 -- Generate 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Model Evaluation -- {type(model).__name__}", fontsize=13)

    idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
    axes[0].scatter(y_test.values[idx], y_pred[idx],
                    alpha=0.3, s=10, color='steelblue', label='Predictions')
    mn = min(y_test.values[idx].min(), y_pred[idx].min())
    mx = max(y_test.values[idx].max(), y_pred[idx].max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
    axes[0].set_xlabel("Actual Next Day Close ($)")
    axes[0].set_ylabel("Predicted Next Day Close ($)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    axes[1].axvline(residuals.mean(), color='orange', linestyle='--', lw=1.5,
                    label=f'Mean={residuals.mean():.2f}')
    axes[1].set_xlabel("Residual (Actual - Predicted) in $")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")
    axes[1].legend()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted DOWN', 'Predicted UP'],
                yticklabels=['Actual DOWN', 'Actual UP'],
                ax=axes[2], linewidths=0.5)
    axes[2].set_title(f"Confusion Matrix\nAccuracy: {round(dir_accuracy*100,2)}%")
    axes[2].set_ylabel("Actual Direction")
    axes[2].set_xlabel("Predicted Direction")

    plt.tight_layout()
    plot_path = "outputs/evaluation_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    logger.info(f"3 plots saved to '{plot_path}'")
    print(f"\nPlots saved to: {plot_path}")
    print("  Plot 1 -- Actual vs Predicted")
    print("  Plot 2 -- Residuals Distribution")
    print("  Plot 3 -- Confusion Matrix (UP/DOWN)")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    run_evaluation()