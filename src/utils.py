"""
utils.py — Utility functions for Yahoo Finance Stock Direction Classifier
Binary Classification: Predict UP (1) or DOWN (0) for next day closing price
"""
import sys
import os
import logging
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────
# 1. LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logger(log_file: str = "logs/app.log", level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes to both console and a log file.

    Args:
        log_file (str): Path to the log file.
        level: Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("StockML")
    logger.setLevel(level)

    # Avoid duplicate handlers on re-runs
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

        console_handler.setFormatter(formatter)

        # File handler
        file_handler = logging.FileHandler(log_file,encoding='utf-8')
        
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ─────────────────────────────────────────────
# 2. CONFIG LOADER
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to config.yaml.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at '{config_path}'. Returning empty config.")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from '{config_path}'")
    return config


# ─────────────────────────────────────────────
# 3. MODEL SAVE / LOAD
# ─────────────────────────────────────────────

def save_model(model, path: str = "models/model_v1.pkl") -> None:
    """
    Saves a trained model to disk using joblib.

    Args:
        model: Trained sklearn/XGBoost model.
        path (str): Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to '{path}'")


def load_model(path: str = "models/model_v1.pkl"):
    """
    Loads a saved model from disk.

    Args:
        path (str): Path to the saved model file.

    Returns:
        Loaded model object.
    """
    if not os.path.exists(path):
        logger.error(f"Model file not found at '{path}'")
        raise FileNotFoundError(f"No model found at '{path}'")

    model = joblib.load(path)
    logger.info(f"Model loaded from '{path}'")
    return model


# ─────────────────────────────────────────────
# 4. METRICS — Binary Classification
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes all key binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for class 1 (optional, needed for AUC-ROC).

    Returns:
        dict: Dictionary of metric names and values.
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_score":  round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


def log_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Logs all metrics in a clean readable format.

    Args:
        metrics (dict): Output from compute_metrics().
        model_name (str): Name of the model (for display).
    """
    logger.info(f"===== {model_name} Metrics =================")
    for metric, value in metrics.items():
        logger.info(f"   {metric:<12}: {value}")
    logger.info("─" * 45)


def print_classification_report(y_true, y_pred, model_name: str = "Model") -> None:
    """
    Prints a detailed sklearn classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name (str): Label for the report header.
    """
    print(f"\n{'='*45}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*45}")
    print(classification_report(y_true, y_pred, target_names=["DOWN (0)", "UP (1)"]))


# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plots and optionally saves a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name (str): Title label.
        save_path (str): If provided, saves the plot to this path.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["DOWN (0)", "UP (1)"],
        yticklabels=["DOWN (0)", "UP (1)"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to '{save_path}'")

    plt.show()


# ─────────────────────────────────────────────
# 6. COMPARE MODELS
# ─────────────────────────────────────────────

def compare_models(results: dict) -> pd.DataFrame:
    """
    Compares multiple models side by side in a DataFrame.
    Works for both Classification (sorts by f1_score)
    and Regression (sorts by R2_Score).

    Args:
        results (dict): Format — { "ModelName": metrics_dict, ... }

    Returns:
        pd.DataFrame: Comparison table sorted by best metric.

    Example (Regression):
        results = {
            "Linear Regression":      {"MAE": 2.3, "RMSE": 3.1, "R2_Score": 0.92},
            "Random Forest Regressor":{"MAE": 1.2, "RMSE": 1.9, "R2_Score": 0.97},
            "XGBoost Regressor":      {"MAE": 1.1, "RMSE": 1.7, "R2_Score": 0.98},
        }
        df = compare_models(results)
        print(df)
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"

    # Auto-detect regression vs classification by column names
    if "R2_Score" in df.columns:
        df = df.sort_values("R2_Score", ascending=False)   # higher is better
    elif "f1_score" in df.columns:
        df = df.sort_values("f1_score", ascending=False)   # higher is better

    logger.info("Model comparison table generated.")
    return df


# ─────────────────────────────────────────────
# 7. TIMESTAMP HELPER
# ─────────────────────────────────────────────

def get_timestamp() -> str:
    """
    Returns a formatted timestamp string — useful for versioning saved files.

    Returns:
        str: e.g., '2025-04-04_14-32-10'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ─────────────────────────────────────────────
# 8. DIRECTORY SETUP
# ─────────────────────────────────────────────

def ensure_dirs(dirs: list) -> None:
    """
    Creates required project directories if they don't exist.

    Args:
        dirs (list): List of directory paths to create.

    Example:
        ensure_dirs(["models/", "logs/", "outputs/"])
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Directories ensured: {dirs}")