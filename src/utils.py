"""
utils.py - Utility functions for Yahoo Finance Stock Price Regression
Project : Predict next day closing price using Random Forest and XGBoost
"""

import sys
import os
import logging
import joblib
import yaml
import numpy as np
import pandas as pd

from datetime import datetime


# 1. LOGGING SETUP

def setup_logger(log_file: str = "logs/app.log", level=logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("StockML")
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.stream = open(sys.stdout.fileno(), mode='w',
                                      encoding='utf-8', buffering=1)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# 2. CONFIG LOADER

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at '{config_path}'. Returning empty config.")
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from '{config_path}'")
    return config


# 3. MODEL SAVE / LOAD

def save_model(model, path: str = "models/model_v1.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to '{path}'")


def load_model(path: str = "models/model_v1.pkl"):
    if not os.path.exists(path):
        logger.error(f"Model file not found at '{path}'")
        raise FileNotFoundError(f"No model found at '{path}'")
    model = joblib.load(path)
    logger.info(f"Model loaded from '{path}'")
    return model


# 4. COMPARE MODELS

def compare_models(results: dict) -> pd.DataFrame:
    """
    Compares multiple models side by side in a DataFrame.
    Auto-detects regression (sorts by R2_Score) or
    classification (sorts by f1_score).
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    if "R2_Score" in df.columns:
        df = df.sort_values("R2_Score", ascending=False)
    elif "f1_score" in df.columns:
        df = df.sort_values("f1_score", ascending=False)
    logger.info("Model comparison table generated.")
    return df


# 5. TIMESTAMP HELPER

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 6. DIRECTORY SETUP

def ensure_dirs(dirs: list) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info(f"Directories ensured: {dirs}")