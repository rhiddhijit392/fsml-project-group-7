import numpy as np
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df