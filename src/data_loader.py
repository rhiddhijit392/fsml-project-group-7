import os
import pandas as pd

def load_data(filepath=None):
    try:
        # Default path if not provided
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            filepath = os.path.join(base_dir, "data", "stock_details_5_years.csv")
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

if __name__ == "__main__":
    df = load_data()
    print("sample data:", df.head())