import os
import pandas as pd

# This starts from the project folder (/Users/zer0/vscode/fsml-project-group-7)
filepath = os.path.join(os.getcwd(), "data/stock_details_5_years.csv")

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: 'stock_details_5_years.csv' not found in {os.getcwd()}")
        return None

if __name__ == "__main__":
    df = load_data(filepath)