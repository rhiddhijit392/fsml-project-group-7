import pandas as pd

def preprocess(df):
    df = df.copy()
    # 1. Clean column names
    df.columns = [col.strip() for col in df.columns]

    # 2. Convert Date column
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    # 3. Sort data
    df = df.sort_values(['Company', 'Date'])

    # 4. Handle missing values per company
    df = df.set_index('Company')
    df = df.groupby(level=0).ffill()

    # 5. Reset index
    df = df.reset_index().reset_index(drop=True)
    print(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")

    return df