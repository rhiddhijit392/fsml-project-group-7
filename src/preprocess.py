import pandas as pd

def preprocess(df):
    df = df.copy()

    # 1. Clean column names
    df.columns = [col.strip() for col in df.columns]

    # 2. Convert Date column
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    # 3. Sort data
    if 'Company' in df.columns:
        df = df.sort_values(['Company', 'Date'])
    else:
        df = df.sort_values('Date')

    # 4. Handle missing values
    df = df.ffill()

    # 5. Reset index
    df = df.reset_index(drop=True)
    print(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")

    return df