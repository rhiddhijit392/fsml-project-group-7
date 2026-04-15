import pandas as pd

def build_features(df):
    df = df.copy()
    # Shift all price data by 1 to represent "Yesterday's" data (ensures no "Today" data is used to predict "Tomorrow")

    # 1. Daily return %
    df['Daily_Return'] = df.groupby('Company')['Close'].pct_change().shift(1)

    # 2. Price range
    df['Price_Range'] = (df['High'] - df['Low']).shift(1)

    # 3. 5 day moving average
    df['MA_5'] = df.groupby('Company')['Close'].transform(
        lambda x: x.rolling(window=5).mean().shift(1)
    )

    # 4. Volume change %
    df['Volume_Change'] = df.groupby('Company')['Volume'].pct_change().shift(1)

    # 5. Target
    df['Next_Day_Close'] = df.groupby('Company')['Close'].shift(-1)

    # Drop NaN rows
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna().reset_index(drop=True)

    return df
