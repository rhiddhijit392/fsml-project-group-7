import pandas as pd

def build_features(df):
    df = df.copy()

    # 1. Daily return
    df['Daily_Return'] = (df['Close'] - df['Open']) / df['Open']

    # 2. Price range
    df['Price_Range'] = df['High'] - df['Low']

    # 3. 5 day moving average per company
    df['MA_5'] = df.groupby('Company')['Close'].transform(
        lambda x: x.rolling(window=5).mean()
    )

    # 4. Volume change per company
    df['Volume_Change'] = df.groupby('Company')['Volume'].pct_change()

    # 5. Tommorow's closing > today's closing
    df['Next_Day_UP'] = (
        df.groupby('Company')['Close'].shift(-1) > df['Close']
    ).astype(int)
    df['Next_Day_Close'] = df.groupby('Company')['Close'].shift(-1)

    # 6. Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df