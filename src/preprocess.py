import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # 1. Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Sort by company and date
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    
    # 3. Encode company name
    le = LabelEncoder()
    df['Company_encoded'] = le.fit_transform(df['Company'])
    
    # 4. Drop unnecessary columns
    df = df.drop(columns=['Company'])
    
    return df