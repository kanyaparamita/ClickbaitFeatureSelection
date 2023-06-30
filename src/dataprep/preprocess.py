import pandas as pd

def preprocess_data(df: pd.DataFrame, label: str) -> pd.DataFrame:
    # Drop rows with empty text
    df.drop(df[df['headline'] == ""].index, inplace=True)
    # Put the labels
    df['label'] = label
    return df

def balance_data(df1: pd.DataFrame, df2: pd.DataFrame, n_samples: int):
    df1 = df1.sample(n=n_samples)
    df2 = df2.sample(n=n_samples)
    return df1, df2