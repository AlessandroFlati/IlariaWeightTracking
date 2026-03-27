import pandas as pd

def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df["days"] = (df["date"] - df["date"].iloc[0]).dt.days.astype(float)
    return df
