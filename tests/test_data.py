import pandas as pd
from src.data import load_weights

def test_load_weights_returns_dataframe():
    df = load_weights("weights.csv")
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "weight" in df.columns
    assert "days" in df.columns
    assert df["days"].iloc[0] == 0
    assert str(df["date"].dtype).startswith("datetime64")
    assert len(df) > 0

def test_load_weights_days_monotonically_increase():
    df = load_weights("weights.csv")
    assert (df["days"].diff().dropna() > 0).all()
