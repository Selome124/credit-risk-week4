import pandas as pd
from src.data_processing import clean_data

def test_clean_data_removes_nulls():
    df = pd.DataFrame({"a": [1, None, 3]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
