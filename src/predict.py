from joblib import load
import pandas as pd

def predict(input_data: pd.DataFrame, model_path: str):
    model = load(model_path)
    return model.predict(input_data)
