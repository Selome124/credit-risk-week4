from data_processing import build_features
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/raw/transactions.csv")

X, y = build_features(df)

model = LogisticRegression()
model.fit(X, y)
