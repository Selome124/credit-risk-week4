import pandas as pd
from data_processing import build_features
from target_engineering import build_proxy_target


def main():
    df = pd.read_csv("data/raw/data.csv")

    # Build proxy target
    df = build_proxy_target(df)

    # Feature engineering
    X, y = build_features(df)

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("Target distribution:")
    print(pd.Series(y).value_counts())


if __name__ == "__main__":
    main()



import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 1. Aggregate Features Transformer
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        agg = X.groupby(self.group_col)[self.agg_col].agg(['sum', 'mean', 'count', 'std']).reset_index()
        agg.columns = [self.group_col, 'total_amount', 'avg_amount', 'transaction_count', 'std_amount']
        X = pd.merge(X, agg, on=self.group_col, how='left')
        X['std_amount'] = X['std_amount'].fillna(0)
        return X

# 2. Date Features Transformer
class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['TransactionHour'] = X[self.date_col].dt.hour
        X['TransactionDay'] = X[self.date_col].dt.day
        X['TransactionMonth'] = X[self.date_col].dt.month
        X['TransactionYear'] = X[self.date_col].dt.year
        return X

def build_feature_pipeline():
    # These columns will be added or used for transformations
    numeric_features = ['Amount', 'Value', 'total_amount', 'avg_amount', 'transaction_count', 'std_amount']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    date_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']

    # Numeric pipeline: impute + scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute + one-hot encode
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Compose all preprocessing
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('date', 'passthrough', date_features)
    ])

    # Full pipeline chaining custom transformers + preprocessing
    pipeline = Pipeline([
        ('aggregate', AggregateFeatures()),
        ('datefeat', DateFeatures()),
        ('preprocessor', preprocessor)
    ])

    return pipeline

def build_features(df):
    target_col = 'FraudResult'
    y = df[target_col].values if target_col in df.columns else None

    pipeline = build_feature_pipeline()
    X_transformed = pipeline.fit_transform(df)

    return X_transformed, y
