import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_proxy_target(df):
    # Convert to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define snapshot date as one day after last transaction
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Compute RFM features by CustomerId
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })

    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Cluster into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)

    rfm['Cluster'] = clusters

    # Analyze cluster means to find high-risk cluster
    cluster_summary = rfm.groupby('Cluster').mean()
    # The high-risk cluster typically has highest Recency and lowest Frequency & Monetary
    high_risk_cluster = cluster_summary.sort_values(
        by=['Recency', 'Frequency', 'Monetary'],
        ascending=[False, True, True]
    ).index[0]

    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

    # Merge is_high_risk back to original dataframe
    df = df.merge(rfm[['is_high_risk']], left_on='CustomerId', right_index=True, how='left')

    return df

# Usage
df = pd.read_csv("data/raw/data.csv")
df_with_target = create_proxy_target(df)
df_with_target.to_csv("data/raw/data_with_target.csv", index=False)
print("New dataset with 'is_high_risk' saved as data_with_target.csv")
