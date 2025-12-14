import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build proxy credit risk target using RFM + KMeans clustering.
    Returns original dataframe with `is_high_risk` column added.
    """

    df = df.copy()

    # Ensure datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Snapshot date (1 day after last transaction)
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    # --------------------
    # RFM CALCULATION
    # --------------------
    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum")
        )
        .reset_index()
    )

    # --------------------
    # SCALING
    # --------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    # --------------------
    # K-MEANS CLUSTERING
    # --------------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # --------------------
    # IDENTIFY HIGH-RISK CLUSTER
    # --------------------
    cluster_summary = (
        rfm.groupby("cluster")[["Frequency", "Monetary"]]
        .mean()
    )

    high_risk_cluster = cluster_summary.sum(axis=1).idxmin()

    # Assign binary target
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    # --------------------
    # MERGE BACK
    # --------------------
    df = df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    return df
