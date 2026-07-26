import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

# Canonical segment IDs. recommendation_tool.py and the orchestrator rely on these
# exact meanings, so raw K-Means indices MUST be remapped onto them before use.
CANONICAL_SEGMENTS = {
    0: "Priority",
    1: "Regular",
    2: "Dormant",
    3: "Overleveraged",
    4: "Wealth/Investor",
}

# Archetype fingerprints expressed as weights over z-scored cluster-mean features.
# "Regular" is handled separately (the cluster closest to the global average).
_ARCHETYPE_SIGNATURES = {
    "Priority":        {"account_balance": 1.0, "transaction_frequency": 1.0,
                        "annual_income": 0.8, "credit_score": 0.8,
                        "online_login_frequency": 0.5},
    "Dormant":         {"account_balance": -0.8, "transaction_frequency": -1.2,
                        "online_login_frequency": -1.0, "annual_income": -0.4},
    "Overleveraged":   {"debt_to_income": 1.2, "credit_card_utilization": 1.2,
                        "account_balance": -0.6, "transaction_frequency": 0.4},
    "Wealth/Investor": {"account_balance": 1.0, "annual_income": 1.0,
                        "investment_balance": 1.4, "transaction_frequency": -0.5},
}


def assign_semantic_labels(df_result, num_clusters, segment_col='segment_id'):
    """
    Remaps arbitrary K-Means cluster indices onto canonical, meaningful segment IDs
    by matching each cluster's average profile to a banking persona.

    - When num_clusters == 5, solves an optimal one-to-one assignment (Hungarian)
      so every canonical persona is used exactly once.
    - Otherwise, falls back to ranking clusters by a composite value score, so
      ID 0 is always the highest-value ("Priority-like") cluster.

    Returns (relabelled_df, mapping) where mapping is {raw_cluster_id: canonical_id}.
    """
    feats = ["account_balance", "transaction_frequency", "annual_income", "credit_score",
             "debt_to_income", "credit_card_utilization", "online_login_frequency",
             "investment_balance"]
    feats = [f for f in feats if f in df_result.columns]

    prof = df_result.groupby(segment_col)[feats].mean()
    raw_ids = list(prof.index)

    # z-score each feature ACROSS clusters so personas are compared on relative standing
    mu = prof.mean(axis=0)
    sd = prof.std(axis=0).replace(0, 1.0)
    z = (prof - mu) / sd

    if num_clusters != 5:
        # Fallback: rank by value; guarantees unique IDs 0..k-1, Priority-like = 0
        value_w = {"account_balance": 1.0, "annual_income": 1.0, "transaction_frequency": 0.7,
                   "credit_score": 0.7, "investment_balance": 0.7,
                   "debt_to_income": -0.8, "credit_card_utilization": -0.6}
        score = sum(z[f] * w for f, w in value_w.items() if f in z.columns)
        order = score.sort_values(ascending=False).index.tolist()
        mapping = {raw: new for new, raw in enumerate(order)}
    else:
        archetype_order = ["Priority", "Regular", "Dormant", "Overleveraged", "Wealth/Investor"]
        S = np.zeros((5, 5))
        for i, raw in enumerate(raw_ids):
            zc = z.loc[raw].values
            for j, arch in enumerate(archetype_order):
                if arch == "Regular":
                    S[i, j] = -float(np.sqrt((zc ** 2).sum()))  # nearest the global mean
                else:
                    sig = _ARCHETYPE_SIGNATURES[arch]
                    vec = np.array([sig.get(f, 0.0) for f in z.columns])
                    norm = np.linalg.norm(vec) or 1.0
                    S[i, j] = float(np.dot(zc, vec) / norm)
        # maximise total similarity -> minimise negative similarity
        row_ind, col_ind = linear_sum_assignment(-S)
        mapping = {raw_ids[i]: int(j) for i, j in zip(row_ind, col_ind)}

    df_out = df_result.copy()
    df_out[segment_col] = df_out[segment_col].map(mapping).astype(int)
    df_out['segment_name'] = df_out[segment_col].map(
        lambda k: CANONICAL_SEGMENTS.get(k, f"Segment {k}"))
    return df_out, mapping

def run_kmeans_clustering(df_engineered, X, features_list, num_clusters=3, seed=42):
    """
    Fits K-Means model on preprocessed features X, adds 'segment_id' and 'segment_name' to df_engineered,
    calculates silhouette score, finds centroids, and flags outliers (edge cases).
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score (subsample to 2000 points if dataset is huge for speed)
    sample_size = min(2000, len(X))
    if sample_size > 10:
        sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
        sil_score = float(silhouette_score(X[sample_indices], labels[sample_indices]))
    else:
        sil_score = 0.0
        
    df_result = df_engineered.copy()
    df_result['segment_id'] = labels
    
    # Identify edge cases (Outliers: top 5% furthest from their respective cluster centroids)
    distances = kmeans.transform(X)
    dist_to_assigned_centroid = distances[np.arange(len(X)), labels]
    
    outlier_threshold = np.percentile(dist_to_assigned_centroid, 95)
    df_result['is_outlier'] = (dist_to_assigned_centroid > outlier_threshold).astype(int)
    
    # Outlier threshold distance per cluster
    outlier_thresholds_by_cluster = {}
    for c in range(num_clusters):
        c_mask = (labels == c)
        if np.any(c_mask):
            c_dists = dist_to_assigned_centroid[c_mask]
            outlier_thresholds_by_cluster[c] = float(np.percentile(c_dists, 95))
            
    # Calculate distances to nearest other cluster center (boundary proxy)
    # The margin between distance to own centroid and distance to second closest centroid.
    sorted_dists = np.sort(distances, axis=1)
    # If num_clusters > 1, margin is dist[1] - dist[0]. Small margin means customer is close to decision boundary.
    if num_clusters > 1:
        boundary_margin = sorted_dists[:, 1] - sorted_dists[:, 0]
        # Flag bottom 5% smallest margins as boundary cases
        boundary_threshold = np.percentile(boundary_margin, 5)
        df_result['is_boundary'] = (boundary_margin < boundary_threshold).astype(int)
    else:
        df_result['is_boundary'] = 0
        
    # Remap arbitrary K-Means indices -> canonical semantic segment IDs.
    # Everything downstream (recommendations, aggregation, conversion) depends on this.
    df_result, label_map = assign_semantic_labels(df_result, num_clusters)
    labels = df_result['segment_id'].to_numpy()
    outlier_thresholds_by_cluster = {
        int(label_map.get(c, c)): v for c, v in outlier_thresholds_by_cluster.items()
    }

    # Calculate cluster sizes (keyed by canonical segment ID)
    cluster_counts = df_result['segment_id'].value_counts().to_dict()
    sizes = {int(k): int(v) for k, v in cluster_counts.items()}

    return {
        'df': df_result,
        'model': kmeans,
        'labels': labels,
        'silhouette_score': sil_score,
        'cluster_sizes': sizes,
        'outlier_thresholds': outlier_thresholds_by_cluster,
        'label_map': label_map,
        'segment_names': CANONICAL_SEGMENTS
    }

def run_dbscan_clustering(df_engineered, X, eps=0.5, min_samples=5):
    """
    Fits DBSCAN model on preprocessed features X. Labels outliers as -1.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Calculate silhouette score (excluding noise points -1 if they make up a huge chunk, or just standard)
    non_noise_mask = (labels != -1)
    if np.sum(non_noise_mask) > 10 and len(np.unique(labels[non_noise_mask])) > 1:
        sample_indices = np.where(non_noise_mask)[0]
        if len(sample_indices) > 2000:
            sample_indices = np.random.choice(sample_indices, size=2000, replace=False)
        sil_score = float(silhouette_score(X[sample_indices], labels[sample_indices]))
    else:
        sil_score = -1.0 # Invalid for single cluster or only noise
        
    df_result = df_engineered.copy()
    df_result['segment_id'] = labels
    df_result['is_outlier'] = (labels == -1).astype(int)
    df_result['is_boundary'] = 0 # DBSCAN doesn't naturally project boundary margin in the same way K-means does
    
    cluster_counts = df_result['segment_id'].value_counts().to_dict()
    sizes = {int(k): int(v) for k, v in cluster_counts.items()}
    
    return {
        'df': df_result,
        'model': dbscan,
        'labels': labels,
        'silhouette_score': sil_score,
        'cluster_sizes': sizes
    }
