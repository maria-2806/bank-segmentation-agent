import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

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
        
    # Calculate cluster sizes
    cluster_counts = df_result['segment_id'].value_counts().to_dict()
    sizes = {int(k): int(v) for k, v in cluster_counts.items()}
    
    return {
        'df': df_result,
        'model': kmeans,
        'labels': labels,
        'silhouette_score': sil_score,
        'cluster_sizes': sizes,
        'outlier_thresholds': outlier_thresholds_by_cluster
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
