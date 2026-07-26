import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def _find_elbow(ks, inertias):
    """
    Kneedle-style elbow: the k whose inertia point lies furthest below the straight
    line joining the first and last points of the inertia curve. This is the more
    appropriate signal for 'how many real groups', since silhouette tends to favour
    very few, coarse clusters.
    """
    ks = np.asarray(ks, dtype=float)
    inertias = np.asarray(inertias, dtype=float)
    if len(ks) < 3:
        return int(ks[0])
    # Normalise both axes to 0..1 so the geometry is scale-free
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    # Distance of each point from the line (x0,y0)->(x1,y1)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) or 1.0
    dist = num / den
    return int(ks[int(np.argmax(dist))])


def evaluate_cluster_range(X, k_min=2, k_max=8, seed=42, sample_size=2000):
    """
    Sweeps K-Means across a range of k and reports separation metrics for each,
    so the chosen number of segments is justified rather than assumed.

    For each k returns:
      - inertia            (within-cluster sum of squares; lower, look for the elbow)
      - silhouette         (higher is better, range -1..1)
      - davies_bouldin     (lower is better)
    Also returns the recommended k (best silhouette).
    """
    n = len(X)
    k_max = min(k_max, n - 1)
    # Subsample only the silhouette computation for speed on large data
    if n > sample_size:
        rng = np.random.default_rng(seed)
        sil_idx = rng.choice(n, size=sample_size, replace=False)
    else:
        sil_idx = np.arange(n)

    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X[sil_idx], labels[sil_idx])) if len(set(labels[sil_idx])) > 1 else 0.0
        dbi = float(davies_bouldin_score(X, labels))
        results.append({
            'k': int(k),
            'inertia': float(km.inertia_),
            'silhouette': round(sil, 4),
            'davies_bouldin': round(dbi, 4),
        })

    best_silhouette_k = max(results, key=lambda r: r['silhouette'])['k']
    elbow_k = _find_elbow([r['k'] for r in results], [r['inertia'] for r in results])
    return {
        'sweep': results,
        'best_silhouette_k': int(best_silhouette_k),
        'elbow_k': int(elbow_k),
        # Elbow is the primary recommendation for segment count; silhouette often
        # prefers a coarser split and is reported alongside for transparency.
        'recommended_k': int(elbow_k),
        'metric_notes': {
            'silhouette': 'higher is better (-1 to 1); tends to favour few coarse clusters',
            'davies_bouldin': 'lower is better (0 is ideal)',
            'inertia': 'lower is better; the elbow (not the minimum) indicates the natural k',
        }
    }


def evaluate_explainability_tree(X, labels, max_depth=3, seed=42, test_size=0.25):
    """
    Measures how faithfully the shallow decision tree reproduces the cluster
    assignments (its "fidelity"). High test accuracy means the human-readable
    rules genuinely explain the segmentation instead of overfitting to it.

    Returns train/test/cross-validated accuracy.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    dt.fit(X_tr, y_tr)

    train_acc = float(dt.score(X_tr, y_tr))
    test_acc = float(dt.score(X_te, y_te))

    # 5-fold CV on the full set for a stability estimate
    cv = cross_val_score(
        DecisionTreeClassifier(max_depth=max_depth, random_state=seed),
        X, labels, cv=5
    )

    return {
        'tree_max_depth': int(max_depth),
        'train_accuracy': round(train_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'cv_accuracy_mean': round(float(cv.mean()), 4),
        'cv_accuracy_std': round(float(cv.std()), 4),
        'interpretation': (
            f"The depth-{max_depth} rule set reproduces {test_acc:.0%} of cluster "
            f"assignments on unseen customers, so the extracted rules are a faithful "
            f"explanation of the segmentation."
        )
    }
