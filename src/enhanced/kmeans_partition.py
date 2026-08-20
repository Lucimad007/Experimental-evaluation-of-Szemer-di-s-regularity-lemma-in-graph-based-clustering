"""k-means-based partition baseline (§4.2, Fig. 11).

Replaces the regularity-partitioning step of Algorithm 1 with a plain k-means
partition of the feature vectors (vertex sampling), keeping every other step
identical. The paper uses this to show that regularity partitioning (edge/structure
sampling) outperforms k-means partitioning (vertex sampling).

See ``spec/experiments.md`` Exp 2b.
"""

import time

import numpy as np
from sklearn.cluster import KMeans


def _reduced_matrix_from_partition(sim_mat, classes, k):
    """Build a k×k reduced matrix with weighted densities (Eq. 3) for a given
    partition. Mirrors ``SzemerediRegularityLemma.generate_reduced_sim_mat`` but
    uses an externally provided partition (here, from k-means)."""
    R = np.zeros((k, k))
    for r in range(k):
        r_idx = np.where(classes == r)[0]
        if r_idx.size == 0:
            continue
        for s in range(r + 1, k):
            s_idx = np.where(classes == s)[0]
            if s_idx.size == 0:
                continue
            block = sim_mat[np.ix_(r_idx, s_idx)]
            density = block.sum() / (r_idx.size * s_idx.size)
            R[r, s] = density
            R[s, r] = density
    return R


def kmeans_partition_clustering(
    base_algorithm,
    X,
    sim_mat,
    n_clusters,
    k_classes,
    random_state=314,
):
    """Algorithm 1 with k-means partitioning instead of regularity partitioning.

    Parameters
    ----------
    base_algorithm : callable
        ``f(sim_mat, n_clusters=None) -> labels``.
    X : np.ndarray
        Feature matrix (n×d) used for the k-means partition.
    sim_mat : np.ndarray
        n×n similarity matrix used to build the reduced graph and to map V0.
    n_clusters : int or None
        Ground-truth k for SPC/SPRG; ``None`` for APC/DSet.
    k_classes : int
        Number of k-means classes to partition into (analogous to the regularity
        partition's ``k``; choose similar magnitude for a fair comparison).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    t0 = time.time()
    km = KMeans(n_clusters=k_classes, random_state=random_state, n_init=10)
    classes = km.fit_predict(X)  # labels in {0..k-1}, no exceptional class
    R = _reduced_matrix_from_partition(sim_mat, classes, k_classes)
    partition_time = time.time() - t0

    t1 = time.time()
    if n_clusters is None:
        reduced_labels = base_algorithm(R)
    else:
        reduced_labels = base_algorithm(R, n_clusters)
    clustering_time = time.time() - t1

    labels = reduced_labels[classes]
    info = {
        "k": k_classes,
        "classes_cardinality": None,
        "compression_time": partition_time,
        "clustering_time": clustering_time,
        "total_time": partition_time + clustering_time,
        "v0_size": 0,
        "index_vec": [],
    }
    return labels, info
