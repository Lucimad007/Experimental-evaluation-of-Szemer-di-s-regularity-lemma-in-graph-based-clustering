"""k-means-based partition baseline (§4.2, fig. 11 of hou et al., pr 171 (2026)).

replaces the regularity-partitioning step of algorithm 1 with a plain k-means
partition of the feature vectors (vertex sampling), keeping every other step
identical. the paper uses this to show that regularity partitioning (edge/structure
sampling) outperforms k-means partitioning (vertex sampling).

see ``spec/experiments.md`` exp 2b.
"""

# partition vs clustering timers
import time

# arrays
import numpy as np
# sklearn k-means on features
from sklearn.cluster import KMeans

# paper §3.3 d₀ (unnamed; 0 = keep all eq. 3 weights)
from ..szemeredi import apply_density_threshold


def _reduced_matrix_from_partition(sim_mat, classes, k, density_threshold=0):
    """build a k×k reduced matrix with weighted densities (eq. 3) for a given
    partition. mirrors ``szemerediRegularityLemma.generate_reduced_sim_mat`` but
    uses an externally provided partition (here, from k-means).
    """
    # r has one vertex per k-means class (no exceptional v0)
    R = np.zeros((k, k))
    # class indices 0..k-1
    for r in range(k):
        r_idx = np.where(classes == r)[0]
        if r_idx.size == 0:
            continue
        # unordered pairs only
        for s in range(r + 1, k):
            s_idx = np.where(classes == s)[0]
            if s_idx.size == 0:
                continue
            # |vr|×|vs| block of original similarities
            block = sim_mat[np.ix_(r_idx, s_idx)]
            # paper eq. 3: mean similarity of the pair (not necessarily equitable)
            density = block.sum() / (r_idx.size * s_idx.size)
            R[r, s] = density
            # r is undirected
            R[s, r] = density
    # same d₀ cutoff as the regularity pipeline
    return apply_density_threshold(R, density_threshold)


def kmeans_partition_clustering(
    base_algorithm,
    X,
    sim_mat,
    n_clusters,
    k_classes,
    random_state=314,
    density_threshold=0,
):
    """algorithm 1 with k-means partitioning instead of regularity partitioning.

    parameters
    ----------
    base_algorithm : callable
        ``f(sim_mat, n_clusters=none) -> labels``.
    X : np.ndarray
        feature matrix (n×d) used for the k-means partition.
    sim_mat : np.ndarray
        n×n similarity matrix used to build the reduced graph and to map v0.
    n_clusters : int or none
        ground-truth k for spc/sprg; ``none`` for apc/dset.
    k_classes : int
        number of k-means classes to partition into (analogous to the regularity
        partition's ``k``; choose similar magnitude for a fair comparison).
    """
    # features for vertex sampling
    X = np.asarray(X, dtype=float)

    # start partition timer
    t0 = time.time()
    # fig. 11: k-means on vertices instead of regularity on edges
    km = KMeans(n_clusters=k_classes, random_state=random_state, n_init=10)
    # labels in {0..k-1}, no exceptional class
    classes = km.fit_predict(X)
    # eq. 3 reduced graph from this partition
    R = _reduced_matrix_from_partition(
        sim_mat, classes, k_classes, density_threshold=density_threshold
    )
    partition_time = time.time() - t0

    # algorithm 1 line 19 on this r
    t1 = time.time()
    if n_clusters is None:
        reduced_labels = base_algorithm(R)
    else:
        reduced_labels = base_algorithm(R, n_clusters)
    clustering_time = time.time() - t1

    # every vertex inherits its class's reduced label (no v0)
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
