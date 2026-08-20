"""Dominant set clustering (DSet) via replicator dynamics (§2.3, Eq. 1).

Extracts dominant sets sequentially: each run of the replicator dynamics on the
remaining (unclustered) vertices converges to a dominant set, which is removed and
recorded as one cluster. The number of clusters is determined automatically.

Follows Pavan & Pelillo (2007) and the reference replicator implementation; see
``spec/base_algorithms.md`` §2.3.
"""

from math import ceil

import numpy as np


def _replicator(A, x, inds, tol, max_iter):
    """Replicator dynamics (Eq. 1), synchronous update.

    The update ``x_i <- x_i (Ax)_i / (x^T A x)`` is a synchronous map (every entry
    is computed from the previous iterate), so it vectorizes exactly as
    ``x <- x * (A @ x)`` followed by normalization by ``sum(x)`` (since
    ``sum_i x_i (Ax)_i = x^T A x``). ``inds`` is kept for API compatibility; the
    already-clustered entries are zero in ``x`` and stay zero.
    """
    error = tol + 1.0
    count = 0
    while error > tol and count < max_iter:
        x_old = x
        x = x_old * (A @ x_old)
        s = x.sum()
        if s <= 0:
            break
        x = x / s
        error = np.linalg.norm(x - x_old)
        count += 1
    return x


def dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000):
    """Cluster ``graph_mat`` (n×n similarity) into dominant sets.

    Parameters
    ----------
    graph_mat : np.ndarray
        Symmetric non-negative similarity matrix.
    max_k : int
        Maximum number of clusters to extract (0 = up to n).
    tol, max_iter : float, int
        Replicator-dynamics stopping criteria.

    Returns
    -------
    np.ndarray of int labels in ``{0, ..., k-1}``.
    """
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    clusters = np.zeros(graph_cardinality, dtype=int)
    already_clustered = np.full(graph_cardinality, False, dtype=bool)

    for k in range(max_k):
        if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
            break
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        x /= x.sum()

        y = _replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters
