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


def _parse_weight_threshold(weight_threshold, n):
    """Decode the paper-silent DSet support cutoff.

    ``None`` → absolute ``1/(1.5 n)`` (Fiorucci / DSLib). A float is an absolute
    cutoff on the replicator weights. A string ``"rel80"`` / ``"rel0.8"`` keeps
    vertices whose weight is at least that fraction of the current maximum —
    a stricter reading of "greater than a threshold" (§2.3) that peels smaller
    dominant sets on a nearly-uniform reduced graph.
    """
    if weight_threshold is None:
        return "abs", 1.0 / (n * 1.5)
    if isinstance(weight_threshold, str):
        key = weight_threshold.strip().lower()
        if not key.startswith("rel"):
            raise ValueError(f"unknown DSet weight_threshold: {weight_threshold}")
        rest = key[3:]
        alpha = float(rest)
        if alpha > 1.0:
            alpha = alpha / 100.0
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"relative DSet threshold must be in (0, 1]: {weight_threshold}")
        return "rel", alpha
    return "abs", float(weight_threshold)


def dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000, weight_threshold=None,
                  leftover_frac=None):
    """Cluster ``graph_mat`` (n×n similarity) into dominant sets.

    Parameters
    ----------
    graph_mat : np.ndarray
        Symmetric non-negative similarity matrix.
    max_k : int
        Maximum number of clusters to extract (0 = up to n).
    tol, max_iter : float, int
        Replicator-dynamics stopping criteria.
    weight_threshold : float, str or None
        After the replicator converges, vertices with weight ≥ this value form
        a dominant set. ``None`` (default) uses ``1/(1.5 n)``. ``"rel95"`` keeps
        vertices at ≥ 95% of the current maximum weight.
    leftover_frac : float or None
        Stop when this fraction of vertices remain unclustered. ``None`` uses
        5% for absolute cutoffs (Fiorucci lineage) and 0% for relative cutoffs
        (extract until every vertex is assigned, matching §2.3's sequential
        "until all clusters are obtained").

    Returns
    -------
    np.ndarray of int labels in ``{0, ..., k-1}``.
    """
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    mode, thresh = _parse_weight_threshold(weight_threshold, graph_cardinality)
    if leftover_frac is None:
        leftover_frac = 0.0 if mode == "rel" else 0.05
    clusters = np.zeros(graph_cardinality, dtype=int)
    already_clustered = np.full(graph_cardinality, False, dtype=bool)
    k = 0

    for k in range(max_k):
        remaining = graph_cardinality - int(already_clustered.sum())
        if remaining <= ceil(leftover_frac * graph_cardinality):
            break
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        denom = x.sum()
        if denom <= 0:
            break
        x /= denom

        y = _replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        y = np.asarray(y, dtype=float).copy()
        y[already_clustered] = 0.0
        if mode == "rel":
            peak = float(y.max()) if y.size else 0.0
            cluster = np.where(y >= thresh * peak)[0] if peak > 0 else np.empty(0, dtype=int)
        else:
            cluster = np.where(y >= thresh)[0]
        cluster = cluster[~already_clustered[cluster]] if cluster.size else cluster
        if cluster.size == 0:
            rest = np.where(~already_clustered)[0]
            if rest.size == 0 or mode == "abs":
                break
            cluster = np.array([rest[int(np.argmax(y[rest]))]], dtype=int)
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters
