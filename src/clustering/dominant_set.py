"""dominant set clustering (dset) via replicator dynamics (§2.3, eq. 1 of hou et al.).

extracts dominant sets sequentially: each run of the replicator dynamics on the
remaining (unclustered) vertices converges to a dominant set, which is removed
and recorded as one cluster. the number of clusters is determined automatically.

follows pavan & pelillo (2007) and the reference replicator implementation; see
``spec/base_algorithms.md`` §2.3.
"""

# leftover-fraction cutoff
from math import ceil

# arrays
import numpy as np


def _replicator(A, x, inds, tol, max_iter):
    """replicator dynamics (paper §2.3 eq. 1), synchronous update.

    the update ``x_i <- x_i (ax)_i / (xᵀ a x)`` is a synchronous map (every
    entry is computed from the previous iterate), so it vectorizes exactly as
    ``x <- x * (a @ x)`` followed by normalization by ``sum(x)`` (since
    ``sum_i x_i (ax)_i = xᵀ a x``). ``inds`` is kept for api compatibility; the
    already-clustered entries are zero in ``x`` and stay zero.
    """
    error = tol + 1.0
    count = 0
    # iterate until the simplex vector stabilizes
    while error > tol and count < max_iter:
        x_old = x
        # paper eq. 1 numerator: x_i (ax)_i
        x = x_old * (A @ x_old)
        s = x.sum()
        # mass collapsed
        if s <= 0:
            break
        # divide by xᵀ a x ≡ sum(x)
        x = x / s
        error = np.linalg.norm(x - x_old)
        count += 1
    return x


def _parse_weight_threshold(weight_threshold, n):
    """decode the paper-silent dset support cutoff.

    ``none`` → absolute ``1/(1.5 n)`` (fiorucci variant). a float is an absolute
    cutoff on the replicator weights. a string ``"rel80"`` / ``"rel0.8"`` keeps
    vertices whose weight is at least that fraction of the current maximum —
    a stricter reading of "greater than a threshold" (§2.3) that peels smaller
    dominant sets on a nearly-uniform reduced graph.
    """
    # paper-silent default used in the experiments
    if weight_threshold is None:
        return "abs", 1.0 / (n * 1.5)
    if isinstance(weight_threshold, str):
        key = weight_threshold.strip().lower()
        if not key.startswith("rel"):
            raise ValueError(f"unknown DSet weight_threshold: {weight_threshold}")
        rest = key[3:]
        alpha = float(rest)
        # allow "rel95" as 0.95
        if alpha > 1.0:
            alpha = alpha / 100.0
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"relative DSet threshold must be in (0, 1]: {weight_threshold}")
        return "rel", alpha
    # explicit absolute cutoff
    return "abs", float(weight_threshold)


def dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000, weight_threshold=None,
                  leftover_frac=None):
    """cluster ``graph_mat`` (n×n similarity) into dominant sets.

    parameters
    ----------
    graph_mat : np.ndarray
        symmetric non-negative similarity matrix.
    max_k : int
        maximum number of clusters to extract (0 = up to n).
    tol, max_iter : float, int
        replicator-dynamics stopping criteria.
    weight_threshold : float, str or none
        after the replicator converges, vertices with weight ≥ this value form
        a dominant set. ``none`` (default) uses ``1/(1.5 n)``. ``"rel95"`` keeps
        vertices at ≥ 95% of the current maximum weight.
    leftover_frac : float or none
        stop when this fraction of vertices remain unclustered. ``none`` (default)
        is 0: peel until every vertex is assigned, matching §2.3 ("repeats this
        process with the remaining unallocated data until all clusters are
        obtained"). pass a positive fraction only to reproduce the old
        fiorucci-lineage 5% dump.

    returns
    -------
    np.ndarray of int labels in ``{0, ..., k-1}``.
    """
    graph_cardinality = graph_mat.shape[0]
    # 0 = extract until leftover
    if max_k == 0:
        max_k = graph_cardinality
    mode, thresh = _parse_weight_threshold(weight_threshold, graph_cardinality)
    # §2.3: extract until all clusters are obtained
    if leftover_frac is None:
        leftover_frac = 0.0
    clusters = np.zeros(graph_cardinality, dtype=int)
    already_clustered = np.full(graph_cardinality, False, dtype=bool)
    k = 0

    # paper §2.3: peel one dominant set at a time
    for k in range(max_k):
        remaining = graph_cardinality - int(already_clustered.sum())
        # leftover dumped into the last cluster
        if remaining <= ceil(leftover_frac * graph_cardinality):
            break
        # paper: x_i^(0) = 1/n on the data still being clustered
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        denom = x.sum()
        if denom <= 0:
            break
        x /= denom

        # eq. 1 until convergence
        y = _replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        y = np.asarray(y, dtype=float).copy()
        y[already_clustered] = 0.0
        # relative: keep near-peak weights
        if mode == "rel":
            peak = float(y.max()) if y.size else 0.0
            cluster = np.where(y >= thresh * peak)[0] if peak > 0 else np.empty(0, dtype=int)
        else:
            # absolute: y_i ≥ threshold (§2.3: "greater than a threshold")
            cluster = np.where(y >= thresh)[0]
        cluster = cluster[~already_clustered[cluster]] if cluster.size else cluster
        # empty support: take the single heaviest leftover vertex
        if cluster.size == 0:
            rest = np.where(~already_clustered)[0]
            if rest.size == 0 or mode == "abs":
                break
            cluster = np.array([rest[int(np.argmax(y[rest]))]], dtype=int)
        already_clustered[cluster] = True
        clusters[cluster] = k
    # leftover vertices share the last label
    clusters[~already_clustered] = k
    return clusters
