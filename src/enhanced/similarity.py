"""gaussian similarity matrix builder (§4 of hou et al., pr 171 (2026)).

implements the paper's similarity function ``s(x, y) = exp(−d(x, y)/(d̄·σ))``
where ``d̄`` is the average pairwise euclidean distance and ``σ`` is a scalar
from the grid ``{0.1, 0.2, 0.5, 1, 2, 5, 10}``. the diagonal is set to 0.

see ``spec/similarity.md``.
"""

# arrays
import numpy as np
# pairwise euclidean distances
from scipy.spatial.distance import cdist

# paper §4.1: 0.1, 0.2, 0.5, 1, 2, 5, 10
from ..config import SIGMA_GRID  # noqa: F401

# paper-silent metric axis: euclidean is the paper cell, cosine / correlation
# are documented alternatives for high-dimensional feature spaces.
GRAPH_METRICS = ("euclidean", "cosine", "correlation")


def pairwise_distances(X, metric="euclidean"):
    """distance matrix for the paper-silent metric axis (euclidean = paper)."""
    metric = str(metric).strip().lower()
    if metric in ("euclidean", "l2", ""):
        return cdist(X, X, "euclidean")
    if metric in ("cosine", "cos"):
        return cdist(X, X, "cosine")
    if metric in ("correlation", "corr"):
        return cdist(X, X, "correlation")
    raise ValueError(f"unknown graph metric: {metric}")


def gaussian_similarity(X, sigma, knn=None, metric="euclidean"):
    """build the n×n similarity matrix for features ``x`` and scalar ``sigma``.

    uses ``s(x, y) = exp(−d(x, y)/(d̄·σ))`` with the paper's euclidean distance
    (``metric`` switches to cosine / correlation as a documented, paper-silent
    graph-construction alternative) and ``d̄`` the mean pairwise distance. the
    paper says "average of all pairwise distances"; we average the n(n−1)
    positive off-diagonal distances (the zero diagonal excluded — an
    ≈(n−1)/n rescaling of d̄, immaterial across the σ grid). diagonal is zero.
    ``knn`` (optional) keeps each vertex's ``k`` strongest similarities and
    symmetrizes; ``none`` is the dense paper graph.
    """
    # feature matrix
    X = np.asarray(X, dtype=float)
    # paper: d(x, y) = euclidean distance; metric axis adds cosine / correlation
    D = pairwise_distances(X, metric)
    # paper d̄ = mean of positive pairwise distances (exclude the zero diagonal)
    d_bar = D[D > 0].mean() if (D > 0).any() else 1.0
    # guard a degenerate all-equal dataset
    d_bar = d_bar if d_bar > 0 else 1e-12
    # paper: s = exp(−d / (d̄ · σ))
    S = np.exp(-D / (d_bar * sigma))
    # no self-loops
    np.fill_diagonal(S, 0.0)
    if knn is not None:
        token = knn
        mutual = False
        if isinstance(token, str):
            t = token.strip().lower()
            if t.startswith("m"):
                mutual = True
                token = int(t[1:])
            else:
                token = int(t)
        S = knn_sparsify(S, int(token), mutual=mutual)
    return S


def knn_sparsify(S, k, mutual=False):
    """keep the ``k`` strongest off-diagonal similarities per row, then symmetrize.

    default is the union of directed kNN (``max(keep, keep.T)``). ``mutual``
    keeps an edge only if both endpoints listed each other.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    k = int(k)
    if k <= 0 or k >= n - 1:
        return np.array(S, copy=True)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    keep = np.zeros_like(S)
    rows = np.arange(n)[:, None]
    keep[rows, idx] = S[rows, idx]
    if mutual:
        mask = (keep > 0) & (keep.T > 0)
        return np.where(mask, np.maximum(keep, keep.T), 0.0)
    return np.maximum(keep, keep.T)
