"""Gaussian similarity matrix builder (§4).

Implements the paper's similarity function ``s(x, y) = exp(−d(x, y)/(d̄·σ))`` where
``d̄`` is the average pairwise Euclidean distance and ``σ`` is a scalar from the grid
``{0.1, 0.2, 0.5, 1, 2, 5, 10}``. The diagonal is set to 0.

See ``spec/similarity.md``.
"""

import numpy as np
from scipy.spatial.distance import cdist

SIGMA_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def gaussian_similarity(X, sigma):
    """Build the n×n similarity matrix for features ``X`` and scalar ``sigma``.

    Uses ``s(x, y) = exp(−d(x, y)/(d̄·σ))`` with Euclidean distance and ``d̄`` the
    mean of all pairwise distances. Diagonal is zero.
    """
    X = np.asarray(X, dtype=float)
    D = cdist(X, X, "euclidean")
    d_bar = D[D > 0].mean() if (D > 0).any() else 1.0
    d_bar = d_bar if d_bar > 0 else 1e-12
    S = np.exp(-D / (d_bar * sigma))
    np.fill_diagonal(S, 0.0)
    return S
