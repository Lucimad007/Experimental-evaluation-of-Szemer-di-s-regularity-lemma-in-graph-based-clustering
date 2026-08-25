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


def gaussian_similarity(X, sigma):
    """build the n×n similarity matrix for features ``x`` and scalar ``sigma``.

    uses ``s(x, y) = exp(−d(x, y)/(d̄·σ))`` with euclidean distance and ``d̄``
    the mean of all pairwise distances. diagonal is zero.
    """
    # feature matrix
    X = np.asarray(X, dtype=float)
    # paper: d(x, y) = euclidean distance
    D = cdist(X, X, "euclidean")
    # paper d̄ = mean of positive pairwise distances (exclude the zero diagonal)
    d_bar = D[D > 0].mean() if (D > 0).any() else 1.0
    # guard a degenerate all-equal dataset
    d_bar = d_bar if d_bar > 0 else 1e-12
    # paper: s = exp(−d / (d̄ · σ))
    S = np.exp(-D / (d_bar * sigma))
    # no self-loops
    np.fill_diagonal(S, 0.0)
    return S
