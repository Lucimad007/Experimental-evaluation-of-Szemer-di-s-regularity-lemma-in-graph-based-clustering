"""Spectral clustering: SPC (Ng–Jordan–Weiss) and SPRG (learned similarity + SPC).

See ``spec/base_algorithms.md`` §2.1. SPC builds the normalized Laplacian
``L_sym = I − D^{−1/2} S D^{−1/2}``, takes the ``k`` smallest eigenvectors, row-
normalizes, and runs k-means. ``k`` is the ground-truth number of clusters.

SPRG (Hou et al., *Towards parameter-free clustering for real-world data*, PR 2023)
learns a robust pairwise similarity by combining subtle similarities from
discriminative feature subspaces, then applies SPC on the learned matrix. It does
not use the ``σ`` parameter.
"""

import numpy as np
from sklearn.cluster import KMeans


def _symmetrize(S):
    return (S + S.T) / 2.0


def spc(sim_mat, n_clusters, random_state=314, n_init=10):
    """Ng–Jordan–Weiss spectral clustering.

    Parameters
    ----------
    sim_mat : np.ndarray
        Non-negative symmetric similarity matrix (diagonal should be 0).
    n_clusters : int
        Number of clusters (ground-truth k).
    """
    S = _symmetrize(np.asarray(sim_mat, dtype=float))
    n = S.shape[0]
    # avoid degenerate (all-zero) degree
    d = S.sum(axis=1)
    d_safe = np.where(d > 0, d, 1e-12)
    D_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L_sym = np.eye(n) - (D_inv_sqrt[:, None] * S * D_inv_sqrt[None, :])
    # symmetrize to clean up numerical asymmetry
    L_sym = _symmetrize(L_sym)

    # k smallest eigenvalues/eigenvectors of a symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    U = eigvecs[:, :n_clusters]
    # row-normalize
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1e-12)
    U = U / norms

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(U)
    return labels


# SPRG (learned structured affinity via the Constrained Laplacian Rank model) lives
# in ``clustering.sprg``; re-exported from ``clustering.__init__``.

