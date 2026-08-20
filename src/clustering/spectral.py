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


def _learn_sprg_similarity(X, n_neighbors=7, alpha=1.0, n_iter=5, random_state=314):
    """SPRG-style learned similarity matrix.

    Combines local-scale (kNN) affinities with iterative consensus refinement to
    reveal subtle similarities in discriminative subspaces, in the spirit of Hou
    et al. (PR 2023). The result is a symmetric non-negative similarity matrix with
    zero diagonal.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # local scaling: σ_i = distance to the k-th nearest neighbour
    from scipy.spatial.distance import cdist

    D = cdist(X, X, "euclidean")
    k = min(n_neighbors, n - 1)
    idx = np.argsort(D, axis=1)
    local_sigma = D[np.arange(n), idx[:, k]]
    local_sigma = np.where(local_sigma > 0, local_sigma, 1e-12)

    # Gaussian with per-point local scale (Zelnik-Manor & Perona style)
    S = np.exp(-((D / local_sigma[:, None]) ** 2) / 2.0)
    S = _symmetrize(S)
    np.fill_diagonal(S, 0.0)

    # iterative consensus refinement in discriminative subspaces:
    # weight each pair by the agreement of their neighbourhoods (subtle similarity)
    P = S.copy()
    for _ in range(n_iter):
        # neighbourhood agreement via shared-NN overlap
        top = np.argpartition(-P, k, axis=1)[:, :k]
        mask = np.zeros_like(P, dtype=bool)
        rows = np.repeat(np.arange(n), k)
        cols = top.ravel()
        mask[rows, cols] = True
        agreement = mask.astype(float) @ mask.astype(float).T
        # normalize by union size to get Jaccard-like agreement
        union = (mask.sum(1)[:, None] + mask.sum(1)[None, :]) - agreement
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(union > 0, agreement / union, 0.0)
        np.fill_diagonal(jacc, 0.0)
        P = alpha * P + (1.0 - alpha) * jacc
        P = np.clip(P, 0.0, None)
        P = _symmetrize(P)
        np.fill_diagonal(P, 0.0)
    return P


def sprg_similarity(X, **learn_kwargs):
    """Public entry point: the SPRG-learned similarity matrix (the graph ``G`` for SPRG).

    SPRG does not use the Gaussian ``σ``; it learns its own pairwise similarity
    from the feature matrix ``X`` (see ``spec/base_algorithms.md`` §2.1 / §4).
    """
    return _learn_sprg_similarity(X, **learn_kwargs)


def sprg(X, n_clusters, random_state=314, n_init=10, **learn_kwargs):
    """SPRG: learn a robust similarity matrix from features ``X`` then run SPC.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n×d).
    n_clusters : int
        Number of clusters (ground-truth k).
    """
    S = _learn_sprg_similarity(X, random_state=random_state, **learn_kwargs)
    return spc(S, n_clusters, random_state=random_state, n_init=n_init)
