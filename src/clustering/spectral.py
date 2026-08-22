"""Spectral clustering (SPC), §2.1 of the paper.

The paper's description, verbatim: "spectral clustering builds the unnormalized
Laplacian L and computes the first k eigenvectors u1, u2, …, uk corresponding
to the k smallest eigenvalues of L. Given the matrix U ∈ R^{n×k} with u1, …, uk
as columns, we use each row of U as a data point and do clustering with
standard methods like k-means".

The two normalized variants the paper mentions ([22] Shi–Malik, [23] Ng–Jordan–
Weiss) are kept available via ``variant="njw"`` (row-normalized NJW), but the
default — and what the experiments use — is the unnormalized Laplacian.
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import KMeans


def _symmetrize(S):
    return (S + S.T) / 2.0


def _k_smallest_eigenvectors(M, k):
    """Eigenvectors of the k smallest eigenvalues of a symmetric matrix.

    Uses LAPACK's ``dsyevr`` (subset driver): O(n^2) memory and it handles even
    USPS-scale (n = 11000) dense Laplacians in minutes. (An earlier version
    fell back to ``eigsh(which="SM")`` for n > 2000, which converges extremely
    slowly on dense graphs and could stall for hours.)
    """
    n = M.shape[0]
    if k >= n:
        k = n
    _, eigvecs = scipy.linalg.eigh(M, subset_by_index=[0, k - 1])
    return eigvecs


def spc(sim_mat, n_clusters, random_state=314, n_init=10, variant="unnormalized"):
    """Spectral clustering on a pairwise similarity matrix.

    Parameters
    ----------
    sim_mat : np.ndarray
        Non-negative symmetric similarity matrix (diagonal should be 0).
    n_clusters : int
        Number of clusters (ground-truth k).
    variant : str
        ``"unnormalized"`` (paper §2.1, default): ``L = D − S``, k smallest
        eigenvectors, rows of U as data points (no row normalization).
        ``"njw"``: normalized Laplacian ``I − D^{−1/2} S D^{−1/2}`` with row
        normalization (Ng–Jordan–Weiss, ref [23] of the paper).
    """
    S = _symmetrize(np.asarray(sim_mat, dtype=float))
    n = S.shape[0]
    d = S.sum(axis=1)

    if variant == "unnormalized":
        L = np.diag(d) - S
        L = _symmetrize(L)
        U = _k_smallest_eigenvectors(L, n_clusters)
    elif variant == "njw":
        d_safe = np.where(d > 0, d, 1e-12)
        D_inv_sqrt = 1.0 / np.sqrt(d_safe)
        L_sym = np.eye(n) - (D_inv_sqrt[:, None] * S * D_inv_sqrt[None, :])
        L_sym = _symmetrize(L_sym)
        U = _k_smallest_eigenvectors(L_sym, n_clusters)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-12)
        U = U / norms
    else:
        raise ValueError(f"unknown SPC variant: {variant}")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(U)
    return labels


# SPRG (clustering-forest affinity + SPC, Zhu–Loy–Gong CVPR 2014 = ref [20])
# lives in ``clustering.sprg``; re-exported from ``clustering.__init__``.
