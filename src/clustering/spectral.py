"""spectral clustering (spc), §2.1 of hou et al., pr 171 (2026).

the paper's description, verbatim: "spectral clustering builds the unnormalized
laplacian l and computes the first k eigenvectors u1, u2, …, uk corresponding
to the k smallest eigenvalues of l. given the matrix u ∈ r^{n×k} with u1, …, uk
as columns, we use each row of u as a data point and do clustering with
standard methods like k-means".

the two normalized variants the paper mentions ([22] shi–malik, [23] ng–jordan–
weiss) are ``variant="shi_malik"`` (generalized eigenproblem ``l u = λ d u``)
and ``variant="njw"`` (row-normalized ``l_sym``). experiments search all three
when ``--clustering-variants all`` (the default); ``paper`` keeps unnormalized.
"""

# arrays
import numpy as np
# dense symmetric eigensolver (dsyevr)
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
# k-means on the rows of u
from sklearn.cluster import KMeans


def _symmetrize(S):
    # force a symmetric similarity / laplacian
    return (S + S.T) / 2.0


def _k_smallest_eigenvectors(M, k):
    """eigenvectors of the k smallest eigenvalues of a symmetric matrix.

    uses lapack's ``dsyevr`` (subset driver): o(n²) memory and it handles even
    usps-scale (n = 11000) dense laplacians in minutes. (an earlier version
    fell back to ``eigsh(which="sm")`` for n > 2000, which converges extremely
    slowly on dense graphs and could stall for hours.)
    """
    n = M.shape[0]
    # cannot request more vectors than the matrix dimension
    if k >= n:
        k = n
    # paper §2.1: first k eigenvectors of l
    _, eigvecs = scipy.linalg.eigh(M, subset_by_index=[0, k - 1])
    return eigvecs


def spc(sim_mat, n_clusters, random_state=314, n_init=10, variant="unnormalized"):
    """spectral clustering on a pairwise similarity matrix.

    parameters
    ----------
    sim_mat : np.ndarray
        non-negative symmetric similarity matrix (diagonal should be 0).
    n_clusters : int
        number of clusters (ground-truth k).
    variant : str
        ``"unnormalized"`` (paper §2.1, default): ``l = d − s``, k smallest
        eigenvectors, rows of u as data points (no row normalization).
        ``"shi_malik"``: shi–malik ncut [22], generalized eigenproblem ``l u = λ d u``.
        ``"njw"``: ng–jordan–weiss [23], ``l_sym`` with row-normalized u.
        ``"row_kmeans"``: k-means on the rows of the similarity matrix itself
        (natural for a small reduced graph r: each meta-node is its density profile).
    """
    # undirected graph
    S = _symmetrize(np.asarray(sim_mat, dtype=float))
    n = S.shape[0]
    # degree vector
    d = S.sum(axis=1)

    # paper §2.1 default: unnormalized laplacian
    if variant == "unnormalized":
        # l = d − s
        L = np.diag(d) - S
        L = _symmetrize(L)
        # u = [u1 … uk]
        U = _k_smallest_eigenvectors(L, n_clusters)
    # paper ref [22]: shi–malik ncut, l u = λ d u
    elif variant == "shi_malik":
        d_safe = np.where(d > 0, d, 1e-12)
        L = np.diag(d_safe) - S
        L = _symmetrize(L)
        Dmat = np.diag(d_safe)
        k = min(n_clusters, n)
        _, U = scipy.linalg.eigh(L, Dmat, subset_by_index=[0, k - 1])
    # paper ref [23]: ng–jordan–weiss
    elif variant == "njw":
        d_safe = np.where(d > 0, d, 1e-12)
        D_inv_sqrt = 1.0 / np.sqrt(d_safe)
        # l_sym = i − d^{−1/2} s d^{−1/2}
        L_sym = np.eye(n) - (D_inv_sqrt[:, None] * S * D_inv_sqrt[None, :])
        L_sym = _symmetrize(L_sym)
        U = _k_smallest_eigenvectors(L_sym, n_clusters)
        # row-normalize u
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-12)
        U = U / norms
    elif variant == "row_kmeans":
        U = S.copy()
        np.fill_diagonal(U, 0.0)
        nrm = np.linalg.norm(U, axis=1, keepdims=True)
        nrm = np.where(nrm > 0, nrm, 1.0)
        U = U / nrm
    elif variant == "row_njw":
        from ..enhanced.similarity import gaussian_similarity
        Sr = gaussian_similarity(S, 1.0)
        return spc(Sr, n_clusters, random_state=random_state, n_init=n_init, variant="njw")
    else:
        raise ValueError(f"unknown SPC variant: {variant}")

    # paper: k-means on the rows of u
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(U)
    return labels


# sprg (clustering-forest affinity + spc, zhu–loy–gong cvpr 2014 = ref [20])
# lives in ``clustering.sprg``; re-exported from ``clustering.__init__``.
