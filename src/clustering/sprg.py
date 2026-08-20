"""SPRG: structured affinity learning via the Constrained Laplacian Rank (CLR) model.

SPRG (Hou, Yuan, Pelillo, *Towards parameter-free clustering for real-world
data*, Pattern Recognition 134:109062, 2023) learns a structured affinity
matrix with exactly k connected components, built on the Constrained Laplacian
Rank algorithm (Nie, Wang, Jordan, Huang, AAAI 2016).

CLR (L2-norm) solves:

    min_{S}  ||S - A||_F^2   s.t.  S >= 0,  S 1 = 1,  rank(L_S) = n - k

where L_S = D_S - (S^T + S)/2 is the Laplacian and A is an initial affinity.
By Ky Fan's theorem, rank(L_S) = n - k  <=>  the k smallest eigenvalues of L_S
are zero, so the problem is solved by alternating:

    1. (fix S) F <- k eigenvectors of L_S for the k smallest eigenvalues;
    2. (fix F) for each i, s_i <- argmin_{s_i>=0, s_i^T 1 = 1} ||s_i - q_i||^2,
       where q_i = a_i - (lam/2) v_i and v_ij = ||f_i - f_j||^2 / 2
       (this is the projection of q_i onto the probability simplex).

The number of clusters k is the ground truth; the cluster labels are the
connected components of the learned S (no k-means post-processing).
The initial affinity A follows Eq. (35) of Nie et al. (an m-nearest-neighbour,
distance-consistent, scale-invariant graph).

See ``spec/base_algorithms.md``. This is faithful to the published CLR basis
of SPRG; SPRG-specific tweaks beyond CLR are not in the public text.
"""

import numpy as np
from scipy.linalg import eigh as _eigh_subset
from scipy.sparse.linalg import eigsh as _eigsh
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.csgraph import connected_components


def _smallest_eigh(L, k):
    """Return the (k+1) smallest eigenpairs of a dense symmetric PSD matrix L.

    Uses a dense partial `eigh` for small n and Lanczos (`eigsh`) for large n,
    so the cost stays O(n^2 k) rather than O(n^3) when n is large.
    """
    n = L.shape[0]
    n_eval = min(k + 1, n)
    if n <= 1500 or n_eval >= n:
        evals, evecs = _eigh_subset(L, subset_by_index=[0, n_eval - 1])
        return evals, evecs
    ncv = min(n, 2 * n_eval + 20)
    LO = LinearOperator((n, n), matvec=lambda x: L @ x, dtype=L.dtype)
    evals, evecs = _eigsh(LO, k=n_eval, which="SA", ncv=ncv, tol=1e-6)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]



def _initial_affinity(X, m=5):
    """CLR initial graph (Nie et al. Eq. 35).

    m-nearest-neighbour, distance-consistent, scale-invariant affinity:
        a_ij = (e_{i,m+1} - e_ij) / (m*e_{i,m+1} - sum_{h=1}^m e_ih)  for j among the m nearest of i
        a_ij = 0                                                otherwise
    where e_ij = ||x_i - x_j||^2 / 2 (half squared distance), the m nearest
    neighbours of i exclude self, and a_ii = 0.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    m = min(m, n - 2)                       # need >= m+1 non-self neighbours
    if m < 1:
        return np.zeros((n, n))
    # half squared distances via the Gram form (avoids an n*n*d temporary)
    sq = (X * X).sum(1)
    D2 = 0.5 * (sq[:, None] + sq[None, :] - 2.0 * (X @ X.T))
    np.fill_diagonal(D2, 0.0)
    D2 = np.maximum(D2, 0.0)
    order = np.argsort(D2, axis=1)          # order[i,0] == i (self, dist 0)
    nn = order[:, 1:m + 1]                   # m nearest of i, excluding self
    e_nn = np.take_along_axis(D2, nn, axis=1)
    e_im1 = D2[np.arange(n), order[:, m + 1]]   # (m+1)-th nearest (excl self)
    denom = m * e_im1 - e_nn.sum(1)
    denom = np.where(denom > 0, denom, 1e-12)
    vals = (e_im1[:, None] - e_nn) / denom[:, None]
    A = np.zeros((n, n))
    np.put_along_axis(A, nn, vals, axis=1)
    np.fill_diagonal(A, 0.0)
    return A


def _laplacian(S):
    """L_S = D_S - (S^T + S)/2 (symmetrized Laplacian); also returns symmetrized S."""
    Ssym = (S + S.T) / 2.0
    d = Ssym.sum(1)
    return np.diag(d) - Ssym, Ssym


def _project_simplex(q):
    """Project q onto {s >= 0, sum(s) = 1} (Wang & Carreira-Perpinan / Duchi)."""
    n = q.size
    if n == 0:
        return q
    u = np.sort(q)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.nonzero(u > cssv / np.arange(1, n + 1))[0]
    rho = idx[-1] if idx.size else 0
    theta = cssv[rho] / (rho + 1)
    return np.maximum(q - theta, 0.0)


def _project_simplex_rows(Q):
    """Project each row of Q (n x m) onto the probability simplex (vectorized)."""
    n, m = Q.shape
    u = np.sort(Q, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    div = np.arange(1, m + 1)
    rho = (u > cssv / div).sum(axis=1) - 1            # last index where u_j > cssv_j/j
    rho = np.clip(rho, 0, m - 1)
    theta = (cssv[np.arange(n), rho] / (rho + 1.0))[:, None]
    return np.maximum(Q - theta, 0.0)



def clr_learn(X, k, m=10, lam=0.1, max_iter=20, tol=1e-6, verbose=False):
    """CLR L2 learning of a structured affinity with exactly k connected components.

    Returns the learned similarity matrix S (n x n, nonneg, row-stochastic, k components).

    Only the k+1 smallest eigenpairs of the Laplacian are computed each iteration
    (MRRR driver), so the cost is roughly O(n^2 k) rather than O(n^3).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    A = _initial_affinity(X, m=m)
    np.fill_diagonal(A, 0.0)

    # initialize F with k smallest eigenvectors of L_A
    L_A, _ = _laplacian(A)
    _, F = _smallest_eigh(L_A, k)
    F = F[:, :k]

    S = (A + A.T) / 2.0
    S = np.clip(S, 0.0, None)
    rs = S.sum(1, keepdims=True)
    S = S / np.where(rs > 0, rs, 1e-12)

    eps_zero = 1e-6
    for _it in range(max_iter):
        S_old = S
        # 1. update each row of S: s_i <- project( a_i - (lam/2) v_i ) onto the simplex
        # v_ij = ||f_i - f_j||^2 / 2  computed via the Gram form to avoid an n*n*k temporary
        G = F @ F.T
        p = np.diag(G)
        V = 0.5 * (p[:, None] + p[None, :] - 2.0 * G)
        Q = A - (lam / 2.0) * V
        S = _project_simplex_rows(Q)
        np.fill_diagonal(S, 0.0)
        # 2. update F = k smallest eigenvectors of L_S
        L_S, Ssym = _laplacian(S)
        evals, evecs = _smallest_eigh(L_S, k)
        F = evecs[:, :k]
        # lambda heuristic (Nie et al.): adjust lam so that #zero eigenvalues of L_S == k
        n_zero = int((evals < eps_zero).sum())
        if n_zero > k:
            lam /= 2.0
        elif n_zero < k:
            lam *= 2.0
        if verbose:
            print(f"  iter={_it} n_zero={n_zero} lam={lam:.4g} "
                  f"diff={np.linalg.norm(S - S_old):.2e}")
        if np.linalg.norm(S - S_old) < tol and n_zero == k:
            break
    return S


def sprg_similarity(X, k, m=10, **kwargs):
    """SPRG-learned structured affinity matrix (the graph G for SPRG).

    Returns the symmetrized learned similarity (so it is a valid undirected
    graph for the regularity-partitioning step).
    """
    S = clr_learn(X, k, m=m, **kwargs)
    return (S + S.T) / 2.0


def sprg(X, n_clusters, m=10, random_state=314, **kwargs):
    """SPRG: learn a structured affinity with exactly n_clusters connected
    components (CLR) and return the component labels.

    By construction (the CLR rank constraint) the learned S has exactly
    n_clusters connected components once the lambda heuristic converges. On
    very large graphs the heuristic may stop early with a different number of
    components; in that case we fall back to spectral clustering (k-means on the
    spectral embedding of the learned affinity) so SPRG always returns exactly
    n_clusters labels (documented deviation, see ``IMPLEMENTATION_PROOF.md``).
    """
    S = clr_learn(X, n_clusters, m=m, **kwargs)
    Ssym = (S + S.T) / 2.0
    ncomp, labels = connected_components(Ssym > 0, directed=False)
    if ncomp == n_clusters:
        return labels
    from .spectral import spc
    return spc(Ssym, n_clusters, random_state=random_state)

