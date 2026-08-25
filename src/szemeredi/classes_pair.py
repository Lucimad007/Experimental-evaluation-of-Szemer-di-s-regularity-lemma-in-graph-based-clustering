"""Bipartite class-pair data structures used by the regularity check.

Each pair of classes ``(r, s)`` from the current partition is treated as a
bipartite graph. ``ClassesPair`` works on a 0/1 adjacency matrix (unweighted);
``WeightedClassesPair`` additionally carries the weighted similarity matrix and is
used when the input graph is edge-weighted (graph-based clustering case).

This follows Fiorucci et al.'s ``dense_graph_reducer`` implementation of the
Alon et al. regularity-partitioning conditions; see
``spec/regularity_partitioning.md``.
"""

import numpy as np


class ClassesPair:
    """A bipartite graph between two equally-sized classes ``r`` and ``s``."""

    def __init__(self, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.epsilon = epsilon
        # indices of nodes in class r (row 0) and class s (row 1)
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_adj_mat.shape[0]
        self.bip_avg_deg = self._bip_avg_degree()
        self.bip_density = self.compute_bip_density()

    def _bip_avg_degree(self):
        return (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        # paper Eq. (2): d(A,B) = e(A,B) / (|A||B|)  — unweighted reduced-graph weight
        return float(self.bip_adj_mat.sum()) / (self.n ** 2.0)

    def classes_vertices_degrees(self):
        c_v_degs = np.sum(self.bip_adj_mat, 0)
        c_v_degs = np.vstack((c_v_degs, np.sum(self.bip_adj_mat, 1)))
        return c_v_degs

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        """Neighbourhood deviation matrix used by Alon condition 3.

        ``M = B^T B`` (or ``B B^T``) with the average-degree term subtracted; also
        returns the diagonal (per-vertex neighbourhood degrees).
        """
        if transpose_first:
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        rs_degrees = np.diag(mat).copy()
        mat = mat - (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        """Greedy certificate Y: pick nodes with the highest inner-sum deviation."""
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        y_card_thresh = int((self.epsilon * self.n) + 1)
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()
        for i in range(y_card_thresh, self.n):
            outer_sum += inner_sums[inner_sums_indices[i]]
            sigma_y = outer_sum / (i ** 2.0)
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:  # Fiorucci/Alon: σ(Y) ≥ (ε³/2) n
                return inner_sums_indices[0:i]
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        sums = np.full((self.n,), -np.inf)
        rest = set(Y_indices) - set(Yp_indices)
        for i in Yp_indices:
            sums[i] = 0.0
            for j in rest:
                sums[i] += nh_dev_mat[i, j]
        return int(np.argmax(sums))

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])  # 2ε⁴ n
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        cert = list(self.index_map[1][outliers_in_Yp])
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl


class WeightedClassesPair(ClassesPair):
    """Paper Eq. (3): reduced-graph edge weight = mean original similarity between two classes."""

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        self.r = r
        self.s = s
        self.epsilon = epsilon
        self.index_map = np.where(classes == r)[0]
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        # |X|×|Y| block of original similarities = all w(x_i, y_j) in paper Eq. (3)
        self.bip_sim_mat = sim_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        self.n = self.bip_sim_mat.shape[0]
        self.bip_avg_deg = self._bip_avg_degree()
        self.bip_density = self.compute_bip_density()

    def _bip_avg_degree(self):
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        # paper Eq. (3): dw(X,Y) = Σ_i Σ_j w(x_i,y_j) / (|X||Y|)
        # equitable partition ⇒ |X|=|Y|=n, so this is sum(w) / n²
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def find_r_cert_and_compl(self, y0):
        indices = np.where(self.bip_adj_mat[:, y0] > 0.0)[0]
        cert = list(self.index_map[0][indices])
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl
