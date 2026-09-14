"""bipartite class-pair used by the regularity check.

paper §3.2: each (vr, vs) is a bipartite graph. eq. 2 = unweighted density;
eq. 3 = weighted (hou et al. pr 2026). certificates follow fiorucci et al. [28]
(dense_graph_reducer) of alon et al. [30].
"""

import numpy as np


class ClassesPair:
    """bipartite graph between two equitable classes r and s."""

    def __init__(self, adj_mat, classes, r, s, epsilon):
        # paper: class indices vr, vs
        self.r = r
        self.s = s
        # paper ε (def. 1 / alon thresholds)
        self.epsilon = epsilon
        # vertices currently labelled r
        self.index_map = np.where(classes == r)[0]
        # stack s under r → 2 × n index map
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        # |vr|×|vs| adjacency block
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        # equitable ⇒ |vr| = |vs| = n
        self.n = self.bip_adj_mat.shape[0]
        # mean degree on this bipartite graph
        self.bip_avg_deg = self._bip_avg_degree()
        # paper eq. 2: d(a,b) = e(a,b) / (|a||b|)
        self.bip_density = self.compute_bip_density()

    def _bip_avg_degree(self):
        # mean of row-sums and column-sums
        return (self.bip_adj_mat.sum(0) + self.bip_adj_mat.sum(1)).sum() / (2.0 * self.n)

    def compute_bip_density(self):
        # paper eq. 2: unweighted edge density of the pair
        return float(self.bip_adj_mat.sum()) / (self.n ** 2.0)

    def vs_degrees(self):
        """column sums of the matrix that produced ``bip_avg_deg`` (Vs into Vr).

        alon2 must compare these to ``bip_avg_deg``. using row sums (Vr) and
        then indexing ``index_map[1]`` was the old mix-up.
        """
        return np.asarray(self.bip_adj_mat.sum(0), dtype=float)

    def neighbourhood_deviation_matrix(self, transpose_first=True):
        """alon condition 3 neighbourhood-deviation matrix m."""
        if transpose_first:
            # m = bᵀb (common-neighbour counts on vs)
            mat = self.bip_adj_mat.T @ self.bip_adj_mat
        else:
            # m = bbᵀ (common-neighbour counts on vr)
            mat = self.bip_adj_mat @ self.bip_adj_mat.T
        # diagonal = per-vertex neighbourhood sizes
        rs_degrees = np.diag(mat).copy()
        # subtract expected common-neighbour term (avg deg)² / n
        mat = mat - (self.bip_avg_deg ** 2.0) / self.n
        return mat, rs_degrees

    def find_Y(self, nh_dev_mat):
        """fiorucci [28] greedy y: nodes with largest inner-sum deviation."""
        # row sums of m minus the diagonal
        inner_sums = nh_dev_mat.sum(1) - np.diag(nh_dev_mat)
        # sort vertices by that score, descending
        inner_sums_indices = np.argsort(inner_sums)[::-1]
        # paper/alon: |y| must exceed εn
        y_card_thresh = int((self.epsilon * self.n) + 1)
        # seed the outer sum with the first |y|-1 vertices
        outer_sum = inner_sums[inner_sums_indices[0:(y_card_thresh - 1)]].sum()
        for i in range(y_card_thresh, self.n):
            # grow y by the next-highest vertex
            outer_sum += inner_sums[inner_sums_indices[i]]
            # σ(y) = mean entry of the y-block of m
            sigma_y = outer_sum / (i ** 2.0)
            # fiorucci/alon: σ(y) ≥ (ε³/2)·n
            if sigma_y >= ((self.epsilon ** 3.0) / 2.0) * self.n:
                return inner_sums_indices[0:i]
        # no y met the threshold → empty certificate
        return np.array([])

    def find_Yp(self, degrees, Y_indices):
        # y' ⊂ y: vertices whose degree is within ε⁴n of the bipartite average
        return Y_indices[np.abs(degrees - self.bip_avg_deg) < ((self.epsilon ** 4.0) * self.n)]

    def compute_y0(self, nh_dev_mat, Y_indices, Yp_indices):
        # score each y' vertex by its deviation mass into y \ y'
        sums = np.full((self.n,), -np.inf)
        rest = set(Y_indices) - set(Yp_indices)
        for i in Yp_indices:
            sums[i] = 0.0
            for j in rest:
                sums[i] += nh_dev_mat[i, j]
        # y0 = argmax of that score
        return int(np.argmax(sums))

    def find_s_cert_and_compl(self, nh_dev_mat, y0, Yp_indices):
        # alon: outliers in vs whose deviation from y0 exceeds 2ε⁴n
        outliers_in_s = set(np.where(nh_dev_mat[y0, :] > 2.0 * (self.epsilon ** 4.0) * self.n)[0])
        # keep those that also sit in y'
        outliers_in_Yp = list(set(Yp_indices) & outliers_in_s)
        # map back to original vertex ids (class s)
        cert = list(self.index_map[1][outliers_in_Yp])
        # complement in vs
        compl = [self.index_map[1][i] for i in range(self.n) if i not in outliers_in_Yp]
        return cert, compl

    def find_r_cert_and_compl(self, y0):
        # neighbours of y0 inside vr
        indices = np.where(self.bip_adj_mat[:, y0] > 0)[0]
        cert = list(self.index_map[0][indices])
        # rest of vr
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl


class WeightedClassesPair(ClassesPair):
    """paper eq. 3: reduced-graph weight = mean original similarity of the pair."""

    def __init__(self, sim_mat, adj_mat, classes, r, s, epsilon):
        # paper: class indices vr, vs
        self.r = r
        self.s = s
        # paper ε
        self.epsilon = epsilon
        # vertices currently labelled r
        self.index_map = np.where(classes == r)[0]
        # stack s under r
        self.index_map = np.vstack((self.index_map, np.where(classes == s)[0]))
        # |x|×|y| block of w(xi, yj) — the two sums in eq. 3
        self.bip_sim_mat = sim_mat[np.ix_(self.index_map[0], self.index_map[1])]
        # 0/1 support of that block (used by alon certificates)
        self.bip_adj_mat = adj_mat[np.ix_(self.index_map[0], self.index_map[1])]
        # equitable ⇒ |vr| = |vs| = n
        self.n = self.bip_sim_mat.shape[0]
        # weighted analogue of mean bipartite degree
        self.bip_avg_deg = self._bip_avg_degree()
        # paper eq. 3
        self.bip_density = self.compute_bip_density()

    def _bip_avg_degree(self):
        # weighted analogue of mean bipartite degree
        return (self.bip_sim_mat.sum(0) + self.bip_sim_mat.sum(1)).sum() / (2.0 * self.n)

    def vs_degrees(self):
        """column sums of ``bip_sim_mat`` — same matrix as ``bip_avg_deg``."""
        return np.asarray(self.bip_sim_mat.sum(0), dtype=float)

    def compute_bip_density(self):
        # paper eq. 3: dw(x,y) = Σi Σj w(xi,yj) / (|x||y|)
        # equitable ⇒ |x|=|y|=n, so this is sum(w) / n²
        return self.bip_sim_mat.sum() / (self.n ** 2.0)

    def find_r_cert_and_compl(self, y0):
        # neighbours of y0 on the 0/1 support
        indices = np.where(self.bip_adj_mat[:, y0] > 0.0)[0]
        cert = list(self.index_map[0][indices])
        # rest of vr
        compl = [self.index_map[0][i] for i in range(self.n) if i not in indices]
        return cert, compl
