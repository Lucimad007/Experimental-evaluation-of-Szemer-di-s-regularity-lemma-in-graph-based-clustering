"""§3.2 Step 2: regularity / irregularity conditions for a class pair.

The paper's operational test: verify (V_r, V_s) as ε-regular, or exhibit
witnesses X ⊂ V_r, Y ⊂ V_s with |X|,|Y| ≥ (ε⁴/16)·n and
|d(X,Y)−d(V_r,V_s)| ≥ ε⁴. Alon et al. [30] give three sufficient conditions
that construct those witnesses; modification 2 uses Fiorucci et al. [28] for
the greedy certificate (condition 3).

Each callable: ``cond(self, cl_pair) -> (is_verified, cert_pair, compl_pair)``.
A pair is **irregular** when ``cert_pair[0]`` is non-empty.
"""

import math

import numpy as np
import scipy.sparse.linalg


def alon1(self, cl_pair):
    """Alon condition 1: verified ε-regular if average degree < ε³·n."""
    return cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.n, [[], []], [[], []]  # paper/Alon: ε³ n


def alon2(self, cl_pair):
    """Alon condition 2: irregular if ≥ (1/16)ε⁴·n vertices deviate by > ε⁴·n.

    Witness sizes match §3.2 Step 2: |X| = n ≥ (ε⁴/16)n, |Y| = #deviators.
    """
    certs = [[], []]
    compls = [[], []]
    s_vertices_degrees = cl_pair.classes_vertices_degrees()[1, :]
    deviation_threshold = (self.epsilon ** 4.0) * cl_pair.n  # paper/Alon: ε⁴ n
    deviated_nodes = np.abs(s_vertices_degrees - cl_pair.bip_avg_deg) > deviation_threshold

    one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg > deviation_threshold)
    is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n  # (ε⁴/16) n

    if not is_irregular:
        one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg < -deviation_threshold)
        is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n

    if is_irregular:
        certs = [
            list(cl_pair.index_map[0][range(cl_pair.n)]),
            list(cl_pair.index_map[1][one_direction_nodes]),
        ]
        compls = [
            [],
            list(cl_pair.index_map[1][~one_direction_nodes]),
        ]
    return is_irregular, certs, compls


def alon3(self, cl_pair, fast_convergence=True):
    """Alon condition 3 + §3.2 modification 2: Fiorucci degree-based greedy cert."""
    is_irregular = False
    cert_s, compl_s = [], []
    y0 = -1

    nh_dev_mat, s_degrees = cl_pair.neighbourhood_deviation_matrix()
    if fast_convergence:
        Y_indices = cl_pair.find_Y(nh_dev_mat)
        if not list(Y_indices):
            # Y spurious: irregularity is suspected but no witness can be built
            # (empty certificate) — the driver therefore tallies the pair as
            # regular. This matches Fiorucci et al.'s dense_graph_reducer, the
            # implementation the paper's modification 2 builds on.
            return True, [[], []], [[], []]
        Y_degrees = s_degrees[Y_indices]
        Yp_indices = cl_pair.find_Yp(Y_degrees, Y_indices)
        if not list(Yp_indices):
            # Y' spurious: not irregular
            return False, [[], []], [[], []]
        y0 = cl_pair.compute_y0(nh_dev_mat, Y_indices, Yp_indices)
        cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, Yp_indices)
    else:
        s_indices = cl_pair.find_Yp(s_degrees, np.arange(cl_pair.n))
        for y0 in s_indices:
            cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, s_indices)
            if cert_s:
                break

    cert_r, compl_r = cl_pair.find_r_cert_and_compl(y0)
    if cert_r and cert_s:
        is_irregular = True
    else:
        cert_r, cert_s = [], []
        compl_r, compl_s = [], []
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]


def frieze_kannan(self, cl_pair):
    """Frieze–Kannan condition: irregular if the top singular value ≥ ε·n."""
    cert_r, cert_s, compl_r, compl_s = [], [], [], []
    W = cl_pair.bip_sim_mat - cl_pair.bip_density if self.is_weighted else cl_pair.bip_adj_mat - cl_pair.bip_density

    # svds requires k < min(dim); fall back gracefully for tiny pairs
    k = 1
    if W.shape[0] <= k or W.shape[1] <= k:
        return False, [[], []], [[], []]
    try:
        _, sv_1, _ = scipy.sparse.linalg.svds(W, k=k, which="LM")
    except Exception:
        return False, [[], []], [[], []]

    is_irregular = sv_1[0] >= self.epsilon * cl_pair.n
    if is_irregular:
        u, _, vt = scipy.sparse.linalg.svds(W, k=1, which="LM")
        x = u.ravel()
        y = vt.ravel()
        beta = 3.0 / self.epsilon
        hat_thresh = beta / math.sqrt(cl_pair.n)
        x_hat = np.where(np.abs(x) <= hat_thresh, x, 0.0)
        y_hat = np.where(np.abs(y) <= hat_thresh, y, 0.0)
        quadratic_threshold = (self.epsilon - 2.0 / beta) * (cl_pair.n / 4.0)

        x_mask = x_hat > 0
        x_plus = np.where(x_mask, x_hat, 0.0)
        x_minus = np.where(~x_mask, x_hat, 0.0)
        y_plus = np.where(y_hat > 0, y_hat, 0.0)
        y_minus = np.where(~(y_hat > 0), y_hat, 0.0)

        q_plus = y_plus * 1.0 / hat_thresh
        q_minus = y_minus * 1.0 / hat_thresh

        r_mask = np.empty((0, 0))
        s_mask = np.empty((0, 0))
        if x_plus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) >= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_plus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) >= 0.0
            s_mask = (r_mask @ W) <= 0.0
        elif x_minus @ W @ y_plus >= quadratic_threshold:
            r_mask = (W @ q_plus) <= 0.0
            s_mask = (r_mask @ W) >= 0.0
        elif x_minus @ W @ y_minus >= quadratic_threshold:
            r_mask = (W @ q_minus) <= 0.0
            s_mask = (r_mask @ W) <= 0.0
        else:
            # no case of the quadratic-form certificate holds: the pair cannot
            # be certified irregular here — report it as regular (no witness)
            return False, [[], []], [[], []]

        cert_r = list(cl_pair.index_map[0][r_mask])
        compl_r = list(cl_pair.index_map[0][~r_mask])
        cert_s = list(cl_pair.index_map[1][s_mask])
        compl_s = list(cl_pair.index_map[1][~s_mask])
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]
