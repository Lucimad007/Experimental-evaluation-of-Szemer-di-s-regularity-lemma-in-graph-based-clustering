"""paper §3.2 step 2: regularity / irregularity of a class pair.

verify (vr, vs) as ε-regular, or exhibit x ⊂ vr, y ⊂ vs with |x|,|y| ≥ (ε⁴/16)·n
and |d(x,y)−d(vr,vs)| ≥ ε⁴. alon et al. [30] give three sufficient tests;
modification 2 uses fiorucci et al. [28] for the greedy certificate (alon 3).

each cond(self, cl_pair) → (is_verified, cert_pair, compl_pair).
irregular ⇔ cert_pair[0] is non-empty.
"""

import math

import numpy as np
import scipy.sparse.linalg


def alon1(self, cl_pair):
    """alon 1: pair is ε-regular if average degree < ε³·n."""
    # paper/alon [30]: ε³ n
    is_regular = cl_pair.bip_avg_deg < (self.epsilon ** 3.0) * cl_pair.n
    # empty cert = verified regular (not irregular)
    return is_regular, [[], []], [[], []]


def alon2(self, cl_pair):
    """alon 2: irregular if ≥ (1/16)ε⁴·n vertices deviate by > ε⁴·n."""
    certs = [[], []]
    compls = [[], []]
    # degrees of vs into vr
    s_vertices_degrees = cl_pair.classes_vertices_degrees()[1, :]
    # paper/alon: ε⁴ n
    deviation_threshold = (self.epsilon ** 4.0) * cl_pair.n
    # vertices whose degree is more than ε⁴n from the mean
    deviated_nodes = np.abs(s_vertices_degrees - cl_pair.bip_avg_deg) > deviation_threshold

    # too-high degree side
    one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg > deviation_threshold)
    # paper: (ε⁴/16) n
    is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n

    if not is_irregular:
        # too-low degree side
        one_direction_nodes = deviated_nodes * (s_vertices_degrees - cl_pair.bip_avg_deg < -deviation_threshold)
        is_irregular = one_direction_nodes.sum() >= (1.0 / 16.0) * (self.epsilon ** 4.0) * cl_pair.n

    if is_irregular:
        # witness x = all of vr, y = the deviant subset of vs
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
    """alon 3 + paper §3.2 mod 2: fiorucci [28] degree-based greedy cert."""
    # whether any certificate was found
    is_irregular = False
    cert_s, compl_s = [], []
    y0 = -1

    # neighbourhood-deviation matrix on vs
    nh_dev_mat, s_degrees = cl_pair.neighbourhood_deviation_matrix()
    if fast_convergence:
        # greedy y
        Y_indices = cl_pair.find_Y(nh_dev_mat)
        if not list(Y_indices):
            # y empty: suspected irregular but no witness — treat as regular
            # (fiorucci dense_graph_reducer; paper modification 2)
            return True, [[], []], [[], []]
        Y_degrees = s_degrees[Y_indices]
        # y' ⊂ y within ε⁴n of mean degree
        Yp_indices = cl_pair.find_Yp(Y_degrees, Y_indices)
        if not list(Yp_indices):
            # y' empty: not irregular
            return False, [[], []], [[], []]
        y0 = cl_pair.compute_y0(nh_dev_mat, Y_indices, Yp_indices)
        cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, Yp_indices)
    else:
        # slow path: try every y' vertex as y0
        s_indices = cl_pair.find_Yp(s_degrees, np.arange(cl_pair.n))
        for y0 in s_indices:
            cert_s, compl_s = cl_pair.find_s_cert_and_compl(nh_dev_mat, y0, s_indices)
            if cert_s:
                break

    # certificate on vr from y0's neighbourhood
    cert_r, compl_r = cl_pair.find_r_cert_and_compl(y0)
    if cert_r and cert_s:
        is_irregular = True
    else:
        cert_r, cert_s = [], []
        compl_r, compl_s = [], []
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]


def frieze_kannan(self, cl_pair):
    """frieze–kannan: irregular if top singular value ≥ ε·n (optional variant)."""
    cert_r, cert_s, compl_r, compl_s = [], [], [], []
    # residual matrix w = similarity (or adj) minus pair density
    W = cl_pair.bip_sim_mat - cl_pair.bip_density if self.is_weighted else cl_pair.bip_adj_mat - cl_pair.bip_density

    # svds needs k < min(dim)
    k = 1
    if W.shape[0] <= k or W.shape[1] <= k:
        return False, [[], []], [[], []]
    try:
        _, sv_1, _ = scipy.sparse.linalg.svds(W, k=k, which="LM")
    except Exception:
        return False, [[], []], [[], []]

    # irregular if σ1 ≥ ε n
    is_irregular = sv_1[0] >= self.epsilon * cl_pair.n
    if is_irregular:
        u, _, vt = scipy.sparse.linalg.svds(W, k=1, which="LM")
        x = u.ravel()
        y = vt.ravel()
        beta = 3.0 / self.epsilon
        hat_thresh = beta / math.sqrt(cl_pair.n)
        # clip large entries of the singular vectors
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
            # no quadratic-form case fires → cannot certify irregular
            return False, [[], []], [[], []]

        cert_r = list(cl_pair.index_map[0][r_mask])
        compl_r = list(cl_pair.index_map[0][~r_mask])
        cert_s = list(cl_pair.index_map[1][s_mask])
        compl_s = list(cl_pair.index_map[1][~s_mask])
    return is_irregular, [cert_r, cert_s], [compl_r, compl_s]
