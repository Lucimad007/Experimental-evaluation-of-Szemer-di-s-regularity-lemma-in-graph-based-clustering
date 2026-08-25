"""paper §3.2 step 4 (refinement) + modification 1.

vanilla alon step 4 yields 1 + k·4^k classes. the paper limits each class to
at most one irregular partner so the split is constant-factor (k roughly doubles).
degree-based certificates: fiorucci et al. [28] (modification 2).
"""

import random

import numpy as np


def _get_s_r_degrees(self, s, r):
    # vertices currently in class s / r
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]
    s_columns = self.adj_mat[:, s_indices]
    r_columns = self.adj_mat[:, r_indices]
    # degrees of s-vertices into r
    s_degs = r_columns[s_indices, :].sum(1)
    # degrees of r-vertices into s
    r_degs = s_columns[r_indices, :].sum(1)
    s_r_degs = np.zeros(len(self.degrees))
    s_r_degs[s_indices] = s_degs
    s_r_degs[r_indices] = r_degs
    return s_r_degs.astype(int)


def degree_based(self):
    """paper §3.2 step 4: refine pi → pi+1. modification 1: ≤1 irregular partner."""
    to_be_refined = list(range(1, self.k + 1))
    irregular_r_indices = []
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    # mod 1: constant split (not alon's k·4^k)
    self.classes_cardinality //= 2

    while to_be_refined:
        s = to_be_refined.pop(0)
        for r in to_be_refined:
            # certificate non-empty ⇒ (s, r) was irregular
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        if irregular_r_indices:
            np.random.seed(314)
            random.seed(314)
            # modification 1: keep one irregular partner
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            irregular_r_indices = []

            s_r_degs = _get_s_r_degrees(self, s, chosen)

            for i in [0, 1]:
                cert_length = len(self.certs_compls_list[chosen - 2][s - 1][0][i])
                compl_length = len(self.certs_compls_list[chosen - 2][s - 1][1][i])
                greater_set_ind = np.argmax([cert_length, compl_length])
                lesser_set_ind = (
                    np.argmin([cert_length, compl_length])
                    if cert_length != compl_length
                    else 1 - greater_set_ind
                )
                greater_set = self.certs_compls_list[chosen - 2][s - 1][greater_set_ind][i]
                lesser_set = self.certs_compls_list[chosen - 2][s - 1][lesser_set_ind][i]

                # dump the smaller side into v0
                self.classes[lesser_set] = 0
                difference = len(greater_set) - self.classes_cardinality
                # trim the larger side to the new class size, highest degree first
                difference_nodes_ordered_by_degree = sorted(
                    greater_set, key=lambda el: s_r_degs[el], reverse=True
                )[0:difference]
                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            # no irregular partner: split the class in two by global degree
            self.k += 1
            global_degrees = self.adj_mat.sum(1)
            s_indices_ordered_by_degree = sorted(
                list(np.where(self.classes == s)[0]), key=lambda el: global_degrees[el], reverse=True
            )
            if is_classes_cardinality_odd:
                # odd leftover goes to v0
                self.classes[s_indices_ordered_by_degree.pop(0)] = 0
            self.classes[s_indices_ordered_by_degree[0:self.classes_cardinality]] = self.k

    # rebuild new classes from v0 in degree order
    C0_cardinality = int(np.sum(self.classes == 0))
    num_of_new_classes = C0_cardinality // self.classes_cardinality
    nodes_in_C0_ordered_by_degree = np.array([x for x in self.degrees if x in np.where(self.classes == 0)[0]])
    for i in range(num_of_new_classes):
        self.k += 1
        self.classes[
            nodes_in_C0_ordered_by_degree[
                (i * self.classes_cardinality):((i + 1) * self.classes_cardinality)
            ]
        ] = self.k

    C0_cardinality = int(np.sum(self.classes == 0))
    if C0_cardinality > self.epsilon * self.N:
        # paper def. 2: a regular partition needs |v0| < ε|v|
        raise RuntimeError(
            "V0 exceeded the epsilon*n bound (not enough nodes in C0 to create "
            "a new class). Try to increase epsilon or decrease the number of "
            "nodes in the graph"
        )
