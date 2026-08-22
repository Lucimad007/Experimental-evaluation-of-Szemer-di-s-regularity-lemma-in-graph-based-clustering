"""Step 4 (refinement) of the Alon algorithm.

Refines the current partition by splitting classes involved in irregular pairs.
For each irregular pair ``(s, r)`` the certificate/complement of each side is used
to move the smaller of the two sets into the exceptional class ``V0`` (label 0);
the remaining nodes of the larger side that exceed the new class cardinality are
also moved to ``V0``. When a class has no irregular partner it is split in two by
degree order. After processing all classes, leftover ``V0`` nodes are grouped into
new classes of the current cardinality.

This is the degree-based refinement from Fiorucci et al.'s
``dense_graph_reducer``; see ``spec/regularity_partitioning.md``.
"""

import random

import numpy as np


def _get_s_r_degrees(self, s, r):
    s_indices = np.where(self.classes == s)[0]
    r_indices = np.where(self.classes == r)[0]
    s_columns = self.adj_mat[:, s_indices]
    r_columns = self.adj_mat[:, r_indices]
    s_degs = r_columns[s_indices, :].sum(1)
    r_degs = s_columns[r_indices, :].sum(1)
    s_r_degs = np.zeros(len(self.degrees))
    s_r_degs[s_indices] = s_degs
    s_r_degs[r_indices] = r_degs
    return s_r_degs.astype(int)


def degree_based(self):
    to_be_refined = list(range(1, self.k + 1))
    irregular_r_indices = []
    is_classes_cardinality_odd = self.classes_cardinality % 2 == 1
    self.classes_cardinality //= 2

    while to_be_refined:
        s = to_be_refined.pop(0)
        for r in to_be_refined:
            if self.certs_compls_list[r - 2][s - 1][0][0]:
                irregular_r_indices.append(r)

        if irregular_r_indices:
            np.random.seed(314)
            random.seed(314)
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

                self.classes[lesser_set] = 0
                difference = len(greater_set) - self.classes_cardinality
                difference_nodes_ordered_by_degree = sorted(
                    greater_set, key=lambda el: s_r_degs[el], reverse=True
                )[0:difference]
                self.classes[difference_nodes_ordered_by_degree] = 0
        else:
            self.k += 1
            # no irregular partner: split in two by (global) degree order
            global_degrees = self.adj_mat.sum(1)
            s_indices_ordered_by_degree = sorted(
                list(np.where(self.classes == s)[0]), key=lambda el: global_degrees[el], reverse=True
            )
            if is_classes_cardinality_odd:
                self.classes[s_indices_ordered_by_degree.pop(0)] = 0
            self.classes[s_indices_ordered_by_degree[0:self.classes_cardinality]] = self.k

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
        raise RuntimeError(
            "V0 exceeded the epsilon*n bound (not enough nodes in C0 to create "
            "a new class). Try to increase epsilon or decrease the number of "
            "nodes in the graph"
        )
