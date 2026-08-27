"""paper §3.2 step 4 (refinement) + modification 1.

vanilla alon step 4 yields 1 + k·4^k classes. the paper limits each class to
at most one irregular partner so the split is constant-factor (k roughly doubles).
degree-based certificates: fiorucci et al. [28] (modification 2).

two realizations, selected by ``degree_mode`` (see spec/regularity_partitioning.md):

- ``degree_based``  — the fiorucci [28] reference mechanics: keep the larger of
  (certificate, complement), dump the smaller side into v0, trim by 0/1 degree,
  then re-chunk v0. degenerate on dense similarities (the 0/1 support is the
  complete graph, so every degree is equal and the split falls back to vertex
  index order).
- ``awdeg_based``   — sperotto & pelillo [16] eq. 15: order each class by the
  average weighted degree awdeg_S(i) = (1/|S|) Σ_{j∈S} w(i,j), decreasing, and
  "subdivide the ordered sequence of elements into the desired number of
  subsets" (here 2, per modification 1). the tail — the least connected
  vertices — joins v0.
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


def _awdeg_order(sim_mat, members):
    """[16] eq. 15: members sorted by intra-set average weighted degree, decreasing."""
    # awdeg_S(i) = (1/|S|) Σ_{j∈S} w(i,j); the 1/|S| factor is constant here
    awdeg = sim_mat[np.ix_(members, members)].mean(1)
    # decreasing: "only the less connected vertices join the exceptional set"
    return members[np.argsort(-awdeg)]


def awdeg_based(self):
    """paper §3.2 step 4 realized with [16] eq. 15 (see module docstring).

    every class is subdivided into 2 subsets of ⌊c/2⌋ along its awdeg-ordered
    sequence (modification 1's "constant number of subclasses"); the tail joins
    v0, and v0 is re-chunked the same way so the partition stays equitable.
    """
    sim_mat = self.sim_mat if self.is_weighted else self.adj_mat
    k_old = self.k
    # modification 1: a constant number of subclasses — here 2
    c_new = self.classes_cardinality // 2
    if c_new < 1:
        raise RuntimeError(
            "class cardinality collapsed below 2; decrease the compression rate"
        )

    new_classes = np.zeros_like(self.classes)
    label = 0
    for s in range(1, k_old + 1):
        members = np.where(self.classes == s)[0]
        if members.size == 0:
            continue
        order = _awdeg_order(sim_mat, members)
        # "subdividing the ordered sequence of elements into the desired number
        # of subsets"; the remainder is the least connected tail → v0
        for half in (order[:c_new], order[c_new:2 * c_new]):
            if half.size == c_new:
                label += 1
                new_classes[half] = label

    # leftovers (including the initial v0) re-chunked by awdeg, decreasing
    v0 = np.where(new_classes == 0)[0]
    if v0.size >= c_new:
        order = _awdeg_order(sim_mat, v0)
        for i in range(order.size // c_new):
            label += 1
            new_classes[order[i * c_new:(i + 1) * c_new]] = label

    self.classes = new_classes
    self.k = label
    self.classes_cardinality = c_new

    if int(np.sum(new_classes == 0)) > self.epsilon * self.N:
        # paper def. 2: a regular partition needs |v0| < ε|v|
        raise RuntimeError(
            "V0 exceeded the epsilon*n bound. Try to increase epsilon or "
            "decrease the number of nodes in the graph"
        )
