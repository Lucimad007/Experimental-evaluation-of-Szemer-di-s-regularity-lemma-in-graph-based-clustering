"""paper §3.2 step 4 (refinement) + modification 1.

vanilla alon step 4 yields 1 + k·4^k classes. the paper limits each class to
at most one irregular partner so the split is constant-factor (k roughly doubles).
degree-based certificates: fiorucci et al. [28] (modification 2).

realizations, selected by ``degree_mode`` (see spec/regularity_partitioning.md):

- ``alon_based``    — alon [30] / hou step 4 with modification 1: cut on
  certificate vs complement; both atoms become equal-size classes; only the
  packing remainder goes to v0.
- ``degree_based``  — the fiorucci [28] reference mechanics: keep the larger of
  (certificate, complement), dump the smaller side into v0, trim by 0/1 degree.
  v0 is then handled by algorithm 2's two-part guard (``apply_v0_guard``):
  leftovers stay in v0 when |c0| ≤ εn; otherwise they are distributed into
  existing classes, never minted as new classes. degenerate on dense
  similarities (the 0/1 support is the complete graph, so every degree is
  equal and the split falls back to vertex index order).
- ``awdeg_based``   — sperotto & pelillo [16] eq. 15: order each class by the
  average weighted degree awdeg_S(i) = (1/|S|) Σ_{j∈S} w(i,j), decreasing, and
  "subdivide the ordered sequence of elements into the desired number of
  subsets" (here 2, per modification 1). the tail — the least connected
  vertices — joins v0 and only accumulates, subject to the same [28] guard.
"""

import random

import numpy as np
import scipy.linalg


def apply_v0_guard(self, c0_ordered):
    """fiorucci [28] algorithm 2 two-part guard. does not mint new classes.

    - |C0| ≤ εn: leave leftovers in v0.
    - |C0| > εn and |C0| > |P|: give each existing class the same number of
      c0 vertices (keeps the partition equitable); remainder stays in v0.
    - otherwise: partition is irregular.
    ``c0_ordered`` is the current v0 vertices in the order used for the
    distribution (degree / awdeg).
    """
    c0_mask = self.classes == 0
    c0_card = int(np.sum(c0_mask))
    eps_n = self.epsilon * self.N
    labels = np.unique(self.classes)
    labels = labels[labels > 0]
    p = int(labels.size)
    if c0_card <= eps_n:
        return
    if p == 0 or c0_card <= p:
        raise RuntimeError(
            "V0 exceeded the epsilon*n bound (not enough nodes in C0 to "
            "distribute). Try to increase epsilon or decrease the number of "
            "nodes in the graph"
        )
    ordered = np.asarray([int(v) for v in c0_ordered if self.classes[int(v)] == 0], dtype=int)
    if ordered.size != c0_card:
        ordered = np.where(c0_mask)[0]
    per_class = c0_card // p
    for i, lab in enumerate(labels):
        chunk = ordered[i * per_class:(i + 1) * per_class]
        self.classes[chunk] = lab
    self.classes_cardinality = int(np.sum(self.classes == labels[0]))
    leftover = int(np.sum(self.classes == 0))
    if leftover > eps_n:
        raise RuntimeError(
            "V0 exceeded the epsilon*n bound after distributing C0. Try to "
            "increase epsilon or decrease the number of nodes in the graph"
        )


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


def _order_atom(atom, degrees):
    """stable order of an atom along ``self.degrees`` (high degree first)."""
    atom = np.asarray(atom, dtype=int)
    if atom.size == 0:
        return atom
    pos = {int(v): i for i, v in enumerate(np.asarray(degrees).ravel())}
    return np.array(sorted(atom.tolist(), key=lambda v: pos.get(int(v), 10 ** 9)), dtype=int)


def _atoms_from_cert(members, cert, compl, degrees=None):
    """alon atoms: cert vs complement, or one atom packed into two slices."""
    members = {int(v) for v in members}
    cert = [int(v) for v in cert if int(v) in members]
    compl = [int(v) for v in compl if int(v) in members and int(v) not in cert]
    if cert and compl:
        a = np.array(cert, dtype=int)
        b = np.array(compl, dtype=int)
        if degrees is not None:
            a, b = _order_atom(a, degrees), _order_atom(b, degrees)
        else:
            a, b = np.array(sorted(cert), dtype=int), np.array(sorted(compl), dtype=int)
        return [a, b]
    atom = np.array(sorted(members), dtype=int)
    if degrees is not None:
        atom = _order_atom(atom, degrees)
    if atom.size < 2:
        return [atom] if atom.size else []
    mid = atom.size // 2
    return [atom[:mid], atom[mid:2 * mid]]


def alon_based(self):
    """alon [30] step 4 with hou modification 1.

    vanilla alon can emit 1 + k·4^k classes. the paper limits each class to
    at most one irregular partner, so step 4 only ever cuts cert vs
    complement: a **constant** (two) subclasses per class. both atoms are
    packed to equal size m; scraps go to v0, then ``apply_v0_guard``.
    """
    to_be_refined = list(range(1, self.k + 1))
    pieces = []

    def take_atoms(class_id, cert, compl):
        members = np.where(self.classes == class_id)[0]
        pieces.extend(_atoms_from_cert(members, cert, compl, degrees=self.degrees))

    while to_be_refined:
        s = to_be_refined.pop(0)
        irregular_r_indices = [
            r for r in to_be_refined
            if self.certs_compls_list[r - 2][s - 1][0][0]
        ]
        if irregular_r_indices:
            np.random.seed(314)
            random.seed(314)
            chosen = random.choice(irregular_r_indices)
            to_be_refined.remove(chosen)
            certs, compls = self.certs_compls_list[chosen - 2][s - 1]
            take_atoms(chosen, certs[0], compls[0])
            take_atoms(s, certs[1], compls[1])
        else:
            take_atoms(s, [], [])

    viable = [p for p in pieces if p.size > 0]
    if not viable:
        raise RuntimeError("alon refine produced no subclasses")
    m = int(min(p.size for p in viable))
    if m < 1:
        raise RuntimeError("alon refine class size collapsed below 1")
    new_classes = np.zeros(self.N, dtype=self.classes.dtype)
    label = 0
    for p in viable:
        label += 1
        p = _order_atom(p, self.degrees)
        new_classes[p[:m]] = label
    self.classes = new_classes
    self.k = label
    self.classes_cardinality = m
    c0 = np.where(new_classes == 0)[0]
    c0_ordered = [x for x in self.degrees if x in set(c0.tolist())]
    apply_v0_guard(self, c0_ordered)


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

    # [28] alg 2: do not mint new classes from v0; two-part guard
    c0_set = set(np.where(self.classes == 0)[0])
    c0_ordered = [x for x in self.degrees if x in c0_set]
    apply_v0_guard(self, c0_ordered)


def _awdeg_order(sim_mat, members):
    """[16] eq. 15: members sorted by intra-set average weighted degree, decreasing."""
    # awdeg_S(i) = (1/|S|) Σ_{j∈S} w(i,j); the 1/|S| factor is constant here
    awdeg = sim_mat[np.ix_(members, members)].mean(1)
    # decreasing: "only the less connected vertices join the exceptional set"
    return members[np.argsort(-awdeg)]


def _fiedler_order(sim_mat, members):
    """order ``members`` by the Fiedler vector of the induced weighted subgraph.

    equitable halving along this order splits a mixed class along its cut
    rather than by degree. sign is canonicalized so the order is stable.
    """
    members = np.asarray(members, dtype=int)
    if members.size <= 2:
        return members
    sub = np.asarray(sim_mat, dtype=float)[np.ix_(members, members)].copy()
    np.fill_diagonal(sub, 0.0)
    d = sub.sum(1)
    L = np.diag(d) - sub
    L = (L + L.T) * 0.5
    try:
        nsub = min(3, members.size - 1)
        _, vecs = scipy.linalg.eigh(L, subset_by_index=[0, nsub])
    except np.linalg.LinAlgError:
        return members
    f = vecs[:, 1] if vecs.shape[1] > 1 else vecs[:, -1]
    if f[int(np.argmax(np.abs(f)))] < 0:
        f = -f
    if vecs.shape[1] > 2:
        return members[np.argsort(np.arctan2(vecs[:, 2], f))]
    return members[np.argsort(f)]


def _halve_by_order(self, order_fn):
    """modification 1: split each class into 2 along ``order_fn``; tail → v0."""
    sim_mat = self.sim_mat if self.is_weighted else self.adj_mat
    k_old = self.k
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
        order = order_fn(sim_mat, members)
        for half in (order[:c_new], order[c_new:2 * c_new]):
            if half.size == c_new:
                label += 1
                new_classes[half] = label

    self.classes = new_classes
    self.k = label
    self.classes_cardinality = c_new
    v0 = np.where(new_classes == 0)[0]
    c0_ordered = order_fn(sim_mat, v0) if v0.size else np.array([], dtype=int)
    apply_v0_guard(self, c0_ordered)


def awdeg_based(self):
    """paper §3.2 step 4 realized with [16] eq. 15 (see module docstring)."""
    _halve_by_order(self, _awdeg_order)


def spectral_based(self):
    """same halving as ``awdeg_based``, but classes are split by Fiedler order."""
    _halve_by_order(self, _fiedler_order)
