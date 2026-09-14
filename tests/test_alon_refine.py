"""alon [30] cert|complement refine: both atoms live; v0 is the size gap only."""

import numpy as np

from src.szemeredi.refinement_step import _atoms_from_cert, alon_based


def _state(classes, k, card, epsilon=0.5):
    n = len(classes)
    st = type("S", (), {})()
    st.N = n
    st.k = k
    st.classes = np.asarray(classes, dtype=float)
    st.classes_cardinality = card
    st.epsilon = epsilon
    st.degrees = np.arange(n)
    return st


def test_atoms_keep_cert_and_complement_apart():
    members = np.arange(10)
    cert = [0, 1, 2, 3]
    compl = [4, 5, 6, 7, 8, 9]
    a, b = _atoms_from_cert(members, cert, compl)
    assert set(a.tolist()) == {0, 1, 2, 3}
    assert set(b.tolist()) == {4, 5, 6, 7, 8, 9}


def test_one_atom_is_packed_into_two_slices():
    members = np.arange(10)
    a, b = _atoms_from_cert(members, list(members), [])
    assert len(a) == 5 and len(b) == 5
    assert set(a.tolist()) | set(b.tolist()) == set(range(10))
    assert set(a.tolist()) & set(b.tolist()) == set()


def test_build_uses_alon_refine_for_every_degree_mode():
    from src.szemeredi import build_regularity_lemma
    S = np.eye(8)
    for dm in ("weighted", "support", "spectral", "alon"):
        alg = build_regularity_lemma(
            "alon", S, 0.15, is_weighted=True,
            random_initialization=False, random_refinement=False,
            drop_edges_between_irregular_pairs=False, degree_mode=dm,
        )
        assert alg.refinement_step.__name__ == "alon_based"


def test_build_uses_alon_refine_on_block_graph():
    from src.szemeredi import build_regularity_lemma

    n = 24
    sim = np.zeros((n, n), dtype=float)
    sim[0:6, 12:18] = 1.0
    sim[12:18, 0:6] = 1.0
    sim[6:12, 18:24] = 1.0
    sim[18:24, 6:12] = 1.0
    np.fill_diagonal(sim, 0.0)
    alg = build_regularity_lemma(
        "alon", sim, epsilon=0.4, is_weighted=True,
        random_initialization=False, random_refinement=False,
        drop_edges_between_irregular_pairs=False, degree_mode="weighted",
    )
    assert alg.refinement_step.__name__ == "alon_based"
    alg.partition_initialization(alg, b=2)
    assert set(alg.classes[0:12]) == {1.0}
    assert set(alg.classes[12:24]) == {2.0}
    alg.check_pairs_regularity()
    k_before = alg.k
    alg.refinement_step(alg)
    assert alg.k >= k_before
    sizes = [int(np.sum(alg.classes == lab)) for lab in np.unique(alg.classes) if lab > 0]
    assert len(set(sizes)) == 1


def test_alon_refine_keeps_both_colors_and_small_v0():
    classes = np.array([1] * 10 + [2] * 10, dtype=float)
    st = _state(classes, k=2, card=10, epsilon=0.5)
    # pair (r=2, s=1): cert sizes 4 and 4, complements 6 and 6
    st.certs_compls_list = [[
        [
            [[10, 11, 12, 13], [0, 1, 2, 3]],
            [[14, 15, 16, 17, 18, 19], [4, 5, 6, 7, 8, 9]],
        ]
    ]]
    alon_based(st)
    assert st.k == 4
    assert st.classes_cardinality == 4
    # scraps: 2 per class of size 6 → |v0|=4, below εn=10
    assert int(np.sum(st.classes == 0)) == 4
    labels = {}
    for v in range(20):
        lab = int(st.classes[v])
        if lab == 0:
            continue
        labels.setdefault(lab, set()).add(v)
    cert_s, compl_s = {0, 1, 2, 3}, {4, 5, 6, 7, 8, 9}
    cert_r, compl_r = {10, 11, 12, 13}, {14, 15, 16, 17, 18, 19}
    for group in labels.values():
        for atom in (cert_s, compl_s, cert_r, compl_r):
            assert group <= atom or group.isdisjoint(atom)
