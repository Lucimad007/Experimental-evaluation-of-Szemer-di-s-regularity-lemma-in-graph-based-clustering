"""Alon on the 0/1 support, Eq. 3 on R, weighted splits as the default."""

import inspect

import numpy as np

from src.enhanced import build_reduced_graph, enhance_clustering
from src.enhanced.similarity import gaussian_similarity
from src.metrics import nmi
from src.szemeredi import build_regularity_lemma, unweighted_adjacency


def _blob(n=80, d=4, k=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 5.0, size=(k, d))
    X = np.vstack([rng.normal(c, 0.6, size=(n // k, d)) for c in centers])
    return X


def test_defaults_are_weighted_theoretical():
    sig = inspect.signature(enhance_clustering)
    assert sig.parameters["degree_mode"].default == "weighted"
    assert sig.parameters["stop_rule"].default == "theoretical"
    assert sig.parameters["adj_threshold"].default == "mean"
    sig_b = inspect.signature(build_regularity_lemma)
    assert sig_b.parameters["degree_mode"].default == "weighted"
    assert sig_b.parameters["adj_threshold"].default == "mean"


def test_median_adj_is_not_complete_on_gaussian():
    X = _blob(n=40, k=4)
    S = gaussian_similarity(X, 1.0)
    n = S.shape[0]
    A0 = unweighted_adjacency(S, 0)
    A = unweighted_adjacency(S, "median")
    assert float(A0.sum()) == n * (n - 1)
    assert float(A.sum()) < n * (n - 1) - 1
    assert float(A.sum()) > 0
    assert np.all(np.diag(A) == 0)


def test_gaussian_support_is_complete_and_alon_finds_zero_irregular():
    X = _blob(n=40, k=4)
    S = gaussian_similarity(X, 1.0)
    n = S.shape[0]
    assert float((S > 0).sum()) == n * (n - 1)
    alg = build_regularity_lemma(
        "alon", S, 0.15, is_weighted=True,
        random_initialization=False, random_refinement=False,
        drop_edges_between_irregular_pairs=False, degree_mode="weighted",
        adj_threshold=0,
    )
    assert alg._complete_support()
    alg.partition_initialization(alg, b=4)
    n_ir = alg.check_pairs_regularity()
    assert n_ir == 0
    # Alon 3 (empty Y on K_n) is the regularity verdict; Alon 1/2 do not fire
    assert [c.__name__ for c in alg.conditions] == ["alon1", "alon3", "alon2"]
    assert alg.condition_verified[0] == 0
    assert alg.condition_verified[1] == 6
    assert alg.condition_verified[2] == 0


def test_reduced_graph_uses_eq3_not_unweighted_density():
    X = _blob(n=40, k=4)
    S = gaussian_similarity(X, 1.0)
    R, _, info = build_reduced_graph(
        S, epsilon=0.15, b=4, compression_rate=0.1, degree_mode="weighted",
        adj_threshold=0,
    )
    assert info["k"] >= 4
    off = R[~np.eye(R.shape[0], dtype=bool)]
    assert off.size
    assert np.all(off > 0)
    # unweighted density of K_n would be identically 1
    assert np.max(np.abs(off - 1.0)) > 1e-6


def test_weighted_complete_stops_when_alon_says_regular():
    X = _blob(n=80, k=4)
    S = gaussian_similarity(X, 1.0)
    _, _, info = build_reduced_graph(
        S, epsilon=0.2, b=2, compression_rate=0.1, degree_mode="weighted",
        adj_threshold=0,
    )
    assert info["k"] == 2


def test_support_mode_stops_at_b_on_complete_graph():
    X = _blob(n=40, k=4)
    S = gaussian_similarity(X, 1.0)
    _, _, info = build_reduced_graph(
        S, epsilon=0.15, b=4, compression_rate=0.2, degree_mode="support",
        stop_rule="theoretical", adj_threshold=0,
    )
    assert info["k"] == 4


def test_weighted_also_stops_at_b_on_complete_graph():
    """τ=0 → K_n → Alon regular; do not dummy-bisect until compression."""
    X = _blob(n=40, k=4)
    S = gaussian_similarity(X, 1.0)
    _, _, info = build_reduced_graph(
        S, epsilon=0.15, b=4, compression_rate=0.05, degree_mode="weighted",
        stop_rule="theoretical", adj_threshold=0,
    )
    assert info["k"] == 4


def test_mean_threshold_enhance_keeps_blob_clusters():
    n, k = 80, 4
    X = _blob(n=n, k=k)
    y = np.repeat(np.arange(k), n // k)
    S = gaussian_similarity(X, 1.0)
    from src.clustering import spc
    orig = nmi(y, spc(S, k))
    labels, info = enhance_clustering(
        lambda sim, n_clusters=None: spc(sim, n_clusters),
        S, n_clusters=k, epsilon=0.15, b=4, compression_rate=0.1,
        adj_threshold="mean",
    )
    assert orig > 0.7
    assert nmi(y, labels) > 0.5
    assert info["k"] >= k


def test_weighted_partition_is_permutation_invariant():
    X = _blob(n=48, k=3, seed=1)
    S = gaussian_similarity(X, 1.0)
    rng = np.random.default_rng(7)
    perm = rng.permutation(S.shape[0])
    S2 = S[np.ix_(perm, perm)]

    def spc_fn(sim, n_clusters=None):
        from src.clustering import spc
        return spc(sim, n_clusters)

    labels_a, info_a = enhance_clustering(
        spc_fn, S, n_clusters=3, epsilon=0.15, b=4, compression_rate=0.1,
    )
    labels_b, info_b = enhance_clustering(
        spc_fn, S2, n_clusters=3, epsilon=0.15, b=4, compression_rate=0.1,
    )
    assert info_a["k"] == info_b["k"]
    mapped = np.empty_like(labels_b)
    mapped[perm] = labels_b
    # alon+mod1 packing follows certificates; [28] greedy tie-breaks by index,
    # so labels need not be a pure permutation of each other.
    assert nmi(labels_a, mapped) > 0.5


def test_fiorucci_dset_none_cutoff_not_replaced_by_hou2023():
    captured = {}

    def fake_dominant_sets(graph_mat, max_k=0, tol=1e-5, max_iter=1000,
                           weight_threshold=None, leftover_frac=None):
        captured["weight_threshold"] = weight_threshold
        captured["leftover_frac"] = leftover_frac
        return np.zeros(graph_mat.shape[0], dtype=int)

    import src.runners as runners
    original = runners.dominant_sets
    runners.dominant_sets = fake_dominant_sets
    try:
        fn = runners.make_base_algorithm("DSet", clustering_variant="fiorucci")
        fn(np.eye(4))
        assert captured["weight_threshold"] is None
        fn_paper = runners.make_base_algorithm("DSet")
        fn_paper(np.eye(4))
        assert captured["weight_threshold"] == 0.0001
    finally:
        runners.dominant_sets = original
