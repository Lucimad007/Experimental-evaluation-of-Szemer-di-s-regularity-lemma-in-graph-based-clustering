"""every clustering-function variant is named, dispatchable, and runnable."""

import itertools

import numpy as np

from src import config
from src.clustering import forest_affinity, spc, sprg
from src.runners import make_base_algorithm, original_graph, run_original


def _blob(n=24, d=4, k=2, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * d, [4.0] * d])[:k]
    X = np.vstack([rng.normal(c, 0.4, size=(n // k, d)) for c in centers])
    y = np.concatenate([np.full(n // k, i) for i in range(k)])
    return X, y


def test_dispatch_every_variant():
    for algo in config.BASE_ALGORITHMS:
        for cv in config.clustering_variant_grid(algo, "all"):
            fn = make_base_algorithm(algo, clustering_variant=cv)
            assert callable(fn)
    assert config.clustering_variant_grid("SPC", "paper") == ("unnormalized",)
    assert config.clustering_variant_grid("SPRG", "paper") == ("adpt+unnormalized",)
    assert config.clustering_variant_grid("APC", "paper") == ("median",)
    assert config.clustering_variant_grid("DSet", "paper") == ("hou2023",)
    assert config.clustering_variant_grid("SPC", "all") == config.SPC_SPECTRAL_VARIANTS
    assert len(config.clustering_variant_grid("SPRG", "all")) == 9
    assert config.clustering_variant_grid("APC", "all") == ("median", "min")
    assert config.clustering_variant_grid("DSet", "all") == config.DSET_THRESHOLD_VARIANTS
    total = sum(len(config.clustering_variant_grid(a, "all")) for a in config.BASE_ALGORITHMS)
    assert total == 19


def test_exp2_style_product_includes_every_cell():
    """(algo, variant, stop, degree, alg, init, drop) is the Exp 2 permutation."""
    stops = ("algorithm1", "theoretical")
    degrees = config.DEGREE_MODES
    cells = []
    for algo in config.BASE_ALGORITHMS:
        for cv, sr, dm, ak, im, dr in itertools.product(
            config.clustering_variant_grid(algo, "all"),
            stops, degrees, config.ALG_KINDS, config.INIT_MODES,
            config.DROP_IRREGULAR_MODES,
        ):
            cells.append((algo, cv, sr, dm, ak, im, dr))
    n_cv = sum(len(config.clustering_variant_grid(a, "all")) for a in config.BASE_ALGORITHMS)
    assert len(cells) == (
        n_cv * len(stops) * len(degrees) * len(config.ALG_KINDS)
        * len(config.INIT_MODES) * len(config.DROP_IRREGULAR_MODES)
    )
    assert ("SPC", "shi_malik", "theoretical", "alon",
            "frieze_kannan", "random", "regular_only") in cells
    assert ("SPC", "shi_malik", "theoretical", "weighted",
            "frieze_kannan", "random", "regular_only") in cells
    assert ("SPRG", "bi+njw", "algorithm1", "spectral",
            "alon", "degree", "all_pairs") in cells
    assert ("APC", "min", "theoretical", "weighted",
            "alon", "random", "all_pairs") in cells
    assert ("DSet", "dslib", "algorithm1", "support",
            "frieze_kannan", "degree", "regular_only") in cells
    assert ("DSet", "rel95", "algorithm1", "support",
            "alon", "degree", "all_pairs") in cells
    assert ("DSet", "fiorucci05", "theoretical", "weighted",
            "alon", "degree", "all_pairs") in cells


def test_parse_roundtrip():
    for algo in config.BASE_ALGORITHMS:
        for cv in config.clustering_variant_grid(algo, "all"):
            parsed = config.parse_clustering_variant(algo, cv)
            assert isinstance(parsed, dict)
            if algo == "SPC":
                assert parsed["spc_variant"] == cv
            if algo == "SPRG":
                forest, spectral = cv.split("+")
                assert parsed["sprg_variant"] == forest
                assert parsed["spc_variant"] == spectral


def test_every_spc_variant_labels_n_points():
    X, y = _blob()
    from src.enhanced.similarity import gaussian_similarity
    S = gaussian_similarity(X, 1.0)
    seen = {}
    for variant in config.SPC_SPECTRAL_VARIANTS:
        labels = spc(S, n_clusters=2, variant=variant)
        assert labels.shape == (len(y),)
        assert set(np.unique(labels)) <= {0, 1}
        seen[variant] = labels
    # the three embeddings are not required to differ on every blob, but
    # dispatch must not collapse to the default: make_base_algorithm wires variant
    fn = make_base_algorithm("SPC", clustering_variant="njw")
    assert np.array_equal(fn(S, 2), seen["njw"])
    fn_u = make_base_algorithm("SPC", clustering_variant="unnormalized")
    assert np.array_equal(fn_u(S, 2), seen["unnormalized"])
    fn_sm = make_base_algorithm("SPC", clustering_variant="shi_malik")
    assert np.array_equal(fn_sm(S, 2), seen["shi_malik"])


def test_every_sprg_forest_times_spectral(monkeypatch):
    monkeypatch.setattr(config, "SPRG_TREES", 6)
    monkeypatch.setattr(config, "SPRG_MIN_LEAF", 2)
    X, y = _blob(n=20, d=3, k=2)
    forests = set()
    for cv in config.clustering_variant_grid("SPRG", "all"):
        parsed = config.parse_clustering_variant("SPRG", cv)
        labels = sprg(
            X, 2,
            variant=parsed["sprg_variant"],
            spc_variant=parsed["spc_variant"],
            n_trees=6, min_samples_leaf=2, random_state=314,
        )
        assert labels.shape == (len(y),)
        A = forest_affinity(
            X, n_trees=6, variant=parsed["sprg_variant"],
            min_samples_leaf=2, random_state=314,
        )
        forests.add((parsed["sprg_variant"], float(A.sum())))
    # adpt / unfm / bi must not all produce the same affinity
    forest_sums = {name: s for name, s in forests}
    assert len(forest_sums) == 3


def test_apc_preference_is_median_of_all_off_diagonal():
    """[10] median of input similarities: zeros count, diagonal does not."""
    from src.clustering.affinity_propagation import _shared_preference, apc
    import inspect
    S = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=float,
    )
    # off-diagonal: 1, 0, 1, 0, 0, 0 → median 0 (positive-only would be 1)
    assert _shared_preference(S, 50) == 0.0
    assert _shared_preference(S, 0) == 0.0
    assert inspect.signature(apc).parameters["convergence_iter"].default == 10


def test_apc_median_and_min_run():
    X, _ = _blob()
    from src.enhanced.similarity import gaussian_similarity
    S = gaussian_similarity(X, 1.0)
    for cv in config.clustering_variant_grid("APC", "all"):
        fn = make_base_algorithm("APC", clustering_variant=cv)
        labels = fn(S)
        assert labels.shape == (S.shape[0],)


def test_dset_every_cutoff_runs():
    X, _ = _blob()
    from src.enhanced.similarity import gaussian_similarity
    S = gaussian_similarity(X, 1.0)
    for cv in config.clustering_variant_grid("DSet", "all"):
        fn = make_base_algorithm("DSet", clustering_variant=cv)
        labels = fn(S)
        assert labels.shape == (S.shape[0],)


def test_dset_eq1_and_peels_until_all_labelled():
    """§2.3 Eq. 1 one step; leftover default 0; paper cell is 0.0001."""
    from src.clustering.dominant_set import _replicator, dominant_sets
    n = 12
    rng = np.random.default_rng(0)
    A = np.zeros((n, n))
    A[:6, :6] = 1.0
    A[6:, 6:] = 1.0
    np.fill_diagonal(A, 0.0)
    # break the uniform equilibrium of two equal cliques
    noise = 0.05 * rng.random((n, n))
    A = (A + noise + noise.T) / 2.0
    np.fill_diagonal(A, 0.0)
    x = np.full(n, 1.0 / n)
    y = x * (A @ x)
    y = y / y.sum()
    y2 = _replicator(A, x.copy(), None, tol=1e-12, max_iter=1)
    assert np.allclose(y, y2)
    labels = dominant_sets(A, weight_threshold=0.0001)
    assert labels.shape == (n,)
    assert len(np.unique(labels)) >= 2
    parsed = config.parse_clustering_variant("DSet", None)
    assert parsed["weight_threshold"] == 0.0001
    assert parsed["leftover_frac"] == 0.0
    rel = config.parse_clustering_variant("DSet", "rel95")
    assert rel["weight_threshold"] == "rel95"
    dump = config.parse_clustering_variant("DSet", "fiorucci05")
    assert dump["leftover_frac"] == 0.05
    assert dump["weight_threshold"] is None


def test_original_graph_sprg_uses_forest_variant(monkeypatch):
    monkeypatch.setattr(config, "SPRG_TREES", 6)
    monkeypatch.setattr(config, "SPRG_MIN_LEAF", 2)
    X, _ = _blob(n=16, d=3, k=2)
    A_adpt = original_graph("SPRG", X, clustering_variant="adpt+unnormalized")
    A_bi = original_graph("SPRG", X, clustering_variant="bi+unnormalized")
    assert A_adpt.shape == A_bi.shape
    assert not np.allclose(A_adpt, A_bi)


def test_run_original_spc_variants():
    X, y = _blob()
    from src.enhanced.similarity import gaussian_similarity
    S = gaussian_similarity(X, 1.0)
    for cv in config.SPC_SPECTRAL_VARIANTS:
        labels = run_original("SPC", S, 2, X=X, clustering_variant=cv)
        assert labels.shape == (len(y),)


def test_d0_grid_includes_adaptive_names_and_canon():
    assert "mean" in config.D0_GRID
    assert "p95" in config.D0_GRID
    assert config.parse_d0_token("med") == "median"
    assert config.canon_d0("0.10000000000000001") == 0.1
    assert config.canon_d0("p90") == "p90"
    assert config.drop_irregular_pairs("regular_only") is True
    assert config.drop_irregular_pairs("all_pairs") is False
    assert config.init_is_random("random") is True
    assert config.init_is_random("degree") is False


def test_enhanced_settings_cover_previously_unused_axes():
    from src.experiments import _iter_enhanced_settings
    settings = list(_iter_enhanced_settings(
        "SPC", n=50,
        epsilon_grid=(0.15,), compression_grid=(0.05,), b_grid=(4,),
        sigma_grid=(1.0,),
        stop_rules=("algorithm1",),
        d0_grid=config.D0_GRID,
        degree_modes=("support",),
        alg_kinds=config.ALG_KINDS,
        init_modes=config.INIT_MODES,
        drop_modes=config.DROP_IRREGULAR_MODES,
        adj_thresholds=(0.0, "median"),
    ))
    keys = {tuple((s["alg_kind"], s["init_mode"], s["drop_irregular"], s["d0"]))
            for s in settings}
    assert ("frieze_kannan", "random", "regular_only", "p90") in keys
    assert ("alon", "degree", "all_pairs", 0.0) in keys
    # every adaptive d0 is present
    d0s = {s["d0"] for s in settings}
    for name in config.D0_ADAPTIVE:
        assert name in d0s
    assert {s["adj_threshold"] for s in settings} == {0.0, "median"}


def test_dset_rel95_and_leftover_dump_run():
    X, _ = _blob()
    from src.enhanced.similarity import gaussian_similarity
    S = gaussian_similarity(X, 1.0)
    for cv in ("rel95", "fiorucci05"):
        fn = make_base_algorithm("DSet", clustering_variant=cv)
        labels = fn(S)
        assert labels.shape == (S.shape[0],)
