"""raw vs z-score feature views; only degenerate columns are dropped."""

import numpy as np

from src.eda import profile_matrix
from src.enhanced.similarity import gaussian_similarity
from src.preprocess import apply_preprocess, parse_preprocess_modes, prepare_dataset


def test_parse_preprocess_modes():
    assert parse_preprocess_modes(None) == ("raw",)
    assert parse_preprocess_modes("raw") == ("raw",)
    assert parse_preprocess_modes("zscore") == ("zscore",)
    assert parse_preprocess_modes("both") == ("raw", "zscore")
    assert parse_preprocess_modes("raw,zscore") == ("raw", "zscore")


def test_raw_is_identity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    Xp, info = apply_preprocess(X, "raw")
    assert info["mode"] == "raw"
    assert info["n_dropped"] == 0
    np.testing.assert_array_equal(Xp, X)
    Xp[0, 0] = 99
    assert X[0, 0] != 99


def test_zscore_mean0_std1_drops_constant_and_duplicate():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    X = np.column_stack([X, np.full(40, 7.0), X[:, 0]])  # const + exact dup of col 0
    Z, info = apply_preprocess(X, "zscore")
    assert info["n_dropped"] == 2
    assert info["n_features"] == 3
    reasons = {r for r, _ in info["dropped"]}
    assert reasons == {"zero_variance", "duplicate"}
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0, ddof=0), 1.0, atol=1e-12)


def test_constant_column_does_not_change_gaussian_on_raw():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 3))
    Xc = np.column_stack([X, np.ones(12)])
    S = gaussian_similarity(X, 1.0)
    Sc = gaussian_similarity(Xc, 1.0)
    np.testing.assert_allclose(S, Sc, atol=1e-12)


def test_profile_flags_constant():
    X = np.column_stack([np.arange(10.0), np.ones(10)])
    y = np.zeros(10, dtype=int)
    rec = profile_matrix(X, y, name="toy")
    assert rec["n_const"] == 1
    assert rec["n_nan"] == 0
    assert rec["zscore_d"] == 1


def test_prepare_segment_drops_constant_column():
    Xz, y, info = prepare_dataset("Segment", "zscore")
    assert info["n_dropped"] == 1
    assert info["dropped"][0][0] == "zero_variance"
    assert Xz.shape[1] == 18
    np.testing.assert_allclose(Xz.mean(0), 0.0, atol=1e-8)


def test_prepare_dutchnumeral_drops_exact_duplicate_columns():
    Xz, y, info = prepare_dataset("Dutchnumeral", "zscore")
    assert info["n_dropped"] == 3
    assert all(r == "duplicate" for r, _ in info["dropped"])
    assert Xz.shape == (2000, 646)


def test_prepare_wine_zscore_keeps_all_columns():
    Xz, y, info = prepare_dataset("Wine", "zscore")
    assert info["n_dropped"] == 0
    assert Xz.shape == (178, 13)
    assert len(np.unique(y)) == 3
    np.testing.assert_allclose(Xz.mean(0), 0.0, atol=1e-10)
