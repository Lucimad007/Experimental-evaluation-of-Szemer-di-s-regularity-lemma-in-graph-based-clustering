"""graph-faithful path that can beat the paper's Wine Reg-SPC NMI 0.76."""

import numpy as np

from src.datasets import load_dataset
from src.enhanced import enhance_clustering
from src.enhanced.similarity import gaussian_similarity, knn_sparsify
from src.metrics import evaluate
from src.preprocess import apply_preprocess
from src.runners import make_base_algorithm, run_original
from src.szemeredi.refinement_step import _fiedler_order


def test_knn_sparsify_symmetric():
    rng = np.random.default_rng(0)
    S = rng.random((12, 12))
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 0.0)
    K = knn_sparsify(S, 3)
    assert np.allclose(K, K.T)
    assert np.count_nonzero(K[0]) <= 6  # 3 plus symmetrized extras
    M = knn_sparsify(S, 3, mutual=True)
    assert np.allclose(M, M.T)
    assert np.count_nonzero(M) <= np.count_nonzero(K)


def test_fiedler_sign_canonical():
    rng = np.random.default_rng(1)
    n = 20
    X = rng.normal(size=(n, 4))
    S = gaussian_similarity(X, 1.0)
    order = _fiedler_order(S, np.arange(n))
    assert order.shape == (n,)
    assert set(order.tolist()) == set(range(n))


def test_wine_spectral_knn_beats_paper():
    X, y = load_dataset("Wine")
    Xz, _ = apply_preprocess(X, "zscore")
    nc = 3
    S = gaussian_similarity(Xz, 2.0, knn=20)
    orig = evaluate(y, run_original("SPC", S, nc, clustering_variant="njw"))["nmi"]
    fn = make_base_algorithm("SPC", clustering_variant="njw")
    labels, info = enhance_clustering(
        fn, S, nc,
        epsilon=0.15, b=8, compression_rate=0.03,
        degree_mode="spectral", stop_rule="theoretical",
        reassign_vertices=True,
        features=Xz,
    )
    nmi = evaluate(y, labels)["nmi"]
    assert info["k"] >= nc
    assert nmi > 0.76, f"enhanced {nmi:.3f} did not beat paper 0.76"
    assert nmi >= orig - 0.06, f"enhanced {nmi:.3f} vs original {orig:.3f}"
