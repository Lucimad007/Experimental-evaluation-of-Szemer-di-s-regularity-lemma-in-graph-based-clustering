import numpy as np
import pytest

from src.szemeredi.fiorucci_ref import is_available
from src.szemeredi.builder import build_regularity_lemma


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="clone .reference/dense_graph_reducer first",
)


def test_fiorucci_partition_small_weighted_graph():
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 4))
    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    S = np.exp(-D / D[D > 0].mean())
    np.fill_diagonal(S, 0.0)
    alg = build_regularity_lemma(
        "fiorucci",
        S,
        epsilon=0.25,
        is_weighted=True,
        random_initialization=False,
        random_refinement=False,
        drop_edges_between_irregular_pairs=False,
    )
    R = alg.run(b=4, compression_rate=0.25, verbose=False)
    assert alg.k >= 2
    assert R.shape == (alg.k, alg.k)
    assert alg.classes.shape == (n,)
    assert set(np.unique(alg.classes)).issubset(set(range(alg.k + 1)))
