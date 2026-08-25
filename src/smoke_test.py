"""quick end-to-end smoke test (no network required).

builds a small synthetic clustered dataset, runs the enhanced spc pipeline
(algorithm 1) and the original spc, and prints nmi/acc for both. this validates
that the regularity partitioning, reduced graph, base clustering and label
mapping all work together.

run: python -m src.smoke_test
"""

# synthetic blobs
import numpy as np

# paper §2.1
from .clustering import spc
# paper algorithm 1
from .enhanced import enhance_clustering
# paper §4 similarity
from .enhanced.similarity import gaussian_similarity
# paper §4 metrics
from .metrics import evaluate


def _make_blobs(n=200, d=8, k=4, seed=314):
    # reproducible synthetic clusters
    rng = np.random.default_rng(seed)
    # well-separated centers
    centers = rng.normal(0, 6, size=(k, d))
    # equal-size gaussian blobs
    X = np.vstack([rng.normal(c, 1.0, size=(n // k, d)) for c in centers])
    y = np.concatenate([np.full(n // k, i) for i in range(k)])
    # shuffle so class order is not spatial
    perm = rng.permutation(n)
    return X[perm], y[perm]


def main():
    X, y = _make_blobs()
    print(f"smoke: n={X.shape[0]} d={X.shape[1]} k={y.max()+1}")
    # σ = 1 (inside the paper grid)
    S = gaussian_similarity(X, 1.0)

    print("running original SPC ...")
    labels_orig = spc(S, int(y.max() + 1))
    m_orig = evaluate(y, labels_orig)
    print(f"  original: NMI={m_orig['nmi']:.3f} ACC={m_orig['acc']:.3f}")

    # recommended-range operating point (ε, b, ϵ) = (0.15, 4, 0.05)
    print("running enhanced SPC (Algorithm 1, eps=0.15, b=4, cr=0.05) ...")
    labels_enh, info = enhance_clustering(
        lambda sim, n_clusters=None: spc(sim, n_clusters),
        S, n_clusters=int(y.max() + 1),
        epsilon=0.15, b=4, compression_rate=0.05,
    )
    m_enh = evaluate(y, labels_enh)
    print(f"  enhanced: NMI={m_enh['nmi']:.3f} ACC={m_enh['acc']:.3f} k={info['k']} "
          f"v0={info['v0_size']} time={info['total_time']:.2f}s")


if __name__ == "__main__":
    main()
