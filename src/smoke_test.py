"""Quick end-to-end smoke test (no network required).

Builds a small synthetic clustered dataset, runs the enhanced SPC pipeline
(Algorithm 1) and the original SPC, and prints NMI/ACC for both. This validates
that the regularity partitioning, reduced graph, base clustering and label
mapping all work together.

Run: python -m src.smoke_test
"""

import numpy as np

from .clustering import spc
from .enhanced import enhance_clustering
from .enhanced.similarity import gaussian_similarity
from .metrics import evaluate


def _make_blobs(n=200, d=8, k=4, seed=314):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(k, d))
    X = np.vstack([rng.normal(c, 1.0, size=(n // k, d)) for c in centers])
    y = np.concatenate([np.full(n // k, i) for i in range(k)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def main():
    X, y = _make_blobs()
    print(f"smoke: n={X.shape[0]} d={X.shape[1]} k={y.max()+1}")
    S = gaussian_similarity(X, 1.0)

    print("running original SPC ...")
    labels_orig = spc(S, int(y.max() + 1))
    m_orig = evaluate(y, labels_orig)
    print(f"  original: NMI={m_orig['nmi']:.3f} ACC={m_orig['acc']:.3f}")

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
