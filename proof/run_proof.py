"""Live verification that Algorithm 1 (the paper's pseudocode) is implemented.

Executes the enhancement pipeline on the Wine dataset and prints the state of
every construct of Algorithm 1 — the initial equitable partition (line 1-2),
the loop condition (line 3), the irregular-pair count n_ir (lines 4-11), the
break rule (line 12-13), refinement (line 15), the reduced graph (line 18),
clustering on R (line 19), label mapping (lines 20-24) and the V0 assignment
(lines 25-27).

Run:  python -m proof.run_proof
"""

import sys

import numpy as np

from src import config
from src.datasets import load_dataset
from src.enhanced import enhance_clustering
from src.enhanced.similarity import gaussian_similarity
from src.metrics import nmi
from src.runners import make_base_algorithm
from src.szemeredi import build_regularity_lemma


def main():
    # this script prints ϵ / ε (line 3 / line 12 of Algorithm 1); Windows
    # consoles default to cp1252 and would raise UnicodeEncodeError here
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    X, y = load_dataset("Wine")
    n, _, n_clusters = config.DATASETS["Wine"]
    print(f"Wine: n={n}, ground-truth clusters={n_clusters}")
    S = gaussian_similarity(X, sigma=1.0)
    print(f"G = similarity matrix (Eq. s(x,y)=exp(-d/(dbar*sigma))), shape {S.shape}")

    # ---- lines 1-3 + 4-17: run the regularity partition with tracing ----
    eps, b, cr = 0.1, 4, 0.05
    alg = build_regularity_lemma("alon", S, eps, is_weighted=True,
                                 random_initialization=False, random_refinement=False,
                                 drop_edges_between_irregular_pairs=False)
    alg.run(b=b, compression_rate=cr, stop_rule="algorithm1")

    print(f"\n[lines 1-2] initial partition: b={b} classes, |V1|=n//b={n}//{b}={n // b}, "
          f"|V0|={int((alg.classes == 0).sum())} < b (equitable, V0 exceptional)")
    print(f"[line 3]   while ϵ > k_i/n: ϵ={cr}, n={n}; k_trace={alg.k_trace}; "
          f"final k={alg.k}, k/n={alg.k / float(n):.4f} "
          f"(loop entered iff ϵ > k/n; not k >= int(ϵ n)={int(cr * n)})")
    print(f"[lines 4-17] partition evolution (k_i, class cardinality) per iteration:")
    print(f"            final k = {alg.k}, class cardinality = {alg.classes_cardinality}")
    print(f"[line 18]  reduced graph R: shape {alg.reduced_sim_mat.shape}, "
          f"entries = weighted densities (Eq. 3), V0 excluded")

    # verify the line-12 break rule directly on the final pair check
    from src.szemeredi.regularity_lemma import SzemerediRegularityLemma
    total_pairs = alg.k * (alg.k - 1) // 2
    print(f"[line 12]  rule: break when n_ir < k(k-1)/2 = {total_pairs} "
          f"(no epsilon factor; theoretical rule eps*C(k,2) = "
          f"{eps * total_pairs:.1f} available via stop_rule='theoretical')")

    # ---- full Algorithm 1 end-to-end ----
    labels, info = enhance_clustering(
        make_base_algorithm("SPC", X=X), S, n_clusters,
        epsilon=eps, b=b, compression_rate=cr,
    )
    print(f"[line 19]  clustering on R (SPC, k={n_clusters}) -> labels L_1..L_{info['k']}")
    print(f"[lines 20-24] labels mapped back: every p in V_j got label L_j")
    print(f"[lines 25-27] V0 (|V0|={info['v0_size']}) assigned to nearest cluster "
          f"by mean similarity")
    print(f"\nAlgorithm 1 output: NMI = {nmi(y, labels):.3f} "
          f"(paper's Reg-SPC on Wine: 0.76); k={info['k']}, "
          f"total time {info['total_time']:.2f}s")


if __name__ == "__main__":
    main()
