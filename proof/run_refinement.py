"""Live verification of Algorithm 1 line 15 (the refinement step).

Runs the pair check (lines 4-11), shows the line-12 decision, then executes
one refinement and shows the new partition. Companion to proof/refinement.md.

Run:  python -m proof.run_refinement
"""

import numpy as np

from src.datasets import load_dataset
from src.enhanced.similarity import gaussian_similarity
from src.szemeredi import build_regularity_lemma


def main():
    X, _ = load_dataset("Wine")
    S = gaussian_similarity(X, 1.0)
    eps, b = 0.15, 4
    alg = build_regularity_lemma("alon", S, eps, is_weighted=True,
                                 random_initialization=False,
                                 random_refinement=False,
                                 drop_edges_between_irregular_pairs=False)

    alg.partition_initialization(alg, b)
    print(f"before refinement: k={alg.k}, class size={alg.classes_cardinality}, "
          f"|V0|={int((alg.classes == 0).sum())}")
    nir = alg.check_pairs_regularity()
    total = alg.k * (alg.k - 1) // 2
    print(f"pair check: n_ir={nir} of C(k,2)={total} pairs -> line 12 "
          f"({nir} < {total}?) -> line 15: Refine")

    partners = [r for r in range(2, alg.k + 1) if alg.certs_compls_list[r - 2][0][0][0]]
    print(f"class 1 has irregular partners {partners}")
    if partners:
        c = alg.certs_compls_list[partners[0] - 2][0]
        print(f"witness driving the split: |X_r|={len(c[0][0])} (r-side cert), "
              f"|complement|={len(c[1][0])}; |Y_s|={len(c[0][1])}")

    alg.refinement_step(alg)
    sizes = np.bincount(alg.classes.astype(int))[1:]
    print(f"after  refinement: k={alg.k} (doubled), class size={alg.classes_cardinality} "
          f"(halved), |V0|={int((alg.classes == 0).sum())} "
          f"(must stay < eps*n={eps * alg.N:.0f})")
    print("non-empty class sizes:", sorted(sizes[sizes > 0].tolist()))


if __name__ == "__main__":
    main()
