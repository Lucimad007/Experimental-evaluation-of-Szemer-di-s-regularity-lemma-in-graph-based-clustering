"""Broader Algorithm-1 mapping search: more graphs, inits, datasets.

Still class-constant labels from clustering R (no full-G candidate).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config, paper_reference
from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.preprocess import apply_preprocess

DATASETS = (
    "Wine", "Seeds", "Thyroid", "Glass", "Appendicitis",
    "Sonar", "Spectf", "Ecoli", "Landmine", "Libras",
)
OUT = Path("results/lemma_claim")
SIGMA = (1.0, 2.0, 5.0)
EPS = (0.1, 0.2)
CR = (0.05, 0.1)
B = (8, 16)
CSV = OUT / "orig_vs_lemma_broad.csv"


def paper_nmi(ds, algo):
    col = {"SPC": "Reg-SPC", "APC": "Reg-APC", "DSet": "Reg-DSet"}[algo]
    return paper_reference.NMI[ds][paper_reference.COLUMNS.index(col)]


def knn_grid(n):
    s = int(np.clip(round(np.sqrt(n)), 8, 25))
    g = [s, 20]
    if s != 15:
        g.append(15)
    g.append(f"m{s}")
    # unique, preserve order
    out, seen = [], set()
    for x in g:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    if CSV.exists():
        old = pd.read_csv(CSV)
        rows = old.to_dict("records")
        done = {(r["dataset"], r["algo"], r["clustering_variant"]) for r in rows}
        print(f"resuming {len(done)} cells")
    for ds in DATASETS:
        X, y = load_dataset(ds)
        X, _ = apply_preprocess(X, "zscore")
        n_clusters = config.DATASETS[ds][2]
        knns = knn_grid(len(y))
        print(f"\n=== {ds} n={len(y)} k={n_clusters} knn={list(knns)} ===")
        cells = (("SPC", "njw"), ("SPC", "row_kmeans"), ("APC", "median"))
        for algo, cv in cells:
            key = (ds, algo, cv)
            if key in done:
                print(f"  skip {algo}/{cv}")
                continue
            orig = _best_original(algo, X, y, n_clusters, clustering_variant=cv, knn=knns)
            orig_nmi = orig[1]["nmi"] if orig else float("nan")
            enh, info, st = _best_over_enhanced_grid(
                algo, X, y, n_clusters,
                EPS, CR, B, SIGMA,
                stop_rules=("theoretical",),
                d0_grid=(0.0, "mean"),
                degree_modes=("spectral", "weighted"),
                alg_kinds=("alon",),
                init_modes=("degree",),
                drop_modes=("all_pairs",),
                adj_thresholds=(0.0,),
                clustering_variant=cv,
                knn=knns,
                reassign_vertices=False,
            )
            if enh is None:
                print(f"  {algo}/{cv}: enhanced failed  orig={orig_nmi:.3f}")
                continue
            p = paper_nmi(ds, algo)
            rec = {
                "dataset": ds, "algo": algo, "clustering_variant": cv,
                "orig_nmi": orig_nmi, "enh_nmi": enh["nmi"],
                "paper_reg_nmi": p,
                "enh_minus_orig": enh["nmi"] - orig_nmi,
                "enh_minus_paper": enh["nmi"] - p,
                "enh_k": info["k"],
                "enh_sigma": st.get("sigma"),
                "enh_epsilon": st.get("epsilon"),
                "enh_compression": st.get("compression"),
                "enh_b": st.get("b"),
                "enh_d0": st.get("d0"),
                "enh_drop": st.get("drop_irregular"),
                "enh_knn": st.get("knn"),
                "enh_degree": st.get("degree_mode"),
            }
            rows.append(rec)
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(
                f"  {algo:4s}/{cv:12s} orig={orig_nmi:.3f}  lemma={enh['nmi']:.3f}  "
                f"paper={p:.3f}  d_orig={rec['enh_minus_orig']:+.3f}  d_paper={rec['enh_minus_paper']:+.3f}  "
                f"(knn={st.get('knn')} deg={st.get('degree_mode')} d0={st.get('d0')} "
                f"eps={st['epsilon']} b={st['b']} sigma={st.get('sigma')} k={info['k']})"
            )
    print(f"\nwrote {CSV}")


if __name__ == "__main__":
    main()
