"""Win-path orig vs enhanced vs Tables 2–5 on small UCI sets."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config, paper_reference
from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.preprocess import apply_preprocess

DATASETS = ("Wine", "Seeds", "Thyroid")
ALGOS = ("SPC", "APC", "DSet")
OUT = Path("results/small_proof")

SIGMA = config.SIGMA_GRID
EPS = config.EPSILON_RECOMMENDED
CR = (0.03, 0.05, 0.1)
B = (4, 8, 16)


def paper_nmi(ds, algo):
    col = {"SPC": "Reg-SPC", "APC": "Reg-APC", "DSet": "Reg-DSet"}[algo]
    return paper_reference.NMI[ds][paper_reference.COLUMNS.index(col)]


def variant_for(algo):
    if algo == "SPC":
        return "njw"
    return config.paper_clustering_variant(algo)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        X, y = load_dataset(ds)
        X, _ = apply_preprocess(X, "zscore")
        n_clusters = config.DATASETS[ds][2]
        print(f"\n=== {ds} n={len(y)} k={n_clusters}  zscore knn={list(config.KNN_GRID)} spectral+weighted polish ===")
        for algo in ALGOS:
            cv = variant_for(algo)
            orig = _best_original(
                algo, X, y, n_clusters, clustering_variant=cv, knn=config.KNN_GRID,
            )
            if orig is None:
                print(f"  {algo}: original failed")
                continue
            orig_nmi = orig[1]["nmi"]
            enh, info, st = _best_over_enhanced_grid(
                algo, X, y, n_clusters,
                EPS, CR, B,
                SIGMA,
                stop_rules=("theoretical",),
                d0_grid=(0.0,),
                degree_modes=("spectral", "weighted"),
                alg_kinds=("alon",),
                init_modes=("degree",),
                drop_modes=("all_pairs",),
                adj_thresholds=(0.0,),
                clustering_variant=cv,
                knn=config.KNN_GRID,
                reassign_vertices=True,
            )
            if enh is None:
                print(f"  {algo}: enhanced failed")
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
            }
            rows.append(rec)
            print(
                f"  {algo:4s} orig={orig_nmi:.3f}  enh={enh['nmi']:.3f}  "
                f"paper={p:.3f}  Δorig={rec['enh_minus_orig']:+.3f}  "
                f"Δpaper={rec['enh_minus_paper']:+.3f}  "
                f"(ε={st['epsilon']} ϵ={st['compression']} b={st['b']} σ={st.get('sigma')} "
                f"knn={st.get('knn')} deg={st.get('degree_mode')} k={info['k']})"
            )
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "orig_enh_paper.csv", index=False)
    print(f"\nwrote {OUT / 'orig_enh_paper.csv'}")


if __name__ == "__main__":
    main()
