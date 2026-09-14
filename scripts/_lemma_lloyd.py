"""Lemma mapping + one Lloyd step on G (still no full-graph clustering)."""
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

DATASETS = ("Wine", "Seeds", "Thyroid", "Appendicitis", "Glass", "Ecoli")
OUT = Path("results/lemma_claim")
CSV = OUT / "lemma_plus_lloyd.csv"
KNN = 20
SIGMA = (1.0, 2.0, 5.0)
EPS = (0.1, 0.2)
CR = (0.05, 0.1)
B = (8, 16)


def paper_nmi(ds, algo):
    col = {"SPC": "Reg-SPC", "APC": "Reg-APC"}[algo]
    return paper_reference.NMI[ds][paper_reference.COLUMNS.index(col)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    done = set()
    if CSV.exists():
        rows = pd.read_csv(CSV).to_dict("records")
        done = {(r["dataset"], r["algo"]) for r in rows}
    for ds in DATASETS:
        X, y = load_dataset(ds)
        X, _ = apply_preprocess(X, "zscore")
        nc = config.DATASETS[ds][2]
        print(f"=== {ds} n={len(y)}  2D-Fiedler init  Alon refine  Lloyd-1 ===")
        for algo, cv in (("SPC", "njw"), ("APC", "median")):
            if (ds, algo) in done:
                print(f"  skip {algo}")
                continue
            orig = _best_original(algo, X, y, nc, clustering_variant=cv, knn=KNN)
            orig_nmi = orig[1]["nmi"] if orig else float("nan")
            enh, info, st = _best_over_enhanced_grid(
                algo, X, y, nc,
                EPS, CR, B, SIGMA,
                stop_rules=("theoretical",),
                d0_grid=(0.0, "mean"),
                degree_modes=("spectral",),
                alg_kinds=("alon",),
                init_modes=("degree",),
                drop_modes=("all_pairs",),
                adj_thresholds=(0.0,),
                clustering_variant=cv,
                knn=KNN,
                lemma_polish=True,
            )
            if enh is None:
                print(f"  {algo} failed orig={orig_nmi:.3f}")
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
                "enh_b": st.get("b"),
                "enh_d0": st.get("d0"),
            }
            rows.append(rec)
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(
                f"  {algo:4s} orig={orig_nmi:.3f}  lemma+lloyd={enh['nmi']:.3f}  "
                f"paper={p:.3f}  d_orig={rec['enh_minus_orig']:+.3f}  "
                f"d_paper={rec['enh_minus_paper']:+.3f}  "
                f"(eps={st['epsilon']} b={st['b']} sigma={st.get('sigma')} k={info['k']})"
            )
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
