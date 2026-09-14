"""One-cell Alon (ours) vs Fiorucci dense_graph_reducer. Not the full lemma grid."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.preprocess import apply_preprocess
from src.szemeredi.fiorucci_ref import is_available

OUT = Path("results/lemma_claim/fiorucci_vs_ours.csv")
DATASETS = ("Wine", "Seeds", "Ecoli")
CLUST = (("SPC", "njw"), ("APC", None))
EPS = (0.2,)
CR = (0.1,)
B = (8,)
SIGMA = (2.0,)
KNN = (20,)


def main():
    if not is_available():
        raise SystemExit("clone .reference/dense_graph_reducer first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    fields = [
        "dataset", "clustering", "partitioner", "orig", "lemma", "k_part", "sec",
    ]
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for name in DATASETS:
            X, y = load_dataset(name)
            X, _ = apply_preprocess(X, "zscore")
            nc = int(np.unique(y).size)
            print(f"\n## {name} n={len(y)}", flush=True)
            for cl, cv in CLUST:
                orig = _best_original(cl, X, y, nc, clustering_variant=cv, knn=20)
                orig_nmi = orig[1]["nmi"] if orig else float("nan")
                for kind in ("alon", "fiorucci"):
                    t0 = time.perf_counter()
                    best, info, _ = _best_over_enhanced_grid(
                        cl, X, y, nc,
                        EPS, CR, B, SIGMA,
                        stop_rules=("theoretical",),
                        d0_grid=(0.0,),
                        degree_modes=("weighted",),
                        alg_kinds=(kind,),
                        init_modes=("degree",),
                        drop_modes=("all_pairs",),
                        adj_thresholds=(0.0,),
                        clustering_variant=cv,
                        knn=KNN,
                        lemma_polish=True,
                    )
                    elapsed = time.perf_counter() - t0
                    enh = best["nmi"] if best else float("nan")
                    k_part = info.get("k", "") if info else ""
                    row = {
                        "dataset": name,
                        "clustering": cl,
                        "partitioner": kind,
                        "orig": f"{orig_nmi:.4f}" if orig_nmi == orig_nmi else "",
                        "lemma": f"{enh:.4f}" if enh == enh else "",
                        "k_part": k_part,
                        "sec": f"{elapsed:.1f}",
                    }
                    w.writerow(row)
                    f.flush()
                    print(
                        f"  {cl:4s} {kind:9s} orig={row['orig']} lemma={row['lemma']} "
                        f"k={k_part} {row['sec']}s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
