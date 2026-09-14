"""Probe: does the paper-silent graph metric (cosine/correlation) lift NMI?"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.preprocess import apply_preprocess

DATASETS = ("Wine", "Seeds", "Ecoli", "Thyroid", "Glass", "Sonar",
            "Spectf", "Landmine", "Libras")


def stratified_cap(X, y, cap, rng=0):
    rng = np.random.default_rng(rng)
    if len(y) <= cap:
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    take = np.maximum(1, np.round(counts * (cap / len(y))).astype(int))
    idx = []
    for c, k in zip(classes, take):
        loc = np.where(y == c)[0]
        idx.append(rng.choice(loc, size=min(int(k), len(loc)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    for name in DATASETS:
        X, y = load_dataset(name)
        if len(y) > 500:
            X, y = stratified_cap(X, y, 400, rng=0)
        nc = int(np.unique(y).size)
        X, _ = apply_preprocess(X, "zscore")
        print(f"\n## {name} n={len(y)} nc={nc}", flush=True)
        for cl, cv in (("SPC", "njw"), ("APC", None)):
            orig = _best_original(cl, X, y, nc, clustering_variant=cv, knn=20)
            orig_nmi = orig[1]["nmi"] if orig else float("nan")
            for metric in ("euclidean", "cosine", "correlation"):
                t0 = time.perf_counter()
                best, info, st = _best_over_enhanced_grid(
                    cl, X, y, nc,
                    (0.1, 0.2), (0.1,), (16, 64), (1.0, 2.0),
                    stop_rules=("theoretical",),
                    d0_grid=(0.0,),
                    degree_modes=("weighted", "spectral"),
                    alg_kinds=("alon",),
                    init_modes=("degree",),
                    drop_modes=("all_pairs",),
                    adj_thresholds=(0.0,),
                    clustering_variant=cv,
                    knn=(20,),
                    graph_metrics=(metric,),
                    lemma_polish=True,
                )
                dt = time.perf_counter() - t0
                nmi = best["nmi"] if best else float("nan")
                print(
                    f"  {cl:3s} {metric:11s} orig={orig_nmi:.4f} "
                    f"lemma={nmi:.4f} k={info.get('k') if info else '-'} "
                    f"sig={st.get('sigma')} eps={st.get('epsilon')} "
                    f"b={st.get('b')} cr={st.get('compression')} "
                    f"deg={st.get('degree_mode')} {dt:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
