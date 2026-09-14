"""Probe: does raising the stratified cap (400 -> 1200) lift NMI? Tight grid."""
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

DATASETS = ("Raisin", "Banknote", "Spambase", "Landsat", "Segment", "Dutchnumeral")
CAPS = (400, 1200)


def stratified_cap(X, y, cap, rng=0):
    rng = np.random.default_rng(rng)
    if len(y) <= cap:
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    take = np.maximum(1, np.round(counts * (cap / len(y))).astype(int))
    while take.sum() > cap:
        i = int(np.argmax(take))
        if take[i] > 1:
            take[i] -= 1
        else:
            break
    idx = []
    for c, k in zip(classes, take):
        loc = np.where(y == c)[0]
        idx.append(rng.choice(loc, size=min(int(k), len(loc)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    for name in DATASETS:
        X0, y0 = load_dataset(name)
        nc = int(np.unique(y0).size)
        print(f"\n## {name} n={len(y0)} nc={nc}", flush=True)
        for cap in CAPS:
            X, y = stratified_cap(X0, y0, cap, rng=0)
            X, _ = apply_preprocess(X, "zscore")
            for cl, cv in (("SPC", "njw"), ("APC", None), ("DSet", None)):
                orig = _best_original(cl, X, y, nc, clustering_variant=cv, knn=20)
                orig_nmi = orig[1]["nmi"] if orig else float("nan")
                t0 = time.perf_counter()
                try:
                    best, info, st = _best_over_enhanced_grid(
                        cl, X, y, nc,
                        (0.1, 0.2), (0.1,), (16, 64), (2.0, 5.0),
                        stop_rules=("theoretical",),
                        d0_grid=(0.0,),
                        degree_modes=("weighted",),
                        alg_kinds=("alon",),
                        init_modes=("degree",),
                        drop_modes=("all_pairs",),
                        adj_thresholds=(0.0,),
                        clustering_variant=cv,
                        knn=(20,),
                        lemma_polish=True,
                    )
                    nmi = best["nmi"] if best else float("nan")
                    kk = info.get("k") if info else "-"
                except Exception as e:
                    nmi, kk = float("nan"), f"ERR {type(e).__name__}"
                dt = time.perf_counter() - t0
                print(
                    f"  n={len(y):5d} {cl:4s} orig={orig_nmi:.4f} lemma={nmi:.4f} "
                    f"k={kk} sig={st.get('sigma') if best else '-'} {dt:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
