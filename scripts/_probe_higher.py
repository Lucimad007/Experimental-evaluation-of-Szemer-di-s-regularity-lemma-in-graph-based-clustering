"""Probe: (a) big b for k>=nc on Leaves; (b) n-cap effect on Rice. Tight grids."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.datasets import load_dataset
from src.experiments import _best_over_enhanced_grid
from src.preprocess import apply_preprocess


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
        k = min(int(k), len(loc))
        idx.append(rng.choice(loc, size=k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def run(tag, algo, X, y, nc, EPS, CR, B, SIGMA, KNN, d0, degree, cv):
    t0 = time.perf_counter()
    best, info, st = _best_over_enhanced_grid(
        algo, X, y, nc, EPS, CR, B, SIGMA,
        stop_rules=("theoretical",),
        d0_grid=d0,
        degree_modes=degree,
        alg_kinds=("alon",),
        init_modes=("degree",),
        drop_modes=("all_pairs",),
        adj_thresholds=(0.0,),
        clustering_variant=cv,
        knn=KNN,
        lemma_polish=True,
    )
    dt = time.perf_counter() - t0
    nmi = best["nmi"] if best else float("nan")
    print(f"[{tag}] {algo} nc={nc} n={len(y)} -> nmi={nmi:.4f} "
          f"k={info.get('k') if info else '-'} st={st} {dt:.1f}s", flush=True)
    return nmi


def main():
    print("=== Leaves: big b vs b=16 (SPC njw, nc=100, n=400) ===", flush=True)
    X, y = load_dataset("Leaves")
    Xs, ys = stratified_cap(X, y, 400, rng=0)
    Xz, _ = apply_preprocess(Xs, "zscore")
    try:
        run("Leaves-16", "SPC", Xz, ys, 100, (0.2,), (0.2,), (16,), (2.0, 5.0),
            (20,), (0.0,), ("weighted",), "njw")
    except Exception as e:
        print(f"[Leaves-16] FAILED {type(e).__name__}: {e}", flush=True)
    for b in (64, 128):
        try:
            run(f"Leaves-{b}", "SPC", Xz, ys, 100, (0.2,), (0.2,), (b,), (2.0, 5.0),
                (20,), (0.0,), ("weighted",), "njw")
        except Exception as e:
            print(f"[Leaves-{b}] FAILED {type(e).__name__}: {e}", flush=True)

    print("=== Rice: n=400 vs n=1200 (SPC njw, APC median; nc=2) ===", flush=True)
    X, y = load_dataset("Rice")
    for cap in (400, 1200):
        Xs, ys = stratified_cap(X, y, cap, rng=0)
        Xz, _ = apply_preprocess(Xs, "zscore")
        run(f"Rice{cap}", "SPC", Xz, ys, 2, (0.1, 0.2), (0.1,), (8, 16),
            (2.0, 5.0), (20,), (0.0,), ("weighted",), "njw")
        run(f"Rice{cap}", "APC", Xz, ys, 2, (0.1, 0.2), (0.1,), (8, 16),
            (2.0, 5.0), (20,), (0.0,), ("weighted",), None)

    print("=== USPS: n=400 vs n=1200 (SPC njw; nc=10) ===", flush=True)
    X, y = load_dataset("USPS")
    for cap in (400, 1200):
        Xs, ys = stratified_cap(X, y, cap, rng=0)
        Xz, _ = apply_preprocess(Xs, "zscore")
        run(f"USPS{cap}", "SPC", Xz, ys, 10, (0.1, 0.2), (0.1,), (8, 16),
            (2.0, 5.0), (20,), (0.0,), ("weighted",), "njw")


if __name__ == "__main__":
    main()
