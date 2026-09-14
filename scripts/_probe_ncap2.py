"""n-cap cause: is fixed kNN=20 too sparse at large n? NMI vs kNN at both n.
Also checks whether the n=400 score is stable across sample seeds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.datasets import load_dataset
from src.enhanced import build_reduced_graph
from src.experiments import _graph_for, _eval_on_reduced
from src.preprocess import apply_preprocess
from src.szemeredi import apply_density_threshold

BANKNOTE = {"sigma": 5.0, "epsilon": 0.1, "compression": 0.2, "b": 8,
            "d0": 0.0, "degree_mode": "spectral", "metric": "euclidean",
            "alg_kind": "alon", "stop_rule": "theoretical",
            "init_mode": "degree", "drop_irregular": "all_pairs",
            "adj_threshold": 0.0}
RAISIN = {"sigma": 5.0, "epsilon": 0.1, "compression": 0.1, "b": 16,
          "d0": 0.0, "degree_mode": "spectral", "metric": "euclidean",
          "alg_kind": "alon", "stop_rule": "theoretical",
          "init_mode": "degree", "drop_irregular": "all_pairs",
          "adj_threshold": 0.0}


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


def evaluate(algo, X, y, nc, cv, cfg):
    S = _graph_for(algo, X, cfg["sigma"], nc, cv,
                   knn=cfg["knn"], metric=cfg["metric"])
    R, classes, info = build_reduced_graph(
        S, cfg["epsilon"], cfg["b"], cfg["compression"],
        alg_kind=cfg["alg_kind"], stop_rule=cfg["stop_rule"],
        density_threshold=0.0, degree_mode=cfg["degree_mode"],
        random_initialization=cfg["init_mode"] == "random",
        drop_edges_between_irregular_pairs=cfg["drop_irregular"] == "regular_only",
        adj_threshold=cfg["adj_threshold"],
    )
    R = apply_density_threshold(R.copy(), cfg["d0"])
    got = _eval_on_reduced(algo, S, y, nc, R, classes, info,
                           clustering_variant=cv, features=X, lemma_polish=True)
    if got is None:
        return float("nan"), ""
    return got[0]["nmi"], got[1].get("k", "")


def main():
    cases = (("Banknote", BANKNOTE, 1372), ("Raisin", RAISIN, 900))
    knns = (10, 20, 40, 60, 80, 120, "m20", "m40", "m60")
    for name, seed_cfg, cap in cases:
        X0, y0 = load_dataset(name)
        nc = int(np.unique(y0).size)
        print(f"\n## {name} nc={nc} n={len(y0)}", flush=True)
        for ncap, tag in ((400, "n400"), (len(y0), "nfull")):
            Xs, ys = stratified_cap(X0, y0, ncap, rng=0)
            Xz, _ = apply_preprocess(Xs, "zscore")
            for knn in knns:
                cfg = dict(seed_cfg, knn=knn)
                t0 = time.perf_counter()
                try:
                    nmi, k = evaluate("SPC", Xz, ys, nc, "njw", cfg)
                except Exception:
                    nmi, k = float("nan"), "ERR"
                print(f"  {tag:5s} knn={str(knn):4s} nmi={nmi:.4f} k={k} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        # n=400 sample-seed stability
        for s in (1, 2, 3):
            Xs, ys = stratified_cap(X0, y0, 400, rng=s)
            Xz, _ = apply_preprocess(Xs, "zscore")
            cfg = dict(seed_cfg, knn="m15")
            try:
                nmi, k = evaluate("SPC", Xz, ys, nc, "njw", cfg)
            except Exception:
                nmi, k = float("nan"), "ERR"
            print(f"  n400  seed={s} m15 nmi={nmi:.4f} k={k}", flush=True)


if __name__ == "__main__":
    main()
