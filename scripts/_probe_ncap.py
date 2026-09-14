"""Why does the 400-cap beat full n? Cross-evaluate winner configs at both n
and run one fair mini-grid (including m15 / sigma=1, which the large-n sweep
never searched) at each n.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config
from src.datasets import load_dataset
from src.enhanced import build_reduced_graph
from src.experiments import _graph_for, _eval_on_reduced
from src.preprocess import apply_preprocess
from src.szemeredi import apply_density_threshold

REFINED = ROOT / "results/lemma_claim/refined_runs.csv"
HIGHER = ROOT / "results/lemma_claim/higher.csv"
OLD = ROOT / "results/lemma_claim/all_small_first.csv"
CLUST = (("SPC", "njw"),)
DATASETS = ("Banknote", "Raisin", "Rice")
GRID_SIGMA = (2.0, 5.0)
GRID_EPS = (0.1, 0.2)


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def cap_of(name, y):
    n = len(y)
    if name == "Rice":
        return 1500, n
    return 400, n


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


def parse_knn(raw):
    s = str(raw).strip()
    if not s or s.lower() in ("none", "nan"):
        return None
    if s.startswith("m") or s.startswith("M"):
        return s.lower()
    return int(float(s))


def winners():
    """(n400 cfg, nfull cfg) per (dataset, clustering) from the stored runs."""
    out = {}
    refined = read(REFINED)
    higher = read(HIGHER)
    old = read(OLD)
    for ds in DATASETS:
        for cl, cv in CLUST:
            small = [r for r in refined
                     if r["dataset"] == ds and r["clustering"] == cl]
            small = [r for r in small if r["nmi"]]
            small = max(small, key=lambda r: float(r["nmi"])) if small else None
            if small is None:
                # old grid rows have no cr/b; assume the modal values
                small = next(
                    (dict(r, cr="0.1", b="8", d0="0.0")
                     for r in old
                     if r["dataset"] == ds and r["clustering"] == cl and r["lemma"]),
                    None,
                )
            big = [r for r in higher
                   if r["dataset"] == ds and r["clustering"] == cl and r["lemma"]]
            big = max(big, key=lambda r: float(r["lemma"])) if big else None
            out[(ds, cl, cv)] = (small, big)
    return out


def cfg_from_row(row, metric="euclidean"):
    return {
        "sigma": float(row["sigma"]), "epsilon": float(row["eps"]),
        "compression": float(row["cr"]), "b": int(float(row["b"])),
        "d0": row["d0"], "knn": parse_knn(row["knn"]),
        "degree_mode": row["degree"], "metric": metric,
        "alg_kind": "alon", "stop_rule": "theoretical",
        "init_mode": "degree", "drop_irregular": "all_pairs",
        "adj_threshold": 0.0,
    }


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
    win = winners()
    for ds in DATASETS:
        X0, y0 = load_dataset(ds)
        nc = int(np.unique(y0).size)
        small_cap, full_cap = cap_of(ds, y0)
        for cl, cv in CLUST:
            small, big = win.get((ds, cl, cv), (None, None))
            print(f"\n## {ds}/{cl} nc={nc} n={len(y0)}", flush=True)
            for ncap in (small_cap, full_cap):
                if ncap == small_cap:
                    Xs, ys = stratified_cap(X0, y0, ncap, rng=0)
                else:
                    Xs, ys = X0, y0
                Xz, _ = apply_preprocess(Xs, "zscore")
                for tag, row in (("n400-winner", small), ("nfull-winner", big)):
                    if row is None:
                        continue
                    stored = row.get("nmi") or row.get("lemma") or "nan"
                    cfg = cfg_from_row(row)
                    t0 = time.perf_counter()
                    try:
                        nmi, k = evaluate(cl, Xz, ys, nc, cv, cfg)
                    except Exception as e:
                        nmi, k = float("nan"), f"ERR {type(e).__name__}"
                    print(
                        f"  n={len(ys):5d} {tag:12s} stored={float(stored):.4f} "
                        f"-> {nmi:.4f} k={k} sig={cfg['sigma']} cr={cfg['compression']} "
                        f"b={cfg['b']} knn={cfg['knn']} {time.perf_counter()-t0:.0f}s",
                        flush=True,
                    )
            # fair mini-grid: includes m15 + sigma in {1,2,5} at both n
            for ncap in (small_cap, full_cap):
                Xs, ys = stratified_cap(X0, y0, ncap, rng=0)
                Xz, _ = apply_preprocess(Xs, "zscore")
                best, best_cfg = -1.0, None
                t0 = time.perf_counter()
                for sigma in GRID_SIGMA:
                    for knn in (20, "m15"):
                        for eps in GRID_EPS:
                            for cr in (0.1, 0.2):
                                for b in (8, 64):
                                    for deg in ("weighted", "spectral"):
                                        cfg = {
                                            "sigma": sigma, "epsilon": eps,
                                            "compression": cr, "b": b, "d0": 0.0,
                                            "knn": knn, "degree_mode": deg,
                                            "metric": "euclidean", "alg_kind": "alon",
                                            "stop_rule": "theoretical",
                                            "init_mode": "degree",
                                            "drop_irregular": "all_pairs",
                                            "adj_threshold": 0.0,
                                        }
                                        try:
                                            nmi, _ = evaluate(cl, Xz, ys, nc, cv, cfg)
                                        except Exception:
                                            continue
                                        if nmi == nmi and nmi > best:
                                            best, best_cfg = nmi, cfg
                print(
                    f"  n={len(ys):5d} mini-grid best={best:.4f} "
                    f"sig={best_cfg['sigma']} knn={best_cfg['knn']} "
                    f"eps={best_cfg['epsilon']} cr={best_cfg['compression']} "
                    f"b={best_cfg['b']} deg={best_cfg['degree_mode']} "
                    f"{time.perf_counter()-t0:.0f}s", flush=True,
                )


if __name__ == "__main__":
    main()
