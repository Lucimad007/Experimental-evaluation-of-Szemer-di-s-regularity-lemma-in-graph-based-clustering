"""Higher-score lemma sweep: finer partitions (k up to ~128) + wider grid.

Same graph-faithful pipeline as ``_lemma_all.py`` (Alon partition on the kNN
support -> Eq. 3 reduced graph -> base clustering on R -> map -> V0 -> Lloyd/
k-means polish). The only change is the searched grid: epsilon up to 0.2 and
b up to 64/128 so the partition can resolve as many classes as the data needs
(the old grid capped k at ~32, which made SPC/SPRG fail on nc>32 datasets).

Resume: results/lemma_claim/higher.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.paper_reference import COLUMNS, NMI as PAPER_NMI
from src.preprocess import apply_preprocess

OUT = Path("results/lemma_claim/higher.csv")
CLUST = (("SPC", "njw"), ("APC", None), ("DSet", None))
HOU_COL = {"SPC": "Reg-SPC", "APC": "Reg-APC", "DSet": "Reg-DSet", "SPRG": "Reg-SPRG"}
MAX_N = 2600
SUB_N = 1200
EPS = (0.1, 0.2)
CR = (0.05, 0.1, 0.2)
SIGMA = (1.0, 2.0, 5.0)
KNN = (20, "m15")

# adaptive grid: the partition check is O(n^2)-ish, so large n gets a smaller
# search. small n gets the full pull. mutual knn >= 20 is included at large n:
# the 2026-09-13 probe showed knn=20 is too sparse there (Banknote n=1372 SPC:
# 0.563 at knn=20 -> 0.835 at m40).
EPS_BIG = (0.1, 0.2)
CR_BIG = (0.1, 0.2)
SIGMA_BIG = (2.0, 5.0)
KNN_BIG = (20, "m20", "m40")


def grid_for(n, nc):
    """(EPS, CR, b, SIGMA, KNN, d0, degree) searched for this size."""
    if n <= 500:
        return EPS, CR, b_grid(n, nc), SIGMA, KNN, (0.0, "mean"), ("weighted", "spectral")
    return EPS_BIG, CR_BIG, b_grid(n, nc), SIGMA_BIG, KNN_BIG, (0.0,), ("weighted", "spectral")


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


def b_grid(n, nc):
    base = [b for b in (8, 16, 64) if b < n]
    if nc > 64 and 128 < n:
        base.append(128)
    return tuple(base)


def hou(name, cl):
    row = PAPER_NMI.get(name)
    if not row:
        return ""
    return f"{row[COLUMNS.index(HOU_COL[cl])]:.2f}"


def done_keys():
    keys = set()
    if not OUT.exists():
        return keys
    with OUT.open(newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["dataset"], row["clustering"]))
    return keys


def append_row(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    sizes = []
    for name in config.DATASETS:
        X, y = load_dataset(name)
        sizes.append((len(y), name))
    sizes.sort()

    keys = done_keys()
    print("order:", ", ".join(f"{n}({s})" for s, n in sizes), flush=True)

    for n_orig, name in sizes:
        X, y = load_dataset(name)
        sampled = ""
        if n_orig > MAX_N:
            X, y = stratified_cap(X, y, SUB_N, rng=0)
            sampled = f"sub{len(y)}"
            print(f"\n## {name} n={n_orig} -> {len(y)}", flush=True)
        else:
            print(f"\n## {name} n={n_orig}", flush=True)
        nc = int(np.unique(y).size)
        X, _ = apply_preprocess(X, "zscore")
        g_eps, g_cr, g_b, g_sigma, g_knn, g_d0, g_deg = grid_for(len(y), nc)
        print(f"   grid: b={g_b} eps={g_eps} cr={g_cr} sigma={g_sigma} "
              f"knn={g_knn} d0={g_d0}", flush=True)

        for cl, cv in CLUST:
            if (name, cl) in keys:
                print(f"  skip {cl}", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                orig = _best_original(cl, X, y, nc, clustering_variant=cv, knn=20)
                orig_nmi = orig[1]["nmi"] if orig else float("nan")
                best, info, st = _best_over_enhanced_grid(
                    cl, X, y, nc,
                    g_eps, g_cr, g_b, g_sigma,
                    stop_rules=("theoretical",),
                    d0_grid=g_d0,
                    degree_modes=g_deg,
                    alg_kinds=("alon",),
                    init_modes=("degree",),
                    drop_modes=("all_pairs",),
                    adj_thresholds=(0.0,),
                    clustering_variant=cv,
                    knn=g_knn,
                    lemma_polish=True,
                )
            except Exception as e:
                print(f"  {cl} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            elapsed = time.perf_counter() - t0
            enh = best["nmi"] if best else float("nan")
            k_part = info.get("k", "") if info else ""
            st = st or {}
            row = {
                "dataset": name,
                "n_orig": n_orig,
                "n_used": len(y),
                "sample": sampled,
                "clustering": cl,
                "orig": f"{orig_nmi:.4f}" if orig_nmi == orig_nmi else "",
                "lemma": f"{enh:.4f}" if enh == enh else "",
                "hou": hou(name, cl),
                "delta_orig": f"{enh - orig_nmi:+.4f}" if orig and best else "",
                "eps": f"{st.get('epsilon'):.4g}" if isinstance(st.get("epsilon"), (int, float)) else "",
                "sigma": f"{st.get('sigma'):.4g}" if isinstance(st.get("sigma"), (int, float)) else "",
                "cr": st.get("compression", ""),
                "b": st.get("b", ""),
                "d0": st.get("d0", ""),
                "knn": st.get("knn", ""),
                "degree": st.get("degree_mode", ""),
                "k_part": k_part,
                "sec": f"{elapsed:.1f}",
            }
            append_row(row)
            print(
                f"  {cl:4s} orig={row['orig']:6s} lemma={row['lemma']:6s} "
                f"hou={row['hou']:4s} {row['delta_orig']:7s} "
                f"k={row['k_part']} b={row['b']} cr={row['cr']} "
                f"sigma={row['sigma']} knn={row['knn']} {row['degree']} {row['sec']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
