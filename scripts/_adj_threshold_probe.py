"""Bigger adj-threshold bakeoff. Small UCI, SPC+APC, paper-lite grids.

Saves CSV under results/adj_threshold_probe/.
"""

import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import config
from src.datasets import load_dataset
from src.enhanced import enhance_clustering
from src.enhanced.similarity import gaussian_similarity
from src.metrics import evaluate
from src.runners import make_base_algorithm, run_original
from src.szemeredi import unweighted_adjacency

DATASETS = ("Appendicitis", "Wine", "Seeds", "Sonar", "Glass", "Ecoli", "Spectf")
ALGOS = ("SPC", "APC")
TAUS = (0, "p25", "median", "mean", "p75")
SIGMAS = config.SIGMA_GRID
EPSILONS = (0.1, 0.15, 0.2)
BS = (4, 8, 16)
CR = 0.05
OUT = Path("results/adj_threshold_probe")


def _fn(algo):
    return make_base_algorithm(algo)


def _orig_best(algo, X, y, k):
    best = None
    for sigma in SIGMAS:
        S = gaussian_similarity(X, sigma)
        try:
            labels = run_original(algo, S, k if algo == "SPC" else None, X=X)
            m = evaluate(y, labels)
        except Exception:
            continue
        if best is None or m["nmi"] > best["nmi"]:
            best = dict(m, sigma=sigma)
    return best


def _enh_best(algo, X, y, k, tau):
    fn = _fn(algo)
    best = None
    n = len(y)
    bs = [b for b in BS if b < n]
    for sigma in SIGMAS:
        S = gaussian_similarity(X, sigma)
        dens = float(unweighted_adjacency(S, tau).sum()) / (n * (n - 1))
        for eps in EPSILONS:
            for b in bs:
                try:
                    labels, info = enhance_clustering(
                        fn, S,
                        n_clusters=(k if algo == "SPC" else None),
                        epsilon=eps, b=b, compression_rate=CR,
                        degree_mode="alon", adj_threshold=tau,
                        stop_rule="theoretical",
                    )
                except Exception:
                    continue
                m = evaluate(y, labels)
                rec = dict(
                    m, sigma=sigma, eps=eps, b=b, kR=info["k"],
                    v0=info["v0_size"], dens=dens, time=info["total_time"],
                )
                if best is None or rec["nmi"] > best["nmi"]:
                    best = rec
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(d, a, t) for d in DATASETS for a in ALGOS for t in TAUS]
    orig_cache = {}
    for dataset, algo, tau in tqdm(jobs, desc="adj-threshold"):
        X, y = load_dataset(dataset)
        k = int(len(np.unique(y)))
        ok = (dataset, algo)
        if ok not in orig_cache:
            orig_cache[ok] = _orig_best(algo, X, y, k)
        orig = orig_cache[ok]
        enh = _enh_best(algo, X, y, k, tau)
        row = {
            "dataset": dataset, "algo": algo, "tau": str(tau),
            "n": len(y), "k": k,
            "orig_nmi": None if orig is None else orig["nmi"],
            "orig_acc": None if orig is None else orig["acc"],
            "orig_sigma": None if orig is None else orig["sigma"],
        }
        if enh is None:
            row.update(enh_nmi=None, status="fail")
        else:
            row.update(
                status="ok",
                enh_nmi=enh["nmi"], enh_acc=enh["acc"],
                dens=enh["dens"], sigma=enh["sigma"], eps=enh["eps"],
                b=enh["b"], kR=enh["kR"], v0=enh["v0"],
                dNMI=enh["nmi"] - (orig["nmi"] if orig else 0.0),
                time=enh["time"],
            )
        rows.append(row)

    csv_path = OUT / "best_per_tau.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {csv_path}")
    print(f"{'dataset':<14} {'algo':<4} {'orig':>6}  "
          + "  ".join(f"{str(t):>7}" for t in TAUS)
          + "  winner")
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["algo"])].append(r)
    wins = {str(t): 0 for t in TAUS}
    for (dataset, algo), rs in grouped.items():
        orig = rs[0]["orig_nmi"]
        by = {r["tau"]: r.get("enh_nmi") for r in rs}
        nums = {t: by.get(str(t)) for t in TAUS}
        valid = {t: v for t, v in nums.items() if v is not None}
        winner = max(valid, key=valid.get) if valid else None
        if winner is not None:
            wins[str(winner)] += 1
        cells = "  ".join(
            f"{(nums[t] if nums[t] is not None else float('nan')):7.3f}"
            if nums[t] is not None else f"{'FAIL':>7}"
            for t in TAUS
        )
        o = f"{orig:6.3f}" if orig is not None else "  n/a "
        print(f"{dataset:<14} {algo:<4} {o}  {cells}  {winner}")
    print("wins (highest enhanced NMI per dataset/algo):", wins)


if __name__ == "__main__":
    main()
