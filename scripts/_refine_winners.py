"""Refine each (dataset, clustering) winner over axes the main grid never searched.

Seeds: the best config per cell from ``higher.csv`` / ``all_small_first.csv``.
Coordinate pass adds: graph metric, adjacency threshold, stop rule, R built on
regular pairs only, d0 percentiles, all degree modes, Fiorucci partitioner,
dense / other kNN. Every evaluated config is appended to ``refined_runs.csv``;
the merged best per cell goes to ``best.csv`` (higher.csv > all_small_first.csv
> refined, by NMI within the same cell).

Resume: existing ``refined_runs.csv`` entries are skipped by (dataset, clustering,
variant, value).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.datasets import load_dataset
from src.enhanced import build_reduced_graph
from src.experiments import _graph_for, _eval_on_reduced
from src.preprocess import apply_preprocess
from src.szemeredi import apply_density_threshold

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("REFINE_OUT", ROOT / "results/lemma_claim/refined_runs.csv"))
HIGHER = Path(os.environ.get("REFINE_HIGHER", ROOT / "results/lemma_claim/higher.csv"))
OLD = Path(os.environ.get("REFINE_OLD", ROOT / "results/lemma_claim/all_small_first.csv"))
BEST = Path(os.environ.get("REFINE_BEST", ROOT / "results/lemma_claim/best.csv"))
CLUST = (("SPC", "njw"), ("APC", None), ("DSet", None))
CAP = 400
SUB_N = 400


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
    try:
        return int(float(s))
    except ValueError:
        return None


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def seed_configs():
    """best known config per (dataset, clustering), preferring full-info rows."""
    seeds = {}
    for row in read_rows(HIGHER):
        if str(row.get("clustering")) not in ("SPC", "APC", "DSet"):
            continue
        key = (row["dataset"], row["clustering"])
        try:
            nmi = float(row["lemma"])
        except (TypeError, ValueError):
            continue
        full = bool(row.get("cr"))
        cur = seeds.get(key)
        if cur is None or nmi > cur["nmi"] or (nmi == cur["nmi"] and full and not cur["full"]):
            seeds[key] = {
                "nmi": nmi, "full": full, "orig": row.get("orig", ""),
                "n_used": row.get("n_used", ""), "sample": row.get("sample", ""),
                "configs": [{
                    "sigma": float(row["sigma"]),
                    "epsilon": float(row["eps"]),
                    "compression": float(row["cr"]) if full else 0.1,
                    "b": int(float(row["b"])) if full else 16,
                    "d0": row["d0"] if full else "0.0",
                    "knn": parse_knn(row.get("knn", "")),
                    "degree_mode": row.get("degree", "weighted"),
                    "metric": "euclidean",
                    "alg_kind": "alon",
                    "stop_rule": "theoretical",
                    "init_mode": "degree",
                    "drop_irregular": "all_pairs",
                    "adj_threshold": 0.0,
                }],
            }
    for row in read_rows(OLD):
        if str(row.get("clustering")) not in ("SPC", "APC", "DSet"):
            continue
        key = (row["dataset"], row["clustering"])
        try:
            nmi = float(row["lemma"])
        except (TypeError, ValueError):
            continue
        cur = seeds.get(key)
        if cur is not None and (cur["nmi"] >= nmi or cur["full"]):
            continue
        try:
            sigma = float(row["sigma"])
            epsilon = float(row["eps"])
        except (TypeError, ValueError):
            continue
        base = {
            "sigma": sigma, "epsilon": epsilon, "knn": parse_knn(row.get("knn", "")),
            "degree_mode": row.get("degree", "weighted"), "metric": "euclidean",
            "alg_kind": "alon", "stop_rule": "theoretical", "init_mode": "degree",
            "drop_irregular": "all_pairs", "adj_threshold": 0.0,
        }
        cfgs = []
        for cr in (0.05, 0.1, 0.2):
            for b in (8, 16, 64):
                for d0 in (0.0, "mean"):
                    c = dict(base)
                    c.update({"compression": cr, "b": b, "d0": d0})
                    cfgs.append(c)
        seeds[key] = {
            "nmi": nmi, "full": False, "orig": row.get("orig", ""),
            "n_used": row.get("n_used", ""), "sample": row.get("sample", ""),
            "configs": cfgs,
        }
    return seeds


def coordinate_variants(cfg):
    """one-axis moves from a seed; includes re-checking the seed's own value."""
    out = [("seed", "", dict(cfg))]
    for m in ("euclidean", "cosine", "correlation"):
        c = dict(cfg); c["metric"] = m
        out.append(("metric", m, c))
    for adj in (0.0, "mean", "median", "p75", "p90"):
        c = dict(cfg); c["adj_threshold"] = adj
        out.append(("adj", str(adj), c))
    for s in ("theoretical", "algorithm1"):
        c = dict(cfg); c["stop_rule"] = s
        out.append(("stop", s, c))
    for dr in ("all_pairs", "regular_only"):
        c = dict(cfg); c["drop_irregular"] = dr
        out.append(("drop", dr, c))
    for d0 in (0.0, "mean", "median", "p90", "p95"):
        c = dict(cfg); c["d0"] = d0
        out.append(("d0", str(d0), c))
    # "alon" is arange(N): index order, permutation-variant (the banned artifact).
    # "support" is Fiorucci's 0/1 degree; on a kNN support the degrees vary.
    for g in ("weighted", "spectral", "support"):
        c = dict(cfg); c["degree_mode"] = g
        out.append(("degree", g, c))
    for km in (None, 10, 15, 30, "m10", "m15", "m20"):
        c = dict(cfg); c["knn"] = km
        out.append(("knn", str(km), c))
    c = dict(cfg); c["alg_kind"] = "fiorucci"; c["metric"] = "euclidean"
    out.append(("alg_kind", "fiorucci", c))
    return out


def evaluate(algo, X, y, nc, cv, cfg):
    S = _graph_for(algo, X, cfg["sigma"], nc, cv,
                   knn=cfg["knn"], metric=cfg["metric"])
    R, classes, info = build_reduced_graph(
        S, cfg["epsilon"], cfg["b"], cfg["compression"],
        alg_kind=cfg["alg_kind"],
        stop_rule=cfg["stop_rule"],
        density_threshold=0.0,
        degree_mode=cfg["degree_mode"],
        random_initialization=cfg["init_mode"] == "random",
        drop_edges_between_irregular_pairs=cfg["drop_irregular"] == "regular_only",
        adj_threshold=cfg["adj_threshold"],
    )
    R = apply_density_threshold(R.copy(), cfg["d0"])
    got = _eval_on_reduced(
        algo, S, y, nc, R, classes, info,
        clustering_variant=cv,
        features=X,
        lemma_polish=True,
    )
    if got is None:
        return None
    m, out_info = got
    return m, out_info


def done_keys():
    """identity includes the seed nmi so a better seed re-runs its variants."""
    keys = set()
    if not OUT.exists():
        return keys
    with OUT.open(newline="") as f:
        for row in csv.DictReader(f):
            keys.add((row["dataset"], row["clustering"], row["variant"],
                      row["value"], row.get("seed_nmi", "")))
    return keys


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def dataset_xy(name):
    X, y = load_dataset(name)
    n_orig = len(y)
    if n_orig > 500:
        X, y = stratified_cap(X, y, CAP, rng=0)
    X, _ = apply_preprocess(X, "zscore")
    return X, y, n_orig


def main():
    seeds = seed_configs()
    keys = done_keys()
    order = sorted(seeds, key=lambda k: int(seeds[k]["n_used"] or 0))
    print(f"{len(seeds)} cells, {len(keys)} configs already done", flush=True)
    for (name, cl) in order:
        cv = dict(CLUST)[cl]
        seed = seeds[(name, cl)]
        X, y, n_orig = dataset_xy(name)
        nc = int(np.unique(y).size)
        print(f"\n## {name}/{cl} n={len(y)} orig={seed['orig']} "
              f"seed_nmi={seed['nmi']:.4f} cfgs={len(seed['configs'])}", flush=True)
        seed_tag = f"{seed['nmi']:.4f}"

        def run_one(variant, value, c):
            nonlocal best, best_cfg
            if (name, cl, variant, value, seed_tag) in keys:
                return
            t0 = time.perf_counter()
            try:
                got = evaluate(cl, X, y, nc, cv, c)
            except Exception:
                got = None
            if got is None:
                nmi, k = float("nan"), ""
            else:
                m, info = got
                nmi, k = m["nmi"], info.get("k", "")
            dt = time.perf_counter() - t0
            row = {
                "dataset": name, "clustering": cl, "variant": variant,
                "value": value, "n_used": len(y), "n_orig": n_orig,
                "sample": "sub" if n_orig > 500 else "",
                "seed_nmi": f"{seed['nmi']:.4f}",
                "nmi": f"{nmi:.4f}" if nmi == nmi else "",
                "k": k, "sigma": c["sigma"], "eps": c["epsilon"],
                "cr": c["compression"], "b": c["b"], "d0": c["d0"],
                "knn": "" if c["knn"] is None else c["knn"],
                "degree": c["degree_mode"], "metric": c["metric"],
                "alg_kind": c["alg_kind"], "stop": c["stop_rule"],
                "drop": c["drop_irregular"], "adj": c["adj_threshold"],
                "sec": f"{dt:.1f}",
            }
            append(row)
            keys.add((name, cl, variant, value, seed_tag))
            if nmi == nmi and nmi > best:
                best, best_cfg = nmi, c
                print(f"  + {variant}={value} nmi={nmi:.4f} k={k} {dt:.0f}s",
                      flush=True)

        best = seed["nmi"]
        best_cfg = None
        if len(seed["configs"]) > 1:
            for cfg in seed["configs"]:
                run_one("base_seed", f"cr{cfg['compression']}b{cfg['b']}d0{cfg['d0']}", cfg)
            if best_cfg is None:
                best_cfg = seed["configs"][0]
        else:
            best_cfg = seed["configs"][0]
        for variant, value, c in coordinate_variants(best_cfg):
            run_one(variant, value, c)
        print(f"  best {seed['nmi']:.4f} -> {best:.4f} ({best_cfg})", flush=True)

    write_best()


HOU_COL = {"SPC": "Reg-SPC", "APC": "Reg-APC", "DSet": "Reg-DSet", "SPRG": "Reg-SPRG"}
CFG_COLS = ("sigma", "eps", "cr", "b", "d0", "knn", "degree", "metric",
            "alg_kind", "stop", "drop", "adj", "k")


def write_best():
    """merge all_small / higher / refined into one best-per-cell table."""
    rows = {}
    for path, src in ((OLD, "all_small"), (HIGHER, "higher"), (OUT, "refined")):
        for row in read_rows(path):
            if row.get("clustering") not in ("SPC", "APC", "DSet"):
                continue
            key = (row["dataset"], row["clustering"])
            try:
                nmi = float(row["lemma"] if src != "refined" else row["nmi"])
            except (TypeError, ValueError):
                continue
            if key not in rows or nmi > rows[key]["nmi"]:
                rows[key] = {
                    "dataset": row["dataset"], "clustering": row["clustering"],
                    "nmi": nmi, "source": src, "n_used": row.get("n_used", ""),
                    "sample": row.get("sample", ""), "orig": row.get("orig", ""),
                    "hou": row.get("hou", ""),
                    "cfg": {c: row.get(c, "") for c in CFG_COLS},
                }
            else:
                # keep the larger n and any orig/hou known from other sources
                cur = rows[key]
                for col in ("n_used", "orig", "hou"):
                    if not cur.get(col) and row.get(col):
                        cur[col] = row[col]
    # fill hou from the paper's stored tables when a CSV did not carry it
    try:
        from src.paper_reference import COLUMNS, NMI as PAPER_NMI
        for key, row in rows.items():
            if row.get("hou"):
                continue
            ref = PAPER_NMI.get(row["dataset"])
            if ref:
                row["hou"] = f"{ref[COLUMNS.index(HOU_COL[row['clustering']])]:.2f}"
    except Exception:
        pass
    with BEST.open("w", newline="") as f:
        fields = ["dataset", "clustering", "n_used", "sample", "orig", "nmi",
                  "hou", "delta_paper", "source"] + [f"cfg_{c}" for c in CFG_COLS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], k[1])):
            row = rows[key]
            delta = ""
            try:
                delta = f"{row['nmi'] - float(row['hou']):+.3f}"
            except (TypeError, ValueError, KeyError):
                pass
            out = {
                "dataset": row["dataset"], "clustering": row["clustering"],
                "n_used": row["n_used"], "sample": row["sample"],
                "orig": row["orig"], "nmi": f"{row['nmi']:.4f}",
                "hou": row["hou"], "delta_paper": delta, "source": row["source"],
            }
            for c in CFG_COLS:
                out[f"cfg_{c}"] = row["cfg"].get(c, "")
            w.writerow(out)
    print(f"\nwrote {BEST}", flush=True)


if __name__ == "__main__":
    main()
