"""Lean real-data demonstration of the paper's main claims.

Runs, on a subset of the locally-available datasets:
  - Exp 2 (enhanced vs original) with a small parameter sweep, all 4 algorithms.
  - Exp 3 (Reg-* vs recent) on a small subset.

Uses a reduced parameter sweep (not the full §4.1 grid) so it finishes quickly;
the full grid is available via `python -m src.main exp2` / `exp3`.
"""

import time

import numpy as np
import pandas as pd

from src import config
from src.datasets import load_dataset
from src.enhanced import enhance_clustering
from src.enhanced.similarity import SIGMA_GRID, gaussian_similarity
from src.clustering import sprg_similarity
from src.metrics import evaluate
from src.runners import make_base_algorithm, run_original
from src.baselines import DATASET_ORDER, RECENT_ALGOS, RECENT_TABLES

EXP2_SET = ["Thyroid", "Wine", "Glass", "Seeds", "Ecoli", "Sonar", "Spectf", "Raisin", "Landmine"]
EXP3_SET = ["Thyroid", "Wine", "Glass", "Seeds", "Ecoli", "Sonar"]
ALGOS = ("SPC", "APC", "DSet", "SPRG")
# lean sweep for the demo
SIGMA_DEMO = (0.5, 1.0, 2.0)
EPS_DEMO = (0.1, 0.2)
CR_DEMO = (0.03, 0.05)
B_DEMO = (4, 8)
# demo-speed forest size (the paper uses Tclust = 1000; see config.SPRG_TREES)
SPRG_TREES_DEMO = 50


def _graph(algo, X, sigma, n_clusters=None, sprg_graph=None):
    if algo == "SPRG":
        return sprg_graph if sprg_graph is not None else sprg_similarity(X, n_clusters)
    return gaussian_similarity(X, sigma)


def main():
    # verbose experiment output uses ε / ϵ / σ, which cp1252 consoles cannot encode
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    config.SPRG_TREES = SPRG_TREES_DEMO
    print("=" * 78)
    print("Exp 2 — Enhanced (Reg-*) vs Original algorithms (NMI / ACC)")
    print("=" * 78)
    rows = []
    t0 = time.time()
    for ds in EXP2_SET:
        X, y = load_dataset(ds)
        n, _, k = config.DATASETS[ds]
        # the SPRG graph is sigma-independent: learn it once per dataset
        sprg_graph = None
        if "SPRG" in ALGOS:
            try:
                sprg_graph = sprg_similarity(X, k)
            except Exception as e:
                print(f"  [skip] {ds}/SPRG graph: {e}")
        for algo in ALGOS:
            # original: best over a small σ sweep (SPRG ignores σ)
            best_o = None
            for sigma in (SIGMA_DEMO if algo != "SPRG" else (1.0,)):
                S = _graph(algo, X, sigma, k, sprg_graph=sprg_graph)
                try:
                    lo = run_original(algo, S, k if algo in ("SPC", "SPRG") else None, X=X)
                    mo = evaluate(y, lo)
                except Exception:
                    continue
                if best_o is None or mo["nmi"] > best_o[0]["nmi"]:
                    best_o = (mo, sigma)
            if best_o is None:
                continue
            mo, best_sigma = best_o
            S = _graph(algo, X, best_sigma, k, sprg_graph=sprg_graph)
            # enhanced: best over a small (ε, ϵ, b) sweep
            best_e = None
            for eps in EPS_DEMO:
                for cr in CR_DEMO:
                    for b in B_DEMO:
                        if b >= n:
                            continue
                        try:
                            lh, info = enhance_clustering(
                                make_base_algorithm(algo, X=X), S,
                                k if algo in ("SPC", "SPRG") else None,
                                epsilon=eps, b=b, compression_rate=cr,
                            )
                            me = evaluate(y, lh)
                        except Exception:
                            continue
                        if best_e is None or me["nmi"] > best_e[0]["nmi"]:
                            best_e = (me, info)
            if best_e is None:
                continue
            me, info = best_e
            rows.append({
                "dataset": ds, "algo": algo,
                "orig_nmi": round(mo["nmi"], 3), "orig_acc": round(mo["acc"], 3),
                "enh_nmi": round(me["nmi"], 3), "enh_acc": round(me["acc"], 3),
                "k_red": info["k"],
            })
    df2 = pd.DataFrame(rows)
    print(df2.to_string(index=False))
    wins = (df2["enh_nmi"] > df2["orig_nmi"]).sum()
    print(f"\nEnhanced NMI > Original NMI on {wins}/{len(df2)} runs")
    print(f"Mean orig NMI={df2['orig_nmi'].mean():.3f} | Mean enh NMI={df2['enh_nmi'].mean():.3f}")
    print(f"(demo took {time.time()-t0:.1f}s)")

    print("\n" + "=" * 78)
    print("Exp 3 — Reg-* (computed) vs recent algorithms (NMI), subset")
    print("=" * 78)
    # reuse the Exp2 enhanced results where available + run any missing
    reg = {ds: {} for ds in EXP3_SET}
    for r in rows:
        if r["dataset"] in reg:
            reg[r["dataset"]][f"Reg-{r['algo']}"] = r["enh_nmi"]
    out_rows = []
    for ds in EXP3_SET:
        row = {"dataset": ds}
        # recent reference values
        for rec in RECENT_TABLES["nmi"]:
            if rec["dataset"] == ds:
                for a in RECENT_ALGOS:
                    row[a] = rec[a]
                break
        for col, val in reg[ds].items():
            row[col] = val
        out_rows.append(row)
    df3 = pd.DataFrame(out_rows)
    for c in df3.columns:
        if c != "dataset":
            df3[c] = df3[c].round(3)
    print(df3.to_string(index=False))


if __name__ == "__main__":
    main()
