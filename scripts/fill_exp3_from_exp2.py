"""Build Exp 3 Tables 2–5 from Exp 2's best (ε, ϵ, b, σ).

Re-runs only the winning config (not the full recommended grid) so we get
NMI/ACC/ARI/RI without duplicating the Exp 2 search. Polls until Exp 2 has
finished Ecoli and Leaves, then writes ``results/exp3_leaves_ecoli/``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, plotting
from src.baselines import RECENT_ALGOS, RECENT_TABLES
from src.datasets import load_dataset
from src.enhanced import enhance_clustering
from src.experiments import _graph_for, _needs_sigma
from src.metrics import evaluate
from src.runners import make_base_algorithm

EXP2_DIR = ROOT / "results" / "exp2_leaves_ecoli" / "exp2_enhanced_vs_original"
OUT_DIR = ROOT / "results" / "exp3_leaves_ecoli"
DATASETS = ("Ecoli", "Leaves")
ALGOS = ("SPC", "APC", "DSet", "SPRG")
METRICS = ("nmi", "acc", "ari", "ri")


def _exp2_frame():
    for name in ("enhanced_vs_original.csv", "enhanced_vs_original_partial.csv"):
        p = EXP2_DIR / name
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if not df.empty and "dataset" in df.columns:
                return df
    return None


def _sigma(row, algo):
    if not _needs_sigma(algo):
        return None
    if "enh_sigma" not in row.index or pd.isna(row["enh_sigma"]):
        return 1.0
    return float(row["enh_sigma"])


def _eval_best_row(row):
    ds, algo = row["dataset"], row["algo"]
    X, y = load_dataset(ds)
    _, _, n_clusters = config.DATASETS[ds]
    sigma = _sigma(row, algo)
    S = _graph_for(algo, X, sigma, n_clusters)
    labels, _info = enhance_clustering(
        make_base_algorithm(algo, X=X),
        S,
        n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
        epsilon=float(row["enh_epsilon"]),
        b=int(row["enh_b"]),
        compression_rate=float(row["enh_compression"]),
    )
    return evaluate(y, labels)


def _recent_row(dataset, metric):
    for rec in RECENT_TABLES[metric]:
        if rec["dataset"] == dataset:
            return {a: rec[a] for a in RECENT_ALGOS}
    return {a: float("nan") for a in RECENT_ALGOS}


def _write_tables(by_ds):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        rows = []
        for ds in DATASETS:
            if ds not in by_ds:
                continue
            row = {"dataset": ds}
            row.update(_recent_row(ds, metric))
            for algo in ALGOS:
                row[f"Reg-{algo}"] = by_ds[ds].get(algo, {}).get(metric, float("nan"))
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            num = df.drop(columns=["dataset"])
            mean_row = {"dataset": "mean"}
            mean_row.update(num.mean(numeric_only=True).to_dict())
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df.to_csv(OUT_DIR / f"table_{metric}.csv", index=False)
    plotting.plot_vs_recent(OUT_DIR)


def main():
    print("[exp3-fill] waiting for Exp 2 best configs (Ecoli, Leaves)", flush=True)
    done = {}
    deadline = time.time() + 36 * 3600
    while time.time() < deadline:
        df = _exp2_frame()
        if df is not None:
            for ds in DATASETS:
                sub = df[df["dataset"] == ds]
                if len(sub) < len(ALGOS):
                    continue
                if ds in done:
                    continue
                print(f"[exp3-fill] {ds}: re-running 4 winning configs for ARI/RI", flush=True)
                done[ds] = {}
                for _, row in sub.iterrows():
                    algo = row["algo"]
                    t0 = time.time()
                    try:
                        m = _eval_best_row(row)
                    except Exception as e:
                        print(f"  FAIL {ds}/{algo}: {e}", flush=True)
                        continue
                    done[ds][algo] = m
                    print(
                        f"  {ds}/{algo} NMI={m['nmi']:.3f} ACC={m['acc']:.3f} "
                        f"ARI={m['ari']:.3f} RI={m['ri']:.3f} "
                        f"({time.time()-t0:.1f}s)",
                        flush=True,
                    )
                _write_tables(done)
                print(f"[exp3-fill] wrote tables after {ds} -> {OUT_DIR}", flush=True)
        if all(ds in done and len(done[ds]) >= 1 for ds in DATASETS):
            print("[exp3-fill] both datasets filled", flush=True)
            return
        time.sleep(30)
    print("[exp3-fill] timed out waiting for Exp 2", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
