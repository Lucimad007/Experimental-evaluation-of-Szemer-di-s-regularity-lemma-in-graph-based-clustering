"""Experiment runner reproducing §4.1, §4.2 and §4.3.

Three experiments (see ``spec/experiments.md``):
  - Exp 1: influence of the three parameters (mean over the other two).
  - Exp 2: enhanced vs original algorithms (NMI + time).
  - Exp 2b: regularity vs k-means partitioning.
  - Exp 3: comparison with recent algorithms (uses ``baselines``).

Results are written to ``results/`` (gitignored) as CSV and PNG figures.
"""

import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .datasets import load_dataset
from .enhanced import enhance_clustering, kmeans_partition_clustering
from .enhanced.similarity import SIGMA_GRID, gaussian_similarity
from .metrics import evaluate
from .runners import make_base_algorithm, original_graph, run_original

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _needs_sigma(name):
    return name in ("SPC", "APC", "DSet")


def _graph_for(algo, X, sigma, n_clusters=None):
    """Original graph G for ``algo``: Gaussian(σ) for SPC/APC/DSet, learned for SPRG."""
    return original_graph(algo, X, sigma, n_clusters=n_clusters)


def _b_grid_for(n, b_grid=None):
    b_grid = b_grid or config.B_GRID
    return [b for b in b_grid if b < n]


# ----------------------------------------------------------------------- Exp 1
def experiment1_parameter_influence(
    dataset_names=None,
    algorithms=None,
    epsilon_grid=None,
    compression_grid=None,
    b_grid=None,
    sigma_grid=None,
    out_dir=None,
    verbose=False,
):
    """Exp 1 (§4.1): influence of ε, ϵ, b on the enhanced algorithms.

    For each algorithm/dataset run the enhanced algorithm over the full grid and
    record (NMI, time). Then aggregate by taking the mean over all combinations of
    the other two parameters to show the influence of the third.
    """
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    epsilon_grid = epsilon_grid or config.EPSILON_GRID
    compression_grid = compression_grid or config.COMPRESSION_GRID
    b_grid = b_grid or config.B_GRID
    sigma_grid = sigma_grid or SIGMA_GRID
    out_dir = Path(out_dir or RESULTS_DIR) / "exp1_parameter_influence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume from the checkpoint (e.g. after a power outage): datasets already
    # present in raw_runs_partial.csv are skipped entirely
    rows = []
    partial_csv = out_dir / "raw_runs_partial.csv"
    if partial_csv.exists():
        try:
            old = pd.read_csv(partial_csv)
            rows = old.to_dict("records")
            done = set(old["dataset"].unique())
            before = len(dataset_names)
            dataset_names = [d for d in dataset_names if d not in done]
            print(f"[exp1] resuming: {len(done)} dataset(s) already checkpointed, "
                  f"{len(dataset_names)}/{before} remaining")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp1 datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        for algo in algorithms:
            sigmas = sigma_grid if _needs_sigma(algo) else [None]
            for sigma in sigmas:
                S = _graph_for(algo, X, sigma, n_clusters)
                bs = _b_grid_for(n, b_grid)
                for eps in epsilon_grid:
                    for cr in compression_grid:
                        for b in bs:
                            try:
                                labels, info = enhance_clustering(
                                    make_base_algorithm(algo, X=X),
                                    S,
                                    n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                                    epsilon=eps,
                                    b=b,
                                    compression_rate=cr,
                                    verbose=verbose,
                                )
                            except Exception as e:
                                if verbose:
                                    print(f"    [err] {ds_name}/{algo}/σ={sigma}/ε={eps}/ϵ={cr}/b={b}: {e}")
                                continue
                            m = evaluate(y, labels)
                            rows.append({
                                "dataset": ds_name, "algo": algo, "sigma": sigma,
                                "epsilon": eps, "compression": cr, "b": b,
                                "nmi": m["nmi"], "acc": m["acc"], "ari": m["ari"], "ri": m["ri"],
                                "time": info["total_time"], "k": info["k"],
                            })
        # checkpoint after each dataset: a crash preserves all completed work
        pd.DataFrame(rows).to_csv(out_dir / "raw_runs_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_runs.csv", index=False)
    (out_dir / "raw_runs_partial.csv").unlink(missing_ok=True)
    _summarize_influence(df, out_dir)
    _summarize_all_vs_selected(df, out_dir)
    return df


def _in_selected(row):
    """Whether a run's (ε, ϵ, b) fall in the paper's recommended narrow ranges."""
    return (
        row["epsilon"] in config.EPSILON_RECOMMENDED
        and row["compression"] in config.COMPRESSION_RECOMMENDED
        and row["b"] in config.B_RECOMMENDED
    )


def _summarize_all_vs_selected(df, out_dir):
    """Fig. 6: mean NMI over ALL parameters vs over the SELECTED (recommended) ranges."""
    if df.empty:
        return
    df = df.copy()
    df["selected"] = df.apply(_in_selected, axis=1)
    all_mean = df.groupby(["dataset", "algo"])["nmi"].mean().reset_index().rename(columns={"nmi": "all_nmi"})
    sel = df[df["selected"]]
    sel_mean = sel.groupby(["dataset", "algo"])["nmi"].mean().reset_index().rename(columns={"nmi": "selected_nmi"})
    cmp = all_mean.merge(sel_mean, on=["dataset", "algo"], how="left")
    cmp.to_csv(out_dir / "all_vs_selected.csv", index=False)


def _summarize_influence(df, out_dir):
    """Mean NMI/time over all combinations of the other two parameters (§4.1).

    For a fixed value of the parameter under study, the mean is taken over all
    combinations of the remaining parameters (the two other partitioning
    parameters and, for the σ-dependent algorithms, σ as well).
    """
    if df.empty:
        return
    for param in ["epsilon", "compression", "b"]:
        agg = df.groupby(["dataset", "algo", param]).agg(
            nmi=("nmi", "mean"), time=("time", "mean")
        ).reset_index()
        agg.to_csv(out_dir / f"influence_{param}.csv", index=False)


# ----------------------------------------------------------------------- Exp 2
def experiment2_enhanced_vs_original(
    dataset_names=None,
    algorithms=None,
    out_dir=None,
    verbose=False,
):
    """Exp 2 (§4.2): enhanced (recommended params) vs original algorithms."""
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    out_dir = Path(out_dir or RESULTS_DIR) / "exp2_enhanced_vs_original"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume: (dataset, algo) pairs already in the partial CSV are skipped
    rows = []
    done_pairs = set()
    partial_csv = out_dir / "enhanced_vs_original_partial.csv"
    if partial_csv.exists():
        try:
            old = pd.read_csv(partial_csv)
            rows = old.to_dict("records")
            done_pairs = set(zip(old["dataset"], old["algo"]))
            dataset_names = [d for d in dataset_names
                             if not all((d, a) in done_pairs for a in algorithms)]
            print(f"[exp2] resuming: {len(rows)} (dataset, algo) rows checkpointed")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp2 datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        for algo in algorithms:
            if (ds_name, algo) in done_pairs:
                continue
            # pick best sigma for the original algorithm (SPRG ignores sigma)
            best_orig = None
            best_orig_nmi = -1
            best_sigma = None
            sigmas = SIGMA_GRID if _needs_sigma(algo) else [1.0]
            for sigma in sigmas:
                S = _graph_for(algo, X, sigma, n_clusters)
                try:
                    t0 = time.time()
                    labels = run_original(algo, S, n_clusters, X=X)
                    orig_time = time.time() - t0
                    m = evaluate(y, labels)
                except Exception:
                    continue
                if m["nmi"] > best_orig_nmi:
                    best_orig_nmi = m["nmi"]
                    best_orig = (labels, m, orig_time)
                    best_sigma = sigma
            if best_orig is None:
                continue
            S = _graph_for(algo, X, best_sigma, n_clusters)
            # enhanced with recommended params (best over the recommended grid)
            best_enh = None
            best_enh_nmi = -1
            for eps in config.EPSILON_RECOMMENDED:
                for cr in config.COMPRESSION_RECOMMENDED:
                    for b in config.B_RECOMMENDED:
                        if b >= n:
                            continue
                        try:
                            labels, info = enhance_clustering(
                                make_base_algorithm(algo, X=X),
                                S,
                                n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                                epsilon=eps, b=b, compression_rate=cr, verbose=verbose,
                            )
                        except Exception:
                            continue
                        m = evaluate(y, labels)
                        if m["nmi"] > best_enh_nmi:
                            best_enh_nmi = m["nmi"]
                            best_enh = (labels, m, info)
            if best_enh is None:
                continue
            rows.append({
                "dataset": ds_name, "algo": algo,
                "orig_nmi": best_orig[1]["nmi"], "orig_time": best_orig[2],
                "enh_nmi": best_enh[1]["nmi"], "enh_time": best_enh[2]["total_time"],
                "enh_k": best_enh[2]["k"],
            })
        # checkpoint after each dataset: a crash preserves all completed work
        pd.DataFrame(rows).to_csv(out_dir / "enhanced_vs_original_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "enhanced_vs_original.csv", index=False)
    (out_dir / "enhanced_vs_original_partial.csv").unlink(missing_ok=True)
    return df


# --------------------------------------------------------------------- Exp 2b
def experiment2b_regularity_vs_kmeans(
    dataset_names=None,
    algorithms=None,
    out_dir=None,
    verbose=False,
):
    """Exp 2b (§4.2, Fig. 11): regularity partitioning vs k-means partitioning."""
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    out_dir = Path(out_dir or RESULTS_DIR) / "exp2b_regularity_vs_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume: (dataset, algo) pairs already in the partial CSV are skipped
    rows = []
    done_pairs = set()
    partial_csv = out_dir / "regularity_vs_kmeans_partial.csv"
    if partial_csv.exists():
        try:
            old = pd.read_csv(partial_csv)
            rows = old.to_dict("records")
            done_pairs = set(zip(old["dataset"], old["algo"]))
            dataset_names = [d for d in dataset_names
                             if not all((d, a) in done_pairs for a in algorithms)]
            print(f"[exp2b] resuming: {len(rows)} (dataset, algo) rows checkpointed")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp2b datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        for algo in algorithms:
            if (ds_name, algo) in done_pairs:
                continue
            best_sigma = 1.0
            best_nmi = -1
            for sigma in (SIGMA_GRID if _needs_sigma(algo) else [1.0]):
                S = _graph_for(algo, X, sigma, n_clusters)
                try:
                    labels, info = enhance_clustering(
                        make_base_algorithm(algo, X=X),
                        S,
                        n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                        epsilon=0.15, b=4, compression_rate=0.05, verbose=verbose,
                    )
                    m = evaluate(y, labels)
                except Exception:
                    continue
                if m["nmi"] > best_nmi:
                    best_nmi = m["nmi"]
                    best_sigma = sigma
            S = _graph_for(algo, X, best_sigma, n_clusters)
            try:
                reg_labels, reg_info = enhance_clustering(
                    make_base_algorithm(algo, X=X), S,
                    n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                    epsilon=0.15, b=4, compression_rate=0.05, verbose=verbose,
                )
                # "keep all the other parts unchanged": the k-means partition
                # uses the same number of classes as the regularity partition
                target_k = int(reg_info["k"])
                km_labels, km_info = kmeans_partition_clustering(
                    make_base_algorithm(algo, X=X), X, S,
                    n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                    k_classes=target_k,
                )
            except Exception as e:
                if verbose:
                    print(f"  [err] {ds_name}/{algo}: {e}")
                continue
            reg_m = evaluate(y, reg_labels)
            km_m = evaluate(y, km_labels)
            rows.append({
                "dataset": ds_name, "algo": algo, "target_k": target_k,
                "reg_nmi": reg_m["nmi"], "reg_time": reg_info["total_time"],
                "km_nmi": km_m["nmi"], "km_time": km_info["total_time"],
            })
        # checkpoint after each dataset: a crash preserves all completed work
        pd.DataFrame(rows).to_csv(out_dir / "regularity_vs_kmeans_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regularity_vs_kmeans.csv", index=False)
    (out_dir / "regularity_vs_kmeans_partial.csv").unlink(missing_ok=True)
    return df


# --------------------------------------------------------------------- Exp 3
def _best_enhanced_metrics(algo, X, y, n_clusters, verbose=False):
    """Best enhanced (Reg-*) metrics over the recommended (ε, ϵ, b, σ) grid.

    Returns a dict with nmi/acc/ari/ri at the best-NMI configuration, or None.
    """
    n = len(y)
    sigmas = SIGMA_GRID if _needs_sigma(algo) else [1.0]
    best = None
    best_nmi = -1.0
    for sigma in sigmas:
        S = _graph_for(algo, X, sigma, n_clusters)
        for eps in config.EPSILON_RECOMMENDED:
            for cr in config.COMPRESSION_RECOMMENDED:
                for b in config.B_RECOMMENDED:
                    if b >= n:
                        continue
                    try:
                        labels, info = enhance_clustering(
                            make_base_algorithm(algo, X=X),
                            S,
                            n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                            epsilon=eps, b=b, compression_rate=cr, verbose=verbose,
                        )
                    except Exception:
                        continue
                    m = evaluate(y, labels)
                    if m["nmi"] > best_nmi:
                        best_nmi = m["nmi"]
                        best = m
    return best


def experiment3_vs_recent(dataset_names=None, out_dir=None, verbose=False):
    """Exp 3 (§4.3, Tables 2–5): enhanced algorithms vs recent algorithms.

    Runs Reg-SPC, Reg-APC, Reg-DSet, Reg-SPRG on each dataset (best config over the
    recommended parameter grid), computes NMI/ACC/ARI/RI, and joins them with the
    8 recent-algorithm reference columns transcribed from Tables 2–5 of the paper.
    Produces one combined table per metric (recent + Reg-* columns + mean row).
    """
    from .baselines import DATASET_ORDER, RECENT_ALGOS, RECENT_TABLES

    out_dir = Path(out_dir or RESULTS_DIR) / "exp3_vs_recent"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = dataset_names or DATASET_ORDER
    reg_algos = ("SPC", "APC", "DSet", "SPRG")
    reg_cols = [f"Reg-{a}" for a in reg_algos]

    # resume: datasets already present in the partial NMI table are skipped
    partial_csv = out_dir / "table_nmi_partial.csv"
    if partial_csv.exists():
        try:
            old = pd.read_csv(partial_csv)
            done = set(old["dataset"].unique())
            dataset_names = [d for d in dataset_names if d not in done]
            print(f"[exp3] resuming: {len(done)} dataset(s) checkpointed")
        except Exception:
            pass

    # per-metric rows: dataset -> {recent algos..., Reg-*...}
    metric_rows = {metric: [] for metric in ("nmi", "acc", "ari", "ri")}

    for ds_name in tqdm(dataset_names, desc="Exp3 datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        _, _, n_clusters = config.DATASETS[ds_name]
        # recent-algorithm reference values for this dataset
        recent_vals = {metric: {} for metric in metric_rows}
        for metric in metric_rows:
            for rec in RECENT_TABLES[metric]:
                if rec["dataset"] == ds_name:
                    for algo in RECENT_ALGOS:
                        recent_vals[metric][algo] = rec[algo]
                    break
        # Reg-* enhanced metrics
        reg_vals = {metric: {} for metric in metric_rows}
        for algo in reg_algos:
            m = _best_enhanced_metrics(algo, X, y, n_clusters, verbose=verbose)
            if m is None:
                for metric in metric_rows:
                    reg_vals[metric][f"Reg-{algo}"] = float("nan")
            else:
                for metric in metric_rows:
                    reg_vals[metric][f"Reg-{algo}"] = m[metric]
        # assemble rows
        for metric in metric_rows:
            row = {"dataset": ds_name}
            row.update(recent_vals[metric])
            row.update(reg_vals[metric])
            metric_rows[metric].append(row)
        # checkpoint after each dataset: a crash preserves all completed work
        _write_exp3_tables(metric_rows, out_dir, final=False)

    _write_exp3_tables(metric_rows, out_dir, final=True)
    return metric_rows


def _write_exp3_tables(metric_rows, out_dir, final=True):
    """Write the per-metric Exp 3 tables (partial during the run, final at end)."""
    suffix = "" if final else "_partial"
    for metric, rows in metric_rows.items():
        df = pd.DataFrame(rows)
        if not df.empty and final:
            num = df.drop(columns=["dataset"])
            mean_row = {"dataset": "mean"}
            mean_row.update(num.mean(numeric_only=True).to_dict())
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df.to_csv(out_dir / f"table_{metric}{suffix}.csv", index=False)
    if final:
        for metric in metric_rows:
            (out_dir / f"table_{metric}_partial.csv").unlink(missing_ok=True)
