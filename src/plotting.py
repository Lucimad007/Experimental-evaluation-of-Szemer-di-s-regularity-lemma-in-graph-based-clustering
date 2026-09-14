"""Plotting helpers for the experiment outputs (§4 figures).

Generates the figures corresponding to Figs. 2–11 of the paper from the CSV
results produced by ``experiments``. Uses matplotlib (Agg backend, headless).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _series_label(df):
    """algo, or algo/variant when the permutation grid is present."""
    if "clustering_variant" in df.columns:
        return df["algo"].astype(str) + "/" + df["clustering_variant"].astype(str)
    return df["algo"]


def plot_parameter_influence(influence_dir):
    """Figs 2–5: one figure per algorithm — NMI (top) and running time
    (bottom, log scale) vs each of ε, ϵ, b, one curve per dataset."""
    influence_dir = Path(influence_dir)
    frames = {}
    for param in ["epsilon", "compression", "b"]:
        csv = influence_dir / f"influence_{param}.csv"
        if csv.exists():
            frames[param] = pd.read_csv(csv)
    if not frames:
        return
    first = next(iter(frames.values()))
    if "clustering_variant" in first.columns:
        keys = sorted({
            (r.algo, r.clustering_variant)
            for df in frames.values()
            for r in df[["algo", "clustering_variant"]].drop_duplicates().itertuples()
        })
    else:
        keys = [(a, None) for a in sorted({a for df in frames.values() for a in df["algo"].unique()})]
    for algo, variant in keys:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for col, param in enumerate(["epsilon", "compression", "b"]):
            if param not in frames:
                continue
            g = frames[param][frames[param]["algo"] == algo]
            if variant is not None:
                g = g[g["clustering_variant"] == variant]
            for row, (val, ylabel) in enumerate((("nmi", "NMI"), ("time", "time (s)"))):
                ax = axes[row, col]
                for ds, sub in g.groupby("dataset"):
                    sub = sub.sort_values(param)
                    ax.plot(sub[param].values, sub[val].values, marker="o", linewidth=1, label=ds)
                ax.set_xlabel(param, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.tick_params(labelsize=7)
                if row == 1:
                    ax.set_yscale("log")
        title = algo if variant is None else f"{algo}/{variant}"
        fig.suptitle(
            f"{title}: NMI and running time vs ε (left), ϵ (middle), b (right)",
            fontsize=12,
        )
        fig.tight_layout()
        suffix = algo if variant is None else f"{algo}_{variant}".replace("+", "_")
        fig.savefig(influence_dir / f"influence_{suffix}.png", dpi=120)
        plt.close(fig)


def plot_all_vs_selected(influence_dir):
    """Fig. 6: mean NMI with ALL parameters vs with the SELECTED (recommended)
    parameter ranges, per dataset and algorithm."""
    influence_dir = Path(influence_dir)
    csv = influence_dir / "all_vs_selected.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    if "clustering_variant" in df.columns:
        df["series"] = df["algo"].astype(str) + "/" + df["clustering_variant"].astype(str)
    else:
        df["series"] = df["algo"]
    series = list(df["series"].unique())
    ncols = min(4, max(1, len(series)))
    nrows = int(np.ceil(len(series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(6, 3.5 * ncols), 4 * nrows), squeeze=False)
    for i, name in enumerate(series):
        ax = axes[i // ncols, i % ncols]
        sub = df[df["series"] == name].set_index("dataset")
        x = np.arange(len(sub))
        width = 0.4
        ax.bar(x - width / 2, sub["all_nmi"], width, label="all parameters")
        ax.bar(x + width / 2, sub["selected_nmi"], width, label="selected parameters")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=90, fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("mean NMI")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(series), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Mean NMI: all parameters vs selected (recommended) ranges")
    fig.tight_layout()
    fig.savefig(influence_dir / "all_vs_selected.png", dpi=120)
    plt.close(fig)


def plot_enhanced_vs_original(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    # too many partition-axis rows to plot raw; collapse to clustering variant
    if "clustering_variant" in df.columns:
        gcols = ["dataset", "algo", "clustering_variant"]
        df = df.groupby(gcols, dropna=False)[["orig_nmi", "enh_nmi"]].mean().reset_index()
    datasets = list(df["dataset"].unique())
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.ravel()
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["orig_nmi"], width, label="original")
        ax.bar(x + width / 2, sub["enh_nmi"], width, label="enhanced")
        ax.set_xticks(x)
        ax.set_xticklabels(_series_label(sub), rotation=45, fontsize=6)
        ax.set_title(ds, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Enhanced vs original (NMI), mean over partition axes")
    fig.tight_layout()
    fig.savefig(csv_path.parent / "enhanced_vs_original.png", dpi=120)
    plt.close(fig)


def plot_raw_vs_zscore(csv_path):
    """Original vs enhanced NMI on raw features vs z-score (Exp-2 protocol)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "preprocess" not in df.columns:
        return
    gcols = [c for c in ("dataset", "algo", "clustering_variant", "preprocess") if c in df.columns]
    df = df.groupby(gcols, dropna=False)[["orig_nmi", "enh_nmi"]].mean().reset_index()
    datasets = list(df["dataset"].unique())
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.ravel()
    width = 0.2
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        if "clustering_variant" in sub.columns:
            series = list((sub["algo"] + "/" + sub["clustering_variant"]).unique())
        else:
            series = list(sub["algo"].unique())
        x = np.arange(len(series))
        def _vals(mode, col):
            out = []
            for s in series:
                if "clustering_variant" in sub.columns:
                    algo, var = s.split("/", 1)
                    hit = sub[(sub["algo"] == algo) & (sub["clustering_variant"] == var)
                              & (sub["preprocess"] == mode)]
                else:
                    hit = sub[(sub["algo"] == s) & (sub["preprocess"] == mode)]
                out.append(float(hit[col].mean()) if len(hit) else np.nan)
            return out
        ax.bar(x - 1.5 * width, _vals("raw", "orig_nmi"), width, label="orig raw")
        ax.bar(x - 0.5 * width, _vals("zscore", "orig_nmi"), width, label="orig zscore")
        ax.bar(x + 0.5 * width, _vals("raw", "enh_nmi"), width, label="enh raw")
        ax.bar(x + 1.5 * width, _vals("zscore", "enh_nmi"), width, label="enh zscore")
        ax.set_xticks(x)
        ax.set_xticklabels(series, rotation=45, fontsize=6)
        ax.set_title(ds, fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Raw vs z-score features — original and enhanced NMI (Exp 2 protocol)")
    fig.tight_layout()
    fig.savefig(csv_path.parent / "raw_vs_zscore.png", dpi=120)
    plt.close(fig)


def plot_regularity_vs_kmeans(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if "clustering_variant" in df.columns:
        gcols = ["dataset", "algo", "clustering_variant"]
        df = df.groupby(gcols, dropna=False)[["reg_nmi", "km_nmi"]].mean().reset_index()
    datasets = list(df["dataset"].unique())
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes = axes.ravel()
    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width / 2, sub["reg_nmi"], width, label="regularity")
        ax.bar(x + width / 2, sub["km_nmi"], width, label="kmeans")
        ax.set_xticks(x)
        ax.set_xticklabels(_series_label(sub), rotation=45, fontsize=6)
        ax.set_title(ds, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Regularity vs k-means partitioning (NMI), mean over partition axes")
    fig.tight_layout()
    fig.savefig(csv_path.parent / "regularity_vs_kmeans.png", dpi=120)
    plt.close(fig)


def plot_vs_recent(exp3_dir):
    """Figures for Exp 3: Reg-* vs the 8 recent algorithms (NMI), per dataset."""
    exp3_dir = Path(exp3_dir)
    csv = exp3_dir / "table_nmi.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    # drop the trailing 'mean' row for the per-dataset figure
    df = df[df["dataset"] != "mean"]
    reg_cols = [c for c in df.columns if c.startswith("Reg-")]
    recent_cols = [c for c in df.columns if c not in ("dataset",) and not c.startswith("Reg-")]
    datasets = list(df["dataset"])
    fig, axes = plt.subplots(4, 5, figsize=(22, 12))
    axes = axes.ravel()
    for i, ds in enumerate(datasets):
        ax = axes[i]
        row = df[df["dataset"] == ds].iloc[0]
        vals = [row[c] if pd.notna(row[c]) else 0.0 for c in recent_cols + reg_cols]
        colors = ["#9ecae1"] * len(recent_cols) + ["#e6550d"] * len(reg_cols)
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_title(ds, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, 1)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Reg-* (orange) vs recent algorithms (blue) — NMI")
    fig.tight_layout()
    fig.savefig(exp3_dir / "vs_recent_nmi.png", dpi=120)
    plt.close(fig)


def _nmi_col(df):
    for name in ("nmi_mean", "enh_nmi_mean", "reg_nmi_mean", "nmi", "enh_nmi", "reg_nmi"):
        if name in df.columns:
            return name
    return None


def plot_axis_comparisons(out_dir):
    """Bar charts from comparison_{axis}.csv written by the experiment runner."""
    out_dir = Path(out_dir)
    for csv in sorted(out_dir.glob("comparison_*.csv")):
        if csv.name == "comparison_leaderboard.csv":
            df = pd.read_csv(csv)
            if df.empty:
                continue
            nmi = _nmi_col(df)
            if nmi is None or "clustering_variant" not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(df)), 5))
            labels = df["algo"].astype(str) + "/" + df["clustering_variant"].astype(str)
            ax.bar(np.arange(len(df)), df[nmi].values)
            ax.set_xticks(np.arange(len(df)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("mean NMI")
            ax.set_title("Clustering-variant leaderboard (mean NMI)")
            fig.tight_layout()
            fig.savefig(out_dir / "comparison_leaderboard.png", dpi=120)
            plt.close(fig)
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        nmi = _nmi_col(df)
        axis = csv.stem.replace("comparison_", "")
        if nmi is None or axis not in df.columns:
            continue
        algos = list(df["algo"].unique()) if "algo" in df.columns else ["all"]
        fig, axes = plt.subplots(
            1, max(1, len(algos)),
            figsize=(max(5, 4 * len(algos)), 4),
            squeeze=False,
        )
        for i, algo in enumerate(algos):
            ax = axes[0, i]
            sub = df[df["algo"] == algo] if "algo" in df.columns else df
            grouped = sub.groupby(axis, dropna=False)[nmi].mean().sort_index()
            ax.bar(np.arange(len(grouped)), grouped.values)
            ax.set_xticks(np.arange(len(grouped)))
            ax.set_xticklabels([str(x) for x in grouped.index], rotation=45, ha="right", fontsize=8)
            ax.set_title(str(algo), fontsize=10)
            ax.set_ylabel("mean NMI")
        fig.suptitle(f"NMI by {axis}")
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison_{axis}.png", dpi=120)
        plt.close(fig)
