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
    algos = sorted({a for df in frames.values() for a in df["algo"].unique()})
    for algo in algos:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for col, param in enumerate(["epsilon", "compression", "b"]):
            if param not in frames:
                continue
            g = frames[param][frames[param]["algo"] == algo]
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
        fig.suptitle(
            f"{algo}: NMI and running time vs ε (left), ϵ (middle), b (right)",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(influence_dir / f"influence_{algo}.png", dpi=120)
        plt.close(fig)


def plot_all_vs_selected(influence_dir):
    """Fig. 6: mean NMI with ALL parameters vs with the SELECTED (recommended)
    parameter ranges, per dataset and algorithm."""
    influence_dir = Path(influence_dir)
    csv = influence_dir / "all_vs_selected.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    algos = list(df["algo"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 6), squeeze=False)
    for i, algo in enumerate(algos):
        ax = axes[0, i]
        sub = df[df["algo"] == algo].set_index("dataset")
        x = np.arange(len(sub))
        width = 0.4
        ax.bar(x - width / 2, sub["all_nmi"], width, label="all parameters")
        ax.bar(x + width / 2, sub["selected_nmi"], width, label="selected parameters")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=90, fontsize=7)
        ax.set_title(algo, fontsize=10)
        ax.set_ylabel("mean NMI")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Mean NMI: all parameters vs selected (recommended) ranges")
    fig.tight_layout()
    fig.savefig(influence_dir / "all_vs_selected.png", dpi=120)
    plt.close(fig)


def plot_enhanced_vs_original(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
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
        ax.set_xticklabels(sub["algo"], rotation=45, fontsize=7)
        ax.set_title(ds, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Enhanced vs original (NMI)")
    fig.tight_layout()
    fig.savefig(csv_path.parent / "enhanced_vs_original.png", dpi=120)
    plt.close(fig)


def plot_regularity_vs_kmeans(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
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
        ax.set_xticklabels(sub["algo"], rotation=45, fontsize=7)
        ax.set_title(ds, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Regularity vs k-means partitioning (NMI)")
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
