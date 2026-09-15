"""Matplotlib figures for THESIS_FA.md from the 2026-09-14 lemma CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "figures"
ALL = ROOT / "results" / "lemma_claim" / "all_small_first.csv"
BEST = ROOT / "results" / "lemma_claim" / "best.csv"
HIGHER = ROOT / "results" / "lemma_claim" / "higher.csv"

NAVY = "#143a6b"
GOLD = "#c4a35a"
FULL = {
    "Appendicitis", "Wine", "Sonar", "Seeds", "Glass", "Thyroid",
    "Spectf", "Ecoli", "Landmine", "Libras",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": NAVY,
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fget(row: dict, key: str):
    try:
        v = float(row[key])
        return v if np.isfinite(v) else None
    except (KeyError, TypeError, ValueError):
        return None


def delta_bar():
    rows = [r for r in read(ALL) if r["dataset"] in FULL]
    labels, vals, colors = [], [], []
    for r in rows:
        o, e = fget(r, "orig"), fget(r, "lemma")
        if o is None or e is None:
            continue
        labels.append(f"{r['dataset'][:6]}-{r['clustering']}")
        d = e - o
        vals.append(d)
        colors.append(NAVY if d >= 0 else "#8b2e2e")
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=0.82)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("ΔNMI  (lemma − orig)")
    ax.set_title("Full-n datasets: Algorithm 1 vs original on the same G")
    fig.tight_layout()
    fig.savefig(OUT / "fig_delta_nmi_full.png", dpi=160)
    plt.close(fig)


def scatter_hou():
    rows = read(ALL)
    xf, yf, xc, yc = [], [], [], []
    for r in rows:
        e, h = fget(r, "lemma"), fget(r, "hou")
        if e is None or h is None:
            continue
        if r["dataset"] in FULL:
            xf.append(h)
            yf.append(e)
        else:
            xc.append(h)
            yc.append(e)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(xf, yf, c=NAVY, s=28, label="full n", zorder=3)
    ax.scatter(xc, yc, c=GOLD, s=28, marker="s", label="capped n", zorder=3)
    ax.plot([0, 1], [0, 1], color="#888", lw=0.9, ls="--")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Hou Tables 2–5  (Reg-*)")
    ax.set_ylabel("ours  (lemma NMI)")
    ax.set_title("Paper vs us (all_small_first.csv)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ours_vs_hou.png", dpi=160)
    plt.close(fig)


def leaves_recovery():
    allr = {(r["dataset"], r["clustering"]): r for r in read(ALL)}
    high = {(r["dataset"], r["clustering"]): r for r in read(HIGHER)}
    best = {(r["dataset"], r["clustering"]): r for r in read(BEST)}
    keys = [("Leaves", "SPC"), ("Rice", "APC"), ("Rice", "DSet"),
            ("Banknote", "APC"), ("Libras", "SPC"), ("USPS", "APC")]
    labels, a, b = [], [], []
    for k in keys:
        labels.append(f"{k[0]}-{k[1]}")
        oa = fget(allr.get(k, {}), "lemma")
        nb = fget(best.get(k, {}), "nmi")
        if nb is None:
            nb = fget(high.get(k, {}), "lemma")
        a.append(0.0 if oa is None else oa)
        b.append(0.0 if nb is None else nb)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, a, w, color="#9aa7b8", label="all_small_first")
    ax.bar(x + w / 2, b, w, color=NAVY, label="best.csv")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("NMI")
    ax.set_ylim(0, 1)
    ax.set_title("Where the finer / refine sweeps moved NMI")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_refine_lifts.png", dpi=160)
    plt.close(fig)


def pipeline():
    """Exact pipeline of scripts/_lemma_all.py, drawn as a two-branch flowchart."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fill=NAVY, fc="white", fs=10.5, lw=1.4, ec=GOLD):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
            linewidth=lw, edgecolor=ec, facecolor=fill, zorder=2,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=fc, zorder=3, linespacing=1.25)

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
            color="#1a1a1a", connectionstyle=f"arc3,rad={rad}", zorder=1,
        ))

    # shared front end (top row)
    w, h, y = 20, 9, 40
    xs = [3, 27, 51, 75]
    labels = ["Load dataset\n(Table 1, 20 sets)", "Cap n > 500\nstratified ~400",
              "z-score\ncolumns", "Gaussian + kNN\ngraph G"]
    for x, t in zip(xs, labels):
        box(x, y, w, h, t)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + w, y + h / 2), (b, y + h / 2))

    # original branch
    box(103, 40, 22, 9, "Original:\ncluster G", fill="#6b7a90")
    arrow((75 + w, y + h / 2), (103, 44.5))

    # enhanced branch (bottom row)
    y2, w2 = 12, 22
    xs2 = [3, 29, 55, 81, 107]
    labels2 = ["Alon partition\n(3 conditions, Hou mods)",
               "reduced graph R\n(Eq. 3 densities, k x k)",
               "cluster R\nSPC / APC / DSet / SPRG",
               "map labels to V_j\n+ V_0 to nearest",
               "Lloyd / k-means\npolish (optional)"]
    for x, t in zip(xs2, labels2):
        box(x, y2, w2, h, t)
    for a, b in zip(xs2[:-1], xs2[1:]):
        arrow((a + w2, y2 + h / 2), (b, y2 + h / 2))
    # fork down from G to Alon: elbow route so it does not cross any label
    gx = 75 + w / 2
    ax_x = 3 + w2 / 2
    ax.plot([gx, gx], [y, 30], color="#1a1a1a", lw=1.3, zorder=1)
    ax.plot([gx, ax_x], [30, 30], color="#1a1a1a", lw=1.3, zorder=1)
    arrow((ax_x, 30), (ax_x, y2 + h))

    # scoring box
    box(103, 1, 26, 7, "NMI vs true labels y\nbest cell over the grid",
        fill="white", fc=NAVY, ec=NAVY)
    arrow((107 + w2 / 2, y2), (116, 8))
    arrow((125, 40), (128, 8), rad=-0.35)

    ax.text(3, 52.5, "Original arm: same algorithm on the full graph",
            fontsize=9.5, color="#6b7a90")
    ax.text(30, 24.5, "Enhanced arm = Algorithm 1 of Hou et al. (partition -> R -> cluster R -> map)",
            fontsize=9.5, color=NAVY)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    delta_bar()
    scatter_hou()
    leaves_recovery()
    print("wrote", list(OUT.glob("fig_*.png")))


if __name__ == "__main__":
    main()
