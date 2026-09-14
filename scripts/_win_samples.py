"""Win-path check on an Ecoli subsample and a Leaves slice (theoretical ε-stop)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.datasets import load_dataset
from src.experiments import _best_original, _best_over_enhanced_grid
from src.paper_reference import NMI as PAPER_NMI
from src.preprocess import apply_preprocess


def stratified_sample(X, y, frac=0.5, rng=None):
    rng = np.random.default_rng(0 if rng is None else rng)
    idx = []
    for c in np.unique(y):
        loc = np.where(y == c)[0]
        k = max(1, int(round(len(loc) * frac)))
        k = min(k, len(loc))
        idx.append(rng.choice(loc, size=k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def leaves_slice(X, y, n_classes=8):
    labels = np.unique(y)[:n_classes]
    mask = np.isin(y, labels)
    X, y = X[mask], y[mask]
    y = np.unique(y, return_inverse=True)[1]
    return X, y


def run_cell(name, X, y, paper_spc=None):
    nc = int(np.unique(y).size)
    X, _ = apply_preprocess(X, "zscore")
    print(f"\n== {name}  n={len(y)} d={X.shape[1]} nc={nc}  stop=theoretical  knn=20 ==")
    orig = _best_original("SPC", X, y, nc, clustering_variant="njw", knn=20)
    orig_nmi = orig[1]["nmi"] if orig else float("nan")
    best, info, st = _best_over_enhanced_grid(
        "SPC", X, y, nc,
        config.EPSILON_RECOMMENDED,
        config.COMPRESSION_RECOMMENDED,
        config.B_RECOMMENDED,
        config.SIGMA_GRID,
        stop_rules=("theoretical",),
        d0_grid=(0.0,),
        degree_modes=("spectral",),
        alg_kinds=("alon",),
        init_modes=("degree",),
        drop_modes=("all_pairs",),
        clustering_variant="njw",
        knn=20,
        reassign_vertices=True,
    )
    if best is None:
        print(f"  orig {orig_nmi:.3f}  enh FAILED")
        return
    enh = best["nmi"]
    tag = "BEATS orig" if enh > orig_nmi else "below orig"
    paper = f"  paper {paper_spc:.2f}" if paper_spc is not None else ""
    vs_paper = ""
    if paper_spc is not None:
        vs_paper = "  BEATS paper" if enh > paper_spc else "  below paper"
    print(
        f"  orig {orig_nmi:.3f}  enh {enh:.3f}  k={info['k']}  "
        f"σ={st['sigma']} ε={st['epsilon']} ϵ={st['compression']} b={st['b']}  "
        f"{tag}{paper}{vs_paper}"
    )


def main():
    Xe, ye = load_dataset("Ecoli")
    Xs, ys = stratified_sample(Xe, ye, frac=0.5)
    run_cell("Ecoli 50% stratified", Xs, ys, paper_spc=PAPER_NMI["Ecoli"][8])

    Xl, yl = load_dataset("Leaves")
    Xl, yl = leaves_slice(Xl, yl, n_classes=8)
    run_cell("Leaves 8 species (16 each)", Xl, yl, paper_spc=PAPER_NMI["Leaves"][8])
    print("\ndone")


if __name__ == "__main__":
    main()
