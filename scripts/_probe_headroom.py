"""Headroom probe: plain k-means / NJW on the same z-scored features vs lemma NMI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.cluster import KMeans

from src import config
from src.clustering.spectral import spc
from src.datasets import load_dataset
from src.enhanced.similarity import gaussian_similarity
from src.metrics import nmi
from src.preprocess import apply_preprocess

DATASETS = ("Appendicitis", "Wine", "Sonar", "Seeds", "Glass", "Thyroid",
            "Spectf", "Ecoli", "Landmine", "Libras", "SCC", "Raisin",
            "Banknote", "Leaves", "Dutchnumeral", "Segment", "Rice",
            "Spambase", "Landsat", "USPS")


def cap(X, y, n=400, rng=0):
    rng = np.random.default_rng(rng)
    if len(y) <= n:
        return X, y
    classes = np.unique(y)
    idx = []
    for c in classes:
        loc = np.where(y == c)[0]
        k = max(1, int(round(len(loc) * n / len(y))))
        idx.append(rng.choice(loc, size=min(k, len(loc)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    for name in DATASETS:
        X, y = load_dataset(name)
        X, y = cap(X, y)
        nc = int(np.unique(y).size)
        X, _ = apply_preprocess(X, "zscore")
        km = KMeans(n_clusters=nc, n_init=10, random_state=0).fit_predict(X)
        km_nmi = nmi(y, km)
        line = f"{name:14s} n={len(y):4d} nc={nc:3d} kmeans={km_nmi:.3f}"
        for sig in (1.0, 2.0, 5.0):
            S = gaussian_similarity(X, sig, knn=20)
            try:
                lab = spc(S, nc, variant="njw")
                line += f" njw_s{sig:g}={nmi(y, lab):.3f}"
            except Exception as e:
                line += f" njw_s{sig:g}=ERR"
        print(line, flush=True)


if __name__ == "__main__":
    main()
