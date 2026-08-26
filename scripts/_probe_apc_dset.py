"""Probe APC preference and DSet extraction on G vs R (Leaves, Ecoli)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AffinityPropagation

from src.clustering.affinity_propagation import apc
from src.clustering.dominant_set import dominant_sets
from src.datasets import load_dataset
from src.enhanced.similarity import gaussian_similarity
from src.metrics import evaluate
from src.szemeredi import build_regularity_lemma


def map_back(classes, k, red_labels, n, sim_mat=None):
    labels = np.full(n, -1, dtype=int)
    for j in range(1, k + 1):
        labels[classes == j] = red_labels[j - 1]
    v0 = np.where(classes == 0)[0]
    if v0.size and sim_mat is not None:
        for p in v0:
            best_lab, best = None, -np.inf
            for j in range(1, k + 1):
                members = np.where(classes == j)[0]
                if members.size == 0:
                    continue
                score = sim_mat[p, members].mean()
                if score > best:
                    best, best_lab = score, red_labels[j - 1]
            labels[p] = best_lab if best_lab is not None else int(red_labels[0])
    elif v0.size:
        labels[v0] = 0
    return labels


def intra_density(sim_mat, classes, k):
    intra = np.zeros(k)
    for j in range(1, k + 1):
        idx = np.where(classes == j)[0]
        if idx.size == 0:
            continue
        block = sim_mat[np.ix_(idx, idx)]
        intra[j - 1] = block.sum() / (idx.size ** 2)
    return intra


def run_ap(S, pref):
    model = AffinityPropagation(
        affinity="precomputed",
        preference=pref,
        random_state=314,
        max_iter=500,
        damping=0.5,
        copy=True,
    )
    lab = model.fit_predict(S)
    lab = np.asarray(lab, dtype=int)
    if (lab < 0).any():
        lab[lab < 0] = lab.max() + 1 if lab.max() >= 0 else 0
    return lab


def calibrate_pref(S, target_k, lo_q=10, hi_q=99.5, n_steps=10):
    """Binary-search a common preference to match target_k (Frey–Dueck)."""
    off = S[~np.eye(S.shape[0], dtype=bool)]
    pos = off[off > 0] if np.any(off > 0) else off
    best_lab, best_diff = None, 10 ** 9
    lo, hi = float(np.percentile(pos, lo_q)), float(np.percentile(pos, hi_q))
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        lab = run_ap(S, mid)
        kh = len(set(lab))
        diff = abs(kh - target_k)
        if diff < best_diff:
            best_diff, best_lab = diff, lab
        if kh == target_k:
            return lab
        if kh < target_k:
            lo = mid
        else:
            hi = mid
    return best_lab


def dset_custom(A, thresh, leftover_frac=0.05):
    from math import ceil

    n = A.shape[0]
    clusters = np.zeros(n, dtype=int)
    already = np.full(n, False)
    k = 0
    while n - already.sum() > ceil(leftover_frac * n) and k < n:
        x = np.full(n, 1.0)
        x[already] = 0.0
        x /= x.sum()
        y = x.copy()
        for _ in range(1000):
            y_old = y
            y = y_old * (A @ y_old)
            s = y.sum()
            if s <= 0:
                break
            y = y / s
            if np.linalg.norm(y - y_old) < 1e-5:
                break
        cluster = np.where(y >= thresh)[0]
        cluster = cluster[~already[cluster]]
        if cluster.size == 0:
            rest = np.where(~already)[0]
            if rest.size == 0:
                break
            cluster = rest[np.argmax(y[rest])][None]
        already[cluster] = True
        clusters[cluster] = k
        k += 1
    clusters[~already] = k
    return clusters


def report(tag, y, labels):
    ev = evaluate(y, labels)
    print(
        "  %-28s k_hat=%4d  NMI=%.3f ACC=%.3f"
        % (tag, len(set(labels)), ev["nmi"], ev["acc"]),
        flush=True,
    )
    return ev["nmi"]


def probe_dataset(name, sigmas, eps=0.1, b=8, cr=0.1):
    X, y = load_dataset(name)
    nc = len(set(y))
    print("\n==== %s n=%d nc=%d ====" % (name, len(y), nc), flush=True)
    for sigma in sigmas:
        S = gaussian_similarity(X, sigma)
        print("\n-- sigma=%s --" % sigma, flush=True)
        report("orig APC median>0", y, apc(S))
        report("orig APC median(all)", y, run_ap(S, np.median(S)))
        report("orig DSet 1/(1.5n)", y, dominant_sets(S))
        report("orig DSet 1e-5", y, dset_custom(S, 1e-5))
        report("orig DSet 1e-4", y, dset_custom(S, 1e-4))

        alg = build_regularity_lemma("alon", S, eps, True, False, False, False)
        alg.run(b=b, compression_rate=cr)
        R, classes, k = alg.reduced_sim_mat, alg.classes.astype(int), int(alg.k)
        off = R[~np.eye(k, dtype=bool)]
        pos = off[off > 0]
        intra = intra_density(S, classes, k)
        print(
            "  R: k=%d |V0|=%d off med=%.3f p95=%.3f intra med=%.3f CV=%.2f"
            % (
                k,
                (classes == 0).sum(),
                np.median(pos) if pos.size else 0,
                np.percentile(pos, 95) if pos.size else 0,
                np.median(intra),
                (off.std() / off.mean()) if off.mean() else 0,
            ),
            flush=True,
        )

        def mapped(lab):
            return map_back(classes, k, lab, len(y), S)

        report("R APC median>0", y, mapped(apc(R)))
        report("R APC median(all)", y, mapped(run_ap(R, np.median(R))))
        report("R APC intra Eq3", y, mapped(run_ap(R, intra)))
        for q in (70, 80, 90, 95, 99):
            report(
                "R APC p%d" % q,
                y,
                mapped(run_ap(R, np.percentile(pos, q))),
            )
        report("R APC k-calibrated", y, mapped(calibrate_pref(R, nc)))

        report("R DSet 1/(1.5n)", y, mapped(dominant_sets(R)))
        report("R DSet 1e-5", y, mapped(dset_custom(R, 1e-5)))
        report("R DSet 1e-4", y, mapped(dset_custom(R, 1e-4)))
        for fac in (0.5, 1.0, 1.5, 3.0, 8.0):
            report(
                "R DSet 1/(%.1fn)" % fac,
                y,
                mapped(dset_custom(R, 1.0 / (fac * k))),
            )

        d0 = float(eps)
        R_eps = R.copy()
        R_eps[R_eps <= d0] = 0.0
        np.fill_diagonal(R_eps, 0.0)
        report("R d0=eps APC med", y, mapped(apc(R_eps)))
        report("R d0=eps APC intra", y, mapped(run_ap(R_eps, intra)))
        report("R d0=eps DSet", y, mapped(dominant_sets(R_eps)))

        d0m = float(np.median(pos)) if pos.size else 0.0
        R_m = R.copy()
        R_m[R_m <= d0m] = 0.0
        np.fill_diagonal(R_m, 0.0)
        report("R d0=med APC med", y, mapped(apc(R_m)))
        report("R d0=med APC intra", y, mapped(run_ap(R_m, intra)))
        report("R d0=med DSet", y, mapped(dominant_sets(R_m)))


if __name__ == "__main__":
    probe_dataset("Ecoli", (0.2, 0.5, 1.0))
    probe_dataset("Leaves", (0.2, 0.5, 1.0, 2.0))
