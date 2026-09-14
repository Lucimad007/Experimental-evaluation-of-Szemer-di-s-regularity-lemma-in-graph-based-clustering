"""Exploratory profile of the Table 1 loaders (no invented features).

Writes ``results/eda/dataset_profile.csv`` and ``column_flags.csv``. Hygiene
that actually changes X lives in ``preprocess.zscore`` (zero-variance and
exact-duplicate columns only). Everything else is reported, not dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .datasets import load_dataset, list_datasets
from .preprocess import apply_preprocess

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _scale_ratio(std):
    pos = std[std > 0]
    if pos.size == 0:
        return float("nan")
    return float(pos.max() / pos.min())


def _max_abs_corr(X, cap=90):
    d = X.shape[1]
    if d < 2 or d > cap:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(d, 1)
    vals = np.abs(C[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else float("nan")


def _n_duplicate_cols(X):
    seen = set()
    n = 0
    for j in range(X.shape[1]):
        key = np.ascontiguousarray(X[:, j]).tobytes()
        if key in seen:
            n += 1
        else:
            seen.add(key)
    return n


def profile_matrix(X, y, name=None):
    """one-row dict of hygiene / scale facts for a loaded ``(X, y)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape
    std = np.nanstd(X, axis=0, ddof=0)
    nunique = np.array([
        len(np.unique(X[:, j][np.isfinite(X[:, j])])) for j in range(d)
    ])
    _, counts = np.unique(y, return_counts=True)
    try:
        n_unique_rows = int(len(np.unique(np.round(X, 12), axis=0)))
    except Exception:
        n_unique_rows = -1
    table = config.DATASETS.get(name) if name else None
    return {
        "dataset": name,
        "n": n,
        "d": d,
        "nc": int(len(np.unique(y))),
        "table_n": None if table is None else table[0],
        "table_d": None if table is None else table[1],
        "table_nc": None if table is None else table[2],
        "n_nan": int(np.isnan(X).sum()),
        "n_inf": int(np.isinf(X).sum()),
        "n_const": int((std == 0).sum()),
        "n_nunique_le2": int((nunique <= 2).sum()),
        "n_dup_cols": _n_duplicate_cols(X),
        "n_unique_rows": n_unique_rows,
        "n_dup_rows": n - n_unique_rows if n_unique_rows >= 0 else -1,
        "std_ratio": _scale_ratio(std),
        "min_std": float(std[std > 0].min()) if (std > 0).any() else 0.0,
        "max_std": float(std.max()) if d else 0.0,
        "class_imbalance": float(counts.max() / counts.min()) if counts.min() else float("nan"),
        "max_abs_corr": _max_abs_corr(X),
        "zscore_d": int(apply_preprocess(X, "zscore")[1]["n_features"]),
    }


def profile_columns(X, name):
    """per-column flags worth looking at (const / duplicate / binary / huge scale)."""
    X = np.asarray(X, dtype=float)
    std = np.nanstd(X, axis=0, ddof=0)
    rows = []
    seen = {}
    med_std = float(np.median(std[std > 0])) if (std > 0).any() else 1.0
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        nunique = int(len(np.unique(finite))) if finite.size else 0
        key = np.ascontiguousarray(col).tobytes()
        flags = []
        if std[j] == 0:
            flags.append("zero_variance")
        if key in seen:
            flags.append(f"duplicate_of_{seen[key]}")
        else:
            seen[key] = j
        if nunique <= 2 and std[j] > 0:
            flags.append("binary_or_two_value")
        if std[j] > 0 and med_std > 0 and std[j] / med_std >= 100:
            flags.append("std_100x_median")
        if not flags:
            continue
        rows.append({
            "dataset": name,
            "col": j,
            "std": float(std[j]),
            "nunique": nunique,
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
            "flags": ",".join(flags),
        })
    return rows


def profile_all(dataset_names=None, out_dir=None):
    """profile every Table 1 dataset; write CSVs under ``results/eda/``."""
    dataset_names = dataset_names or list_datasets()
    out_dir = Path(out_dir or RESULTS_DIR) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    col_rows = []
    skipped = []
    for name in dataset_names:
        try:
            X, y = load_dataset(name)
        except Exception as e:
            skipped.append({"dataset": name, "error": str(e)})
            print(f"  [skip] {name}: {e}")
            continue
        rec = profile_matrix(X, y, name)
        rows.append(rec)
        col_rows.extend(profile_columns(X, name))
        print(
            f"{name:14s} n={rec['n']:5d} d={rec['d']:3d} nc={rec['nc']:3d} "
            f"const={rec['n_const']} dupcol={rec['n_dup_cols']} "
            f"std_ratio={rec['std_ratio']:.3g} zscore_d={rec['zscore_d']}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset_profile.csv", index=False)
    pd.DataFrame(col_rows).to_csv(out_dir / "column_flags.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out_dir / "skipped.csv", index=False)
    return df


def original_preview(dataset_names=None, algorithms=None, n_max=3000, d_max=100, out_dir=None):
    """best-σ original NMI on raw vs z-score (no regularity).

    skip datasets with ``n > n_max`` or ``d > d_max`` (USPS / Dutchnumeral / …).
    """
    from .metrics import evaluate
    from .runners import original_graph, run_original

    dataset_names = dataset_names or list_datasets()
    algorithms = algorithms or ("SPC",)
    out_dir = Path(out_dir or RESULTS_DIR) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in dataset_names:
        try:
            X, y = load_dataset(name)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        n, d, n_clusters = config.DATASETS[name]
        if n > n_max or d > d_max:
            print(f"  [skip] {name}: n={n} d={d} (preview cap n≤{n_max}, d≤{d_max})")
            continue
        for mode in ("raw", "zscore"):
            Xp, info = apply_preprocess(X, mode)
            for algo in algorithms:
                if algo == "SPRG":
                    continue
                best_nmi = -1.0
                best_acc = float("nan")
                best_sigma = None
                for sigma in config.SIGMA_GRID:
                    S = original_graph(algo, Xp, sigma, n_clusters=n_clusters)
                    try:
                        labels = run_original(algo, S, n_clusters, X=Xp)
                        m = evaluate(y, labels)
                    except Exception:
                        continue
                    if m["nmi"] > best_nmi:
                        best_nmi = m["nmi"]
                        best_acc = m["acc"]
                        best_sigma = sigma
                if best_nmi < 0:
                    continue
                rows.append({
                    "dataset": name, "algo": algo, "preprocess": mode,
                    "n_features": info["n_features"],
                    "orig_nmi": best_nmi, "orig_acc": best_acc,
                    "best_sigma": best_sigma,
                })
                print(f"  {name}/{algo}/{mode}: NMI={best_nmi:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "original_nmi_preview.csv", index=False)
    return df
