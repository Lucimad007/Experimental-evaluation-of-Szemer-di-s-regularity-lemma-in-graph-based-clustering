"""Raw vs z-score feature views.

The paper never names a preprocessor. ``raw`` is the paper cell (UCI values as
loaded). ``zscore`` is a documented ablation, not a silent default:

1. drop zero-variance columns (they add nothing to Euclidean distance and
   make a z-score undefined) — Segment col 2 is REGION-PIXEL-COUNT ≡ 9;
2. drop exact duplicate columns (keep the first) — Dutchnumeral fac view
   has three exact copies;
3. column z-score, population std (ddof=0).

No new features. Rare-but-real indicators (Ecoli lip/chg) stay. Duplicate
*rows* stay so NP matches Table 1.
"""

import numpy as np

from .datasets import load_dataset

PREPROCESS_MODES = ("raw", "zscore")


def parse_preprocess_modes(spec):
    """``raw`` / ``zscore`` / ``both``. default is the paper cell."""
    if spec is None:
        return ("raw",)
    s = str(spec).strip().lower()
    if not s or s in ("raw", "none", "paper"):
        return ("raw",)
    if s in ("zscore", "z-score", "standardize", "standard"):
        return ("zscore",)
    if s in ("both", "all", "compare"):
        return PREPROCESS_MODES
    parts = tuple(p.strip().lower() for p in s.split(",") if p.strip())
    out = []
    for p in parts:
        got = parse_preprocess_modes(p)
        for m in got:
            if m not in out:
                out.append(m)
    if not out:
        raise ValueError(f"unknown preprocess spec: {spec!r}")
    return tuple(out)


def _zero_variance_cols(X):
    std = np.nanstd(X, axis=0, ddof=0)
    return np.where(std == 0)[0]


def _duplicate_cols(X):
    """column indices that are exact copies of an earlier column."""
    drop = []
    seen = {}
    for j in range(X.shape[1]):
        key = np.ascontiguousarray(X[:, j]).tobytes()
        if key in seen:
            drop.append(j)
        else:
            seen[key] = j
    return np.asarray(drop, dtype=int)


def apply_preprocess(X, mode="raw"):
    """return ``(X_out, info)``. ``info['dropped']`` is ``[(reason, col), ...]``."""
    mode = str(mode).strip().lower()
    if mode in ("z-score", "standardize", "standard"):
        mode = "zscore"
    if mode in ("none", "paper"):
        mode = "raw"
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-d, got shape {X.shape}")
    if mode == "raw":
        return X.copy(), {
            "mode": "raw",
            "dropped": [],
            "n_dropped": 0,
            "n_features": int(X.shape[1]),
        }
    if mode != "zscore":
        raise ValueError(f"unknown preprocess mode: {mode!r}")

    keep = np.ones(X.shape[1], dtype=bool)
    dropped = []
    for j in _zero_variance_cols(X):
        keep[int(j)] = False
        dropped.append(("zero_variance", int(j)))
    Xk = X[:, keep]
    dup_local = _duplicate_cols(Xk)
    if dup_local.size:
        orig = np.where(keep)[0][dup_local]
        for j in orig:
            keep[int(j)] = False
            dropped.append(("duplicate", int(j)))
    Xp = X[:, keep]
    mu = Xp.mean(axis=0)
    sd = Xp.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (Xp - mu) / sd
    return Z, {
        "mode": "zscore",
        "dropped": dropped,
        "n_dropped": len(dropped),
        "n_features": int(Z.shape[1]),
    }


def prepare_dataset(name, mode="raw"):
    """load Table 1 dataset ``name`` and apply ``mode``. returns ``(X, y, info)``."""
    X, y = load_dataset(name)
    Xp, info = apply_preprocess(X, mode)
    info["dataset"] = name
    info["n"] = int(X.shape[0])
    info["n_features_raw"] = int(X.shape[1])
    return Xp, y, info
