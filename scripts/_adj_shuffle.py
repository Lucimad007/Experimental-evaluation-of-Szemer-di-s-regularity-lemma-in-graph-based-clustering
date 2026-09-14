import importlib.util
from pathlib import Path

import numpy as np

from src.datasets import load_dataset

_spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("_adj_threshold_probe.py")
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

print("shuffle |dNMI|  SPC, same inner grid, degree_mode=alon")
print(f"{'dataset':<10} {'tau':>8} {'file':>7} {'shuf':>7} {'abs_d':>7}")
rng = np.random.default_rng(7)
for name in ("Wine", "Seeds", "Ecoli"):
    X, y = load_dataset(name)
    k = int(len(np.unique(y)))
    perm = rng.permutation(len(y))
    Xp, yp = X[perm], y[perm]
    for tau in probe.TAUS:
        a = probe._enh_best("SPC", X, y, k, tau)
        b = probe._enh_best("SPC", Xp, yp, k, tau)
        if a is None or b is None:
            print(f"{name:<10} {str(tau):>8} FAIL")
            continue
        d = abs(a["nmi"] - b["nmi"])
        print(f"{name:<10} {str(tau):>8} {a['nmi']:7.3f} {b['nmi']:7.3f} {d:7.3f}")

print("shuffle |dNMI|  SPC, same inner grid, degree_mode=alon")
print(f"{'dataset':<10} {'tau':>8} {'file':>7} {'shuf':>7} {'abs_d':>7}")
rng = np.random.default_rng(7)
for name in ("Wine", "Seeds", "Ecoli"):
    X, y = load_dataset(name)
    k = int(len(np.unique(y)))
    perm = rng.permutation(len(y))
    Xp, yp = X[perm], y[perm]
    for tau in TAUS:
        a = _enh_best("SPC", X, y, k, tau)
        b = _enh_best("SPC", Xp, yp, k, tau)
        if a is None or b is None:
            print(f"{name:<10} {str(tau):>8} FAIL")
            continue
        d = abs(a["nmi"] - b["nmi"])
        print(f"{name:<10} {str(tau):>8} {a['nmi']:7.3f} {b['nmi']:7.3f} {d:7.3f}")
