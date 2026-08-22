"""UCI dataset loaders for the 20 datasets in Table 1.

Datasets are loaded from the local zip archives in ``data/`` (no network needed).
Each loader returns ``(X, y)`` with ``X ∈ R^{n×d}`` (float64) and integer ``y`` in
``{0, …, NC−1}``. Results are cached as ``.npz`` under ``data/``.

Coverage from the local zips (14 datasets):
    Banknote, Sonar, Ecoli, Glass, Segment, Libras, Raisin, Rice, Seeds,
    Spambase, Spectf, Landsat, Thyroid, Wine

Not in the local folder (fall back to ``ucimlrepo`` if installed, else a synthetic
stand-in so the code still runs):
    Appendicitis, SCC, USPS, Landmine (rar), Leaves (images), Dutchnumeral
    (the provided tunadromd.zip is a different Android-malware dataset, not the
    MPEG-7 Dutch numeral dataset used by the paper).

See ``spec/datasets.md``.
"""

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _zip(name):
    return zipfile.ZipFile(DATA_DIR / name)


def _text(zf, inner):
    return zf.read(inner).decode("utf-8", "replace")


def _cache_path(name):
    return DATA_DIR / f"_cache_{name}.npz"


def _save_cache(name, X, y):
    np.savez_compressed(_cache_path(name), X=X, y=y)


def _load_cache(name):
    p = _cache_path(name)
    if p.exists():
        d = np.load(p)
        return d["X"], d["y"]
    return None


def _labels_to_int(y):
    return pd.factorize(np.asarray(y))[0]


def _arff_data_rows(text):
    """Return the rows after the @data line of an ARFF file as a list of strings."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip().lower().startswith("@data"):
            return lines[i + 1:]
    return []


# ---------------------------------------------------------------------------
# per-dataset parsers (read from local zips)
# ---------------------------------------------------------------------------
def _load_wine():
    df = pd.read_csv(io.StringIO(_text(_zip("wine.zip"), "wine.data")), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()


def _load_seeds():
    df = pd.read_csv(io.StringIO(_text(_zip("seeds.zip"), "seeds_dataset.txt")),
                      sep=r"\s+", header=None)
    return df.iloc[:, :7].to_numpy(dtype=float), df.iloc[:, 7].to_numpy()


def _load_ecoli():
    df = pd.read_csv(io.StringIO(_text(_zip("ecoli.zip"), "ecoli.data")),
                      sep=r"\s+", header=None)
    return df.iloc[:, 1:8].to_numpy(dtype=float), df.iloc[:, 8].to_numpy()


def _load_glass():
    df = pd.read_csv(io.StringIO(_text(_zip("glass+identification.zip"), "glass.data")),
                      header=None)
    return df.iloc[:, 1:10].to_numpy(dtype=float), df.iloc[:, 10].to_numpy()


def _load_sonar():
    text = _text(_zip("connectionist+bench+sonar+mines+vs+rocks.zip"), "sonar.all-data")
    rows = [l for l in text.splitlines() if l and not l.startswith(";;;")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    return df.iloc[:, :60].to_numpy(dtype=float), df.iloc[:, 60].to_numpy()


def _load_banknote():
    df = pd.read_csv(io.StringIO(_text(_zip("banknote+authentication.zip"),
                                   "data_banknote_authentication.txt")), header=None)
    return df.iloc[:, :4].to_numpy(dtype=float), df.iloc[:, 4].to_numpy()


def _load_spambase():
    df = pd.read_csv(io.StringIO(_text(_zip("spambase.zip"), "spambase.data")), header=None)
    return df.iloc[:, :57].to_numpy(dtype=float), df.iloc[:, 57].to_numpy()


def _load_libras():
    df = pd.read_csv(io.StringIO(_text(_zip("libras+movement.zip"), "movement_libras.data")),
                      header=None)
    # class is the LAST column (numeric 1..15); 90 features
    return df.iloc[:, :90].to_numpy(dtype=float), df.iloc[:, 90].to_numpy()


def _load_segment():
    zf = _zip("image+segmentation.zip")
    txt_tr = _text(zf, "segmentation.data")
    txt_te = _text(zf, "segmentation.test")
    rows_tr = [l for l in txt_tr.splitlines()
               if l and not l.startswith(";;;") and not l.startswith("REGION")]
    rows_te = [l for l in txt_te.splitlines()
               if l and not l.startswith(";;;") and not l.startswith("REGION")]
    df = pd.read_csv(io.StringIO("\n".join(rows_tr + rows_te)), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


def _load_spectf():
    zf = _zip("spectf+heart.zip")
    tr = _text(zf, "SPECTF.train")
    te = _text(zf, "SPECTF.test")
    df = pd.read_csv(io.StringIO("\n".join([tr, te])), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


def _load_landsat():
    zf = _zip("statlog+landsat+satellite.zip")
    tr = _text(zf, "sat.trn")
    te = _text(zf, "sat.tst")
    df = pd.read_csv(io.StringIO("\n".join([tr, te])), sep=r"\s+", header=None)
    return df.iloc[:, :36].to_numpy(dtype=float), df.iloc[:, 36].to_numpy()  # class last


def _load_thyroid():
    df = pd.read_csv(io.StringIO(_text(_zip("thyroid+disease.zip"), "new-thyroid.data")),
                      header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


def _load_rice():
    text = _text(_zip("rice+cammeo+and+osmancik.zip"), "Rice_Cammeo_Osmancik.arff")
    rows = [l for l in _arff_data_rows(text) if l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    return df.iloc[:, :7].to_numpy(dtype=float), df.iloc[:, 7].to_numpy()  # class last (string)


def _load_raisin():
    outer = _zip("raisin.zip")
    inner = outer.read("Raisin_Dataset.zip")
    zf2 = zipfile.ZipFile(io.BytesIO(inner))
    text = _text(zf2, "Raisin_Dataset/Raisin_Dataset.arff")
    rows = [l for l in _arff_data_rows(text) if l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    return df.iloc[:, :7].to_numpy(dtype=float), df.iloc[:, 7].to_numpy()  # class last (string)


# --- loaders needing extra extraction (rar / images) -------------------------
def _load_landmine():
    """Landmine (338, 3, 5): extract Mine Dataset.rar with bsdtar, read the
    'Normalized_Data' sheet (V, H, S = features, M = class)."""
    import tempfile
    import subprocess
    import shutil

    import xlrd

    outer = _zip("land+mines-1.zip")
    rar_bytes = outer.read("Mine Dataset.rar")
    tmp = tempfile.mkdtemp()
    try:
        rar_path = os.path.join(tmp, "mine.rar")
        open(rar_path, "wb").write(rar_bytes)
        out = os.path.join(tmp, "out")
        os.makedirs(out, exist_ok=True)
        subprocess.run(["bsdtar", "-xf", rar_path, "-C", out],
                        capture_output=True, text=True)
        xls = os.path.join(out, "Mine Dataset", "Mine_Dataset.xls")
        wb = xlrd.open_workbook(xls)
        sh = wb.sheet_by_name("Normalized_Data")
        data = [sh.row_values(i) for i in range(1, sh.nrows)]  # skip header
        arr = np.array(data, dtype=float)
        return arr[:, :3], arr[:, 3]  # V,H,S features ; M class
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_leaves():
    """Leaves (1600, 64, 100): 100 plant species × 16 images each.

    The zip holds 1600 JPG images; the paper uses 64-dim features extracted from
    them. We extract a 64-dim feature vector per image by resizing to 8×8 grayscale
    and flattening (an approximation of the paper's exact 64-dim features, which
    are not specified in the paper text). Label = species folder name.
    """
    from PIL import Image

    zf = _zip("one+hundred+plant+species+leaves+data+set.zip")
    jpgs = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))
    X = np.zeros((len(jpgs), 64), dtype=float)
    y = []
    for i, name in enumerate(jpgs):
        img = Image.open(io.BytesIO(zf.read(name))).convert("L").resize((8, 8))
        X[i] = np.asarray(img, dtype=float).flatten()
        # species = parent folder, e.g. ".../Acer_Campestre/Acer_Campestre_01.ab.jpg"
        y.append(name.split("/")[-2])
    return X, np.array(y)


# ---------------------------------------------------------------------------
# fallback loaders (not available locally)
# ---------------------------------------------------------------------------
def _load_via_ucimlrepo(uciid, target_col):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uciid)
    X = ds.data.features.to_numpy(dtype=float)
    y = ds.data.targets[target_col].to_numpy()
    return X, y


def _synthetic(n, d, nc, seed=314):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(nc, d))
    X = np.vstack([rng.normal(c, 1.0, size=(n // nc, d)) for c in centers])
    y = np.concatenate([np.full(n // nc, i) for i in range(nc)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


_FALLBACK = {
    # name: (kind, args, synthetic-shape)
    # ``synthetic-shape`` is only used when the primary source is unreachable
    # (e.g. no network for ucimlrepo); see the fallback chain in load_dataset.
    "Appendicitis":  ("ucimlrepo", (544, "Class"), (106, 7, 2)),
    "SCC":           ("ucimlrepo", (840, "Class"), (600, 60, 6)),
    "USPS":          ("synthetic", (11000, 256, 10), None),
    "Dutchnumeral":  ("synthetic", (2000, 649, 10), None),
}

_LOCAL_LOADERS = {
    "Wine": _load_wine, "Seeds": _load_seeds, "Ecoli": _load_ecoli,
    "Glass": _load_glass, "Sonar": _load_sonar, "Banknote": _load_banknote,
    "Spambase": _load_spambase, "Libras": _load_libras, "Segment": _load_segment,
    "Spectf": _load_spectf, "Landsat": _load_landsat, "Thyroid": _load_thyroid,
    "Rice": _load_rice, "Raisin": _load_raisin,
    "Landmine": _load_landmine, "Leaves": _load_leaves,
}


def load_dataset(name):
    """Load a dataset by name; returns ``(X, y)`` with integer labels.

    Uses the local zip in ``data/`` when available; otherwise falls back to
    ``ucimlrepo`` (if installed) or a synthetic stand-in. Results are cached.
    """
    cached = _load_cache(name)
    if cached is not None:
        return cached

    if name in _LOCAL_LOADERS:
        X, y = _LOCAL_LOADERS[name]()
    elif name in _FALLBACK:
        kind, args, syn_shape = _FALLBACK[name]
        if kind == "ucimlrepo":
            try:
                X, y = _load_via_ucimlrepo(*args)
            except Exception:
                if syn_shape is None:
                    raise
                X, y = _synthetic(*syn_shape)
        else:
            X, y = _synthetic(*args)
    else:
        raise ValueError(f"unknown dataset: {name}")

    y = _labels_to_int(y)
    X = np.asarray(X, dtype=float)
    _save_cache(name, X, y)
    return X, y


def list_datasets():
    """Return the list of all 20 dataset names (Table 1)."""
    return list(_LOCAL_LOADERS.keys()) + list(_FALLBACK.keys())
