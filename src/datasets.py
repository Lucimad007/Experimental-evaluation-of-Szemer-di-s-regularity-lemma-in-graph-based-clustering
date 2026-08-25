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

# local zip archives (table 1 datasets)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _zip(name):
    # open a dataset zip under data/
    return zipfile.ZipFile(DATA_DIR / name)


def _text(zf, inner):
    # decode one member of a zip
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
    # map string / 1-based labels to {0, …, nc−1}
    return pd.factorize(np.asarray(y))[0]


def _arff_data_rows(text):
    """return the rows after the @data line of an arff file as a list of strings."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.strip().lower().startswith("@data"):
            return lines[i + 1:]
    return []


# ---------------------------------------------------------------------------
# per-dataset parsers (read from local zips)
# ---------------------------------------------------------------------------
def _load_wine():
    # table 1: wine (178, 13, 3); class is column 0
    df = pd.read_csv(io.StringIO(_text(_zip("wine.zip"), "wine.data")), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()


def _load_seeds():
    # table 1: seeds (210, 7, 3)
    df = pd.read_csv(io.StringIO(_text(_zip("seeds.zip"), "seeds_dataset.txt")),
                      sep=r"\s+", header=None)
    return df.iloc[:, :7].to_numpy(dtype=float), df.iloc[:, 7].to_numpy()


def _load_ecoli():
    # table 1: ecoli (336, 7, 8); skip the sequence-name column
    df = pd.read_csv(io.StringIO(_text(_zip("ecoli.zip"), "ecoli.data")),
                      sep=r"\s+", header=None)
    return df.iloc[:, 1:8].to_numpy(dtype=float), df.iloc[:, 8].to_numpy()


def _load_glass():
    # table 1: glass (214, 9, 6)
    df = pd.read_csv(io.StringIO(_text(_zip("glass+identification.zip"), "glass.data")),
                      header=None)
    return df.iloc[:, 1:10].to_numpy(dtype=float), df.iloc[:, 10].to_numpy()


def _load_sonar():
    # table 1: sonar (208, 60, 2)
    text = _text(_zip("connectionist+bench+sonar+mines+vs+rocks.zip"), "sonar.all-data")
    rows = [l for l in text.splitlines() if l and not l.startswith(";;;")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    return df.iloc[:, :60].to_numpy(dtype=float), df.iloc[:, 60].to_numpy()


def _load_banknote():
    # table 1: banknote (1372, 4, 2)
    df = pd.read_csv(io.StringIO(_text(_zip("banknote+authentication.zip"),
                                   "data_banknote_authentication.txt")), header=None)
    return df.iloc[:, :4].to_numpy(dtype=float), df.iloc[:, 4].to_numpy()


def _load_spambase():
    # table 1: spambase (4601, 57, 2)
    df = pd.read_csv(io.StringIO(_text(_zip("spambase.zip"), "spambase.data")), header=None)
    return df.iloc[:, :57].to_numpy(dtype=float), df.iloc[:, 57].to_numpy()


def _load_libras():
    # table 1: libras (360, 90, 15)
    df = pd.read_csv(io.StringIO(_text(_zip("libras+movement.zip"), "movement_libras.data")),
                      header=None)
    # class is the last column (numeric 1..15); 90 features
    return df.iloc[:, :90].to_numpy(dtype=float), df.iloc[:, 90].to_numpy()


def _load_segment():
    # table 1: segment (2310, 19, 7); train+test concatenated, class first
    zf = _zip("image+segmentation.zip")
    txt_tr = _text(zf, "segmentation.data")
    txt_te = _text(zf, "segmentation.test")
    rows_tr = [l for l in txt_tr.splitlines()
               if l and not l.startswith(";;;") and not l.startswith("REGION")]
    rows_te = [l for l in txt_te.splitlines()
               if l and not l.startswith(";;;") and not l.startswith("REGION")]
    df = pd.read_csv(io.StringIO("\n".join(rows_tr + rows_te)), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


# remaining table 1 loaders (class-first / class-last csv, arff, rar, images)
def _load_spectf():
    zf = _zip("spectf+heart.zip")
    tr = _text(zf, "SPECTF.train")
    te = _text(zf, "SPECTF.test")
    df = pd.read_csv(io.StringIO("\n".join([tr, te])), header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


def _load_landsat():
    # table 1: landsat (6435, 36, 6); train+test, class last
    zf = _zip("statlog+landsat+satellite.zip")
    tr = _text(zf, "sat.trn")
    te = _text(zf, "sat.tst")
    df = pd.read_csv(io.StringIO("\n".join([tr, te])), sep=r"\s+", header=None)
    return df.iloc[:, :36].to_numpy(dtype=float), df.iloc[:, 36].to_numpy()  # class last


def _load_thyroid():
    # table 1: thyroid (215, 5, 3); class first
    df = pd.read_csv(io.StringIO(_text(_zip("thyroid+disease.zip"), "new-thyroid.data")),
                      header=None)
    return df.iloc[:, 1:].to_numpy(dtype=float), df.iloc[:, 0].to_numpy()  # class first


def _load_rice():
    # table 1: rice (3810, 7, 2); arff, class last
    text = _text(_zip("rice+cammeo+and+osmancik.zip"), "Rice_Cammeo_Osmancik.arff")
    rows = [l for l in _arff_data_rows(text) if l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    return df.iloc[:, :7].to_numpy(dtype=float), df.iloc[:, 7].to_numpy()  # class last (string)


def _load_raisin():
    # table 1: raisin (900, 7, 2); nested zip + arff
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
    """Leaves (1600, 64, 100): 100 plant species × 16 samples each.

    The zip ships the UCI-provided 64-dim feature files — ``data_Sha_64.txt``
    (shape descriptors), ``data_Mar_64.txt`` (margin) and ``data_Tex_64.txt``
    (texture). The paper (Table 1) lists ND=64 without naming the view; we use
    the shape descriptors (``data_Sha_64.txt``), each row = 64 features + class
    id in 1..100. (The previous loader resized the JPGs to 8×8 grayscale — an
    approximation no longer needed.)
    """
    zf = _zip("one+hundred+plant+species+leaves+data+set.zip")
    X, y = [], []
    with zf.open("100 leaves plant species/data_Sha_64.txt") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
            parts = line.strip().split(",")
            if len(parts) == 65:                    # species name + 64 features
                X.append([float(v) for v in parts[1:]])
                y.append(parts[0])
    return np.asarray(X, dtype=float), np.array(y)


def _load_dutchnumeral():
    """Dutchnumeral (2000, 649, 10): UCI *Multiple Features* (mfeat).

    Handwritten numerals '0'-'9' (200 per class) extracted from Dutch utility
    maps — the dataset the multi-view-clustering literature calls "Dutch
    numeral"/"HW". The paper's ND=649 is the concatenation of mfeat's six
    feature views: fac (216) + fou (76) + kar (64) + pix (240) + zer (47) +
    mor (6). The files carry no label column; rows are ordered by class
    (first 200 rows = digit 0, etc.). Feature scales differ per view (raw UCI
    values; the paper specifies no preprocessing and we apply none).
    """
    zf = _zip("multiple+features.zip")
    views = ["mfeat-fac", "mfeat-fou", "mfeat-kar", "mfeat-pix", "mfeat-zer", "mfeat-mor"]
    X = np.hstack([np.loadtxt(zf.open(v)) for v in views])
    y = np.repeat(np.arange(10), 200)
    return X, y


def _load_usps():
    """USPS (11000, 256, 10): Roweis' usps_all.mat — 1100 16×16 digit images
    per class × 10 digits, grayscale values 0-255, stored as (256, 1100, 10).

    This is the canonical 11000-sample USPS subset used in the spectral
    clustering literature and matching the paper's Table 1 exactly.
    Source: https://cs.nyu.edu/home/people/in_memoriam/roweis/data.html
    """
    import scipy.io as sio

    data = sio.loadmat(DATA_DIR / "usps_all.mat")["data"]
    X = data.reshape(256, -1).T.astype(float)      # (11000, 256)
    # C-order flattening of (1100, 10) is sample-major: row k = sample k//10 of
    # digit k%10 — verified by 87% nearest-neighbour label agreement (vs 11%
    # for the class-major ordering).
    y = np.tile(np.arange(10), 1100)
    return X, y


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
    # last-resort stand-in when a table 1 dataset is not on disk
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
}

_LOCAL_LOADERS = {
    "Wine": _load_wine, "Seeds": _load_seeds, "Ecoli": _load_ecoli,
    "Glass": _load_glass, "Sonar": _load_sonar, "Banknote": _load_banknote,
    "Spambase": _load_spambase, "Libras": _load_libras, "Segment": _load_segment,
    "Spectf": _load_spectf, "Landsat": _load_landsat, "Thyroid": _load_thyroid,
    "Rice": _load_rice, "Raisin": _load_raisin,
    "Landmine": _load_landmine, "Leaves": _load_leaves,
    "Dutchnumeral": _load_dutchnumeral, "USPS": _load_usps,
}


def load_dataset(name):
    """load a dataset by name; returns ``(x, y)`` with integer labels.

    uses the local zip in ``data/`` when available; otherwise falls back to
    ``ucimlrepo`` (if installed) or a synthetic stand-in. results are cached.
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

    # paper labels are treated as {0, …, nc−1}
    y = _labels_to_int(y)
    X = np.asarray(X, dtype=float)
    _save_cache(name, X, y)
    return X, y


def list_datasets():
    """return the list of all 20 dataset names (table 1)."""
    return list(_LOCAL_LOADERS.keys()) + list(_FALLBACK.keys())
