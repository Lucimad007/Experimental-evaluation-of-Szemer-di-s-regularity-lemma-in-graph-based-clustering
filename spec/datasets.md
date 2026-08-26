# Spec: Datasets (§4, Table 1)

Sync status: implemented in `src/datasets.py`

20 real datasets from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/). NP = number of points, ND = data dimension,
NC = number of clusters.

| Dataset       | NP    | ND  | NC  |
|---------------|-------|-----|-----|
| Thyroid       | 215   | 5   | 3   |
| Wine          | 178   | 13  | 3   |
| Glass         | 214   | 9   | 6   |
| Leaves        | 1600  | 64  | 100 |
| Seeds         | 210   | 7   | 3   |
| Segment       | 2310  | 19  | 7   |
| Libras        | 360   | 90  | 15  |
| Ecoli         | 336   | 7   | 8   |
| Appendicitis  | 106   | 7   | 2   |
| SCC           | 600   | 60  | 6   |
| USPS          | 11000 | 256 | 10  |
| Rice          | 3810  | 7   | 2   |
| Raisin        | 900   | 7   | 2   |
| Spambase      | 4601  | 57  | 2   |
| Sonar         | 208   | 60  | 2   |
| Banknote      | 1372  | 4   | 2   |
| Landsat       | 6435  | 36  | 6   |
| Landmine      | 338   | 3   | 5   |
| Dutchnumeral  | 2000  | 649 | 10  |
| Spectf        | 267   | 44  | 2   |

## Loading

- Datasets are fetched from UCI (via `ucimlrepo` where the dataset id is known, or
  from local files in `data/` if present). `data/` is gitignored.
- Each loader returns `(X, y)` with `X ∈ R^{n×d}` (float64) and `y` integer labels in
  `{0, …, NC−1}`.
- A small cache in `data/` avoids re-downloading.
- The large variances in NP (106–11000), ND (3–649) and NC (2–100) are intentional
  and make the experimental conclusions broad, per §4.
- Only **real** datasets are used: §4 notes that regular-partitioning enhancement
  does not work well on synthetic data because regular pairs demand random edge
  distribution found in real data.

### Local-zip coverage (18 of 20)

`src/datasets.py` parses 18 datasets from local files in `data/` (no network):
Banknote, Sonar, Ecoli, Glass, Segment, Libras, Raisin, Rice, Seeds, Spambase,
Spectf, Landsat, Thyroid, Wine, **Landmine**, **Leaves**, **Dutchnumeral**,
**USPS** — handling CSV / whitespace-delimited / ARFF / XLS / image formats,
string-label columns and header rows. Specifically:
- **Landmine** is extracted from `Mine Dataset.rar` (inside the zip) via
  `bsdtar`, reading the `Normalized_Data` sheet (V, H, S features; M class).
- **Leaves** reads the UCI zip's 64-dim shape-descriptor file
  (`100 leaves plant species/data_Sha_64.txt`; species name + 64 features per
  row), matching Table 1's ND=64 without image resizing.
- **Dutchnumeral** concatenates the six UCI *Multiple Features* (mfeat) views —
  fac (216) + fou (76) + kar (64) + pix (240) + zer (47) + mor (6) = 649 dims,
  200 rows per digit 0–9 in file order.
- **USPS** loads Roweis' `usps_all.mat` (11000 × 256, 1100 images per digit).

### Fallback (2)

Appendicitis, SCC via `ucimlrepo` (id 544 / 840) if installed; without network
these degrade to a synthetic stand-in so the code still runs.
