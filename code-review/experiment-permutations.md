# Experiment permutation grid

`--clustering-variants all` (the default) is a Cartesian product over every
implemented clustering variant and every partition / reduced-graph axis.
`--profile paper` pins the Tables 2–5 path instead.

## Axes

| Axis | Values | Where |
|------|--------|--------|
| Base algorithm | SPC, APC, DSet, SPRG | `--algorithms` |
| Clustering variant | see counts below | `--clustering-variants all` (default) |
| Stop rule | `algorithm1`, `theoretical` | `--stop-rule both` (default) |
| Degree mode | `alon`, `support`, `weighted`, `spectral` | `--degree-mode all` (default) |
| Regularity kind | `alon`, `frieze_kannan` | `--alg-kind both` (default) |
| Partition init | `degree`, `random` | `--init both` (default) |
| Reduced-graph pairs | `all_pairs`, `regular_only` | `--drop-irregular both` (default) |
| \(d_0\) | numeric grid + `mean`/`median`/`p90`/`p95` | `--d0 grid` (default) |
| Alon adj \(\tau\) | `0`, `p25`, `median`, `mean`, `p75` | `--adj-threshold grid` (default) |
| \(\sigma\) | paper §4.1 grid (not SPRG) | always searched when the algo uses it |
| \(\varepsilon,\epsilon,b\) | full grid (Exp 1) or recommended (Exp 2/3) | paper §4.1 |

`--profile paper` pins Alon + degree init + all-pairs \(R\) + support + line-12
stop (Tables 2–5 path). It does **not** pin `--clustering-variants paper`.

## Clustering-variant counts (`all`)

| Algorithm | Variants | Strings |
|-----------|----------|---------|
| SPC | 3 | `unnormalized`, `njw`, `shi_malik` |
| SPRG | 9 | `{adpt,unfm,bi}+{unnormalized,njw,shi_malik}` |
| APC | 2 | `median`, `min` |
| DSet | 5 | `fiorucci`, `hou2023`, `dslib`, `rel95`, `fiorucci05` |
| **Total** | **19** | |

`--clustering-variants paper` is 4 cells (one per algorithm).

## What each experiment writes

**Exp 1.** One CSV row per
`(dataset, algo, clustering_variant, σ, ε, ϵ, b, stop, d0, degree, alg_kind, init, drop, adj_threshold)`.
Plus `comparison_{axis}.csv` / `.png` for every axis (mean/max NMI). Resume
keys include the new axes, so an old paper-default checkpoint does **not**
skip Frieze–Kannan / random init / `rel95` / …

**Exp 2.** One row per
`(dataset, algo, clustering_variant, stop, degree, alg_kind, init, drop, adj_threshold)`.
Original NMI is the best-\(\sigma\) original **for that variant**. Enhanced
NMI is the best recommended-grid run **for that cell** (\(d_0\) searched
inside). `comparison_leaderboard.csv` ranks clustering variants.
`best_per_dataset.csv` is the single highest-NMI row on that dataset;
`best_per_dataset_algo.csv` is the winner per algorithm.

**Exp 2b.** Same product as Exp 2, regularity vs k-means partition, \(d_0=0\).

**Exp 3.** Tables 2–5 stay paper-shaped (`Reg-SPC`, …): each column is the
**best-NMI clustering variant** on that dataset (partition axes searched
inside). The un-collapsed clustering-variant search is
`results/.../exp3_vs_recent/reg_variants.csv`.

**Invariance.** Replays each Exp 2 row with that row's clustering and
partition settings.

## CLI

```powershell
# every clustering variant × both stop rules × both degree modes × Alon/FK × …
python -m src.main exp2 --datasets Wine --out results/all_variants

# paper §2 cells only (unnormalized SPC, adpt+unnormalized SPRG, …)
python -m src.main exp2 --clustering-variants paper --datasets Wine
```
