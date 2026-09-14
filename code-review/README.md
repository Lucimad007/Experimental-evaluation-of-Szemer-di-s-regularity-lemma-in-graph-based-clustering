# Code review: clustering functions vs Hou et al. (PR 171, 2026)

This folder is the audit of every **clustering function** used in the
experiments — what the paper (and its cited references) specify, what this
repo implements, and which variants are searched.

The experiment runners take the **Cartesian product** of every implemented
variant with the partition axes (stop rule, degree mode, \(d_0\), Alon vs
Frieze–Kannan, init, drop-irregular). See
[experiment-permutations.md](experiment-permutations.md).

| File | Covers |
|------|--------|
| [spc.md](spc.md) | SPC §2.1: unnormalized Laplacian, Shi–Malik [22], Ng–Jordan–Weiss [23] |
| [sprg.md](sprg.md) | SPRG [20]: forest weighting `adpt`/`unfm`/`bi` × SPC embedding |
| [apc.md](apc.md) | APC §2.2 vs Frey–Dueck [10] (sklearn message passing) |
| [dset.md](dset.md) | DSet §2.3 Eq. 1, leftover peel, threshold lineages |
| [apc-dset.md](apc-dset.md) | APC preference (median / min) and DSet weight cutoffs |
| [experiment-permutations.md](experiment-permutations.md) | Exact product the CLI searches; `--clustering-variants all\|paper` |
| [paper-silent.md](paper-silent.md) | Constants that are **not** variant axes (φ, APC iters, kNN, …) |

## Paper-faithful cell

Hou et al. §2 names four algorithms. The single cell that matches the paper's
prose is:

| Algorithm | `clustering_variant` | Meaning |
|-----------|----------------------|---------|
| SPC | `unnormalized` | \(L = D-S\), \(k\) smallest eigenvectors, k-means on rows of \(U\) (no row-norm) |
| SPRG | `adpt+unnormalized` | ClustRF-Strct-Adpt [20] + unnormalized SPC |
| APC | `median` | Frey–Dueck shared preference = median of similarities [10] |
| DSet | `hou2023` | replicator cutoff \(0.0001\) ([25]) |

Force only that cell with `--clustering-variants paper`. The default is
`--clustering-variants all`.
