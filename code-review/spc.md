# SPC (§2.1 of Hou et al., refs [22] and [23])

## What the paper says

Verbatim: spectral clustering “builds the unnormalized Laplacian \(L\) and
computes the first \(k\) eigenvectors \(u_1,\ldots,u_k\) corresponding to the
\(k\) smallest eigenvalues of \(L\). Given the matrix \(U\in\mathbb{R}^{n\times k}\)
with \(u_1,\ldots,u_k\) as columns, we use each row of \(U\) as a data point and
do clustering with standard methods like k-means.”

It cites Shi–Malik [22] and Ng–Jordan–Weiss [23] as the two well-known
normalized recipes, but the experimental path in this paper is the
unnormalized recipe above.

## What the code does

Implementation: `src/clustering/spectral.py` → `spc(..., variant=...)`.

| `variant` | Laplacian / eigenproblem | Embedding | Paper status |
|-----------|-------------------------|-----------|--------------|
| `unnormalized` | \(L = D-S\), standard eigendecomposition | rows of \(U\), **no** row-norm | **§2.1 default** |
| `shi_malik` | generalized \(Lu=\lambda Du\) (NCut) | rows of \(U\), no extra row-norm | cited [22], not the §2.1 recipe |
| `njw` | \(L_{\mathrm{sym}}=I-D^{-1/2}SD^{-1/2}\) | row-normalized \(U\) | cited [23] |

k-means on the rows of \(U\) is sklearn `KMeans` with `n_init=10`,
`random_state=314`. Isolated vertices get a tiny degree floor \(10^{-12}\) so
the normalized variants stay defined.

## Experiments

`--clustering-variants all` (default) records **three Exp 2/3 rows per dataset
and partition setting**, one per SPC variant.

`--clustering-variants paper` keeps only `unnormalized`.
