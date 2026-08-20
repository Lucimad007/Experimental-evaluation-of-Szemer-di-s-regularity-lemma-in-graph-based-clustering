# Spec: Base clustering algorithms (§2)

Sync status: implemented in `src/clustering/`

Four representative graph-based clustering algorithms are used in the experiments.
Each takes a pairwise similarity matrix as input.

## 2.1 Spectral clustering (SPC) and SPRG

**SPC.** Given the similarity matrix `S ∈ R^{n×n}`, build the (normalized) Laplacian
and compute the first `k` eigenvectors `u_1, …, u_k` corresponding to the `k` smallest
eigenvalues. Form `U ∈ R^{n×k}` with those eigenvectors as columns, treat each row of
`U` as a point, and cluster with k-means. We use the Ng–Jordan–Weiss normalized
Laplacian variant (`L_sym = I − D^{−1/2} S D^{−1/2}`), rows of `U` normalized to unit
length, as in the standard formulation. `k` is set to the ground-truth number of
clusters (consistent with the experimental protocol).

**SPRG** (Hou et al., *Towards parameter-free clustering for real-world data*, PR
2023). A spectral-clustering variant that *learns* the similarity matrix by combining
subtle similarities in discriminative feature subspaces, instead of using the
Gaussian similarity. It does **not** use the `σ` parameter. The paper notes it
performs much better than NCut but at a much larger computation cost. We implement a
faithful iterative similarity-learning + spectral clustering pipeline (see
`src/clustering/spectral.py`).

## 2.2 Affinity propagation clustering (APC)

Takes the pairwise similarity matrix as input and exchanges messages between data
points to gradually identify exemplars (cluster centers) and members. All points
are initially potential exemplars; messages (responsibility and availability) are
updated by minimizing an energy function. The number of clusters is determined
automatically (no `k` needed). We use the standard Frey–Dueck algorithm
(`sklearn.cluster.AffinityPropagation`), feeding the similarity matrix as the
preference/similarity input with preferences set to the median similarity.

## 2.3 Dominant set clustering (DSet)

Extracts clusters **sequentially**. A dominant set is a maximal subset with internal
coherency — the edge-weighted analogue of a clique. Given the `n×n` similarity
matrix `A`, the weight of point `p_i` is updated by the replicator dynamics (Eq. 1):

```
x_i^{(t+1)} = x_i^{(t)} * (A x^{(t)})_i / ( x^{(t)T} A x^{(t)} )
```

with `x_i^{(0)} = 1/n`. After convergence, points whose weight exceeds a threshold
form a dominant set (one cluster); the cluster is removed and the process repeats on
the remaining points. The number of clusters is determined automatically. We follow
the replicator-dynamics implementation (Pavan & Pelillo 2007; Bulo, Pelillo & Bomze
2011) used in the reference code, with the weight threshold
`1/(n * 1.5)` and a stop when fewer than 5% of points remain unclustered.
