# Spec: Base clustering algorithms (§2)

Sync status: implemented in `src/clustering/`

Four representative graph-based clustering algorithms are used in the experiments.
Each takes a pairwise similarity matrix as input.

## 2.1 Spectral clustering (SPC) and SPRG

**SPC.** Given the similarity matrix `S ∈ R^{n×n}`, build the **unnormalized**
Laplacian `L = D − S` and compute the first `k` eigenvectors `u_1, …, u_k`
corresponding to the `k` smallest eigenvalues. Form `U ∈ R^{n×k}` with those
eigenvectors as columns, treat each row of `U` as a point (no row
normalization), and cluster with k-means. This is the paper's verbatim §2.1
description; the normalized variants it mentions (Shi–Malik [22], Ng–Jordan–
Weiss [23]) are available as `variant="njw"` in `src/clustering/spectral.py`.
`k` is set to the ground-truth number of clusters.

**SPRG** (Zhu, Loy & Gong, *Constructing robust affinity graphs for spectral
clustering*, CVPR 2014 — reference **[20]** of the paper; also cited as "SPRG"
by Hou et al. PR 2023, ref [25]). A spectral-clustering variant that *learns*
the similarity matrix — "combining subtle similarity in discriminative feature
subspaces" — instead of using the Gaussian similarity. It does **not** use the
`σ` parameter and takes the ground-truth `k`. Method (equations from
Zhu–Loy–Gong):

1. Train a **clustering random forest** of `Tclust = 1000` trees, each on a
   random subset of `X`, unsupervised via the pseudo two-class algorithm
   (synthetic uniform samples labelled class 1 vs real samples class 0), with
   Gini information-gain splits (Eq. 1–3) over `mtry = √d` candidate features
   and mid-point thresholds; stop at ≤ `φ` real samples per node.
2. Compute **structure-aware affinities** (Eq. 7): `a^t_ij` = (sum of shared
   path-node weights) / (sum of longer-path node weights), with variants Bi
   (Eq. 9–10), Unfm (Eq. 11, weights 1) and Adpt (Eq. 12–14, `w_κ = 1/|S_κ|`,
   `1/|Λ_b̂|` at the leaf — default).
3. Average over trees (Eq. 8) → the learned affinity `A`; cluster with SPC
   (unnormalized) on `A` with the ground-truth `k`.

Implementation: `src/clustering/sprg.py` (`forest_affinity`, `sprg`,
`sprg_on_graph`). Paper-silent choices (`src/config.py`): `φ = 5`, variant
`adpt`, bootstrap subsets, seed 314; `config.SPRG_TREES` can be reduced for
fast runs. For Reg-SPRG on the reduced graph R (no features), each R-vertex is
represented by its row of R and the forest is grown on those profiles
(`sprg_on_graph`, `φ = 1`).

## 2.2 Affinity propagation clustering (APC)

Takes the pairwise similarity matrix as input and exchanges messages between data
points to gradually identify exemplars (cluster centers) and members. All points
are initially potential exemplars; messages (responsibility and availability) are
updated by minimizing an energy function. The number of clusters is determined
automatically (no `k` needed). We use the standard Frey–Dueck algorithm
(`sklearn.cluster.AffinityPropagation`). The shared preference is the **median**
of the off-diagonal input similarities ([10]: "median of the input similarities").
Stop when exemplar decisions are unchanged for **10** iterations ([10] p.973).

## 2.3 Dominant set clustering (DSet)

Extracts clusters **sequentially**. A dominant set is a maximal subset with internal
coherency — the edge-weighted analogue of a clique. Given the `n×n` similarity
matrix `A`, the weight of point `p_i` is updated by the replicator dynamics (Eq. 1):

```
x_i^{(t+1)} = x_i^{(t)} * (A x^{(t)})_i / ( x^{(t)T} A x^{(t)} )
```

with `x_i^{(0)} = 1/n`. After convergence, points whose weight exceeds a threshold
form a dominant set (one cluster); the cluster is removed and the process repeats on
the remaining points **until all clusters are obtained**. The number of clusters is
determined automatically. We follow the replicator of Pavan & Pelillo [9]
(Eq. 1), not the faster dynamics of Bulo et al. [26]. Hou §2.3 does not name the
numeric threshold; the paper-faithful cell is `0.0001` from Hou et al. PR 2023
[25] (same first author, the same sentence). Vascon et al. [27] (the library the
paper cites) use `1e-5`; `1/(1.5n)` is a Fiorucci-lineage extra. Leftover dump of
5% of points is the searched extra `fiorucci05` — the paper cell peels until all
clusters are obtained (leftover 0). Relative cutoff `rel95` is also searched.
