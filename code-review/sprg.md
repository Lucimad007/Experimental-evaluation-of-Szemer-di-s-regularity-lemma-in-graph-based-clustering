# SPRG (Zhu–Loy–Gong CVPR 2014 = Hou et al. ref [20])

## What the paper says

Hou et al. call SPRG “a variant of SPC” that “learns the similarity by combining
subtle similarity in discriminative feature subspaces” and does not use
\(\sigma\). The equations are **not** in the 2026 paper; they are in [20].

## What [20] specifies

1. Clustering forest (`Tclust` trees, `mtry=\sqrt{d}`, stop at \(\le\varphi\)
   real samples).
2. Structure-aware affinity (Eq. 7) with three node-weightings:
   - `bi` — ClustRF-Bi (Eqs. 9–10): 1 iff same leaf.
   - `unfm` — ClustRF-Strct-Unfm (Eq. 11): uniform node weights.
   - `adpt` — ClustRF-Strct-Adpt (Eqs. 12–13): \(w_\kappa=1/|S_\kappa|\). Best
     in [20]; this is our paper-faithful default.
3. Forest consensus \(A=(1/T_{\mathrm{clust}})\sum_t A^t\) (Eq. 8).
4. Spectral clustering on \(A\) with ground-truth \(k\). [20] uses
   unnormalized SPC; we also search NJW and Shi–Malik as the last step.

## What the code does

- Forest: `src/clustering/sprg.py` (`forest_affinity`, `_accumulate_tree_affinity`).
- Original SPRG: `sprg(X, n_clusters, variant=forest, spc_variant=...)`.
- Reg-SPRG (Algorithm 1 line 19): `sprg_on_graph(R, ...)` grows the forest on
  **rows of \(R\)** (\(\varphi=1\) because \(|R|\) is small).

`clustering_variant` strings are `forest+spectral`, e.g. `adpt+unnormalized`.

| Forest | Spectral | Count |
|--------|----------|-------|
| `adpt`, `unfm`, `bi` | `unnormalized`, `njw`, `shi_malik` | **9** |

Paper cell: `adpt+unnormalized`.

## Paper-silent (not searched)

`Tclust=1000`, `mtry=\sqrt{d}`, \(\varphi=5\) on the original graph, bootstrap
per tree, one fixed seed (the paper of [20] averages 5 trials). See
[paper-silent.md](paper-silent.md).
