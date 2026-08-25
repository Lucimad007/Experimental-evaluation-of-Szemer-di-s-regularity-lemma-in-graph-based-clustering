# Spec: Building the reduced graph (§3.3)

Sync status: implemented in `src/szemeredi/regularity_lemma.py` (`generate_reduced_sim_mat`)

## Weighted edge density (Eq. 3)

For an edge-weighted original graph (as in graph-based clustering, where edge weights
are pairwise similarities), Eq. 2 is extended to:

```
dw(X, Y) = ( Σ_{i=1..|X|} Σ_{j=1..|Y|} w(x_i, y_j) ) / (|X| |Y|)
```

where `x_i ∈ X`, `y_j ∈ Y`, and `w(x_i, y_j)` is the weight between vertices `x_i`
and `y_j`.

## Reduced graph construction

Given the regularity partition `V = V0 ∪ V1 ∪ … ∪ Vk`:

- The reduced graph `R` has `k` vertices, one per class `V1, …, Vk`.
- Two vertices are adjacent if the corresponding classes form an `ε`-regular pair
  with edge density above a threshold `d0`.
- The edge weight between the vertices for `Vr` and `Vs` is `dw(Vr, Vs)` from Eq. 3.
- The exceptional class `V0` is **not** included in `R` (it forms no regular pairs).

## Key Lemma (Lemma 2, Komlós et al.)

`R` is a reduced graph of `G`; a small subgraph of `R(t)` (the `t`-blow-up of `R`) is
also a subgraph of `G`. Since `R(1) = R`, a small subgraph of `R` is also in `G`. So
`R` is a compressed, compact, sampled (by *structures*/edges rather than vertices)
version of `G` that preserves essential structure. This is the justification for
doing clustering on `R` instead of `G`.

## Implementation notes

- `bip_density` for a weighted pair = `bip_sim_mat.sum() / n²` where `n` is the
  common class cardinality (matches Eq. 3 with `|X| = |Y| = n`).
- `drop_edges_between_irregular_pairs` controls whether the reduced matrix is fully
  connected (all pairs) or only regular pairs. The paper builds `R` from regular
  pairs with density above `d0` (Lemma 2: `d0 > ε`; the numeric value is
  unspecified). `density_threshold` zeros Eq. 3 weights at or below a float or an
  adaptive name (`p90`, `p95`, `mean`, `median`). Default `d0 = 0` keeps every
  pair density; Exp 2/3 search `d0` for APC/DSet.
