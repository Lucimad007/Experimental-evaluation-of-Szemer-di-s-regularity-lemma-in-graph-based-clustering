# Spec: Similarity matrix (§4)

Sync status: implemented in `src/enhanced/similarity.py`

## Gaussian similarity (paper formula)

For SPC, APC and DSet the pairwise similarity is computed with

```
s(x, y) = exp( − d(x, y) / (d̄ · σ) )
```

where:
- `d(x, y)` is the **Euclidean distance** between points `x` and `y` (note: linear
  distance, not squared, per the paper),
- `d̄` is the **average of all pairwise distances** over the dataset,
- `σ` is selected from `{0.1, 0.2, 0.5, 1, 2, 5, 10}`.

The diagonal is set to 0 (no self-similarity), matching the reference implementation.

## Metric axis (paper-silent extension)

`src/enhanced/similarity.py` also accepts `metric ∈ {euclidean, cosine,
correlation}` (default `euclidean` = the paper cell). Cosine and correlation
change only the distance `d(x, y)` that feeds the same Gaussian kernel; the
regularity partitioning, reduced graph and Algorithm 1 are untouched. The axis
is exposed to `_best_over_enhanced_grid(..., graph_metrics=(...))` and to the
`_probe_*` / `_refine_winners` scripts, and is documented as a graph-construction
ablation, not as part of the paper protocol.

## SPRG

SPRG learns its own similarity matrix and does **not** use `σ`. See
`base_algorithms.md` §2.1.

## Selection of σ

The paper's §4.1 aggregation protocol ("the mean result of all the
combinations of two parameters") fixes the partitioning parameters under study
but does not state how σ is handled in that mean. Our implementation (documented
interpretation) averages over the remaining parameters *including σ* — see
`experiments.py::_summarize_influence`. For the comparison experiments (§4.2,
§4.3) the original algorithm uses its best σ by NMI over the grid, and the
enhanced algorithm is evaluated over the σ × recommended-(ε, ϵ, b) grid with the
best NMI kept.
