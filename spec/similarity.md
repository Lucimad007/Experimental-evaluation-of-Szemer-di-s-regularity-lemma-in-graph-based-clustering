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

## SPRG

SPRG learns its own similarity matrix and does **not** use `σ`. See
`base_algorithms.md` §2.1.

## Selection of σ

In the parameter-influence experiments (§4.1) the best `σ` per dataset/algorithm is
selected by the clustering quality (NMI) over the grid. For the comparison
experiments (§4.2, §4.3) the best `σ` from the parameter study is used.
