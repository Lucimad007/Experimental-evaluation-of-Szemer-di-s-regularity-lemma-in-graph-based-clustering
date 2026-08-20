# Spec: Evaluation metrics (§4)

Sync status: implemented in `src/metrics.py`

The paper reports NMI as the main criterion and ACC, ARI, RI in the supplementary
material / Tables 2–5. All four are implemented.

## NMI — normalized mutual information

Standard symmetric NMI between ground-truth labels `U` and predicted labels `V`:

```
NMI(U, V) = 2 · I(U; V) / (H(U) + H(V))
```

(`I` = mutual information, `H` = entropy). Equivalent to
`sklearn.metrics.normalized_mutual_info_score` with `average_method='arithmetic'`.

## ACC — clustering accuracy

Best label permutation matching predictions to ground truth, computed via the
Hungarian algorithm on the confusion matrix. Reported as a fraction in [0, 1].

## ARI — adjusted rand index

`sklearn.metrics.adjusted_rand_score` (chance-corrected pair-counting measure).

## RI — rand index

Plain pair-counting agreement:
`RI = (a + d) / (a + b + c + d)` where `a` = same-pair same-cluster, `d` =
different-pair different-cluster, etc. Equivalent to
`sklearn.metrics.rand_score`.

## Notes

- All metrics expect integer label vectors of equal length.
- For algorithms that determine the number of clusters automatically (APC, DSet),
  the metrics are computed directly against ground truth (no `k` alignment needed
  for NMI/ARI/RI; ACC uses the Hungarian permutation).
