# APC and DSet variants

Full APC vs §2.2 / [10] audit: [apc.md](apc.md).

## APC (§2.2, Frey–Dueck [10])

Affinity propagation has no cluster count. Frey–Dueck: a shared preference
“could be the median of the input similarities (moderate number of clusters)
or their minimum (a small number of clusters).”

| `clustering_variant` | Preference | Status |
|-----------------------|------------|--------|
| `median` | median of off-diagonal input similarities | **[10] default** |
| `min` | minimum of those similarities | [10] alternative |

Implementation: `src/clustering/affinity_propagation.py`
(`preference_quantile=50` or `0`). Convergence: 10 equal iterates ([10] p.973).

## DSet (§2.3, Eq. 1)

Full audit: [dset.md](dset.md). Replicator dynamics as written. The **weight
cutoff** is unnamed in 2026; leftover 5% dump was removed (it contradicted
“until all clusters are obtained”).

| `clustering_variant` | Cutoff | Status |
|-----------------------|--------|--------|
| `hou2023` | \(0.0001\) | **paper cell** — Hou et al. PR 2023 [25] |
| `dslib` | \(10^{-5}\) | Vascon et al. [27] |
| `fiorucci` | \(1/(1.5n)\) | no textual source; leftover 0 |
| `rel95` | \(\ge 95\%\) of peak weight | relative reading of “greater than a threshold” |
| `fiorucci05` | \(1/(1.5n)\), leftover 5% | old Fiorucci dump; compared, not the paper cell |

`--clustering-variants paper` uses `hou2023` only.
