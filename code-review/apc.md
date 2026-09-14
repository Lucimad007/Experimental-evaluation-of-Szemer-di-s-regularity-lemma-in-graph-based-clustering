# APC (§2.2 of Hou et al., Frey–Dueck [10])

## Verdict

Hou et al. §2.2 is **prose, not an algorithm**. There is no energy function,
no responsibility/availability update, no preference, and no stopping rule
in the 2026 paper. Every sentence of that paragraph matches our code.

The **equations** are Frey–Dueck [10]. Two knobs used to contradict [10];
they now match:

- shared preference = median (or min) of **off-diagonal** input similarities
  (zeros count; we no longer drop non-positive entries)
- stop when exemplar decisions are unchanged for **10** iterations ([10] p.973)

Message passing is `sklearn.cluster.AffinityPropagation` (`affinity="precomputed"`).

## What Hou et al. §2.2 actually says (mapped to code)

| §2.2 claim | Code |
|------------|------|
| “takes the pair-wise data similarity matrix at input” | `apc(sim_mat)` — Gaussian \(S\) on original G, reduced graph \(R\) for Reg-APC |
| “exchanges messages between data points to generate cluster centers and cluster members gradually” | sklearn `_affinity_propagation`: responsibility \(R\) and availability \(A\) |
| “treats all data points as potential cluster centers initially” | \(A=0\), \(R=0\); preference on the diagonal of \(S\) so every vertex can be an exemplar |
| “messages are updated by minimizing an appropriately chosen energy function” | Frey–Dueck net-similarity / energy; sklearn implements the damped updates |
| “each message reflects the affinity that one data point has for choosing another data point as its cluster center” | responsibility \(r(i,k)\): how well \(k\) is suited to be exemplar for \(i\) |
| “does not require to specify the number of clusters” | `n_clusters` is ignored; \(k\) = number of vertices with \(a(i,i)+r(i,i)>0\) |

Implementation: `src/clustering/affinity_propagation.py`.

## [10] preference

| `clustering_variant` | What we set | Source |
|----------------------|-------------|--------|
| `median` (paper cell) | median of off-diagonal \(S\) | [10] “median of the input similarities” |
| `min` | minimum of those values | [10] alternative |

## Remaining paper-silent (not in Hou, not a contradiction)

| Knob | We use | Notes |
|------|--------|--------|
| Damping | \(0.5\) | [10] and sklearn default |
| `max_iter` | 500 | [10] does not cap; sklearn default is 200 |
| Degeneracy noise | sklearn \(\varepsilon\)-scale Gaussian noise on \(S\) | [10] says to break ties; this is sklearn’s method |
| `random_state` | 314 | not in Hou; pins the noise |

sklearn’s updates are the standard Frey–Dueck map. We do not reimplement them.

## Experiments

`--clustering-variants paper` → `median` only.
`--clustering-variants all` also runs `min`.
