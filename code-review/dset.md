# DSet (§2.3 of Hou et al., Eq. 1, refs [9] [25] [26] [27])

## Verdict

Hou et al. **§2.3 writes the algorithm**: sequential dominant-set extraction
via replicator dynamics **Eq. 1**, \(x_i^{(0)}=1/n\), then points with weight
above a threshold form a cluster, remove them, repeat on the rest until all
clusters are obtained. \(k\) is automatic.

Eq. 1, the uniform start, and sequential peeling **already matched**. Two
things did **not**:

1. **Leftover 5% dump** — contradicted “repeats this process with the remaining
   unallocated data until all clusters are obtained.” Default leftover
   fraction is now **0**.
2. **Threshold \(1/(1.5n)\)** — §2.3 does not name a number. That cutoff
   matches **no** cited source. Same first author in [25]: “weights above a
   threshold (**0.0001** in this paper).” The library they point at [27]
   uses \(10^{-5}\). The paper-faithful cell is now `hou2023` = \(0.0001\).
   `fiorucci` and `dslib` remain search variants.

We use Eq. 1, not the faster dynamics of Bulo et al. [26] (the paper
describes Eq. 1 and only mentions [26] as “a more efficient method”).

## What §2.3 says (mapped to code)

| §2.3 claim | Code |
|------------|------|
| Sequential, not all-at-once like SPC | outer loop in `dominant_sets` |
| Input: pairwise similarity \(A\) | `graph_mat` |
| \(x=(x_1,\ldots,x_n)^\top\) data weights | simplex vector `x` |
| \(x_i^{(0)}=1/n\) | uniform on the **remaining** vertices (after a peel, \(n\) is \(\lvert\mathrm{rest}\rvert\)) |
| Eq. 1: \(x_i^{(t+1)}=x_i^{(t)}\,(Ax^{(t)})_i\big/\bigl(x^{(t)\top}Ax^{(t)}\bigr)\) | `_replicator`: `x <- x*(A@x)` then `/sum(x)` because \(\sum_i x_i(Ax)_i=x^\top Ax\) |
| After convergence, \(x_i\) greater than a threshold → one cluster | `y >= thresh` |
| Remove those points; repeat on remaining | `already_clustered`; next start zeros them |
| \(k\) automatic | peel until nothing remains or a peel is empty |
| [26] faster extraction | **not used** (Eq. 1 as written) |
| [27] open-source library | cited; we do not wrap DSLib; cutoff `dslib` = \(10^{-5}\) is their theta |

Implementation: `src/clustering/dominant_set.py`.

## Threshold variants (`--clustering-variants`)

| `clustering_variant` | Cutoff | Status |
|-----------------------|--------|--------|
| `hou2023` | \(0.0001\) | **paper cell** — Hou et al. PR 2023 [25] |
| `dslib` | \(10^{-5}\) | Vascon et al. [27] |
| `fiorucci` | \(1/(1.5n)\) | no textual source; leftover 0 |
| `rel95` | \(\ge 95\%\) of peak replicator weight | relative cutoff implemented in `_parse_weight_threshold` |
| `fiorucci05` | same as `fiorucci`, leftover fraction \(0.05\) | previously unused leftover dump |

## Remaining paper-silent (not a contradiction)

| Knob | We use | Notes |
|------|--------|--------|
| Replicator `tol` | \(10^{-5}\) | [27] uses \(10^{-6}\) |
| `max_iter` | 1000 | [27] also 1000 |
| Empty peel | stop; leftover share the last label | §2.3 does not say what to do if no \(x_i\) exceeds the threshold |

Zeros on already-clustered coordinates make \(A@x\) equal to the induced
submatrix on the remainder, so we do not copy \(A\) each peel.
