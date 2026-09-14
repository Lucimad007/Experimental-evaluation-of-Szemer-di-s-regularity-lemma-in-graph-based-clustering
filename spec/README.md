# Specification

This folder is the **authoritative specification** for the implementation in `src/`.
It is derived directly from the paper:

> Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo.
> *Experimental evaluation of Szemerédi's regularity lemma in graph-based clustering.*
> Pattern Recognition 171 (2026) 112205.

The implementation in `src/` must stay in sync with the documents here. When the code
or the spec change, update both and keep the mapping below accurate.

## Document map

| Spec document                | Paper section        | Implementation module                          |
|------------------------------|----------------------|------------------------------------------------|
| `regularity_partitioning.md` | §3.1, §3.2           | `src/szemeredi/`                               |
| `reduced_graph.md`           | §3.3                 | `src/szemeredi/regularity_lemma.py` (reduced)  |
| `algorithm1.md`              | §3.4, Algorithm 1    | `src/enhanced/algorithm1.py`                   |
| `base_algorithms.md`         | §2                   | `src/clustering/`                              |
| `similarity.md`              | §4 (similarity)      | `src/enhanced/similarity.py`                   |
| `parameters.md`              | §4.1                 | `src/experiments.py`, `src/config.py`          |
| `metrics.md`                 | §4 (NMI/ACC/ARI/RI)  | `src/metrics.py`                               |
| `datasets.md`                | §4, Table 1          | `src/datasets.py`                              |
| `experiments.md`             | §4.1, §4.2, §4.3     | `src/experiments.py`                           |

## Source of truth

- The paper PDF lives in `papers/` (gitignored, local only).
- The regularity-partitioning algorithm follows Alon et al. (1994) with the practical
  modifications described in §3.2 and the degree-based greedy certificate method of
  Fiorucci et al. (2020). The reference implementation studied while building this is
  `MarcoFiorucci/dense_graph_reducer` (clone into `.reference/`, gitignored).
  Opt-in at runtime: `--alg-kind fiorucci`.

## Status

Implementation in progress. Each spec doc has a `Sync status` line tracking whether
the code currently matches it.
