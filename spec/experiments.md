# Spec: Experiments (§4.1, §4.2, §4.3)

Sync status: implemented in `src/experiments.py`

## Setup

- 20 real UCI datasets (see `datasets.md`).
- 4 base algorithms: SPC, APC, DSet, SPRG (see `base_algorithms.md`).
- Hardware in the paper: Intel Core i7-10510U (1.8 GHz), 16 GB RAM. We record wall
  time per run; absolute times will differ but relative comparisons hold.
- Main criterion: NMI. ACC/ARI/RI also computed (supplementary).

## Exp 1 — Influence of parameters (§4.1, Figs. 2–5)

For each algorithm and each dataset, run the **enhanced** algorithm over the full
grid `ε × ϵ × b × σ` (σ only for SPC/APC/DSet). For each value of one parameter,
report the **mean NMI and mean running time over all combinations of the other
parameters** (not the best combo). Produce, per algorithm, three panels (one per
parameter: ε, ϵ, b) showing NMI curves across datasets, plus running-time panels.

Then verify the recommended narrow ranges (ε ∈ {0.1, 0.15, 0.2}, ϵ ∈ {0.02, 0.03, 0.04, 0.05, 0.1},
b ≤ 16) by comparing "all parameters" vs "selected parameters" results (Fig. 6).

## Exp 2 — Comparison with original algorithms (§4.2, Figs. 7–10)

For each dataset and algorithm, compare:
- **Original**: run the base algorithm directly on the full similarity matrix `G`
  (best `σ` by NMI).
- **Enhanced (Reg-*)**: run Algorithm 1, taking the **best NMI** over the
  recommended `(ε, ϵ, b)` grid **and** `σ` (searched independently of the original
  algorithm).

Report NMI and running time; the enhanced version should be better on both on the
majority of datasets.

## Raw vs z-score (not in the paper)

Same Exp 2 protocol, twice: `raw` features (paper cell) and `zscore` (drop
zero-variance / exact-duplicate columns, then standardize). Command:
`python -m src.main preprocess`. The paper path (`exp2`) is unchanged and
always uses `raw`.

## Exp 2b — Regularity vs k-means partitioning (§4.2, Fig. 11)

Replace the regularity-partitioning step in Algorithm 1 with a **k-means-based**
partition (partition the feature vectors into the same number of classes, keep all
other steps identical). The paper does not name `(ε, ϵ, b)` for this figure; we use
the single recommended-range point `(ε, b, ϵ) = (0.15, 4, 0.05)` and search `σ`.
Compare the two partitioning strategies across the 4 algorithms. Regularity
partitioning should win on most datasets, demonstrating the advantage of
edge-structure sampling over vertex sampling.

## Exp 3 — Comparison with recent algorithms (§4.3, Tables 2–5)

Compare Reg-SPC, Reg-APC, Reg-DSet, Reg-SPRG against 8 recent algorithms: 3W-DPET,
DenMune, FSDPC, DPC-FSC, LDP-SC, KSF-DPC, ICKDP, BP. Report NMI/ACC/ARI/RI per
dataset; the enhanced (old) algorithms should beat the recent ones on most datasets.

> **Scope note:** the 8 recent algorithms are external third-party methods whose
> per-dataset values are taken verbatim from Tables 2–5 of the paper as fixed
> reference columns (`src/baselines.py`). The Reg-* columns are **computed here**
> by running the enhanced algorithms (best config over the recommended parameter
> grid) and joined with the reference columns to reproduce Tables 2–5. The core
> contribution — Algorithm 1 + the 4 base algorithms + the k-means-partitioning
> baseline — is fully implemented from code.
