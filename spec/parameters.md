# Spec: Parameters (§4.1)

Sync status: implemented in `src/config.py` and `src/experiments.py`

## Three regularity-partitioning parameters

| Param | Meaning | Range tested | Recommended (paper) |
|-------|---------|--------------|---------------------|
| `ε`   | regularity constant (Def. 1); share of a class used to check regularity | {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6} | 0.1 – 0.2 |
| `ϵ`   | max compression ratio `|R|/|G|`; `⌊1/ϵ⌋` = min class size | {0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2} | 0.02 – 0.1 |
| `b`   | initial number of classes (initial cardinality of `R`) | {2,…,10, 16, 32, 64, 128, 256, 512, 1024} with `b < |G|` | `b ≤ 16` |

## σ grid (similarity)

`σ ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10}` (SPC, APC, DSet only).

## Evaluation protocol (§4.1)

Instead of fixing two parameters and varying the third (grid-search best combo), the
paper uses the **mean result over all combinations of the other two parameters** to
show the influence of the third. Example: for `ε = 0.1`, average NMI over all
`(ϵ, b)` with `ϵ` from its grid and `b` from its grid.

## Findings (§4.1)

- **ε**: larger `ε` → worse results in general (degraded on 13 datasets, slight on
  5, slight improvement on 2). Small `ε` gives better results but more running time.
  Evident degradation at `ε > 0.2` on several datasets. Recommend 0.1–0.2.
- **ϵ**: running time rises quickly with `ϵ`; best at `ϵ = 0.1` or smaller on 16/20
  datasets. Both too-large (0.2) and too-small (0.01) degrade results. Recommend
  0.02–0.1.
- **b**: `b > 16` does not save running time (compensation in later iterations
  outweighs skipped iterations) and does not improve results. With `b ≤ 16` results
  vary in a complex pattern; recommend `b ≤ 16`.

## Paper-silent knobs searched in Exp 2 / Exp 3 (best NMI)

These are not in the paper's σ / ε / ϵ / b grids. They complete algorithms the
paper specifies only qualitatively, and are searched only for the *enhanced*
(Reg-*) runs — original APC/DSet keep the Frey–Dueck median and `1/(1.5 n)`.

| Knob | Algorithms | Values | Source |
|------|------------|--------|--------|
| APC preference quantile | APC | {50, 90, 95, 99} | Frey–Dueck [10]: shared preference may be the median or varied for more/fewer clusters |
| DSet support cutoff | DSet | {`None`=`1/(1.5n)`, `rel50`, `rel80`, `rel95`} | §2.3 "greater than a threshold" (value unnamed) |
| Reduced-graph `d₀` | APC, DSet | {0, `p90`, `p95`} | §3.3 / Lemma 2: edges only if density > `d₀` (value unnamed). SPC/SPRG keep `d₀=0` |
