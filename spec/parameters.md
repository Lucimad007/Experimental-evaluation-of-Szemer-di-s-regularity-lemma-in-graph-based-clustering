# Spec: Parameters (§4.1)

Sync status: implemented in `src/config.py` and `src/experiments.py`

Quoted from Hou et al., Pattern Recognition 171 (2026) 112205, §4.1.
Reproduction searches **only** these values.

## Three regularity-partitioning parameters (tested)

| Param | Meaning | Values tested (§4.1) | Recommended / "selected" (Fig. 6, Exp 2/3) |
|-------|---------|----------------------|--------------------------------------------|
| `ε`   | regularity constant (Def. 1); share of a class used to check regularity | `{0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6}` | `{0.1, 0.15, 0.2}` ("0.1 to 0.2") |
| `ϵ`   | max compression ratio `|R|/|G|`; `⌊1/ϵ⌋` = min class size | `{0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2}` | `{0.02, 0.03, 0.04, 0.05, 0.1}` ("0.02 to 0.1") |
| `b`   | initial number of classes (initial cardinality of `R`) | `{2,…,10, 16, 32, 64, 128, 256, 512, 1024}` with `b < |G|` | `{2,…,10, 16}` (`b ≤ 16`) |

## σ grid (similarity)

`σ ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10}` (SPC, APC, DSet only). SPRG does not use `σ`.

## Which experiment uses which grid

| Experiment | Search | How a number is reported |
|------------|--------|--------------------------|
| Exp 1 (Figs. 2–5) | full `ε × ϵ × b × σ` × stop rule × `d₀` | **mean** NMI/time over the other parameters (σ, stop rule, `d₀`) |
| Exp 1 (Fig. 6) | same runs | mean over **all** tested `(ε,ϵ,b)` vs mean over **selected** ranges |
| Exp 2 (Figs. 7–10) | original: best `σ`; enhanced: best NMI over selected `(ε,ϵ,b) × σ` × stop rule × `d₀` | one original vs one enhanced per dataset/algorithm |
| Exp 2b (Fig. 11) | `σ` grid at the single recommended point `(ε, b, ϵ) = (0.15, 4, 0.05)`; `d₀ = 0` fixed (ablation) | regularity vs k-means at the same `k` |
| Exp 3 (Tables 2–5) | same enhanced search as Exp 2 | best-NMI Reg-* joined to the paper's 8 recent-algorithm columns |

## Evaluation protocol (§4.1)

Instead of fixing two parameters and varying the third (grid-search best combo), the
paper uses the **mean result over all combinations of the other two parameters** to
show the influence of the third. Example: for `ε = 0.1`, average NMI over all
`(ϵ, b)` with `ϵ` from its grid and `b` from its grid (and `σ` for SPC/APC/DSet).

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

## Not searched (paper-silent defaults)

The paper does not name APC preference, DSet weight cutoff, or the reduced-graph
adjacency threshold `d₀`'s value. Reproduction does **not** grid-search APC
preference or DSet cutoff (APC uses the Frey–Dueck median, DSet uses
`1/(1.5 n)`). `d₀` **is** searched in Exp 1/2/3 as a documented paper-silent
extension (`config.D0_GRID`, which includes 0 = the pure-paper setting of keeping
every Eq. 3 weight); Exp 2b fixes `d₀ = 0` because the ablation must hold every
other part fixed.
