# Results — corrected Alon / V0 vs the paper

Hou, Ge, Yuan, Pelillo, *Experimental evaluation of Szemerédi's regularity lemma
in graph-based clustering*, Pattern Recognition 171 (2026) 112205.

**Verdict:** After fixing `alon2` (Vs degrees, 1/8 both-direction trigger) and
the [28] Algorithm 2 V0 guard, the honest weighted partitioner is still
permutation-invariant and still does **not** recover Tables 2–5 on Wine,
Seeds, or Ecoli. Appendicitis enhanced NMI is 0.865 (paper 0.82) for all three
algorithms — the same all-Reg-* identity, while original SPC/DSet already score
1.0.

## What changed in the partitioner

- `alon2` uses Vs column sums of the same matrix as `bip_avg_deg`; trigger is
  more than `(1/8)ε⁴n` deviant vertices (both directions); certificate is
  `A' = Vr`, `B'` = larger one-sided set.
- V0 follows [28] Alg 2: stay in V0 if `|C0| ≤ εn`; else distribute into
  existing classes if `|C0| > |P|`. Weighted V0 only accumulates.
- If a refinement does not increase `k`, the loop stops (otherwise
  `degree_based` can cycle when C0 is poured back into the same labels).

Tests: `python -m pytest tests` (**48 passed**, 2026-09-14). Smoke (honest
path): n=200 blobs, original and enhanced SPC both NMI/ACC 1.000. The old
support-path smoke (enhanced NMI 0.026, `k` stuck at `b`) is the index-order
artifact, not the pipeline used below.

## What was run

- **Partitioner:** `degree_mode=weighted` ([16] Eq. 15).
- **Protocol:** paper Exp 2 recommended grid, `d₀ = 0`, both stop rules,
  SPC/APC/DSet, Wine/Seeds/Ecoli/Appendicitis.
- **Output:** `results/corrected/exp2_enhanced_vs_original/` (gitignored).
  The previous graph-faithful CSVs are in `results/weighted/`; the
  index-order artifact is in `results/support_artifact/`.

```text
python -m src.main exp2 --profile honest --d0 0
  --datasets Wine,Seeds,Ecoli,Appendicitis --algorithms SPC,APC,DSet
  --out results/corrected

python -m src.main invariance --out results/corrected
```

## Shuffle invariance

**24/24 passed** with `|Δ NMI| = 0` (tolerance 0.05).

## NMI: original vs corrected enhanced vs old weighted vs paper

One row per (dataset, algorithm). Enhanced is the best of the two stop rules
on the recommended grid. Old weighted is the 2026-08-27 pilot (identical stop
rules then). Paper values are Tables 2–5.

| dataset | algo | original | corrected | old weighted | paper | enh − orig | enh − paper |
|---|---|---|---|---|---|---|---|
| Wine | SPC | 0.441 | 0.443 | 0.421 | 0.76 | +0.002 | −0.32 |
| Wine | APC | 0.326 | 0.415 | 0.415 | 0.58 | +0.089 | −0.16 |
| Wine | DSet | 0.431 | 0.424 | 0.401 | 0.60 | −0.007 | −0.18 |
| Seeds | SPC | 0.525 | 0.460 | 0.420 | 0.81 | −0.065 | −0.35 |
| Seeds | APC | 0.474 | 0.605 | 0.548 | 0.81 | +0.131 | −0.20 |
| Seeds | DSet | 0.559 | 0.581 | 0.462 | 0.81 | +0.022 | −0.23 |
| Ecoli | SPC | 0.441 | 0.489 | 0.456 | 0.78 | +0.048 | −0.29 |
| Ecoli | APC | 0.515 | 0.509 | 0.492 | 0.77 | −0.006 | −0.26 |
| Ecoli | DSet | 0.528 | 0.487 | 0.454 | 0.71 | −0.041 | −0.22 |
| Appendicitis | SPC | 1.000 | 0.865 | 0.775 | 0.82 | −0.135 | +0.04 |
| Appendicitis | APC | 0.802 | 0.865 | 0.775 | 0.82 | +0.063 | +0.04 |
| Appendicitis | DSet | 1.000 | 0.865 | 0.775 | 0.82 | −0.135 | +0.04 |

The two stop rules now **disagree** on Wine SPC (0.421 vs 0.443) and Seeds SPC
(0.443 vs 0.460). That is new: the 2026-08-27 pilot had identical best NMI on
every cell because `alon2` marked every pair irregular.

Enhanced NMI beats original on **6/12** unique cells (Wine SPC/APC, Seeds
APC/DSet, Ecoli SPC, Appendicitis APC). Wine/Seeds/Ecoli remain 0.16–0.35 NMI
below the paper. Appendicitis is above the paper and identical across SPC/APC/DSet.

## How to read this

1. The 2026-08-30 fixes make Alon internally consistent and stop V0 from minting
   classes. They raise several honest cells versus the 08-27 pilot (Seeds APC
   0.548 → 0.605, Seeds DSet 0.462 → 0.581, Appendicitis 0.775 → 0.865).
2. They do not turn the Gaussian complete graph into the partition the paper’s
   tables imply. Tables 2–5 are still not a graph-faithful reproduction.
3. kNN / `w > τ` remain unpublished and unused.

Not mixed into this verdict: DSet cutoff `1/(1.5n)` vs Hou 2023 `0.0001` /
DSLib `1e-5`; APC 15 vs 10 stable iterations; SPRG `φ`.

## Fresh honest rerun (2026-09-14)

Previous `results/lemma_claim/*.csv` files were moved to
`results/lemma_claim/archive_pre_rerun/` so resume keys were empty, then:

```text
python scripts/_lemma_all.py      # 80 cells, ~1.6 h wall
python scripts/_lemma_higher.py   # 60 SPC/APC/DSet cells, ~2.1 h wall
python scripts/_refine_winners.py # merge → best.csv, ~5 min
```

**`all_small_first.csv` (20 datasets × 4 algorithms, SPRG included):** 80 rows,
**78 valid** NMI cells (Leaves SPC and Leaves SPRG had no lemma score — SPC
is recovered in `higher.csv`). Vs own original: **60 better, 3 tie, 15 worse**.
Vs Hou Reg-*: **22 better, 56 worse**.

Full-\(n\) highlights (lemma / orig / paper): Appendicitis all 1.000 (paper
0.82); Wine SPC 0.911 / 0.909 / 0.76, APC/DSet/SPRG 0.893; Seeds SPC 0.775 /
0.752 / 0.81; Ecoli SPC 0.701 / 0.623 / 0.78. Weak: Sonar, Spectf, Landmine.

**`best.csv` (60 SPC/APC/DSet cells):** vs `all_small_first` **21/59 improved,
mean ΔNMI +0.013, max +0.110**. Beats paper Reg-* on **18/60**. Sources:
mostly `all_small`, then `higher`/`refined`. Highlights vs the same-run
`all_small` column: Banknote APC 0.435→0.544, Rice APC/DSet 0.486→0.596,
Libras +0.04..0.05, USPS APC 0.521→0.579, Leaves SPC recovered at 0.736
(paper 0.82). Remaining gaps: Sonar, Glass, Spambase, capped Rice/Raisin/USPS.

Quote **`results/lemma_claim/best.csv`** for SPC/APC/DSet and
`all_small_first.csv` for SPRG. Capped \(n>500\) cells are not comparable to
Hou’s full-\(n\) tables.

### Why capped n > full n (probe, 2026-09-13, still valid)

The 2026-09-13 n-cap probes were not re-run. Their conclusion stands: cap
effects are seed luck at n=400 and kNN under-search at large n, not “more
graph data hurts”. See `archive_pre_rerun/` and the notes below.

### Why capped n > full n (probe, 2026-09-13)

Cross-evaluating the n=400 winners at full n (and vice versa) plus one fair
mini-grid per n level (`scripts/_probe_ncap.py`, `_probe_ncap2.py`) shows the
cap effect is **two things, neither of them "more graph data hurts"**:

1. **Seed luck at small n.** Banknote SPC with its winner config scores 0.929,
   0.382, 0.834, 0.737 under stratified caps with seeds 0–3. The 400-cap
   numbers are a favorable draw, not a floor; comparisons against the old
   grid stay apples-to-apples because both use seed 0.
2. **kNN under-search at large n.** The large-n sweep restricted kNN to 20.
   At n=1372 the *same* Banknote config improves from 0.563 (knn=20) to
   **0.835 with mutual kNN 40**; at n=400, m40 gives 0.729 (m60 collapses the
   graph). Fixed kNN does not scale: the grid for n>500 now also searches
   `m20`/`m40` (`scripts/_lemma_higher.py`).

Full-n Rice (n=3810) is stable at 0.569 regardless of the n=400 config; the
n=400 0.626 again reflects the favorable cap draw.
