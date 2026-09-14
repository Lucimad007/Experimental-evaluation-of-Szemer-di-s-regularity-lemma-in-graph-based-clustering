# VERIFICATION — is the paper implemented, and does it work?

This document proves, at two levels, that

> Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo.
> *Experimental evaluation of Szemerédi's regularity lemma in graph-based clustering.*
> Pattern Recognition 171 (2026) 112205.

is implemented here faithfully and that its claims hold on our runs:

1. **Fidelity** — every implementable statement of the paper maps to code, and the
   mapping has been machine-checked (not merely eyeballed).
2. **Empirics** — our computed results reproduce the paper's reported numbers
   on the datasets we could run, and the paper's causal claim (regularity
   partitioning *enhances* clustering) is demonstrated by its own ablation.

> ## ⚠ 2026-08-27 — every empirical claim below is INVALID under `degree_mode="support"`
>
> A permutation test showed the default partitioner never reads the graph. Because
> `adj_mat = (sim_mat > 0.0)` is the complete graph on a Gaussian kernel, `alon2`
> certifies every pair irregular with the whole class as certificate, `alon3` never
> runs, `n_ir` is always `C(k,2)`, and the class split degenerates to **vertex index
> order**. UCI files are sorted by class label, so the classes came out class-pure
> for free.
>
> Permuting the rows of `X` leaves the similarity graph unchanged up to relabelling,
> yet (best NMI over the paper's own σ × ε × ϵ × b grid):
>
> | dataset | algo | original | enhanced, file order | enhanced, rows permuted |
> |---|---|---|---|---|
> | Ecoli | SPC  | 0.441 | 0.799 | 0.095 |
> | Ecoli | APC  | 0.500 | 0.844 | 0.103 |
> | Ecoli | DSet | 0.514 | 0.751 | 0.109 |
> | Seeds | SPC  | 0.512 | 0.764 | 0.052 |
> | Seeds | APC  | 0.446 | 0.867 | 0.084 |
> | Seeds | DSet | 0.472 | 0.820 | 0.075 |
> | Wine  | SPC  | 0.428 | 0.577 | 0.064 |
> | Wine  | APC  | 0.314 | 0.728 | 0.072 |
> | Wine  | DSet | 0.339 | 0.644 | 0.063 |
>
> So section 2 below ("the paper's claims hold on our runs") measured row order, not
> the regularity lemma. The fidelity table in section 1 is unaffected — the code does
> match the paper's text; the text's procedure is what collapses on dense graphs.
>
> `degree_mode="weighted"` (Sperotto & Pelillo [16] Eq. 15) is exactly
> permutation-invariant (pilot: 24/24 winning configs, `|Δ NMI| = 0` after a
> row shuffle). Re-running Exp 2 on that path with the paper's recommended grid
> **does not restore the tables**: see [`RESULTS.md`](RESULTS.md). Section 2
> below remains invalid as a reproduction of Hou et al.
> Mechanism, quotes and the acceptance test: `spec/regularity_partitioning.md`.

The complete line-by-line audit with code excerpts is in
[`IMPLEMENTATION_PROOF.md`](IMPLEMENTATION_PROOF.md); per-topic executable
proofs live in [`proof/`](proof/) (start with
[`proof/algorithm1.md`](proof/algorithm1.md) — the paper's Algorithm 1
pseudocode mapped line-by-line to code with a live run); computed-vs-paper
tables land in [`RESULTS.md`](RESULTS.md) as runs complete.

---

## 1. Fidelity — paper statement → code → machine check

| Paper element | Where implemented | How verified |
|---|---|---|
| Similarity `s(x,y)=exp(−d(x,y)/(d̄σ))`, Euclidean, σ grid {0.1,0.2,0.5,1,2,5,10} (§4.1) | `src/enhanced/similarity.py` | formula + grid diffed against paper text |
| SPC: **unnormalized** Laplacian, k smallest eigenvectors, rows of U, k-means (§2.1 verbatim) | `src/clustering/spectral.py::spc` | word-by-word match; NJW variant kept as option |
| SPRG = Zhu–Loy–Gong CVPR 2014, **ref [20]**: clustering forest (pseudo two-class, Gini gain Eq. 1–3, mtry=√d, Tclust=1000, φ-stop) + structure-aware affinity (Eq. 7–14, Bi/Unfm/Adpt) + forest average (Eq. 8) | `src/clustering/sprg.py` | **fast implementation vs a naive O(n²) transcription of Eq. 7/11/13: identical to 2e-16** (both Unfm and Adpt) |
| SPRG does not use σ (§4.1) | `src/runners.py::original_graph` | code inspection |
| APC message passing (§2.2) | `src/clustering/affinity_propagation.py` | paper-silent constants (median preference) documented |
| DSet replicator dynamics Eq. 1, x⁰=1/n, sequential extraction (§2.3) | `src/clustering/dominant_set.py` | vectorized update algebraically identical (Σxᵢ(Ax)ᵢ = xᵀAx) |
| Eq. 2 edge density; Eq. 3 weighted density | `src/szemeredi/classes_pair.py` | formula match |
| Alon conditions ε³n / ε⁴n / ε⁴n/16 / (ε³/2)n / 2ε⁴n (§3.2) | `src/szemeredi/conditions.py`, `classes_pair.py` | thresholds diffed vs paper + Fiorucci reference |
| Algorithm 1 line 12: break when `n_ir < k(k−1)/2` (**no ε factor**) | `regularity_lemma.py::check_partition_regularity(stop_rule="algorithm1")` | condition verified character-level in the PDF; theoretical `ε·C(k,2)` rule (§3.2 Step 3) kept as option |
| Modification 1: ≤1 irregular partner per class | `refinement_step.py` | matches Fiorucci reference exactly |
| Modification 2: degree-based greedy certificates (Fiorucci et al. [28]) | `conditions.py::alon3` | identical to the reference code the paper adopts |
| Modification 3 + Algorithm 1 line 3: `while ϵ > k_i/n` | `regularity_lemma.py::_line3_compressible` | float test `ϵ > k/n` (not `k >= int(ϵ·N)`) |
| Algorithm 1 lines 18–27: R ∈ R^{k×k} (Eq. 3 weights), cluster R, map labels, V0→nearest cluster | `regularity_lemma.py::generate_reduced_sim_mat`, `enhanced/algorithm1.py` | pipeline order matches the 27 pseudocode lines |
| Parameter grids ε/ϵ/b, recommended ranges, b<\|G\| (§4.1) | `src/config.py` | all values diffed against paper |
| Table 1 datasets (all 20, NP/ND/NC) | `src/config.py`, `src/datasets.py` | **all 20 loaders verified against Table 1 with real data** (USPS=Roweis usps_all.mat 11000×256 with NN-validated label order; Dutchnumeral=UCI mfeat, 6 views=649 dims; Leaves=zip's 64-D shape descriptors) |
| Metrics NMI/ACC/ARI/RI (§4) | `src/metrics.py` | NMI arithmetic, ACC Hungarian |
| Tables 2–5 reference values | `src/baselines.py`, `src/paper_reference.py` | **640/640 recent-algorithm cells cross-validated against the PDF; our parsed Reg-* means equal the paper's stated means (NMI 0.76/0.72/0.73/0.76 …)** |

Paper-silent constants (APC preference, DSet threshold, d₀, SPRG φ/variant,
preprocessing) are individually documented in `IMPLEMENTATION_PROOF.md` §"Gaps"
with their provenance — nothing is silently invented. They are **not**
grid-searched; experiments use the paper's ε / ϵ / b / σ values only.

## 2. Empirics — the paper's claims hold on our runs

**(a) The causal claim: regularity partitioning *enhances* clustering (§4.2, Fig. 11).**
The paper's own ablation — replace regularity partitioning with k-means
partitioning at the *same* reduced-graph size k, everything else unchanged —
isolates the mechanism. Our run (SPC, 6 datasets):

| dataset | original | + regularity partition | + k-means partition (same k) |
|---|---|---|---|
| Thyroid | 0.138 | **0.682** | 0.138 |
| Wine | 0.062 | **0.553** | 0.398 |
| Glass | 0.077 | **0.559** | 0.093 |
| Seeds | 0.525 | **0.764** | 0.624 |
| Sonar | 0.045 | **0.383** | 0.008 |
| Spectf | 0.003 | **0.125** | 0.029 |

Regularity beats k-means partitioning **6/6** and beats the original algorithm
**6/6** — compression alone (same k!) does not do it; the structure-aware
partition does. This is the paper's central claim, demonstrated.

**(b) Our computed values match the paper's own reported values.**
Best-of-recommended-grid, fixed code, ours vs paper (NMI):
Thyroid SPC 0.83/0.81, Thyroid SPRG 0.92/0.89 (full-grid checkpoint:
SPC 0.83, APC 0.88/0.73, DSet 0.85/0.74, SPRG 0.92/0.89);
Wine SPC 0.84/0.76, APC 0.73/0.58. Same regime everywhere; where we differ we
are *above* the paper, the expected direction for best-of-grid selection.
Full per-dataset tables: `RESULTS.md` (written by `python -m src.make_results`).

**Known per-dataset divergence (documented, not hidden):** on *Appendicitis*
(n=106, the smallest dataset) the original SPC already scores NMI = 1.000 at
every σ in our run, while our enhanced version tops out at 0.15 (the paper
reports Reg-* = 0.91 there). The dataset is trivially separable; partitioning
106 points into ≥5 classes destroys the two-block structure of the reduced
graph in our pipeline (with k = 2 the DSet step is degenerate). The paper
itself reports the enhancement degrading one dataset of twenty (Landsat);
ours appears to be Appendicitis. All other datasets match the paper's regime.
Full per-dataset tables: `RESULTS.md` (written by `python -m src.make_results`).

*Update after targeted Leaves/Ecoli runs (`scripts/_verify_apc_dset_fix.py`,
narrow σ∈{0.2,0.5,1} × ε∈{0.1,0.2} × ϵ∈{0.05,0.1} × b∈{8,16} grid plus a
diagnostic sweep up to ϵ=0.2):*

- **Ecoli reproduces fully**: Reg-SPC/APC/DSet reach NMI 0.80–0.89 vs paper's
  0.71–0.78 and beat all eight recent algorithms on every metric; only
  Reg-SPRG sits below its paper value (0.65 vs 0.69), consistent with the
  documented on-R adaptation.
- **Leaves splits**: Reg-SPC reaches 0.76 vs paper 0.82 and still beats all
  recent baselines (best recent ≈0.74); Reg-APC/Reg-DSet plateau at
  0.56–0.64 vs the paper's 0.75/0.79 across *every* legitimate setting tried
  (recommended ranges and up to ϵ=0.2). The line-12 stop fires early there
  (k settles at 64–133 while the data has 100 classes), so refinement stops
  before the reduced graph can represent all classes — a per-dataset
  behaviour we could not reproduce past, documented rather than tuned away.

**(c) Mechanism.** The reduced graph R was verified to carry ground-truth-aligned
block structure (majority-pure classes; intra-block density ≈0.8 vs ≈0.2–0.35
between blocks) — the structure-preserving compression the lemma guarantees.

## 2b. The paper's empirical claims, checked one by one

`python -m proof.verify_claims` re-verifies every §4 claim from our computed
results (re-run as more datasets land). Status on the current data
(12-dataset exp2/exp2b, 2-dataset exp3, 5-dataset exp1 grid subset):

| Claim (paper) | Result |
|---|---|
| C1 Enhancement improves clustering (Figs 7–10; "better on 19/20") | **PASS** — 40/48 runs, mean NMI 0.375 → 0.534 |
| C2 Regularity beats k-means partitioning (Fig. 11) | **PASS** — 36/48 runs |
| C3 Reg-* beats the 8 recent algorithms (Tables 2–5) | **PASS** — SPC/APC/DSet each beat all 8 on 3/4 datasets so far; SPRG 1/4 (its on-graph adaptation is our one documented SPRG deviation) |
| C4 ε ∈ 0.1–0.2 recommended; degradation above 0.2 | **FAIL on the current subset** — aggregate NMI is nearly flat in ε (0.329–0.337) with the max at ε=0.4 vs 0.332 within ε≤0.2; differences are ≤0.008, so no paper-style degradation above 0.2 is observed here. In the reference-faithful operationalization ε has weak influence (the ε⁴ thresholds certify nearly all pairs irregular for any ε), so the partition barely depends on ε. |
| C5 ϵ best ≤ 0.1; 0.01 never best; 0.2 costs time without gains | **PASS** — best at 0.03; 0.01 worst-but-one; 0.2 lowest NMI and 1.8× the median time |
| C6 b ≤ 16 suffices; b > 16 doesn't help | **FAIL on the current subset** — on the 5-dataset exp1 grid the b>16 mean (max 0.408) exceeds b≤16 (0.363), reversing the earlier PASS-on-4/5 state; per-dataset behaviour is mixed, and Appendicitis recovers NMI ≈ 0.87 only at large b (classes of 1–3 points; see divergence note below). |
| C7 larger ε → less time; larger ϵ → more time | **PASS** — 2.23s (ε=0.05) → 1.77s (ε=0.6); ϵ=0.2 costs the most |
| C8 recommended ranges rarely degrade vs all parameters (Fig. 6) | **PASS** — selected-range mean 0.360 vs all-parameter 0.341 (selected is *better*) |

## 3. Scope, honestly

- **Run here**: the 12 small datasets (n ≤ 1000) — full §4.1 parameter grid
  (exp1), enhanced-vs-original (exp2), regularity-vs-k-means (exp2b),
  vs-recent-algorithms (exp3).
- **Not run here**: the 5 large datasets (Segment, Rice, Spambase, Landsat,
  USPS) — each needs hours of *uninterrupted* power (USPS ≈ a day for its full
  grid). All code paths are identical; only compute time differs. Commands are
  provided in `RUN_AFTER_OUTAGE.bat` (commented block at the bottom).
- Everything is **checkpointed per dataset and resumable**: after a power
  outage, double-click `RUN_AFTER_OUTAGE.bat` — completed work is skipped
  automatically.

## 4. How to reproduce the checks

```bash
python -m src.smoke_test                  # end-to-end pipeline check
python -m src.demo_run                    # 9-dataset demonstration (fast)
python -m src.make_results                # ours-vs-paper tables -> RESULTS.md
RUN_AFTER_OUTAGE.bat                      # (Windows) resume all small-set runs
```
