# CONTEXT PROMPT — everything suspicious / vague / paper-silent in this codebase

Paste this whole file into an LLM to bring it up to speed on every questionable
point in the reproduction. Keep this file updated as the codebase evolves.

> ## RESOLVED, 2026-08-27 — read this first
>
> Item **A3** below was not a cosmetic degeneracy. It disabled the partitioner.
> Because `adj_mat = (sim_mat > 0.0)` is the complete graph on a Gaussian kernel,
> `alon2` certifies **every** pair irregular with the whole class as certificate,
> `alon3` (the paper's modification 2) **never runs at all**, `n_ir` is always
> `C(k,2)` so **neither stop rule ever fires** (settling A1/G1), and the class
> split falls back to **vertex index order**. Since UCI files are sorted by class
> label, the partition came out class-pure for free and every "enhancement"
> number in `results/` was an artifact of row order:
>
> | dataset | algo | original | enhanced, file order | enhanced, rows permuted |
> |---|---|---|---|---|
> | Ecoli | SPC | 0.441 | 0.799 | 0.095 |
> | Ecoli | APC | 0.500 | 0.844 | 0.103 |
> | Seeds | APC | 0.446 | 0.867 | 0.084 |
> | Wine  | APC | 0.314 | 0.728 | 0.072 |
>
> Fix: a `degree_mode` flag. `"support"` keeps the reference behaviour;
> `"weighted"` implements Sperotto & Pelillo [16] Eq. 15 (average weighted
> degree, decreasing) and is exactly permutation-invariant. Both are searched and
> recorded by the experiments (`degree_mode` / `enh_degree_mode` columns).
> Details and quotes: `spec/regularity_partitioning.md`.
>
> Open: `"weighted"` is honest but scores ~0.15–0.46 NMI at a single parameter
> point, well below the paper's Reg-* values. Whether a variant exists that is
> both permutation-invariant and reproduces Tables 2–5 is **unresolved** (G5).
>
> Circumstantial: in the paper's own Tables 2–5, all four Reg-* algorithms give
> identical values on Seeds, Appendicitis, Rice and Banknote across all four
> metrics (ACC = 1.00 on Rice/Raisin/Banknote). Four unrelated clustering
> algorithms agreeing to two decimals is what you would expect if the labels come
> from the partition rather than from the clustering step.

```
CONTEXT: I am working on a Python reproduction of:

Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo,
"Experimental evaluation of Szemerédi's regularity lemma in graph-based
clustering", Pattern Recognition 171 (2026) 112205. [THE PAPER]

The paper's official GitHub (https://github.com/dr-houjian/eval-regcluster/)
returns 404, so the reproduction was verified against: (a) the full PDF text,
(b) all 40 cited references (kept locally in references/), (c) the adopted
reference code Fiorucci et al. [28] `dense_graph_reducer` (kept locally in
.reference/, bit-identical differential-tested). Implementation: src/,
specs: spec/, self-verification: VERIFICATION.md + IMPLEMENTATION_PROOF.md.

Help me reason about the following list of EVERYTHING that is suspicious,
vague, or paper-silent in the codebase. For each item I give the paper text,
what the code does, and my open question.

════════════════════════════════════════════════════════════════════════
A. ALGORITHM / PSEUDOCODE AMBIGUITIES
════════════════════════════════════════════════════════════════════════

A1. Algorithm 1 line 12 stop rule — TYPO SUSPECT (MOOT in practice: n_ir is
always C(k,2) because alon2 certifies every pair irregular, so neither rule
ever fires and the loop always exits on the ϵ compression guard — which is
exactly what §4.1 describes. Verified: both rules give byte-identical results).
Paper Algorithm 1 line 12: "if n_ir < k_i(k_i − 1)/2 then Break" (NO ε factor,
strict <). But §3.2 Step 3 of the same paper: "If at most ε(k_i choose 2)
pairs are not verified as regular pairs, then P_i is a regular partition"
(WITH ε, ≤). Every ancestor ([16] Sperotto-Pelillo, [17] Sárközy, [28]
Fiorucci, [21] the paper's own ICPR 2024 version) uses the ε·C(k,2) rule.
As printed, line 12 fires as soon as ONE pair verifies regular.
Code: both rules implemented — default `stop_rule="algorithm1"` = as-printed
(n_ir < C(k,2)); `stop_rule="theoretical"` = n_ir ≤ ε·C(k,2). Experiments
currently run BOTH and record which one won (columns stop_rule/enh_stop_rule).
Q: Is line 12 a typo (dropped ε)? Which rule did the authors' experiments
actually use? Empirically the ε⁴-threshold conditions certify almost all
pairs irregular (n_ir = C(k,2)), so the loop usually exits via the ϵ
compression guard anyway ("terminated when k > ϵ·|G| in most cases", §4.1).

A2. Line 3 loop guard semantics.
Algorithm 1 line 3: "while ϵ > k_i/n do". Code: float test `compression_rate
> k/N` (`_line3_compressible`). The Fiorucci reference instead breaks when
`k >= int(ϵ·N)`. Difference is ≤1 iteration at the boundary. §4.1 also says
"⌊1/ϵ⌋ = minimum class size" and "terminating when the number of classes is
greater than ϵ|G|" — three slightly different phrasings of the same guard.
Q: Is the float test the right reading of "while ϵ > k_i/n"?

A3. The 0/1 support construction — RESOLVED, WAS FATAL (see banner at top).
Paper §3.4: "we calculate the pairwise similarity matrix W, obtaining the
original graph G" — NO threshold ever defined. Code (inherited verbatim from
Fiorucci's dense_graph_reducer): `adj_mat = (sim_mat > 0.0).astype(float)`.
The Gaussian kernel is strictly positive, so this support is the COMPLETE
graph. Measured consequences: alon2 fires on every pair (it compares an
UNWEIGHTED degree against a WEIGHTED mean degree), alon3 never runs, n_ir is
always C(k,2), and the split degenerates to vertex index order — which is why
the enhanced numbers tracked the class-sorted row order of UCI files rather
than the graph. Answer to Q2: YES, the ordering must be [16] Eq. 15's average
weighted degree; that is now `degree_mode="weighted"` and it is exactly
permutation-invariant. Q1 (kNN / w > tau) remains unused: it is a paper-silent
preprocessing choice we decline to invent.
Still open: alon2's row/col mix-up (it derives the deviation mask from V_r's
degrees, then applies it to index_map[1] = V_s) is inherited from the
reference and NOT yet fixed — harmless while all degrees are equal.

A4. Refinement (Step 4) mechanics — NOT IN ANY PAPER TEXT.
Paper only says: "divide current classes to obtain a new partition P′ with
1 + k_i·4^k_i classes, with a refinement algorithm, e.g., [30]" then
modification 1: "limit the number of irregular pairs containing each class
to be at most one... allowing to divide the classes into a constant number
of subclasses". The concrete mechanism exists ONLY in the reference code:
per class with an irregular partner: keep the LARGER of (cert, complement),
dump the smaller side wholesale into V0, trim the larger side to exactly
⌊c/2⌋ by degree (highest first); classes with NO partner are split in two
by degree; then V0 leftovers are re-chunked into NEW classes of the same
⌊c/2⌋ size (degree-ordered) so the partition stays equitable and |V0| < εn.
Note ~half the graph flows through V0 each refinement by construction.
Q: Is this faithful to "a constant number of subclasses"? Is the V0
re-chunking (which is also only in the reference code, not the paper)
justified, or should leftover vertices stay in V0?

A5. Certificate vs degree for the split.
Code splits by certificates (cert/complement) with degree only for trimming;
pure degree split only for classes with no irregular partner.
Q: Theoretically certificate splitting is what raises the Szemerédi index
([30]); degree splitting is the [16] heuristic. Is the hybrid correct?
With the complete-support degeneracy, certificate sizes are all-or-nothing
(hence splits are effectively balanced index-order) — does that matter?

A6. Condition 2 constant: 1/8 vs 1/16.
[16] and [28] TEXTS: "more than (1/8)ε⁴n vertices whose degrees deviate from
d by at least ε⁴n". The reference CODE and this repo: `≥ (1/16)ε⁴n` in ONE
direction (high side then low side). Resolution found: the theorem counts
deviants in BOTH directions (1/8 total); by pigeonhole at least half deviate
in one direction → 1/16 per direction is the operational form ([16] p.18
spells this out). Verified equivalent, but flag for review.

A7. d₀ (reduced-graph adjacency threshold) — NEVER VALUED, NOW SEARCHED.
§3.3: "two vertices are adjacent if the corresponding classes are ε-regular
with the edge density above a threshold d₀". Lemma 2 requires d₀ > ε.
The paper never gives d₀ a value. The repo defaulted to 0 (keep every Eq. 3
density) and NOW grid-searches config.D0_GRID = (0, 0.05, 0.1, 0.2, 0.3, 0.5)
in Exp 1/2/3 (recorded as columns d0 / enh_d0); Exp 2b fixes d₀ = 0 because
the fig. 11 ablation must hold every other part fixed. Also §3.3's adjacency
requires "ε-regular pairs" — code default uses ALL pairs
(`drop_edges_between_irregular_pairs=False`), justified by §3.3's own
softening "every pair of classes forms an approximately regular pair".
Q: Is searching d₀ a legitimate extension or a deviation? Should values
below ε be excluded (Lemma 2 requires d₀ > ε)? Is the all-pairs reading
acceptable?

A8. V0 → nearest cluster — METRIC UNSPECIFIED.
Algorithm 1 lines 25–27: "Assign p to the nearest cluster". §3.4 calls it
trivial and says V0 is small. Code: nearest = class with highest MEAN
similarity to p (ties → lowest label). [16] used "a predefined distance
measure" (unspecified).
Q: Is mean-similarity the right metric?

════════════════════════════════════════════════════════════════════════
B. PAPER-SILENT CONSTANTS (all documented; APC/DSet/SPRG ones NOT searched)
════════════════════════════════════════════════════════════════════════

B1. APC preference = median of POSITIVE off-diagonal similarities (Frey–Dueck
suggest median; Gaussian sims are all positive, so this ≈ plain median).
B2. DSet weight threshold = 1/(1.5·n) (Fiorucci lineage); alternatives known:
DSLib [27] uses 1e-5, Hou et al. PR 2023 [25] uses 0.0001. The paper text
just says "greater than a threshold".
B3. DSet leftover cutoff = 5% of points (Fiorucci lineage); "rel" modes exist
but unused.
B4. SPRG: φ (min samples per node) = 5 (paper [20] picks φ by cross-validation),
node-weighting variant "adpt" (ClustRF-Strct-Adpt, [20]'s best), bootstrap
subsets, seed 314, Tclust=1000, mtry=√d (both from [20]).
B5. Reg-SPRG on the reduced graph R: R has no feature vectors, so the forest
is grown on R's ROWS as k-dim profiles (φ=1). This is the repo's ONE
documented adaptation; the paper doesn't specify how SPRG runs on R.
B6. d̄ (similarity denominator): paper says "average of all pairwise
distances"; code averages the n(n−1) POSITIVE distances (excludes the zero
diagonal) — a ≈(n−1)/n factor, immaterial across the σ grid.
B7. NMI averaging = "arithmetic" (sklearn default). Paper doesn't specify.
ACC = Hungarian best-permutation (standard).
B8. k for SPC/SPRG = ground-truth number of clusters. Paper says "computes
the first k eigenvectors" without stating how k is chosen.
B9. k-means n_init=10, random_state=314; seeds fixed at 314 everywhere
(partition, refinement partner choice, SPRG forest).
B10. σ selection protocol: Exp 2/3 original algorithm = best σ by NMI;
enhanced = best over σ × recommended (ε,ϵ,b). Paper shows per-dataset bars
but never states the selection protocol.
B11. Exp 2b operating point (ε,b,ϵ)=(0.15,4,0.05) not named in the paper
("keep all the other parts unchanged" only).
B12. Exp 1 aggregation: mean NMI over all other parameters INCLUDING σ,
stop rule and d₀; paper says "mean over the combinations of the other two
parameters" (ε,ϵ,b only, σ unmentioned).
B13. SPRG has no σ slot (dummy None) so the experiment grid stays uniform.

════════════════════════════════════════════════════════════════════════
C. DEVIATIONS FROM THE REFERENCE CODE (all documented in spec/)
════════════════════════════════════════════════════════════════════════

C1. np.diag(M).copy(): reference's `rs_degrees = np.diag(mat)` returns a VIEW
(confirmed empirically), which the reference then corrupts in place with
`mat -= avg²/n`; the repo keeps true degrees. (Can change results vs the
reference in edge cases; differential tests still matched on tested inputs.)
C2. No-partner refinement branch sorts by GLOBAL degree; the reference sorts
by a stale/possibly-undefined `s_r_degs` (NameError or stale-data bug).
C3. Reference calls sys.exit("not enough nodes in C0...") on |V0| > εn; the
repo raises a catchable RuntimeError (needed for per-config experiment loops).
C4. Loop guard: float test ϵ > k/n (paper line 3) vs reference int cap
k >= int(ϵ·N).
C5. `random_refinement` (a no-op `pass` stub in the reference) is NOT ported.
C6. Post-loop refresh: if the loop exits right after a refinement, the repo
re-runs the pair check on the final partition so certificates/R match the
classes (the reference would use stale certificates in that case).
C7. Frieze–Kannan: reference has latent bugs (ambiguous truth value of the
svds array, sys.exit on no quadratic form); the repo guards and falls back.

════════════════════════════════════════════════════════════════════════
D. DATA / PIPELINE ASSUMPTIONS (verified shapes vs Table 1)
════════════════════════════════════════════════════════════════════════

D1. Leaves (1600×64, 100 classes): features = the zip's `data_Sha_64.txt`
(64-D shape descriptors). The paper never names which of the UCI views
(shape/margin/texture) ND=64 refers to.
D2. Dutchnumeral (2000×649, 10): UCI Multiple Features concatenation of all
six views (fac 216 + fou 76 + kar 64 + pix 240 + zer 47 + mor 6); labels
implied by row order (200 per digit). The provided `tunadromd.zip` is a
different dataset and was rejected.
D3. USPS (11000×256, 10): Roweis usps_all.mat; label order = C-order
flattening of (1100,10) (sample-major), validated by 87% NN label agreement.
D4. NO preprocessing/normalization anywhere (features as distributed).
D5. Landmine: extracted from a RAR via bsdtar (Normalized_Data sheet).
D6. Appendicitis/SCC: loaded via ucimlrepo (no local files).
D7. Synthetic stand-ins exist as last-resort fallback (documented; the paper
itself says RL enhancement does not work on synthetic data).

════════════════════════════════════════════════════════════════════════
E. EMPIRICAL DIVERGENCES vs THE PAPER (honest, documented in VERIFICATION.md)
════════════════════════════════════════════════════════════════════════

E1. Appendicitis: original algorithms already score NMI = 1.0; enhanced tops
out ~0.15 vs paper's Reg-* 0.82–0.91.
E2. SCC: same phenomenon (orig 1.0, enhanced ~0.11 vs paper 0.93+). (The doc
had only flagged Appendicitis; SCC is a second instance.)
E3. Leaves: Reg-SPC 0.76 vs paper 0.82 (still beats all 8 recent); Reg-APC/
Reg-DSet stuck at ~0.56–0.64 vs paper 0.75/0.79 across every setting tried,
including ϵ=0.2. Hypothesis: line-12 stop fires early (k settles at 64–133
while NC=100), refinement stops before R can represent all classes.
E4. Claim C4 (ε recommended range): on the current 5-dataset exp1 subset the
aggregate is nearly flat in ε with the max at ε=0.4 (0.337 vs 0.332 within
ε≤0.2) — paper says small ε best.
E5. Claim C6 (b≤16): on the same subset b>16 averages higher (0.408 vs 0.363)
— paper says b>16 doesn't help.
E6. Reg-SPRG consistently below its paper values on most datasets (the on-R
adaptation limitation; C3 passes only 2/2 datasets so far).
E7. Ecoli reproduces fully (Reg-SPC/APC/DSet at or above paper; all beat the
8 recent algorithms on all four metrics).

════════════════════════════════════════════════════════════════════════
F. EXTERNAL / UNIMPLEMENTED
════════════════════════════════════════════════════════════════════════

F1. The 8 recent algorithms (3W-DPET, DenMune, FSDPC, DPC-FSC, LDP-SC,
KSF-DPC, ICKDP, BP) are NOT implemented — their Tables 2–5 values are used as
fixed reference columns (transcription verified: 640/640 + 960/960 cells).
F2. DSLib [27] (open-source DSet library) not used — not pip-installable
(CMake C++), different default threshold (1e-5), and the paper only says the
library "is provided", never that its experiments used it. (Pelillo is a
co-author of both the paper and DSLib — plausible the authors did use it.)
F3. The paper's GitHub 404 — no author code to compare with; verification
relies on PDF + references + the Fiorucci reference implementation.
F4. `references/30.pdf` (Alon et al. 1994) is a scanned PDF with no text
layer — its constants were triangulated via [16], [17], [28] restatements.

════════════════════════════════════════════════════════════════════════
G. OPEN QUESTIONS I MOST WANT ANSWERS TO
════════════════════════════════════════════════════════════════════════

G1. Is Algorithm 1 line 12 (`n_ir < k(k−1)/2`) a typo for `n_ir ≤ ε·C(k,2)`?
Which one reproduces the paper's Tables 2–5 better?
G2. What threshold/support did the authors actually use to build the graph
from W? Does the complete-support degeneracy matter for the results?
G3. Is the V0 re-chunking + ⌊c/2⌋ halving refinement (reference mechanics,
not in the paper text) the right realization of "divide classes into a
constant number of subclasses"?
G4. Is grid-searching d₀ legitimate? Lemma 2 requires d₀ > ε — should values
below ε be excluded?
G5. Why do Leaves Reg-APC/DSet not reach the paper's values — is it the
line-12 early stop (k < NC), the DSet threshold, σ selection, or the
similarity/preprocessing assumptions?
G6. Would switching the DSet cutoff to DSLib's 1e-5 (or Hou 2023's 0.0001)
change Reg-DSet on the reduced graph materially?
G7. Does the mean-similarity V0 assignment (vs distance-based) matter?
G8. Which SPRG variant (adpt/unfm/bi) and φ would best reproduce the paper's
Reg-SPRG numbers?
```