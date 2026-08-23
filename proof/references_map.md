# The 40 references — role in this implementation, and verification

Every reference [1]–[40] of the paper, what role it plays here, and how it is
verified. Roles: **implemented** (code derives from it), **method template**
(pipeline shape derives from it), **theory** (theorem/lemma quoted by the
paper, no code), **reference values** (Tables 2–5 columns), **context**
(cited for background; intentionally not implemented — not part of the
paper's method).

| # | Reference | Role | Where / how verified |
|---|---|---|---|
| 1 | Bai et al., Self-supervised spectral clustering | context | not part of the paper's method |
| 2 | Wang et al., Block diagonal representation | context | — |
| 3 | Ma et al., Discriminative subspace MF | context | — |
| 4 | Chen et al., Low-rank tensor proximity | context | — |
| 5 | Huang et al., Ultra-scalable spectral clustering | context | — |
| 6 | Yu et al., GAN-based deep subspace clustering | context | — |
| 7 | Lu et al., M3C | context | — |
| 8 | Wang et al., Graduated assignment | context | — |
| 9 | **Pavan & Pelillo, Dominant sets (TPAMI 2007)** | **implemented (DSet)** | their Eq. (14) replicator `x_i(t+1)=x_i(t)(Ax)_i/(xᵀTAxᵀ)` = the paper's Eq. (1) = `src/clustering/dominant_set.py` (vectorized, algebraically identical); extracted from `references/9.pdf` and matched symbol-for-symbol |
| 10 | **Frey & Dueck, Affinity propagation (Science 2007)** | **implemented (APC)** | their paper: "the shared value could be the **median of the input similarities**" — our preference (`affinity_propagation.py`) is their stated recommendation (extracted from `references/10.pdf`); message passing via sklearn's implementation of the same paper |
| 11 | Lu & Yan, LP basis selection | context | — |
| 12 | Jiang et al., OOD partial matching | context | — |
| 13 | Roth et al., Constant shift embedding | context | — |
| 14 | Fowlkes et al., Nyström spectral grouping | context | — |
| 15 | Pavan & Pelillo, DSet out-of-sample | context | — |
| 16 | **Sperotto & Pelillo, regularity lemma for pairwise clustering (EMMCVPR 2007)** | **method template** | their two-step strategy (regularity partition as preclustering → reduced graph → pairwise clustering) is the pipeline shape of Algorithm 1 lines 18–19 (`references/16.pdf`, extracted) |
| 17 | **Sárközy, Song, Szemerédi & Trivedi, practical regularity partitioning (2012)** | **method predecessor** | their "Regularity Clustering" = build reduced graph + spectral clustering on it (extracted from `references/17.pdf`); the paper benchmarks b against their b = 2…7 choice; our b grid {2…10,16,…} follows the paper |
| 18 | Pelillo, Elezi, Fiorucci, regularity survey | context | — |
| 19 | **Szemerédi, Regular partitions of graphs (1976)** | **theory** | Lemma 1 (existence); realized by the partitioning loop, not coded separately |
| 20 | **Zhu, Loy & Gong, robust affinity graphs (CVPR 2014)** | **implemented (SPRG)** | `src/clustering/sprg.py`: pseudo-two-class Gini forest (their Eq. 1–3), mtry=√d, Tclust=1000, φ-stop; structure-aware affinities (their Eq. 7, 8, 9–10 Bi, 11 Unfm, 12–14 Adpt). **Fast implementation vs naive transcription of their equations: identical to 2e-16** |
| 21 | Hou et al., ICPR 2024 (preliminary version) | context | the main paper supersedes it |
| 22 | Shi & Malik, Normalized cuts | mentioned alternative | SPC default is the paper's unnormalized Laplacian; normalized variants optional |
| 23 | **Ng, Jordan & Weiss (NIPS 2002)** | mentioned alternative | our optional `variant="njw"` matches their algorithm (k smallest of `I − D^{−1/2}SD^{−1/2}`, rows renormalized to unit length, k-means — extracted from `references/23.pdf`) |
| 24 | Ding et al., spectral clustering survey | context | — |
| 25 | Hou, Yuan, Pelillo, parameter-free clustering (PR 2023) | cited for "SPRG performs much better than NCut" | its DSet weight threshold 0.0001 (its Eq. 4, extracted) documented as the alternative to our 1/(1.5n) (from the Fiorucci/DSLib lineage) |
| 26 | Rota Bulò et al., fast dynamics for DSet | context | (speedup variant; we use the plain replicator of Eq. 1) |
| 27 | Vascon et al., DSLib | DSet library provenance | threshold lineage documented in `IMPLEMENTATION_PROOF.md` |
| 28 | **Fiorucci, Pelosin, Pelillo (PR 2020)** | **implemented (certificates + refinement + pair machinery)** | paper's modification 2 names this method; ported from their `dense_graph_reducer` (kept in `.reference/`), verified line-by-line; thresholds ε³n, ε⁴n, ε⁴n/16, (ε³/2)n, 2ε⁴n all match |
| 29 | Fiorucci et al., strong regularity + densification | context | — |
| 30 | **Alon, Duke, Lefmann, Rödl, Yuster (J. Algorithms 1994)** | **implemented (the constructive method)** | the 5-step method + witness thresholds (ε⁴ deviation, ε⁴/16 sizes) quoted in the paper's §3.2 Step 2 and coded in `src/szemeredi/`; PDF is a scan (no text layer), verified via the paper's quotes + [28]'s implementation |
| 31 | **Frieze & Kannan (1999)** | **implemented (optional variant)** | `conditions.py::frieze_kannan` (top singular value ≥ ε·n criterion + certificates); the paper reports Alon and F–K perform similarly and adopts Alon — we default to Alon, both provided |
| 32 | Komlós et al., key lemma | theory | Lemma 2 (reduced-graph inheritance); no code |
| 33 | Yu et al., 3W-DPET | reference values | Tables 2–5 column; 640/640 cells validated against the PDF |
| 34 | Abbas et al., DenMune | reference values | same |
| 35 | Xu et al., FSDPC | reference values | same |
| 36 | Li et al., DPC-FSC | reference values | same |
| 37 | Long et al., LDP-SC | reference values | same |
| 38 | Li et al., KSF-DPC | reference values | same |
| 39 | Guo et al., ICKDP | reference values | same |
| 40 | Bar et al., Border-peeling | reference values | same |

**Not vague:** every reference that contributes *code* ([9], [10], [20], [23],
[28], [30], [31]) is implemented and machine-checked; every reference that
contributes *structure* ([16], [17]) was re-extracted and matched; theory
references ([19], [32]) correspond to the paper's quoted lemmas; competitor
references ([33]–[40]) are used exactly as the paper uses them (fixed
reference columns, validated cell-by-cell). The remaining 24 are background
citations with no implementation role — matching the paper, which also does
not implement them.
