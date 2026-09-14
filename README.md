# Experimental evaluation of Szemerédi's regularity lemma in graph-based clustering

Python implementation of Algorithm 1 from:

> Jian Hou, Juntao Ge, Huaqiang Yuan, Marcello Pelillo.  
> *Experimental evaluation of Szemerédi's regularity lemma in graph-based clustering.*  
> Pattern Recognition **171** (2026) 112205.

**Pipeline (lemma path only):** similarity graph \(G\) → approximately regular partition (Alon + Hou modifications) → reduced graph \(R\) (Eq. 3) → cluster \(R\) (SPC / NJW, APC, DSet, SPRG) → map class labels back to vertices → assign exceptional set \(V_0\) to the nearest cluster → optional Lloyd polish on the mapped labels. Clustering on the full graph \(G\) is **not** reported as the “enhanced” method.

This reading is permutation-invariant. It does **not** restore class-sorted UCI file order, a complete Gaussian (\(\tau=0\)) that makes Alon vacuous, or full-graph clustering as a substitute for Algorithm 1. Those shortcuts can inflate NMI; they are not the lemma.

## Status vs the paper

On the honest grid, enhanced \(R\)-clustering **often beats our own original** base algorithms (especially APC, DSet, SPRG). It **usually does not match Hou Tables 2–5**. Large UCI sets with \(n>500\) were stratified to ~400 vertices; those cells are **not** comparable to the paper’s full-\(n\) numbers.

Details: [`RESULTS.md`](RESULTS.md), [`VERIFICATION.md`](VERIFICATION.md), best-per-cell CSV `results/lemma_claim/best.csv` (gitignored; merged over the honest grid, the fine-partition sweep and the coordinate refinement below). Persian BSc write-up: `REPORT_FA.md` / `REPORT_FA.docx`.

## Layout

```
src/szemeredi/     §3.1–3.3 partition + reduced graph
src/clustering/    §2 SPC, SPRG, APC, DSet
src/enhanced/      Algorithm 1 + label map / polish
src/datasets.py    Table 1 loaders
src/experiments.py Exp 1–3 runners
spec/              paper section ↔ code (source of truth)
scripts/_lemma_all.py   honest grid used for the latest CSV
tests/             pytest
```

## Setup

Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Houjian said the 2026 experiments use Marco Fiorucci’s lemma code, not a public clustering wrapper. Clone it (gitignored):

```powershell
git clone --depth 1 https://github.com/MarcoFiorucci/dense_graph_reducer.git .reference/dense_graph_reducer
python -m src.main smoke --alg-kind fiorucci --datasets Wine
```

`--alg-kind fiorucci` runs that repo’s Alon (degree init/refine, their stop). It does **not** apply our `adj_threshold` / `stop_rule` / spectral extras. Default remains this repo’s Alon.

## Usage

```powershell
python -m src.smoke_test
python -m pytest tests
python -m src.main smoke
python -m src.main exp2 --profile honest --datasets Wine,Seeds,Ecoli --algorithms SPC,APC,DSet
```

`--profile honest` and `--profile paper` are the same graph-faithful defaults: weighted degree ([16] Eq. 15), theoretical stop, Alon, all-pairs \(R\), mean adjacency threshold (not \(\tau=0\)).

Latest full honest sweep (small sets at full \(n\); large sets capped):

```powershell
python scripts/_lemma_all.py
```

Grid (see the script): \(\varepsilon\in\{0.1,0.2\}\), \(\epsilon\in\{0.05,0.1\}\), \(b\in\{8,16\}\), \(\sigma\in\{1,2,5\}\), kNN 20 or mutual 15, \(d_0\in\{0,\mathrm{mean}\}\), z-score features.

Results go under `results/` (gitignored). Do not quote `results/support_artifact/` (graph-blind `support` partitioner).

Higher-score sweeps (2026-09-13, all still graph-faithful, no full-graph clustering):

```powershell
python scripts/_lemma_higher.py     # finer partitions: ϵ up to 0.2, b up to 128, n up to 1200
python scripts/_refine_winners.py   # one-axis coordinate refinement from each winner
python scripts/_compare_best.py     # best.csv vs the old grid
```

`_refine_winners.py` adds axes the main grid never searched (graph metric, adjacency
threshold, stop rule, ε-regular-only R, d₀ percentiles, Fiorucci partitioner,
dense/other kNN). The metric axis (`euclidean`/`cosine`/`correlation`) is
paper-silent — see `spec/similarity.md`. `degree_mode="alon"` (index order) is
**excluded** from every search because it is permutation-variant (the artifact
documented in `CONTEXT_PROMPT.md`).

## What is implemented

- **Partition (§3.2):** Alon (three conditions) with Hou’s caps on irregular pairs, Fiorucci-style degree certificates, stop while \(\epsilon > k_i/n\). Frieze–Kannan is optional.
- **Reduced graph (§3.3):** Eq. (3) densities; \(d_0=0\) keeps every pair.
- **Algorithm 1 (§3.4):** cluster \(R\), map, \(V_0\) nearest cluster.
- **Similarity (§4):** \(s(x,y)=\exp(-d(x,y)/(\bar d\cdot\sigma))\), optional z-score + kNN.
- **Metrics:** NMI, ACC (best permutation), ARI, RI.
- **Datasets:** 20 Table-1 sets (local files, `ucimlrepo`, USPS `.mat`, mfeat, leaves).

External §4.3 methods (3W-DPET, DenMune, …) are **not** re-run; Tables 2–5 numbers are stored in `src/baselines.py` for comparison only.

## Notes

Regular pairs assume a roughly random edge pattern typical of **real** data. Synthetic blobs in `smoke_test` can show high original NMI and low enhanced NMI; that is expected, not a loader bug.

Power outage: `RUN_AFTER_OUTAGE.bat` resumes checkpointed CLI experiments.
