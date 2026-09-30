# Experimental evaluation of Szemerédi's regularity lemma in graph-based clustering

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A7C3A?logo=pytest&logoColor=white)](pytest.ini)
[![License: MIT](https://img.shields.io/badge/License-MIT-A33B24)](LICENSE)
[![Paper](https://img.shields.io/badge/Pattern%20Recognition-171%20%7C%20112205-1A334A)](https://doi.org/10.1016/j.patcog.2025.112205)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.patcog.2025.112205-6B645C)](https://doi.org/10.1016/j.patcog.2025.112205)

Independent Python implementation of **Algorithm 1** from Hou, Ge, Yuan, and Pelillo, *Pattern Recognition* **171** (2026), Art. 112205.

A similarity graph is expensive: memory grows like \(n^2\) and several graph clustering algorithms like \(n^3\). Szemerédi’s regularity lemma replaces most of that graph by a much smaller reduced graph. This repository builds that reduced graph, clusters it, and compares the result with clustering the original graph.

## Result on this grid

On **78 valid cells** (20 datasets × SPC, APC, DSet, SPRG; two Leaves cells empty), clustering the reduced graph versus the same algorithm on the original graph:

| Outcome | Cells |
|---|---:|
| Reduced graph better | 60 |
| Equal | 3 |
| Original graph better | 15 |

That comparison is against **this repository’s own baseline**, on the same similarity graph. It is not a copy of Hou’s printed tables. Ten datasets run at full \(n\). Ten with \(n>500\) use a stratified cap of about 400 vertices, so those cells are not comparable to the paper’s full-data numbers. See [`RESULTS.md`](RESULTS.md) and [`VERIFICATION.md`](VERIFICATION.md).

## Pipeline

```text
table  →  z-score  →  Gaussian similarity + kNN  →  graph G
                                                      ├─ original: cluster G
                                                      └─ enhanced: Alon partition → reduced graph R
                                                                    → cluster R → lift labels → V0 → polish
                                                      both scored with NMI against true labels
```

The enhanced branch clusters \(R\), not \(G\). Polish (Lloyd or k-means on the lifted labels) runs after Algorithm 1 and is not part of the lemma.

## Layout

| Path | Role |
|---|---|
| [`src/szemeredi/`](src/szemeredi/) | §3.1–3.3 partition and reduced graph |
| [`src/clustering/`](src/clustering/) | SPC (NJW), APC, DSet, SPRG |
| [`src/enhanced/`](src/enhanced/) | Algorithm 1, label lift, polish |
| [`src/datasets.py`](src/datasets.py) | Table 1 loaders (20 datasets) |
| [`src/experiments.py`](src/experiments.py) | Experiment runners |
| [`spec/`](spec/) | Paper section ↔ code |
| [`scripts/_lemma_all.py`](scripts/_lemma_all.py) | Honest grid behind the latest CSV |
| [`tests/`](tests/) | pytest |
| [`THESIS_FA.md`](THESIS_FA.md) | Persian BSc thesis |
| [`THESIS_FA_slides.pptx`](THESIS_FA_slides.pptx) | Defense slides |

`results/` is gitignored. Do not quote `results/support_artifact/` (the graph-blind `support` partitioner).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Hou’s 2026 runs use Marco Fiorucci’s lemma code, not a public clustering wrapper. The clone stays gitignored:

```powershell
git clone --depth 1 https://github.com/MarcoFiorucci/dense_graph_reducer.git .reference/dense_graph_reducer
python -m src.main smoke --alg-kind fiorucci --datasets Wine
```

`--alg-kind fiorucci` runs that repository’s Alon partition. It does not apply this repo’s `adj_threshold`, `stop_rule`, or spectral extras. The default partitioner is this repository’s Alon.

## Usage

```powershell
python -m src.smoke_test
python -m pytest tests
python -m src.main smoke
python -m src.main exp2 --profile honest --datasets Wine,Seeds,Ecoli --algorithms SPC,APC,DSet
```

`--profile honest` and `--profile paper` share the same graph-faithful defaults: weighted degree, theoretical stop, Alon, all-pairs \(R\), mean adjacency threshold. They do not use a complete Gaussian (\(\tau=0\)), which makes the Alon tests vacuous.

Full honest sweep (small sets at full \(n\); large sets capped):

```powershell
python scripts/_lemma_all.py
```

Grid in that script: \(\varepsilon\in\{0.1,0.2\}\), \(\epsilon\in\{0.05,0.1\}\), \(b\in\{8,16\}\), \(\sigma\in\{1,2,5\}\), kNN 20 or mutual 15, \(d_0\in\{0,\mathrm{mean}\}\), z-scored features.

Later sweeps, still graph-faithful and still clustering \(R\):

```powershell
python scripts/_lemma_higher.py    # finer partitions: epsilon up to 0.2, b up to 128, n up to 1200
python scripts/_refine_winners.py  # one-axis refinement from each winner
python scripts/_compare_best.py    # best.csv against the main grid
```

`_refine_winners.py` also searches axes the main grid never touched (graph metric, adjacency threshold, stop rule, ε-regular-only \(R\), \(d_0\) percentiles, Fiorucci’s partitioner, other kNN). The metric axis is paper-silent; see [`spec/similarity.md`](spec/similarity.md). `degree_mode="alon"` (index order) is excluded from every search: it is permutation-variant.

## What is implemented

- **Partition (§3.2).** Alon’s three tests, with Hou’s cap of one irregular partner per class, Fiorucci-style degree certificates, and a stop when classes are small enough. The partition is approximately regular. Frieze–Kannan is optional.
- **Reduced graph (§3.3).** Pair densities. \(d_0=0\) keeps every pair.
- **Algorithm 1 (§3.4).** Cluster \(R\), copy the label onto each class, assign \(V_0\) to the nearest cluster.
- **Similarity (§4).** \(s(x,y)=\exp(-d/(\bar d\cdot\sigma))\), optional z-score and kNN.
- **Metrics.** NMI, ACC (best label permutation), ARI, RI.
- **Datasets.** The 20 sets in Table 1 (local files, `ucimlrepo`, USPS `.mat`, mfeat, leaves).

External §4.3 methods are not re-run. Hou’s printed numbers live in [`src/baselines.py`](src/baselines.py) for comparison only.

## Citation

```bibtex
@article{hou2026szemeredi,
  author  = {Hou, Jian and Ge, Juntao and Yuan, Huaqiang and Pelillo, Marcello},
  title   = {Experimental evaluation of {Szemerédi's} regularity lemma in graph-based clustering},
  journal = {Pattern Recognition},
  volume  = {171},
  pages   = {112205},
  year    = {2026},
  doi     = {10.1016/j.patcog.2025.112205}
}
```

## License

[MIT](LICENSE).
