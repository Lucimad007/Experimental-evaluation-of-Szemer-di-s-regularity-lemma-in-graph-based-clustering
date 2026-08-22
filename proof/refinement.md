# Proof: the refinement step (Algorithm 1 line 15) is implemented

> Algorithm 1, line 15: "**Refine Pᵢ to obtain Pᵢ₊₁ with kᵢ₊₁ classes**, and i ← i + 1."

## Where the paper (and references) specify it

| Source | What it says |
|---|---|
| Paper §3.2, Step 4 | "divide current classes to obtain a new partition P′ with 1 + kᵢ·4^{kᵢ} classes, with a refinement algorithm, e.g., [30]" |
| Paper §3.2, Step 5 | "Let k_{i+1} = kᵢ·4^{kᵢ}, P_{i+1} = P′, i ← i + 1, and go to Step 2" |
| Paper §3.2, **modification 1** | "to avoid the exponential growth of classes, **we limit the number of irregular pairs containing each class to at most one** in Step 2 … this allows to divide the classes into a **constant number of subclasses** in Step 4" |
| Alon et al. 1994 ([30], `references/30.pdf`) | the constructive refinement: split each class by the witnesses of its irregular pairs (the witness X separates two vertex groups of different density toward the partner class) |
| Fiorucci et al. 2020 ([28], `references/28.pdf`) | the concrete degree-based operationalization the paper adopts (modification 2); ported from `dense_graph_reducer` in `.reference/` |

## How the code implements it

`src/szemeredi/refinement_step.py:33` (`degree_based`), invoked from
`regularity_lemma.py:213` when line 12 does not break:

1. **Halve the class cardinality** — each class splits into 2 subclasses (the
   "constant number" of modification 1; k roughly doubles per iteration,
   instead of Alon's k·4^k explosion).
2. **Classes with an irregular partner** (`refinement_step.py:46-72`): one
   partner is chosen (modification 1's "at most one"; seeded choice identical
   to the reference code), and each side is split by **certificate vs.
   complement** of the irregular pair: the lesser set → V0; the greater set is
   trimmed to the new cardinality by dropping its lowest s–r-degree vertices
   (excess → V0).
3. **Classes without an irregular partner** (`refinement_step.py:73-82`): split
   in two by (global) degree order so the partition stays equitable.
4. **V0 regrouping** (`refinement_step.py:84-91`): leftover V0 vertices are
   packed into new full-size classes; `|V0| > ε·n` raises (Definition 2
   requires `|V0| < ε|V|`).
5. **i ← i+1** is the loop iteration itself: the partition state advances and
   Algorithm 1 re-tests line 3's condition `ϵ > kᵢ/n`.

## Live verification (runs the actual code)

`python -m proof.run_refinement` — one full lines-4→15 pass on Wine:

```text
before refinement: k=4, class size=44, |V0|=2
pair check: n_ir=6 of C(k,2)=6 pairs -> line 12 (6 < 6?) -> line 15: Refine
class 1 has irregular partners [2, 3, 4]
witness driving the split: |X_r|=44 (r-side cert), |complement|=0; |Y_s|=44
after  refinement: k=8 (doubled), class size=22 (halved), |V0|=2 (deficiency, must stay < eps*n=27)
non-empty class sizes: [22, 22, 22, 22, 22, 22, 22, 22]
```

k doubles (not k·4^k — modification 1), the partition stays equitable
(all classes exactly 22), and |V0| stays below the Definition-2 bound.
