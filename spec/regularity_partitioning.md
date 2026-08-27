# Spec: Regularity partitioning (§3.1, §3.2)

Sync status: implemented in `src/szemeredi/`

## §3.1 Definitions (used by §3.2)

Given an undirected graph `G = (V, E)` without self-loops.

**Edge density** (Eq. 2): `d(A, B) = e(A, B) / (|A| |B|)`.

**Regular pair** (Definition 1). Disjoint `A, B ⊆ V` and `ε > 0`: `(A, B)` is
`ε`-regular if for every `X ⊆ A`, `Y ⊆ B` with `|X| > ε|A|`, `|Y| > ε|B|`,
`|d(X, Y) − d(A, B)| < ε`.

**Equitable partition.** `V = V0 ∪ V1 ∪ … ∪ Vk` with `|V1| = … = |Vk|`; `V0` is
the exceptional class (may be empty).

**Regular partition** (Definition 2). Equitable, `|V0| < ε|V|`, and all but at
most `ε k²` pairs `(Vi, Vj)` are `ε`-regular.

**Lemma 1** (regularity lemma). Existence of an `ε`-regular partition into
`k+1` classes with `m ≤ k ≤ M` for large enough graphs.

## §3.2 Partitioning the graph (verbatim)

The paper uses **Alon et al. [30]** in experiments (Frieze–Kannan [31] is said
to perform similarly). Theorem 1 is the algorithmic form: such a partition can
be found in `O(n^2.376)` sequential time. That bound is the theorem statement,
not a second implementation.

### Five steps of Alon et al. (paper text)

1. **Partition initialization.** Divide `V` arbitrarily into an equitable
   partition `P1` with `1 + b` classes `V0, V1, …, Vb`, with `|V1| = ⌊n/b⌋` and
   therefore `|V0| < b`. Denote `k1 = b`.
2. **Regularity checking.** For each pair `(Vr, Vs)` of `Pi`, `1 ≤ r < s ≤ ki`,
   verify it as a regular pair, or show that it is not by finding `X ⊂ Vr` and
   `Y ⊂ Vs` with `|X| ≥ (ε⁴/16)|Vr|` and `|Y| ≥ (ε⁴/16)|Vs|` such that
   `|d(X, Y) − d(Vr, Vs)| ≥ ε⁴`.
   *(The PDF typesets the second bound as `|Vr|` twice; Alon [30] uses `|Vs|`
   for `Y`. Classes are equitable, so `|Vr| = |Vs|` anyway.)*
3. If at most `ε · C(ki, 2)` pairs are not verified as regular, then `Pi` is a
   regular partition and the process stops.
4. Otherwise, divide current classes to obtain a new partition `P'` with
   `1 + ki · 4^{ki}` classes, with a refinement algorithm (e.g. [30]).
5. Let `k_{i+1} = ki · 4^{ki}`, `P_{i+1} = P'`, `i ← i+1`, go to Step 2.

### Three practical modifications (paper text)

> First, in order to avoid the exponential growth of classes, we limit the
> number of irregular pairs containing each class to be at most one in Step 2.
> … this modification allows to divide the classes into a constant number of
> subclasses in Step 4. Second, we generate the certificates to check
> regularity of pairs with the degree-based greedy method proposed in Fiorucci
> et al. [28]. Third, we terminate the iteration when the class size is
> sufficiently small. … Based on these modifications, we obtain approximately,
> but not provably, regular partitions in general.

Parameters: `ε` (regular pair), `b` (initial number of classes), maximum
compression ratio `ϵ = |R|/|G|`, with `⌊1/ϵ⌋` the minimum class size.

### How the code maps onto that

| Paper | Code | Notes |
|---|---|---|
| Step 1 | `partition_initialization.py` | leftover vertices stay in `V0`; `|V0| = n mod b < b` |
| Step 2 | `check_pairs_regularity` + `conditions.py` alon1/2/3 | scans **every** pair (needed for Algorithm 1's `n_ir`); witnesses from Alon/Fiorucci |
| Step 3 | `check_partition_regularity(..., stop_rule="theoretical")` | `n_ir ≤ ε · C(k,2)`. Experiments use Algorithm 1 line 12 instead (`"algorithm1"`, default) |
| Step 4 | `refinement_step.py` | **not** `k·4^k` — modification 1: at most one partner, cardinality halved |
| Step 5 | `while` in `regularity_lemma.py::run` | `i ← i+1` is the next loop pass |
| Mod 1 | `random.choice` of one irregular partner | applied at refinement; counting still walks all pairs |
| Mod 2 | `conditions.alon3` + `classes_pair.find_Y / Yp / y0` | Fiorucci greedy; Alon 1/2 are the other two sufficient conditions of [30] |
| Mod 3 | `_line3_compressible`: `ϵ > k/n` | min class size `⌊1/ϵ⌋` |

Original Alon `k_{i+1} = k_i · 4^{k_i}` is **intentionally not** implemented —
that is the exponential growth the paper's first modification exists to avoid.

## Implementation notes (Fiorucci [28])

- Condition 1 (regular): `bip_avg_deg < ε³ · n`.
- Condition 2 (irregular): ≥ `(1/16)·ε⁴·n` vertices whose degree deviates from
  the average by more than `ε⁴ · n`.
- Condition 3 (irregular): neighbourhood-deviation matrix, greedy `Y` until
  `σ(Y) ≥ (ε³/2)·n`, then `Y'`, `y0`, certificates at `2ε⁴·n`.
- On a class with no irregular partner, Fiorucci's reference sorts by a stale
  `s_r_degs`; we sort by global adjacency degree (the reference's own stated
  intent). `np.diag(M).copy()` keeps neighbourhood degrees uncorrupted when `M`
  is updated in place.
- Random seeds are fixed (`314`), matching the reference.

## The 0/1 support: why `adj = (sim_mat > 0.0)` (paper-silent, from the reference)

The paper never specifies how the weighted similarity matrix becomes the graph
Alon's algorithm operates on: §3.4 says "we calculate the pairwise similarity
matrix `W`, obtaining the original graph `G`" — i.e., the similarity matrix *is*
the (edge-weighted) graph, and no threshold is ever named. The 0/1 matrix used
by the regularity tests is the **support** of `W`:

```
adj_mat = (sim_mat > 0.0).astype(float)
```

This is inherited verbatim from Fiorucci et al.'s `dense_graph_reducer` (the
code the paper adopts for the greedy certificates; `(sim_mat > 0.0)` is a real
filter there because their inputs — 0/1 block matrices, image-similarity graphs —
have genuine sparsity). Sperotto–Pelillo [16] justify running the partitioning
on the support: "the algorithms are not influenced by edge-weights" — the
weights are consumed only by the reduced graph (Eq. 3).

**Degeneracy on dense similarities (documented, inherited, not a bug):** the
Gaussian kernel `exp(−d/(d̄σ))` is strictly positive, so the support of every
paper dataset is the **complete graph**. Consequences:

- `degrees = argsort(adj_mat.sum(0))` and the refinement's `s_r_degs` are all
  equal → the "degree-based" initialization/refinement ordering degenerates to
  vertex-index tie-breaking (deterministic but content-free).
- Certificates are all-or-nothing: `find_r_cert_and_compl` returns *all* of
  `Vr` (the complete-support "neighbourhood" of `y0`), so splits are driven by
  certificate sizes, not degree structure.
- The pipeline still works because the informative signal rides the **weighted**
  quantities: alon1's weighted average degree, the weighted residual in
  Frieze–Kannan, and above all the Eq. 3 weighted reduced graph `R`, from which
  the structure is recovered by clustering. The degenerate ordering only
  permutes vertices within balanced classes.

**The alternative the lineage specifies but the reference does not implement:**
Sperotto–Pelillo [16] Eq. 15 order elements by the *average weighted degree*
`awdeg_S(i) = (1/|S|) Σ_{j∈S} ω(i,j)` — this stays informative on dense
similarities. Fiorucci's code instead sorts by the 0/1 degree, which the repo
reproduces for fidelity. Any other adjacency threshold (kNN, `w > τ`) is a
paper-silent preprocessing choice and is **not** used; `drop_edges_between_irregular_pairs`
and `density_threshold` prune only the reduced graph, never the partition input.
