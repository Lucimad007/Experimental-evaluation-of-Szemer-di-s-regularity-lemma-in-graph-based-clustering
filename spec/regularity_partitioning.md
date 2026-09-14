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
| Step 2 | `check_pairs_regularity` + `conditions.py` alon1/2/3 | Alon always on the **0/1 support** (`ClassesPair`), never on Gaussian weights. Eq. 3 is used only when building `R` |
| Step 3 | `check_partition_regularity(..., stop_rule="theoretical")` | default: `n_ir ≤ ε · C(k,2)` ([16]/[21]/[28]). `"algorithm1"` is the printed line 12 (no ε) |
| Step 4 | `refinement_step.alon_based` | Alon [30] cert vs complement, **modification 1**: at most one irregular partner so each class yields a **constant** number of subclasses (2), not \(k\cdot 4^k\) |
| Step 5 | `while` in `regularity_lemma.py::run` | `i ← i+1` is the next loop pass |
| Mod 1 | `random.choice` of one irregular partner | applied at refinement; counting still walks all pairs |
| Mod 2 | `conditions.alon3` first among irregularity tests | Fiorucci [28] greedy cert; `alon2` is a [30] fallback |
| Mod 3 | `_line3_compressible`: `ϵ > k/n` | min class size `⌊1/ϵ⌋` |

Original Alon `k_{i+1} = k_i · 4^{k_i}` is **intentionally not** implemented —
that is the exponential growth the paper's first modification exists to avoid.

## Implementation notes (Fiorucci [28])

- Condition 1 (regular): `bip_avg_deg < ε³ · n`.
- Condition 2 (irregular): more than `(1/8)·ε⁴·n` vertices (both directions)
  whose degree deviates from the average by more than `ε⁴ · n`; certificate
  `A' = Vr`, `B'` = the larger one-sided set. Degrees are Vs column sums of
  the same matrix as `bip_avg_deg`.
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

### Degeneracy on dense similarities — MEASURED, and it is fatal

The Gaussian kernel `exp(−d/(d̄σ))` is strictly positive, so the support of every
paper dataset is the **complete graph** (measured on Ecoli: off-diagonal support
density `1.000000`, every degree exactly `335`). The consequences are not
cosmetic — they disable the partitioner completely:

1. `vs_degrees()` returns the column sums of the **same** matrix as
   `bip_avg_deg`. The old helper compared **unweighted** degrees (constant, =
   class size `n`) to the **weighted** mean `n·d̄w`. On Ecoli that was `41.4`
   against `0.0425`, so every vertex "deviated".
2. Therefore **the old `alon2` certified every pair irregular**, with certificate
   = the whole class and empty complement. Measured across 3 datasets × 6 σ
   values, every pair was decided by `alon1` or `alon2` and **`alon3` never
   executed** — so the paper's modification 2 (Fiorucci's degree-based greedy
   certificate) was dead code. After 2026-08-30, unweighted complete pairs no
   longer fire `alon2` (all degrees equal the mean). Weighted Eq. 3 pairs can
   still fire because `ε⁴n` is small.
3. `n_ir` is therefore always `C(k,2)`, so **neither stop rule ever fires** and the
   loop always exits on the `ϵ` compression guard. This makes the Algorithm 1
   line 12 vs §3.2 Step 3 question (the suspected typo) moot.
4. In refinement, `greater_set` is the whole class and `s_r_degs` are all equal, so
   the "sort by degree" is a stable sort on constant keys: the class is cut into
   "first half by vertex index". The `V0` re-chunk sorts by `argsort` of a constant
   vector, which is again index order.
5. Net effect: **the partition is a deterministic chunking of the vertex index
   order and never reads the graph.** Real UCI files are sorted by class label, so
   the classes come out class-pure for free.

**Acceptance test.** A graph algorithm must be invariant to vertex relabelling.
Permuting the rows of `X` (which leaves the similarity graph unchanged up to
relabelling) collapses the result under `degree_mode="support"`:

| dataset | algo | original | enhanced, file order | enhanced, rows permuted |
|---------|------|----------|----------------------|-------------------------|
| Ecoli | SPC  | 0.441 | 0.799 | 0.095 |
| Ecoli | APC  | 0.500 | 0.844 | 0.103 |
| Ecoli | DSet | 0.514 | 0.751 | 0.109 |
| Seeds | APC  | 0.446 | 0.867 | 0.084 |
| Wine  | APC  | 0.314 | 0.728 | 0.072 |

(best NMI over σ × ε × ϵ × b, i.e. the paper's own protocol). Every reported gain
is an artifact of row order. A second symptom: under `"support"` the parameter `b`
has *no* effect on the result (identical NMI to six decimals for b = 2, 4, 8),
so Exp 1's "influence of b" curve measures nothing.

### `degree_mode`: the two readings

Alon conditions always run on the 0/1 support. `degree_mode` only orders
vertices for init / refine. On `K_n` (the Gaussian kernel) every equitable
partition is already ε-regular, so Alon would stop at `k = b` and ignore `ϵ`.
For `"weighted"` / `"spectral"` the driver then keeps splitting by awdeg /
Fiedler until the compression guard — Sperotto's practical size threshold,
not a fake irregularity certificate. `"support"` still stops when Alon says
regular (index-order init, `k = b` on `K_n`).

- **`"weighted"`** (default): Sperotto & Pelillo [16] Eq. 15, the *average weighted degree*

  ```
  awdeg_S(i) = (1/|S|) Σ_{j∈S} ω(i,j),   S ⊆ V
  ```

  [16] p. 22: "All elements in the current subset S are listed in decreasing
  order by average weighted degree. In so doing, the partition of S takes place
  simply by subdividing the ordered sequence of elements into the desired number
  of subsets. The decreasing order has been preferred because of the presence of
  the exceptional set: in this way we assume that only the less connected
  vertices join the exceptional set. Hence, the obtained regular partition
  contains classes the elements of which can already be considered similar to
  each other."

  This is implemented by `refinement_step.awdeg_based`: each class is subdivided
  into 2 subsets (modification 1's "constant number of subclasses") along its
  awdeg-ordered sequence, and the least-connected tail joins `V0`. Note `awdeg`
  is **intra-class** (`S` = the class being split), which is what makes classes
  progressively homogeneous. This mode is permutation-invariant.

- **`"support"`** (ablation): Fiorucci [28] 0/1 degree of `sim_mat > 0`. On
  `K_n` every degree is equal, so init is vertex index order and Alon stops at
  `k = b`. Kept to compare against the reference code, not used by default.

  Two remaining deviations from [16] in this mode, both documented. First, [16]
  uses Eq. 15 in the **refinement only** — its Step 1 is "Arbitrarily divide
  the set V into an equitable partition P1" — whereas we also order the initial
  partition by `awdeg` over `S = V`, so that `b` selects coherent seed classes
  instead of index blocks. Second, [16] lets `V0` only accumulate, which we now
  do: `apply_v0_guard` never mints new classes from the tail. If `|C0| > εn`
  and `|C0| > |P|`, C0 is distributed into existing classes ([28] Algorithm 2);
  if `|C0| ≤ εn` it stays in V0.

If a refinement does not increase `k` (typical for `degree_based` when every
pair is irregular and C0 is poured back into the same labels),
`regularity_lemma.run` stops. Otherwise the loop would never hit the `ϵ > k/n`
guard.

  The "constant number of subclasses" being **2** is not a guess: [28] §4 always
  produces exactly two subclasses per class, "filled up to `|C_i|/2`", and [21]
  (the authors' own earlier version) adopts [28]'s method by name. [17] is the
  outlier, splitting into `l ∈ {3,4}` with `m = ⌊|V_i|/l⌋`.

Note that [16] also states "the algorithms are not influenced by edge-weights",
which is what licenses running the Alon *conditions* on the support. Conditions
always see the 0/1 pair (`ClassesPair`). Eq. 3 weights only `R`. In `"weighted"`
mode the split itself is Eq. 15 rather than certificates.

Any other adjacency threshold (kNN, `w > τ`) is a paper-silent preprocessing
choice and is **not** used; `drop_edges_between_irregular_pairs` and
`density_threshold` prune only the reduced graph, never the partition input.
