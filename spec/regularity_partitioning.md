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
