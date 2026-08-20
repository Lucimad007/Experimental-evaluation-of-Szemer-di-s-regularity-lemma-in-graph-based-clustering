# Spec: Regularity partitioning (§3.1, §3.2)

Sync status: implemented in `src/szemeredi/`

## Definitions (§3.1)

Given an undirected graph `G = (V, E)` without self-loops.

**Edge density** between `A ⊆ V` and `B ⊆ V` (Eq. 2):

```
d(A, B) = e(A, B) / (|A| |B|)
```

where `e(A, B)` is the number of edges with one endpoint in `A` and one in `B`.

**Regular pair** (Definition 1). Disjoint `A, B ⊆ V` and constant `ε > 0`: the pair
`(A, B)` is `ε`-regular if for every `X ⊆ A` with `|X| > ε|A|` and `Y ⊆ B` with
`|Y| > ε|B|`,

```
|d(X, Y) − d(A, B)| < ε
```

**Equitable partition**. `V = V0 ∪ V1 ∪ … ∪ Vk` with `|V1| = |V2| = … = |Vk|`; `V0`
is the exceptional class (may be empty).

**Regular partition** (Definition 2). An equitable partition is an `ε`-regular
partition if `|V0| < ε|V|` and all but at most `ε * C(k,2)` pairs `(Vi, Vj)` are
`ε`-regular.

**Regularity Lemma** (Lemma 1). For any `m ≥ 1` and `ε > 0` there exist `N(ε,m)` and
`M(ε,m)` such that every graph with `|V| > N` has an `ε`-regular partition into `k+1`
classes with `m ≤ k ≤ M`.

## Algorithm (§3.2, Alon et al. with modifications)

Parameters: `ε` (regularity), `b` (initial number of classes), `ϵ` (max compression
ratio `|R|/|G|`; `⌊1/ϵ⌋` is the minimum class size).

1. **Partition initialization.** Divide `V` arbitrarily into an equitable partition
   `P1` with `1 + b` classes `V0, V1, …, Vb`, with `|V1| = ⌊n/b⌋` and therefore
   `|V0| < b`. Set `k1 = b`.
2. **Regularity checking.** For each pair `(Vr, Vs)` of `Pi`, `1 ≤ r < s ≤ ki`,
   verify it as a regular pair, or show it is not regular by finding
   `X ⊂ Vr` and `Y ⊂ Vs` with `|X| ≥ (ε⁴/16)|Vr|` and `|Y| ≥ (ε⁴/16)|Vs|` such that
   `|d(X,Y) − d(Vr,Vs)| ≥ ε⁴`.
3. If at most `ε * C(ki,2)` pairs are not verified as regular, `Pi` is a regular
   partition — stop.
4. Otherwise, refine the current classes into a new partition `P'` with
   `1 + ki·4·ki` classes using a refinement algorithm (Alon et al.).
5. Set `ki+1 = ki·4·ki`, `Pi+1 = P'`, `i ← i+1`, go to Step 2.

### Practical modifications used in this paper (§3.2)

1. **Limit irregular pairs per class to at most one** in Step 2 (instead of finding
   all irregular pairs). This keeps the number of subclasses produced in Step 4
   bounded by a constant.
2. **Degree-based greedy certificates** (Fiorucci et al. 2020) to generate the
   witnesses `X, Y` that show a pair is not regular.
3. **Terminate when the class size is sufficiently small** (i.e. when
   `k > ϵ·|G|`), to avoid the very large number of iterations needed for a provable
   regular partition. The result is an *approximately*, not provably, regular
   partition.

## Implementation notes

- The Alon 3-condition regularity check and the degree-based greedy certificate
  construction follow Fiorucci et al.'s `dense_graph_reducer`:
  - Condition 1 (regular): `bip_avg_deg < ε³ · n`.
  - Condition 2 (irregular): vertices in `s` whose degree deviates from the average
    by more than `ε⁴ · n`; irregular if at least `(1/16)·ε⁴·n` such vertices in one
    direction.
  - Condition 3 (irregular): greedy certificate built from the neighbourhood
    deviation matrix, with the `Y / Y' / y0` construction; threshold
    `(ε³/2)·n` on `σ(Y)`.
- The reduced graph uses the **weighted** edge density (Eq. 3) when the input is a
  similarity matrix; see `reduced_graph.md`.
- Random seeds are fixed (`314`) for reproducibility, matching the reference.
