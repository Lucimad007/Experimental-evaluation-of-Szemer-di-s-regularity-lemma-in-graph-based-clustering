# Spec: Algorithm 1 — Enhancement algorithm based on the regularity lemma (§3.4)

Sync status: implemented in `src/enhanced/algorithm1.py` (lines 18–27) and
`src/szemeredi/regularity_lemma.py` (lines 1–18).

## Algorithm 1 (verbatim from the paper)

```
Input:  original graph G = (V, E), regularity partitioning parameters ε, b, ϵ
Output: labels of n data l_i, i = 1, …, n

 1: Obtain V = V0 ∪ V1 ∪ … ∪ Vb, by dividing V arbitrarily.
 2: k1 ← b, i ← 1.
 3: while ϵ > k_i / n do
 4:     n_ir ← 0
 5:     for r = 1, …, k_i − 1 do
 6:         for s = r + 1, …, k_i do
 7:             if (V_r, V_s) is not verified as ε-regular then
 8:                 n_ir ← n_ir + 1
 9:             end if
10:         end for
11:     end for
12:     if n_ir < k_i(k_i − 1)/2 then
13:         Break
14:     else
15:         Refine P_i to obtain P_{i+1} with k_{i+1} classes, and i ← i + 1.
16:     end if
17: end while
18: Build the reduced graph R ∈ R^{k_i × k_i}, based on V = V0 ∪ V1 ∪ … ∪ V_{k_i}.
19: Perform graph-based clustering on R, obtaining labels L_1, …, L_{k_i}.
20: for j = 1, …, k_i do
21:     for p ∈ V_j do
22:         Assign the label of p as L_j.
23:     end for
24: end for
25: for p ∈ V0 do
26:     Assign p to the nearest cluster.
27: end for
```

## Loop condition and break (as written)

- **Line 3** is a float test: keep iterating while `ϵ > k_i / n`. The driver is
  `_line3_compressible` (`compression_rate > k / N`). This is modification (3)
  in §3.2 ("terminate when the class size is sufficiently small"). It is **not**
  `k >= int(ϵ·n)`: that integer cap stopped one refine too early and, via
  `max(max_k, b)`, entered the loop even when `b/n ≥ ϵ` (the paper never enters).
- **Lines 7–8** increment `n_ir` whenever the pair is **not verified as
  ε-regular**: an irregularity witness **or** no Alon condition decided the pair.
  Only a positive regularity verdict (Alon 1, or Alon 3 with an empty
  certificate — Fiorucci's "Y spurious") leaves `n_ir` unchanged.
- **Line 12** `n_ir < k(k−1)/2` is the experimental stop (no `ε` factor). The
  formal §3.2 Step 3 rule `n_ir ≤ ε·C(k,2)` is `stop_rule="theoretical"`.
- Modification (1) (at most one irregular **partner** per class) applies at
  **refinement** (line 15), not when counting `n_ir`. Counting still walks every
  pair, so `n_ir` can equal `C(k,2)` and the loop does refine.

If the while-guard fails after a refine, the last pair count belongs to the
pre-refine partition. Line 18 needs certificates for the **current** classes, so
the driver re-runs the pair check without changing the partition.

## "Nearest cluster" for V0

The paper states this is "trivial" and that `V0` is typically small so the method
has little influence. Each `p ∈ V0` is assigned to the class with highest average
similarity to `p` (ties broken by lowest label).

## Complexity (§3.4)

- Regularity partitioning: `O(n^2.376)` worst case (Theorem 1); the practical
  modifications reduce the constant but not the worst-case bound.
- Clustering on `R`: SPC `O(|R|³)`, APC `O(|R|² · n_iter)`, DSet `O(|R|² · n_c)`,
  SPRG `O(|R|³)`, where `|R|` is the reduced graph size.
- Since `n ≫ |R|`, partitioning can dominate for large datasets — acknowledged as
  the main drawback in §5.
