# Spec: Algorithm 1 — Enhancement algorithm based on the regularity lemma (§3.4)

Sync status: implemented in `src/enhanced/algorithm1.py`

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

## Notes on the loop condition and break

- The outer `while ϵ > k_i / n` enforces the maximum compression ratio: stop when
  the number of classes exceeds `ϵ·n` (i.e. classes would become smaller than
  `⌊1/ϵ⌋`). This is modification (3) in §3.2.
- The inner `if n_ir < k_i(k_i−1)/2` (i.e. at least one pair is regular) breaks the
  loop — the partition is accepted as approximately regular. With modification (1)
  (at most one irregular pair per class) the count `n_ir` is bounded.
- After the loop, the reduced graph `R` of size `k_i × k_i` is built (Step 18), the
  chosen base clustering algorithm is run on `R` (Step 19), and the labels are
  mapped back: every vertex in `V_j` gets label `L_j` (Steps 20–24). Vertices in the
  exceptional class `V0` are assigned to the nearest cluster (Steps 25–27).

## "Nearest cluster" for V0

The paper states this is "trivial" and that `V0` is typically small so the method has
little influence. We assign each `p ∈ V0` to the cluster whose representative (mean
of original feature vectors of members, or the class with highest average similarity
to `p`) is closest. The implementation uses the average similarity of `p` to each
class's members and picks the argmax (ties broken by lowest label).

## Complexity (§3.4)

- Regularity partitioning: `O(n^2.376)` worst case (Theorem 1); the practical
  modifications reduce the constant but not the worst-case bound.
- Clustering on `R`: SPC `O(|R|³)`, APC `O(|R|² · n_iter)`, DSet `O(|R|² · n_c)`,
  SPRG `O(|R|³)`, where `|R|` is the reduced graph size.
- Since `n ≫ |R|`, partitioning can dominate for large datasets — acknowledged as the
  main drawback in §5.
