# Proof: Algorithm 1 is implemented

![Algorithm 1 from the paper](algorithm1_paper.png)

**Algorithm 1 — "Enhancement algorithm based on the regularity lemma"** (the
paper's §3.4 pseudocode, shown above) is implemented line-for-line. Every one of
its 27 lines maps to code.

## Line-by-line mapping

| Algorithm 1 (paper) | Implementation | Notes |
|---|---|---|
| **Input:** G=(V,E), parameters ε, b, ϵ | `src/enhanced/algorithm1.py` `enhance_clustering(…, epsilon, b, compression_rate)` | G is the similarity matrix; ε=`epsilon`, ϵ=`compression_rate` |
| **Output:** labels l₁…lₙ | `enhance_clustering` returns `labels` | |
| **1:** Obtain V = V0 ∪ V1 ∪ … ∪ Vb, dividing V arbitrarily | `src/szemeredi/partition_initialization.py` `degree_based` / `random` | any equitable split; degree-based is the default |
| **2:** k₁ ← b, i ← 1 | `partition_initialization.py` (`self.k = b`) | |
| **3:** while ϵ > kᵢ/n do | `regularity_lemma.py` `_line3_compressible`: `compression_rate > k/N` | float test, not `k >= int(ϵ n)` |
| **4:** n_ir ← 0 | `check_pairs_regularity` | |
| **5-6:** for r = 1…kᵢ−1, s = r+1…kᵢ | `check_pairs_regularity` (pairs `s < r`; same set) | |
| **7:** if (V_r, V_s) not verified as ε-regular | `conditions.py` alon1 / alon2 / alon3 | increment unless a positive regularity verdict |
| **8:** n_ir ← n_ir + 1 | `if not verified_regular: n_ir += 1` | irregular witness **or** undecided |
| **12:** if n_ir < kᵢ(kᵢ−1)/2 then Break | `check_partition_regularity(…, stop_rule="algorithm1")` | **no ε factor**; §3.2 Step 3 is `stop_rule="theoretical"` |
| **13:** Break | `break` in `run()` | |
| **15:** Refine Pᵢ → Pᵢ₊₁ with kᵢ₊₁ classes | `refinement_step.py` `degree_based` | ≤1 irregular partner per class (modification 1) |
| **17:** end while | `while self._line3_compressible(...)` | |
| **18:** Build the reduced graph R ∈ ℝ^{kᵢ×kᵢ} | `generate_reduced_sim_mat` | Eq. 3 weighted densities; V0 excluded |
| **19:** Perform graph-based clustering on R → L₁…L_k | `algorithm1.py` `base_algorithm(R, …)` | SPC / APC / DSet / SPRG in `src/runners.py` |
| **20-24:** for j, for p ∈ V_j: label(p) = L_j | `assign_from_reduced` | |
| **25-27:** for p ∈ V0: assign to nearest cluster | `_map_v0_to_nearest` | nearest = highest mean similarity to a class |

## Live verification

`python -m proof.run_proof` executes Algorithm 1 end-to-end on Wine and prints
the state of each construct.

## Machine checks

- Line 12 `n_ir < k_i(k_i−1)/2` was verified character-level against the paper PDF.
- Line 3 is the float guard `ϵ > k/n` (modification 3).
- `python -m src.smoke_test` exercises lines 1–27 end-to-end.
- Full audit: [`../IMPLEMENTATION_PROOF.md`](../IMPLEMENTATION_PROOF.md) §3.4.
