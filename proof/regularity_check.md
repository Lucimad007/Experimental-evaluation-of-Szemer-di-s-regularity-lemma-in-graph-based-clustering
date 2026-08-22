# Proof: "if (V_r, V_s) is not verified as ε-regular" is implemented

> Algorithm 1, line 7. This is the only black-box line of the pseudocode; the
> paper delegates its internals to three sources, all implemented.

## Where the paper (and references) specify it

| Source | What it says |
|---|---|
| Paper Definition 1 (§3.1) | the idealized criterion: (A,B) is ε-regular if for **every** X⊂A, Y⊂B with \|X\|>ε\|A\|, \|Y\|>ε\|B\|: \|d(X,Y)−d(A,B)\| < ε — uncheckable as stated |
| Paper §3.2, **Step 2** | the operational version: "verify it as a regular pair, or **show that it is not a regular pair by finding X ⊂ V_r and Y ⊂ V_s** with \|X\| ≥ (ε⁴/16)\|V_r\| and \|Y\| ≥ (ε⁴/16)\|V_s\|, such that **\|d(X,Y) − d(V_r,V_s)\| ≥ ε⁴**" |
| Alon et al. 1994 ([30], `references/30.pdf`) | the constructive lemma whose proof yields the three sufficient conditions and the ε³n / ε⁴n / ε⁴n/16 thresholds |
| Paper §3.2, **modification 2** | "we generate the certificates to check regularity of pairs with the **degree-based greedy method proposed in Fiorucci et al. [28]**" (`references/28.pdf`; code kept in `.reference/dense_graph_reducer`) |

## How the code implements it

`src/szemeredi/conditions.py` — three conditions tried in order by the driver
(`regularity_lemma.py:88-97`); the pair counts as "not verified as ε-regular"
(line 8 of Algorithm 1, counter at `regularity_lemma.py:93`) iff the deciding
condition produced a non-empty r-side witness:

| Condition | Verdict logic | Witnesses produced |
|---|---|---|
| `alon1` (`conditions.py:21`) | regular if the bipartite average degree < **ε³·n** (too sparse to misbehave) | — |
| `alon2` (`conditions.py:26`) | irregular if ≥ **(1/16)·ε⁴·n** vertices on one side deviate from the average degree by > **ε⁴·n** | X = all of V_r, Y = the deviating vertices |
| `alon3` (`conditions.py:53`, Fiorucci greedy) | neighbourhood-deviation matrix `BᵀB − d̄²/n` (`classes_pair.py:43`); greedy `Y` until σ(Y) ≥ **(ε³/2)·n** (`find_Y`); `Y′` = degrees within ε⁴·n of average (`find_Yp`); `y₀` = max deviation sum (`compute_y0`); s-certificate = Y′ vertices deviating from y₀ by > **2ε⁴·n** (`find_s_cert_and_compl`); r-certificate = V_r-neighbours of y₀ (`find_r_cert_and_compl`) | both non-empty ⇒ irregular |

## Live demo (runs the actual code)

A constructed bipartite pair (20+20 vertices, B split into dense/sparse
connectivity) — `alon2` fires with the deviating vertices as witnesses:

```text
alon1 (regular if sparse)    -> undecided for this pair |X|=0 |Y|=0
alon2 (deviation count)      -> IRREGULAR (witness found) |X|=20 |Y|=20
alon3 (greedy certificate)   -> (not reached: alon2 decided)

B-side degrees vs weighted bipartite average:
  all 20 (0/1 adjacency) vs avg 13.59 -> deviation 6.41 > eps^4*n = 0.032
  count 20 >= (1/16)eps^4*n = 0.002   -> irregular, witnesses = the 20 vertices
```

## The weighted-graph nuance (paper- and reference-faithful)

For Gaussian similarities every off-diagonal weight is positive, so the 0/1
adjacency degrees are all n−1 while the average degree is weighted — nearly
every vertex "deviates" and `alon2` certifies almost every pair irregular.
This is exactly what the Fiorucci reference computes (verified line-by-line)
and explains the paper's remark that its modifications yield "approximately,
but not provably, regular partitions", with the loop typically ending at the
ϵ·\|G\| cap (line 3) rather than the line-12 break.
