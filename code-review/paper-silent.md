# Paper-silent constants (not permutation axes)

These exist in the code and are documented so they are not mistaken for
forgotten variants. They are **not** crossed in `--clustering-variants all`.

| Item | Value | Why it is not a grid |
|------|-------|----------------------|
| SPRG \(\varphi\) (min leaf) | 5 on \(G\), 1 on \(R\) | [20] selects \(\varphi\) by CV; we do not invent a \(\varphi\) grid |
| SPRG `Tclust` | 1000 | [20] Sec. 4; reduce only for smoke |
| SPRG `mtry` | \(\sqrt{d}\) | [20] Sec. 4 |
| APC `max_iter` | 500 | [10] does not name a cap; sklearn default is 200 |
| APC damping | 0.5 | [10] and sklearn default |
| DSet replicator `tol` / `max_iter` | \(10^{-5}\) / 1000 | [27] |
| `is_weighted=False` (Eq. 2) | unused | the paper is edge-weighted similarities (Eq. 3) |
| `alon3` slow path | unused | Fiorucci greedy (`fast_convergence=True`) is modification 2 |
| kNN graph / \(w>\tau\) | unused | Hou et al. never state them; do not invent |
| Similarity | Gaussian \(s=\exp(-d/(\bar d\sigma))\) | paper §4; not a function variant |

DSet leftover 5% dump **is** a searched clustering variant (`fiorucci05`); the
paper cell still peels until empty (`hou2023` leftover 0). Adaptive \(d_0\)
names, Frieze–Kannan, random init, and Lemma 2 adjacency **are** searched
(see [experiment-permutations.md](experiment-permutations.md)).

To add a new function form (a fourth SPC embedding, another DSet cutoff,
…): put it in `src/config.py` (`SPC_SPECTRAL_VARIANTS` / …), decode it in
`parse_clustering_variant`, and it is automatically in the experiment product.
