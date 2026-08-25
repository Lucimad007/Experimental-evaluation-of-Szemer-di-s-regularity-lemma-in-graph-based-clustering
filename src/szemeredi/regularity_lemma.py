"""szemerédi regularity lemma: partitioning + reduced graph.

this is the core of §3.2–§3.3 of hou et al., pattern recognition 171 (2026)
112205. ``szemerediRegularityLemma.run`` implements alon et al. [30]
partitioning (§3.2 steps 1–5) with the three practical modifications, which is
also algorithm 1 lines 1–18. then it builds the reduced similarity matrix
``r ∈ r^{k×k}`` (§3.3, eq. 3).

see ``spec/regularity_partitioning.md`` and ``spec/reduced_graph.md``.
"""

# used to pin the refinement rng (modification 1)
import random

# arrays for the reduced graph and class labels
import numpy as np

# bipartite pair objects (eq. 2 unweighted, eq. 3 weighted)
from .classes_pair import ClassesPair, WeightedClassesPair


def apply_density_threshold(R, density_threshold):
    """paper §3.3 / lemma 2 d₀: drop r-edges whose eq. 3 density is ≤ d₀.

    the paper never names a number. ``0`` / none = no cutoff (algorithm 1's
    full weighted r). else a float, or ``mean`` / ``median`` / ``p90`` / ``p95``
    computed from off-diagonal eq. 3 entries of this r.
    """
    # paper-silent default: keep every eq. 3 weight
    if density_threshold in (None, 0, 0.0, "0"):
        # return r unchanged
        return R
    # work on a float copy
    R = np.asarray(R, dtype=float).copy()
    # number of reduced-graph vertices
    n = R.shape[0]
    # empty r has nothing to threshold
    if n == 0:
        return R
    # off-diagonal mask (no self-loops on r)
    mask = ~np.eye(n, dtype=bool)
    # densities of all unordered pairs
    off = R[mask]
    # named cutoffs computed from this r
    if isinstance(density_threshold, str):
        # normalize the name
        key = density_threshold.lower()
        # mean off-diagonal density
        if key in ("mean",):
            d0 = float(off.mean()) if off.size else 0.0
        # median off-diagonal density
        elif key in ("median", "med"):
            d0 = float(np.median(off)) if off.size else 0.0
        # percentile, e.g. p90 / p95
        elif key.startswith("p"):
            d0 = float(np.percentile(off, float(key[1:]))) if off.size else 0.0
        else:
            raise ValueError(f"unknown density_threshold: {density_threshold}")
    else:
        # numeric d₀ as written
        d0 = float(density_threshold)
    # §3.3: adjacent only if density > d₀
    R[mask & (R <= d0)] = 0.0
    return R


class SzemerediRegularityLemma:
    """alon et al. regularity partitioning + reduced graph builder.

    the partitioning method, refinement step and regularity conditions are
    assigned as attributes (callables) by ``builder.build_regularity_lemma`` so
    that the alon and frieze–kannan variants share the same driver code.
    """

    # filled by the builder (random or degree-based)
    partition_initialization = None
    # filled by the builder (degree-based refinement)
    refinement_step = None
    # filled by the builder (alon 1/2/3 or frieze–kannan)
    conditions = []

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,
                 density_threshold=0):
        # keep the similarity matrix only when using eq. 3
        if is_weighted:
            self.sim_mat = sim_mat
        # 0/1 support of the original graph g
        self.adj_mat = (sim_mat > 0.0).astype(float)
        # paper ε (regular-pair parameter, def. 1)
        self.epsilon = epsilon
        # |v| = n
        self.N = self.adj_mat.shape[0]
        # vertices ordered by degree (used by degree-based init / refine)
        self.degrees = np.argsort(self.adj_mat.sum(0))

        # true → eq. 3 densities; false → eq. 2
        self.is_weighted = is_weighted
        # true = lemma 2 adjacency (ε-regular pairs only). default false = alg 1 / eq. 3 on all pairs
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs
        # paper d₀ (unnamed; 0 = keep all)
        self.density_threshold = density_threshold

        # populated during run()
        # current number of non-exceptional classes k_i
        self.k = 0
        # |v1| = ⌊n / k⌋ after each split
        self.classes_cardinality = 0
        # per-vertex class labels (0 = v0)
        self.classes = np.empty((0,), dtype=float)
        # reduced similarity r ∈ r^{k×k}
        self.reduced_sim_mat = np.empty((0, 0))
        # regularity-index trace (mean squared pair density)
        self.index_vec = []
        # certificates / complements for each pair (used by refinement)
        self.certs_compls_list = []
        # lists of ε-regular partners per class (lemma 2 adjacency)
        self.regularity_list = []
        # how often each alon condition fired
        self.condition_verified = []
        # k at each iteration (debug)
        self.k_trace = []

    # ------------------------------------------------------------------ reduced
    def generate_reduced_sim_mat(self):
        """algorithm 1 line 18: reduced graph ``r ∈ r^{k×k}`` from ``v1..vk``.

        each off-diagonal entry is the eq. 3 weighted density of that class
        pair. ``v0`` is excluded (standard reduced-graph construction). when
        ``drop_edges_between_irregular_pairs`` is true, only pairs verified as
        ε-regular keep a weight (§3.3 theoretical adjacency).
        """
        # paper §3.3: r has k vertices = v1..vk. v0 is not a vertex of r.
        # lemma 2's blow-up r(t) is a proof, not implemented — we never enlarge r.
        self.reduced_sim_mat = np.zeros((self.k, self.k))
        # class indices 1..k (skip v0 = label 0)
        for r in range(2, self.k + 1):
            # all pairs: algorithm 1 / eq. 3; else ε-regular pairs only (lemma 2)
            s_iter = (
                range(1, r)
                if not self.drop_edges_between_irregular_pairs
                else self.regularity_list[r - 2]
            )
            # fill each unordered pair (vr, vs)
            for s in s_iter:
                # weighted pair → eq. 3
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    # unweighted pair → eq. 2
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                # paper eq. (3) [or eq. (2) if unweighted]: edge weight of r between vr and vs
                self.reduced_sim_mat[r - 1, s - 1] = cl_pair.bip_density
                # r is undirected
                self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density
        # paper §3.3 d₀ (unnamed). 0 keeps the full eq. 3 matrix.
        self.reduced_sim_mat = apply_density_threshold(
            self.reduced_sim_mat, self.density_threshold
        )

    # ----------------------------------------------------------- regularity chk
    def check_pairs_regularity(self, track_index=True):
        """§3.2 step 2: for each pair (vr, vs), verify regular or produce a witness.

        paper: verify as a regular pair, or show it is not by finding x ⊂ vr,
        y ⊂ vs with |x|,|y| ≥ (ε⁴/16)·|class| and |d(x,y)−d(vr,vs)| ≥ ε⁴.
        the constructive test is alon 1/2/3 (via fiorucci [28], modification 2).
        ``n_ir`` counts pairs **not verified as ε-regular** (algorithm 1 lines 4–11).
        """
        # reset pair certificates
        self.certs_compls_list = []
        # reset ε-regular partner lists
        self.regularity_list = []
        # algorithm 1 line 4: n_ir ← 0
        n_ir = 0
        # one counter per condition plus a "none fired" bin
        self.condition_verified = [0] * (len(self.conditions) + 1)
        # regularity index accumulator
        index = 0.0

        # lines 5–6: every unordered pair (vr, vs), 1 ≤ r < s ≤ k
        # (loop order is s < r; the set of pairs is identical)
        for r in range(2, self.k + 1):
            # certificates for pairs involving class r
            self.certs_compls_list.append([])
            # regular partners of class r
            self.regularity_list.append([])
            # s = 1 .. r-1
            for s in range(1, r):
                # eq. 3 pair
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    # eq. 2 pair
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                # whether any alon/frieze test produced an answer
                is_verified = False
                # whether that answer was "ε-regular" (empty cert)
                verified_regular = False
                # try alon 1, then 2, then 3 (first hit wins)
                for i, cond in enumerate(self.conditions):
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        # store the witness for the refinement step
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])
                        # non-empty cert ⇒ verified *irregular*
                        if cert_pair[0]:
                            pass
                        else:
                            # empty cert ⇒ ε-regular
                            verified_regular = True
                            # record s as a regular partner of r
                            self.regularity_list[r - 2].append(s)
                        # which condition fired
                        self.condition_verified[i] += 1
                        # stop at the first decisive test
                        break

                # no condition fired → treat as not regular, empty cert
                if not is_verified:
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])
                    self.condition_verified[-1] += 1

                # algorithm 1 lines 7–8: if not verified as ε-regular, n_ir++
                if not verified_regular:
                    n_ir += 1

                # regularity-index contribution d(vr,vs)²
                index += cl_pair.compute_bip_density() ** 2.0

        # mean over k² (including the unused diagonal)
        index *= 1.0 / self.k ** 2.0
        # keep the trace unless this is a post-refine refresh
        if track_index:
            self.index_vec.append(index)
        # algorithm 1 line 11: return n_ir
        return n_ir

    def check_partition_regularity(self, n_ir, stop_rule="algorithm1"):
        """§3.2 step 3, or algorithm 1 line 12.

        ``"theoretical"`` is step 3 as written: stop when at most
        ``ε · c(k, 2)`` pairs are not verified as regular.
        ``"algorithm1"`` (default, the experimental pipeline) is line 12:
        ``n_ir < k(k−1)/2``.
        """
        # total unordered pairs among v1..vk
        total_pairs = (self.k * (self.k - 1)) / 2.0
        # alg 1 line 12: n_ir < k(k−1)/2  (no ε)
        if stop_rule == "algorithm1":
            return n_ir < total_pairs
        # §3.2 step 3: n_ir ≤ ε·c(k,2)
        if stop_rule == "theoretical":
            return n_ir <= self.epsilon * total_pairs
        raise ValueError(f"unknown stop rule: {stop_rule}")

    def _line3_compressible(self, compression_rate):
        """§3.2 modification 3: keep iterating while class size is not yet small.

        paper: ``⌊1/ϵ⌋`` is the minimum class size, i.e. ``ϵ = |r|/|g|``.
        equivalent loop guard (algorithm 1 line 3): ``while ϵ > k_i / n``.
        """
        # paper: while ϵ > k_i/n
        if 0.0 < compression_rate <= 1.0:
            return compression_rate > (self.k / float(self.N))
        # convenience: ϵ > 1 is an absolute cap on k (not in the paper)
        if compression_rate > 1.0:
            return self.k < int(compression_rate)
        raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")

    # --------------------------------------------------------------------- run
    def run(self, b=2, compression_rate=0.05, iteration_by_iteration=False, verbose=False,
            stop_rule="algorithm1"):
        """§3.2 steps 1–5 (with modifications 1–3), then §3.3 reduced graph r.

        parameters
        ----------
        b : int
            step 1: initial number of classes (v0 excluded).
        compression_rate : float
            modification 3 / algorithm 1 line 3's ``ϵ``. in ``(0, 1]`` the
            loop is ``while ϵ > k_i / n``. values ``> 1`` are an absolute cap
            on ``k`` (not in the paper).
        stop_rule : str
            step 3 (``"theoretical"``) or algorithm 1 line 12 (``"algorithm1"``,
            default — what the experiments actually run).
        """
        # pin numpy rng for reproducible partitions
        np.random.seed(314)
        # pin python rng for modification 1's partner choice
        random.seed(314)

        # §3.2 step 1: equitable p1 with 1+b classes, k1 ← b
        if verbose:
            print("Performing partition initialization")
        # algorithm 1 lines 1–2
        self.partition_initialization(self, b)

        # §3.2 steps 2–5. modification 3 is the while-guard (class size).
        self.k_trace = []
        # true iff the last n_ir belongs to the current classes
        pairs_match_partition = False
        # iteration counter (debug)
        iteration = 0
        # algorithm 1 line 3: while ϵ > k_i / n
        while self._line3_compressible(compression_rate):
            iteration += 1
            self.k_trace.append(int(self.k))
            if verbose:
                print("Iteration " + str(iteration))
                print("Performing pairs regularity check")
            # step 2 / lines 4–11
            n_ir = self.check_pairs_regularity()
            # this n_ir matches the current partition
            pairs_match_partition = True
            if verbose:
                total_pairs = (self.k * (self.k - 1)) / 2.0
                print(
                    "irregular pairs / total pairs = "
                    + str(n_ir)
                    + " / "
                    + str(int(total_pairs))
                )
                print(
                    "k = "
                    + str(self.k)
                    + ". Class cardinality = "
                    + str(self.classes_cardinality)
                    + ". Index = "
                    + str(self.index_vec[-1])
                )
                print("conditions verified = " + str(self.condition_verified))

            # §3.2 step 3: if at most ε·c(k,2) (or algorithm 1 line 12) not-regular, stop
            if self.check_partition_regularity(n_ir, stop_rule=stop_rule):
                if verbose:
                    print("The partition is regular (line 12): Break")
                # algorithm 1 line 13: break
                break
            if verbose:
                print("The partition is irregular, proceed to refinement (line 15)")
            # step 4 + modification 1 (≤1 irregular partner). step 5 is the loop.
            self.refinement_step(self)
            # classes changed; last n_ir is stale
            pairs_match_partition = False
            if iteration_by_iteration:
                input("Press Enter to continue...")
            if verbose:
                print()

        # line 18 needs pair certificates for the *current* classes. if the
        # while-guard failed after a refinement, the last count belongs to the
        # pre-refine partition — refresh without changing the classes.
        if not pairs_match_partition:
            self.check_pairs_regularity(track_index=False)

        # algorithm 1 line 18: build r
        self.generate_reduced_sim_mat()
        return self.reduced_sim_mat
