"""Szemerédi regularity lemma: partitioning + reduced graph.

This is the core of §3.2–§3.3 of the paper. ``SzemerediRegularityLemma.run``
implements Alon et al. partitioning (§3.2 Steps 1–5) with the three practical
modifications, which is also Algorithm 1 lines 1–18. Then it builds the reduced
similarity matrix ``R ∈ R^{k×k}`` (§3.3, Eq. 3).

See ``spec/regularity_partitioning.md`` and ``spec/reduced_graph.md``.
"""

import random

import numpy as np

from .classes_pair import ClassesPair, WeightedClassesPair


def apply_density_threshold(R, density_threshold):
    """Paper §3.3 / Lemma 2 ``d₀``: drop R-edges whose Eq. 3 density is ≤ d₀.

    The paper never names a number. ``0`` / None = no cutoff (Algorithm 1's
    full weighted R). Else a float, or ``mean`` / ``median`` / ``p90`` / ``p95``
    computed from off-diagonal Eq. 3 entries of this R.
    """
    if density_threshold in (None, 0, 0.0, "0"):
        return R  # paper-silent default: keep every Eq. 3 weight
    R = np.asarray(R, dtype=float).copy()
    n = R.shape[0]
    if n == 0:
        return R
    mask = ~np.eye(n, dtype=bool)
    off = R[mask]
    if isinstance(density_threshold, str):
        key = density_threshold.lower()
        if key in ("mean",):
            d0 = float(off.mean()) if off.size else 0.0
        elif key in ("median", "med"):
            d0 = float(np.median(off)) if off.size else 0.0
        elif key.startswith("p"):
            d0 = float(np.percentile(off, float(key[1:]))) if off.size else 0.0
        else:
            raise ValueError(f"unknown density_threshold: {density_threshold}")
    else:
        d0 = float(density_threshold)
    R[mask & (R <= d0)] = 0.0  # §3.3: adjacent only if density > d₀
    return R


class SzemerediRegularityLemma:
    """Alon et al. regularity partitioning + reduced graph builder.

    The partitioning method, refinement step and regularity conditions are
    assigned as attributes (callables) by ``builder.build_regularity_lemma`` so
    that the Alon and Frieze–Kannan variants share the same driver code.
    """

    partition_initialization = None
    refinement_step = None
    conditions = []

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,
                 density_threshold=0):
        if is_weighted:
            self.sim_mat = sim_mat
        self.adj_mat = (sim_mat > 0.0).astype(float)
        self.epsilon = epsilon  # paper ε (regular-pair parameter)
        self.N = self.adj_mat.shape[0]
        self.degrees = np.argsort(self.adj_mat.sum(0))

        self.is_weighted = is_weighted
        # True = Lemma 2 adjacency (ε-regular pairs only). Default False = Alg 1 / Eq. 3 on all pairs.
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs
        self.density_threshold = density_threshold  # paper d₀ (unnamed; 0 = keep all)

        # populated during run()
        self.k = 0
        self.classes_cardinality = 0
        self.classes = np.empty((0,), dtype=float)
        self.reduced_sim_mat = np.empty((0, 0))
        self.index_vec = []
        self.certs_compls_list = []
        self.regularity_list = []
        self.condition_verified = []
        self.k_trace = []

    # ------------------------------------------------------------------ reduced
    def generate_reduced_sim_mat(self):
        """Algorithm 1 line 18: reduced graph ``R ∈ R^{k×k}`` from ``V1..Vk``.

        Each off-diagonal entry is the Eq. 3 weighted density of that class
        pair. ``V0`` is excluded (standard reduced-graph construction). When
        ``drop_edges_between_irregular_pairs`` is True, only pairs verified as
        ε-regular keep a weight (§3.3 theoretical adjacency).
        """
        # paper §3.3: R has k vertices = V1..Vk. V0 is NOT a vertex of R.
        # Lemma 2's blow-up R(t) is a proof, not implemented — we never enlarge R.
        self.reduced_sim_mat = np.zeros((self.k, self.k))
        for r in range(2, self.k + 1):  # class indices 1..k (skip V0 = label 0)
            s_iter = (
                range(1, r)  # all pairs: Algorithm 1 / Eq. 3
                if not self.drop_edges_between_irregular_pairs
                else self.regularity_list[r - 2]  # ε-regular pairs only (Lemma 2)
            )
            for s in s_iter:
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                # paper Eq. (3) [or Eq. (2) if unweighted]: edge weight of R between Vr and Vs
                self.reduced_sim_mat[r - 1, s - 1] = cl_pair.bip_density
                self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density
        # paper §3.3 d₀ (unnamed). 0 keeps the full Eq. 3 matrix.
        self.reduced_sim_mat = apply_density_threshold(
            self.reduced_sim_mat, self.density_threshold
        )

    # ----------------------------------------------------------- regularity chk
    def check_pairs_regularity(self, track_index=True):
        """§3.2 Step 2: for each pair (V_r, V_s), verify regular or produce a witness.

        Paper: verify as a regular pair, or show it is not by finding X ⊂ V_r,
        Y ⊂ V_s with |X|,|Y| ≥ (ε⁴/16)·|class| and |d(X,Y)−d(V_r,V_s)| ≥ ε⁴.
        The constructive test is Alon 1/2/3 (via Fiorucci, modification 2).
        ``n_ir`` counts pairs **not verified as ε-regular** (Algorithm 1 lines 4–11).
        """
        self.certs_compls_list = []
        self.regularity_list = []
        # line 4
        n_ir = 0
        self.condition_verified = [0] * (len(self.conditions) + 1)
        index = 0.0

        # lines 5–6: every unordered pair (V_r, V_s), 1 ≤ r < s ≤ k
        # (loop order is s < r; the set of pairs is identical)
        for r in range(2, self.k + 1):
            self.certs_compls_list.append([])
            self.regularity_list.append([])
            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                is_verified = False
                verified_regular = False
                for i, cond in enumerate(self.conditions):
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])
                        if cert_pair[0]:
                            # verified *irregular* → not verified as ε-regular
                            pass
                        else:
                            verified_regular = True
                            self.regularity_list[r - 2].append(s)
                        self.condition_verified[i] += 1
                        break

                if not is_verified:
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])
                    self.condition_verified[-1] += 1

                # lines 7–8
                if not verified_regular:
                    n_ir += 1

                index += cl_pair.compute_bip_density() ** 2.0

        index *= 1.0 / self.k ** 2.0
        if track_index:
            self.index_vec.append(index)
        return n_ir

    def check_partition_regularity(self, n_ir, stop_rule="algorithm1"):
        """§3.2 Step 3, or Algorithm 1 line 12.

        ``"theoretical"`` is Step 3 as written: stop when at most
        ``ε · C(k, 2)`` pairs are not verified as regular.
        ``"algorithm1"`` (default, the experimental pipeline) is line 12:
        ``n_ir < k(k−1)/2``.
        """
        total_pairs = (self.k * (self.k - 1)) / 2.0
        if stop_rule == "algorithm1":
            return n_ir < total_pairs  # Alg 1 line 12: n_ir < k(k−1)/2  (no ε)
        if stop_rule == "theoretical":
            return n_ir <= self.epsilon * total_pairs  # §3.2 Step 3: n_ir ≤ ε·C(k,2)
        raise ValueError(f"unknown stop rule: {stop_rule}")

    def _line3_compressible(self, compression_rate):
        """§3.2 modification 3: keep iterating while class size is not yet small.

        Paper: ``⌊1/ϵ⌋`` is the minimum class size, i.e. ``ϵ = |R|/|G|``.
        Equivalent loop guard (Algorithm 1 line 3): ``while ϵ > k_i / n``.
        """
        if 0.0 < compression_rate <= 1.0:
            return compression_rate > (self.k / float(self.N))  # paper: while ϵ > k_i/n
        if compression_rate > 1.0:
            # convenience: ϵ > 1 is an absolute cap on k (not in the paper)
            return self.k < int(compression_rate)
        raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")

    # --------------------------------------------------------------------- run
    def run(self, b=2, compression_rate=0.05, iteration_by_iteration=False, verbose=False,
            stop_rule="algorithm1"):
        """§3.2 Steps 1–5 (with modifications 1–3), then §3.3 reduced graph R.

        Parameters
        ----------
        b : int
            Step 1: initial number of classes (V0 excluded).
        compression_rate : float
            Modification 3 / Algorithm 1 line 3's ``ϵ``. In ``(0, 1]`` the
            loop is ``while ϵ > k_i / n``. Values ``> 1`` are an absolute cap
            on ``k`` (not in the paper).
        stop_rule : str
            Step 3 (``"theoretical"``) or Algorithm 1 line 12 (``"algorithm1"``,
            default — what the experiments actually run).
        """
        np.random.seed(314)
        random.seed(314)

        # §3.2 Step 1: equitable P1 with 1+b classes, k1 ← b
        if verbose:
            print("Performing partition initialization")
        self.partition_initialization(self, b)

        # §3.2 Steps 2–5. Modification 3 is the while-guard (class size).
        self.k_trace = []
        pairs_match_partition = False
        iteration = 0
        while self._line3_compressible(compression_rate):
            iteration += 1
            self.k_trace.append(int(self.k))
            if verbose:
                print("Iteration " + str(iteration))
                print("Performing pairs regularity check")
            n_ir = self.check_pairs_regularity()  # Step 2
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

            # §3.2 Step 3: if at most ε·C(k,2) (or Algorithm 1 line 12) not-regular, stop
            if self.check_partition_regularity(n_ir, stop_rule=stop_rule):
                if verbose:
                    print("The partition is regular (line 12): Break")
                break
            if verbose:
                print("The partition is irregular, proceed to refinement (line 15)")
            # Step 4 + modification 1 (≤1 irregular partner). Step 5 is the loop.
            self.refinement_step(self)
            pairs_match_partition = False
            if iteration_by_iteration:
                input("Press Enter to continue...")
            if verbose:
                print()

        # Line 18 needs pair certificates for the *current* classes. If the
        # while-guard failed after a refinement, the last count belongs to the
        # pre-refine partition — refresh without changing the classes.
        if not pairs_match_partition:
            self.check_pairs_regularity(track_index=False)

        # line 18
        self.generate_reduced_sim_mat()
        return self.reduced_sim_mat
