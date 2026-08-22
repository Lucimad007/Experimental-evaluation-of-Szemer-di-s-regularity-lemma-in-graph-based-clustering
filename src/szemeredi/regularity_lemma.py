"""Szemerédi regularity lemma: partitioning + reduced graph.

This is the core of §3.2–§3.3 of the paper. ``SzemerediRegularityLemma.run``
performs the Alon et al. partitioning loop with the practical modifications
(limit irregular pairs per class to ≤1, degree-based greedy certificates,
terminate when ``k > ϵ·|V|``), then builds the reduced similarity matrix
``R ∈ R^{k×k}`` whose entries are the weighted edge densities (Eq. 3) between
classes.

See ``spec/regularity_partitioning.md`` and ``spec/reduced_graph.md``.
"""

import random

import numpy as np

from .classes_pair import ClassesPair, WeightedClassesPair


class SzemerediRegularityLemma:
    """Alon et al. regularity partitioning + reduced graph builder.

    The partitioning method, refinement step and regularity conditions are
    assigned as attributes (callables) by ``builder.build_regularity_lemma`` so
    that the Alon and Frieze–Kannan variants share the same driver code.
    """

    partition_initialization = None
    refinement_step = None
    conditions = []

    def __init__(self, sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs):
        if is_weighted:
            self.sim_mat = sim_mat
        self.adj_mat = (sim_mat > 0.0).astype(float)
        self.epsilon = epsilon
        self.N = self.adj_mat.shape[0]
        self.degrees = np.argsort(self.adj_mat.sum(0))

        self.is_weighted = is_weighted
        self.drop_edges_between_irregular_pairs = drop_edges_between_irregular_pairs

        # populated during run()
        self.k = 0
        self.classes_cardinality = 0
        self.classes = np.empty((0,), dtype=float)
        self.reduced_sim_mat = np.empty((0, 0))
        self.index_vec = []
        self.certs_compls_list = []
        self.regularity_list = []
        self.condition_verified = []

    # ------------------------------------------------------------------ reduced
    def generate_reduced_sim_mat(self):
        """Build the reduced similarity matrix ``R`` of size ``k×k`` (Eq. 3)."""
        self.reduced_sim_mat = np.zeros((self.k, self.k))
        for r in range(2, self.k + 1):
            s_iter = (
                range(1, r)
                if not self.drop_edges_between_irregular_pairs
                else self.regularity_list[r - 2]
            )
            for s in s_iter:
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)
                self.reduced_sim_mat[r - 1, s - 1] = cl_pair.bip_density
                self.reduced_sim_mat[s - 1, r - 1] = cl_pair.bip_density

    # ----------------------------------------------------------- regularity chk
    def check_pairs_regularity(self):
        """Step 2: determine regular/irregular pairs and their certificates."""
        self.condition_verified = [0] * (len(self.conditions) + 1)
        num_of_irregular_pairs = 0
        index = 0.0

        for r in range(2, self.k + 1):
            self.certs_compls_list.append([])
            self.regularity_list.append([])
            for s in range(1, r):
                if self.is_weighted:
                    cl_pair = WeightedClassesPair(self.sim_mat, self.adj_mat, self.classes, r, s, self.epsilon)
                else:
                    cl_pair = ClassesPair(self.adj_mat, self.classes, r, s, self.epsilon)

                is_verified = False
                for i, cond in enumerate(self.conditions):
                    is_verified, cert_pair, compl_pair = cond(self, cl_pair)
                    if is_verified:
                        self.certs_compls_list[r - 2].append([cert_pair, compl_pair])
                        if cert_pair[0]:
                            num_of_irregular_pairs += 1
                        else:
                            self.regularity_list[r - 2].append(s)
                        self.condition_verified[i] += 1
                        break

                if not is_verified:
                    # no condition verified -> treat as regular
                    self.certs_compls_list[r - 2].append([[[], []], [[], []]])
                    self.condition_verified[-1] += 1

                index += cl_pair.compute_bip_density() ** 2.0

        index *= 1.0 / self.k ** 2.0
        self.index_vec.append(index)
        return num_of_irregular_pairs

    def check_partition_regularity(self, num_of_irregular_pairs, stop_rule="algorithm1"):
        """Stopping rule of the partitioning loop.

        ``"algorithm1"`` (default) is the practical rule of Algorithm 1
        (line 12): break as soon as the number of non-ε-regular pairs drops
        below half of all pairs, i.e. ``n_ir < k(k−1)/2`` — no ε factor.
        ``"theoretical"`` is the rule of §3.2 Step 3: the partition is regular
        when at most ``ε·C(k,2)`` pairs are not verified as regular.
        """
        total_pairs = (self.k * (self.k - 1)) / 2.0
        if stop_rule == "algorithm1":
            return num_of_irregular_pairs < total_pairs
        if stop_rule == "theoretical":
            return num_of_irregular_pairs <= self.epsilon * total_pairs
        raise ValueError(f"unknown stop rule: {stop_rule}")

    # --------------------------------------------------------------------- run
    def run(self, b=2, compression_rate=0.05, iteration_by_iteration=False, verbose=False,
            stop_rule="algorithm1"):
        """Run the Alon algorithm and return the reduced similarity matrix.

        Parameters
        ----------
        b : int
            Initial number of classes (cardinality of the initial partition, V0
            excluded).
        compression_rate : float
            Maximum compression ratio ``ϵ = |R|/|G|``. If in ``(0, 1]`` the
            algorithm stops when ``k > int(ϵ·|V|)`` (modification 3); if ``> 1.0``
            it is interpreted as an absolute cap on ``k``.
        stop_rule : str
            ``"algorithm1"`` (paper's Algorithm 1, default) or ``"theoretical"``
            (§3.2 Step 3).
        """
        np.random.seed(314)
        random.seed(314)

        if 0.0 < compression_rate <= 1.0:
            max_k = int(compression_rate * self.N)
        elif compression_rate > 1.0:
            max_k = int(compression_rate)
        else:
            raise ValueError("incorrect compression rate. Only float greater than 0.0 are accepted")
        max_k = max(max_k, b)

        iteration = 0
        if verbose:
            print("Performing partition initialization")
        self.partition_initialization(self, b)

        # Algorithm 1 lines 3-17. The pair check runs at the top of each pass
        # so that, when the loop ends (via the break rule of line 12 or the
        # ϵ > k/n condition of line 3), the reduced graph is built from pairs
        # verified at the *final* partition.
        while True:
            self.certs_compls_list = []
            self.regularity_list = []
            self.condition_verified = [0] * len(self.conditions)
            iteration += 1
            if verbose:
                print("Iteration " + str(iteration))
                print("Performing pairs regularity check")
            num_of_irregular_pairs = self.check_pairs_regularity()
            if verbose:
                total_pairs = (self.k * (self.k - 1)) / 2.0
                print(
                    "irregular pairs / total pairs = "
                    + str(num_of_irregular_pairs)
                    + " / "
                    + str(int(total_pairs))
                )
                print(
                    "irregular pairs ratio = "
                    + str(num_of_irregular_pairs / (self.epsilon * total_pairs))
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
                print("Performing partition regularity check")

            if self.check_partition_regularity(num_of_irregular_pairs, stop_rule=stop_rule):
                if verbose:
                    print("The partition is regular")
                break
            if self.k >= max_k:
                # Algorithm 1 line 3's while condition ϵ > k_i/n is no longer
                # satisfied: stop iterating (modification 3, "terminate when
                # the class size is sufficiently small").
                if verbose:
                    print(
                        "Either the classes cardinality is too low or the number of "
                        "classes is too high. Stopping iterations"
                    )
                break
            if verbose:
                print("The partition is irregular, proceed to refinement")
                print("Performing refinement")
            self.refinement_step(self)
            if iteration_by_iteration:
                input("Press Enter to continue...")
            if verbose:
                print()

        self.generate_reduced_sim_mat()
        return self.reduced_sim_mat
