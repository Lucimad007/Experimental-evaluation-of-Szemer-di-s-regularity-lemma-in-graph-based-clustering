"""algorithm 1: enhancement of a graph-based clustering algorithm via the
regularity lemma (§3.4 of hou et al., pattern recognition 171 (2026) 112205).

pipeline (mirrors algorithm 1 in the paper, line numbers as in the figure):
  lines 1–17  partition g (alon et al. with modifications) until line 12 or
              the compression guard ``ϵ > k_i/n`` fails;
  line 18     build the reduced graph ``r`` (k×k, eq. 3 densities);
  line 19     graph-based clustering on ``r`` → labels ``l1..lk``;
  lines 20–24 every vertex in ``vj`` gets ``lj``;
  lines 25–27 vertices in the exceptional class ``v0`` → nearest cluster.

see ``spec/algorithm1.md``.
"""

# wall-clock timing of partition vs clustering
import time

# arrays for labels and similarity
import numpy as np

# alon factory used by lines 1–18
from ..szemeredi import build_regularity_lemma


def _map_v0_to_nearest(sim_mat, classes, reduced_labels, k):
    """algorithm 1 lines 25–27: each ``p ∈ v0`` → nearest cluster.

    "nearest" = the class whose members have the highest average similarity to
    the vertex (ties broken by lowest label). this mirrors the paper's "assign
    to the nearest cluster" for v0, which it notes is trivial and has little
    influence since |v0| is typically small.
    """
    # paper v0: exceptional class, not a vertex of r
    v0_idx = np.where(classes == 0)[0]
    # nothing to map
    if v0_idx.size == 0:
        return
    # placeholder labels (filled for v1..vk then v0)
    labels = np.full(classes.shape[0], -1, dtype=int)
    # copy lj onto every p ∈ vj (needed to score v0 against clusters)
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    # paper: "assign p to the nearest cluster" (unspecified); we use mean similarity
    for p in v0_idx:
        # row of g for vertex p
        sims = sim_mat[p]
        # best cluster so far
        best_label = None
        best_score = -np.inf
        # score each class vj by mean similarity of p to its members
        for j in range(1, k + 1):
            members = np.where(classes == j)[0]
            if members.size == 0:
                continue
            score = sims[members].mean()
            if score > best_score:
                best_score = score
                best_label = reduced_labels[j - 1]
        # empty partition fallback: first reduced label
        if best_label is None:
            best_label = reduced_labels[0]
        labels[p] = best_label
    return labels


def build_reduced_graph(
    sim_mat,
    epsilon,  # paper ε (regular pair)
    b,  # paper b (initial number of classes)
    compression_rate,  # paper ϵ = |r|/|g|  (alg 1 line 3: while ϵ > k/n)
    alg_kind="alon",
    is_weighted=True,  # true → eq. (3); false → eq. (2)
    random_initialization=False,
    random_refinement=False,
    drop_edges_between_irregular_pairs=False,  # false = alg 1 all pairs; true = lemma 2
    density_threshold=0,  # paper d₀ (unnamed). 0 = keep every eq. 3 weight
    stop_rule="algorithm1",
    verbose=False,
):
    """algorithm 1 lines 1–18: regularity partition + reduced graph r.

    returns ``(r, classes, info)`` so the same partition can be clustered with
    several base-algorithm settings (apc preference, dset threshold) without
    recomputing the expensive partition.
    """
    # ensure float similarity
    sim_mat = np.asarray(sim_mat, dtype=float)
    # start compression timer
    t0 = time.time()
    # wire alon (paper) or frieze–kannan
    alg = build_regularity_lemma(
        alg_kind,
        sim_mat,
        epsilon,
        is_weighted=is_weighted,
        random_initialization=random_initialization,
        random_refinement=random_refinement,
        drop_edges_between_irregular_pairs=drop_edges_between_irregular_pairs,
        density_threshold=density_threshold,
    )
    # algorithm 1 lines 1–17 then line 18
    alg.run(b=b, compression_rate=compression_rate, verbose=verbose, stop_rule=stop_rule)
    # time of partitioning + building r
    compression_time = time.time() - t0
    # current k = |r|
    k = int(alg.k)
    # stats consumed by experiments / assign_from_reduced
    info = {
        "k": k,
        "classes_cardinality": int(alg.classes_cardinality),
        "compression_time": compression_time,
        # paper |v0|; these points are not vertices of r
        "v0_size": int(np.sum(alg.classes == 0)),
        "index_vec": list(alg.index_vec),
    }
    return alg.reduced_sim_mat, alg.classes.astype(int), info


def assign_from_reduced(sim_mat, classes, reduced_labels, k):
    """algorithm 1 lines 20–27. not the lemma 2 blow-up r(t) — we never enlarge r."""
    # |g|
    n = sim_mat.shape[0]
    # labels for every original vertex
    labels = np.full(n, -1, dtype=int)
    # paper alg 1 lines 20–24: every p ∈ vj gets label lj from clustering on r
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    # paper alg 1 lines 25–27: v0 was not in r → nearest cluster
    v0_idx = np.where(classes == 0)[0]
    if v0_idx.size > 0:
        mapped = _map_v0_to_nearest(sim_mat, classes, reduced_labels, k)
        if mapped is not None:
            labels[v0_idx] = mapped[v0_idx]
    return labels


def enhance_clustering(
    base_algorithm,
    sim_mat,
    n_clusters,
    epsilon,  # paper ε
    b,  # paper b
    compression_rate,  # paper ϵ
    alg_kind="alon",
    is_weighted=True,
    random_initialization=False,
    random_refinement=False,
    drop_edges_between_irregular_pairs=False,  # false = alg 1 eq. 3; true = lemma 2
    density_threshold=0,  # paper d₀; 0 = unnamed / keep all
    stop_rule="algorithm1",
    verbose=False,
):
    """run algorithm 1 to enhance ``base_algorithm`` on ``sim_mat``.

    parameters
    ----------
    base_algorithm : callable
        ``f(sim_mat, n_clusters=none) -> labels`` (the base clustering algorithm).
        for algorithms that determine the cluster count automatically (apc, dset)
        pass ``n_clusters=none`` to the algorithm; otherwise the ground-truth ``k``.
    sim_mat : np.ndarray
        n×n pairwise similarity matrix (the original graph g).
    n_clusters : int or none
        ground-truth number of clusters, used for spc/sprg; ``none`` for apc/dset.
    epsilon, b, compression_rate : float, int, float
        regularity-partitioning parameters ``ε``, ``b``, ``ϵ``.
    drop_edges_between_irregular_pairs : bool
        if true, r keeps weights only for ε-regular pairs (the theoretical
        adjacency of §3.3). default false: the practical pipeline builds r as
        the fully weighted matrix of eq. 3 densities, then applies ``d₀``.
    density_threshold : float or str
        ``d₀`` of §3.3 / lemma 2: two reduced-graph vertices are adjacent only
        if their eq. 3 density exceeds this threshold. ``0`` (default) keeps
        every pair; ``"p90"`` / ``"p95"`` / ``"mean"`` set ``d₀`` from r itself.
    stop_rule : str
        partition-loop stopping rule: ``"algorithm1"`` (default, algorithm 1
        line 12) or ``"theoretical"`` (§3.2 step 3).

    returns
    -------
    labels : np.ndarray of int (length n)
    info : dict with timing and partition statistics
    """
    # original graph g
    sim_mat = np.asarray(sim_mat, dtype=float)

    # algorithm 1 lines 1–18
    R, classes, info = build_reduced_graph(
        sim_mat,
        epsilon,
        b,
        compression_rate,
        alg_kind=alg_kind,
        is_weighted=is_weighted,
        random_initialization=random_initialization,
        random_refinement=random_refinement,
        drop_edges_between_irregular_pairs=drop_edges_between_irregular_pairs,
        density_threshold=density_threshold,
        stop_rule=stop_rule,
        verbose=verbose,
    )
    # |r|
    k = int(info["k"])

    # start clustering timer
    t1 = time.time()
    # paper alg 1 line 19: cluster on r (k×k), not on g — labels l1 … lk
    if n_clusters is None:
        reduced_labels = base_algorithm(R)
    else:
        reduced_labels = base_algorithm(R, n_clusters)
    clustering_time = time.time() - t1

    # paper alg 1 lines 20–27: copy lj onto vj; leftover v0 → nearest cluster
    labels = assign_from_reduced(sim_mat, classes, reduced_labels, k)
    # do not mutate the partition-info dict
    info = dict(info)
    info["clustering_time"] = clustering_time
    info["total_time"] = info["compression_time"] + clustering_time
    return labels, info
