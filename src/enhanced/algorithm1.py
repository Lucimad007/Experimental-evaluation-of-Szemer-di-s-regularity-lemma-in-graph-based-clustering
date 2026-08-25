"""Algorithm 1: enhancement of a graph-based clustering algorithm via the
regularity lemma (§3.4).

Pipeline (mirrors Algorithm 1 in the paper, line numbers as in the figure):
  lines 1–17  partition G (Alon et al. with modifications) until line 12 or
              the compression guard ``ϵ > k_i/n`` fails;
  line 18     build the reduced graph ``R`` (k×k, Eq. 3 densities);
  line 19     graph-based clustering on ``R`` → labels ``L1..Lk``;
  lines 20–24 every vertex in ``Vj`` gets ``Lj``;
  lines 25–27 vertices in the exceptional class ``V0`` → nearest cluster.

See ``spec/algorithm1.md``.
"""

import time

import numpy as np

from ..szemeredi import build_regularity_lemma


def _map_v0_to_nearest(sim_mat, classes, reduced_labels, k):
    """Algorithm 1 lines 25–27: each ``p ∈ V0`` → nearest cluster.

    "Nearest" = the class whose members have the highest average similarity to the
    vertex (ties broken by lowest label). This mirrors the paper's "assign to the
    nearest cluster" for V0, which it notes is trivial and has little influence
    since |V0| is typically small.
    """
    v0_idx = np.where(classes == 0)[0]  # paper V0: exceptional class, not a vertex of R
    if v0_idx.size == 0:
        return
    labels = np.full(classes.shape[0], -1, dtype=int)
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    # paper: "assign p to the nearest cluster" (unspecified); we use mean similarity
    for p in v0_idx:
        sims = sim_mat[p]
        best_label = None
        best_score = -np.inf
        for j in range(1, k + 1):
            members = np.where(classes == j)[0]
            if members.size == 0:
                continue
            score = sims[members].mean()
            if score > best_score:
                best_score = score
                best_label = reduced_labels[j - 1]
        if best_label is None:
            best_label = reduced_labels[0]
        labels[p] = best_label
    return labels


def build_reduced_graph(
    sim_mat,
    epsilon,  # paper ε (regular pair)
    b,  # paper b (initial number of classes)
    compression_rate,  # paper ϵ = |R|/|G|  (Alg 1 line 3: while ϵ > k/n)
    alg_kind="alon",
    is_weighted=True,  # True → Eq. (3); False → Eq. (2)
    random_initialization=False,
    random_refinement=False,
    drop_edges_between_irregular_pairs=False,  # False = Alg 1 all pairs; True = Lemma 2
    density_threshold=0,  # paper d₀ (unnamed). 0 = keep every Eq. 3 weight
    stop_rule="algorithm1",
    verbose=False,
):
    """Algorithm 1 lines 1–18: regularity partition + reduced graph R.

    Returns ``(R, classes, info)`` so the same partition can be clustered with
    several base-algorithm settings (APC preference, DSet threshold) without
    recomputing the expensive partition.
    """
    sim_mat = np.asarray(sim_mat, dtype=float)
    t0 = time.time()
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
    alg.run(b=b, compression_rate=compression_rate, verbose=verbose, stop_rule=stop_rule)
    compression_time = time.time() - t0
    k = int(alg.k)
    info = {
        "k": k,
        "classes_cardinality": int(alg.classes_cardinality),
        "compression_time": compression_time,
        "v0_size": int(np.sum(alg.classes == 0)),  # paper |V0|; these points are not vertices of R
        "index_vec": list(alg.index_vec),
    }
    return alg.reduced_sim_mat, alg.classes.astype(int), info


def assign_from_reduced(sim_mat, classes, reduced_labels, k):
    """Algorithm 1 lines 20–27. Not the Lemma 2 blow-up R(t) — we never enlarge R."""
    n = sim_mat.shape[0]
    labels = np.full(n, -1, dtype=int)
    # paper Alg 1 lines 20–24: every p ∈ V_j gets label L_j from clustering on R
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    # paper Alg 1 lines 25–27: V0 was not in R → nearest cluster
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
    drop_edges_between_irregular_pairs=False,  # False = Alg 1 Eq. 3; True = Lemma 2
    density_threshold=0,  # paper d₀; 0 = unnamed / keep all
    stop_rule="algorithm1",
    verbose=False,
):
    """Run Algorithm 1 to enhance ``base_algorithm`` on ``sim_mat``.

    Parameters
    ----------
    base_algorithm : callable
        ``f(sim_mat, n_clusters=None) -> labels`` (the base clustering algorithm).
        For algorithms that determine the cluster count automatically (APC, DSet)
        pass ``n_clusters=None`` to the algorithm; otherwise the ground-truth ``k``.
    sim_mat : np.ndarray
        n×n pairwise similarity matrix (the original graph G).
    n_clusters : int or None
        Ground-truth number of clusters, used for SPC/SPRG; ``None`` for APC/DSet.
    epsilon, b, compression_rate : float, int, float
        Regularity-partitioning parameters ``ε``, ``b``, ``ϵ``.
    drop_edges_between_irregular_pairs : bool
        If True, R keeps weights only for ε-regular pairs (the theoretical
        adjacency of §3.3). Default False: the practical pipeline builds R as
        the fully weighted matrix of Eq. 3 densities, then applies ``d₀``.
    density_threshold : float or str
        ``d₀`` of §3.3 / Lemma 2: two reduced-graph vertices are adjacent only
        if their Eq. 3 density exceeds this threshold. ``0`` (default) keeps
        every pair; ``"p90"`` / ``"p95"`` / ``"mean"`` set ``d₀`` from R itself.
    stop_rule : str
        Partition-loop stopping rule: ``"algorithm1"`` (default, Algorithm 1
        line 12) or ``"theoretical"`` (§3.2 Step 3).

    Returns
    -------
    labels : np.ndarray of int (length n)
    info : dict with timing and partition statistics
    """
    sim_mat = np.asarray(sim_mat, dtype=float)

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
    k = int(info["k"])

    t1 = time.time()
    # paper Alg 1 line 19: cluster on R (k×k), not on G — labels L_1 … L_k
    if n_clusters is None:
        reduced_labels = base_algorithm(R)
    else:
        reduced_labels = base_algorithm(R, n_clusters)
    clustering_time = time.time() - t1

    # paper Alg 1 lines 20–27: copy L_j onto V_j; leftover V0 → nearest cluster
    labels = assign_from_reduced(sim_mat, classes, reduced_labels, k)
    info = dict(info)
    info["clustering_time"] = clustering_time
    info["total_time"] = info["compression_time"] + clustering_time
    return labels, info
