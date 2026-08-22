"""Algorithm 1: enhancement of a graph-based clustering algorithm via the
regularity lemma (§3.4).

Pipeline (mirrors Algorithm 1 in the paper):
  1. partition the similarity graph ``G`` into an (approximately) regular
     partition ``V = V0 ∪ V1 ∪ … ∪ Vk`` (Alon et al. with modifications);
  2. build the reduced graph ``R`` (k×k) with weighted edge densities (Eq. 3);
  3. run the chosen base clustering algorithm on ``R`` → labels ``L1..Lk``;
  4. map labels back: every vertex in ``Vj`` gets ``Lj``; vertices in the
     exceptional class ``V0`` are assigned to the nearest cluster.

See ``spec/algorithm1.md``.
"""

import time

import numpy as np

from ..szemeredi import build_regularity_lemma


def _map_v0_to_nearest(sim_mat, classes, reduced_labels, k):
    """Assign each vertex in the exceptional class V0 to the nearest cluster.

    "Nearest" = the class whose members have the highest average similarity to the
    vertex (ties broken by lowest label). This mirrors the paper's "assign to the
    nearest cluster" for V0, which it notes is trivial and has little influence
    since |V0| is typically small.
    """
    v0_idx = np.where(classes == 0)[0]
    if v0_idx.size == 0:
        return
    labels = np.full(classes.shape[0], -1, dtype=int)
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    # representative similarity of each class = mean sim of its members to a point
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


def enhance_clustering(
    base_algorithm,
    sim_mat,
    n_clusters,
    epsilon,
    b,
    compression_rate,
    alg_kind="alon",
    is_weighted=True,
    random_initialization=False,
    random_refinement=False,
    drop_edges_between_irregular_pairs=False,
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
        adjacency of §3.3, with the unspecified d₀ treated as 0). Default
        False: the practical pipeline — following Sperotto–Pelillo's original
        regularity-clustering template and the Fiorucci et al. code base the
        paper's modification 2 adopts — builds R as the fully weighted matrix
        of Eq. 3 densities. (With the strict Alon conditions on small real
        graphs almost every pair tests irregular, so an edge-dropped R would
        be nearly empty and clustering on it impossible.)
    stop_rule : str
        Partition-loop stopping rule: ``"algorithm1"`` (default, Algorithm 1
        line 12) or ``"theoretical"`` (§3.2 Step 3).

    Returns
    -------
    labels : np.ndarray of int (length n)
    info : dict with timing and partition statistics
    """
    sim_mat = np.asarray(sim_mat, dtype=float)
    n = sim_mat.shape[0]

    t0 = time.time()
    alg = build_regularity_lemma(
        alg_kind,
        sim_mat,
        epsilon,
        is_weighted=is_weighted,
        random_initialization=random_initialization,
        random_refinement=random_refinement,
        drop_edges_between_irregular_pairs=drop_edges_between_irregular_pairs,
    )
    alg.run(b=b, compression_rate=compression_rate, verbose=verbose, stop_rule=stop_rule)
    compression_time = time.time() - t0

    k = int(alg.k)
    R = alg.reduced_sim_mat
    classes = alg.classes.astype(int)

    t1 = time.time()
    if n_clusters is None:
        reduced_labels = base_algorithm(R)
    else:
        reduced_labels = base_algorithm(R, n_clusters)
    clustering_time = time.time() - t1

    # map labels back
    labels = np.full(n, -1, dtype=int)
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    v0_idx = np.where(classes == 0)[0]
    if v0_idx.size > 0:
        mapped = _map_v0_to_nearest(sim_mat, classes, reduced_labels, k)
        if mapped is not None:
            labels[v0_idx] = mapped[v0_idx]

    info = {
        "k": k,
        "classes_cardinality": int(alg.classes_cardinality),
        "compression_time": compression_time,
        "clustering_time": clustering_time,
        "total_time": compression_time + clustering_time,
        "v0_size": int(v0_idx.size),
        "index_vec": list(alg.index_vec),
    }
    return labels, info
