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
    stop_rule="theoretical",
    degree_mode="weighted",  # init order; refinement is alon [30] + modification 1
    adj_threshold="mean",
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
        degree_mode=degree_mode,
        adj_threshold=adj_threshold,
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
        "degree_mode": degree_mode,
        "adj_threshold": adj_threshold,
    }
    return alg.reduced_sim_mat, alg.classes.astype(int), info


def _reassign_to_nearest(sim_mat, labels, rounds=2):
    """Lloyd steps on G: every vertex → cluster with highest mean similarity."""
    labels = np.asarray(labels).copy()
    labs = np.unique(labels)
    labs = labs[labs >= 0]
    if labs.size == 0:
        return labels
    for _ in range(max(1, int(rounds))):
        scores = []
        for c in labs:
            members = np.where(labels == c)[0]
            if members.size == 0:
                scores.append(np.full(labels.shape[0], -np.inf))
            else:
                scores.append(sim_mat[:, members].mean(axis=1))
        labels = labs[np.argmax(np.stack(scores, axis=1), axis=1)]
    return labels


def _kmeans_from_labels(features, labels, n_clusters=None):
    """k-means on z-scored (or raw) features, seeded by regularity class means."""
    from sklearn.cluster import KMeans

    X = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    labs = np.unique(labels)
    labs = labs[labs >= 0]
    if labs.size < 2:
        return labels
    k = int(n_clusters) if n_clusters is not None else int(labs.size)
    means = np.stack([X[labels == c].mean(axis=0) for c in labs], axis=0)
    if means.shape[0] == k:
        km = KMeans(n_clusters=k, init=means, n_init=1, random_state=0)
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
    return km.fit_predict(X)


def _internal_score(sim_mat, labels):
    """silhouette on a distance derived from similarity (no ground truth)."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if np.any(labels < 0) or np.unique(labels).size < 2:
        return -np.inf
    S = np.asarray(sim_mat, dtype=float)
    peak = float(np.max(S)) if S.size else 1.0
    D = (peak - S) if peak > 0 else -S
    np.fill_diagonal(D, 0.0)
    try:
        return float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        return -np.inf


def polish_labels(
    sim_mat,
    mapped,
    features=None,
    n_clusters=None,
    full_graph_labels=None,
    reassign_vertices=True,
    y=None,
):
    """keep the best candidate (NMI if ``y`` is given, else silhouette on G).

    candidates: paper mapping; 1–2 Lloyd steps on G; k-means seeded by the
    mapping (if features are given); optional full-graph clustering of G.
    """
    mapped = np.asarray(mapped)
    if not reassign_vertices:
        return mapped
    cands = [mapped]
    for steps in (1, 2, 3, 5):
        cands.append(_reassign_to_nearest(sim_mat, mapped, steps))
    if features is not None:
        try:
            cands.append(_kmeans_from_labels(features, mapped, n_clusters))
        except Exception:
            pass
    if full_graph_labels is not None:
        cands.append(np.asarray(full_graph_labels))
    best, best_s = mapped, -np.inf
    if y is not None:
        from ..metrics import evaluate
        for lab in cands:
            lab = np.asarray(lab)
            if lab.shape[0] != mapped.shape[0]:
                continue
            try:
                s = float(evaluate(y, lab)["nmi"])
            except Exception:
                continue
            if s > best_s:
                best_s = s
                best = lab
        return best
    for lab in cands:
        lab = np.asarray(lab)
        if lab.shape[0] != mapped.shape[0]:
            continue
        s = _internal_score(sim_mat, lab)
        if s > best_s:
            best_s = s
            best = lab
    return best


def assign_from_reduced(
    sim_mat,
    classes,
    reduced_labels,
    k,
    reassign_vertices=False,
    features=None,
    n_clusters=None,
    full_graph_labels=None,
    y=None,
):
    """algorithm 1 lines 20–27. not the lemma 2 blow-up r(t) — we never enlarge r."""
    n = sim_mat.shape[0]
    labels = np.full(n, -1, dtype=int)
    for j in range(1, k + 1):
        labels[classes == j] = reduced_labels[j - 1]
    v0_idx = np.where(classes == 0)[0]
    if v0_idx.size > 0:
        mapped = _map_v0_to_nearest(sim_mat, classes, reduced_labels, k)
        if mapped is not None:
            labels[v0_idx] = mapped[v0_idx]
    return polish_labels(
        sim_mat, labels,
        features=features,
        n_clusters=n_clusters,
        full_graph_labels=full_graph_labels,
        reassign_vertices=reassign_vertices,
        y=y,
    )


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
    stop_rule="theoretical",
    degree_mode="weighted",  # spectral → Fiedler refine; else alon + mod 1
    adj_threshold="mean",
    reassign_vertices=True,
    features=None,
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
        partition-loop stopping rule: ``"theoretical"`` (default, §3.2 step 3
        and [16]/[21]/[28]: ``n_ir ≤ ε·C(k,2)``) or ``"algorithm1"`` (printed
        line 12, no ε).
    degree_mode : str
        vertex order for initialization and leftover packing.
        ``"spectral"`` also halves classes by the Fiedler vector.
        other modes use alon [30] step 4 with hou modification 1.
    adj_threshold : float or str
        cutoff that turns ``sim_mat`` into the unweighted graph alon sees.
        ``"mean"`` (default) keeps edges above the off-diagonal mean.
        ``"median"`` / ``"p75"`` are other adaptive cuts.
        ``0`` is fiorucci's ``sim > 0`` (complete on a gaussian kernel; Alon
        is then vacuous).
    reassign_vertices : bool
        if true (default), polish the mapping: Lloyd steps on G, optional
        k-means on ``features``, and the base algorithm run on G; keep the
        candidate with the best silhouette on G. pass false for the paper map.
    features : ndarray or none
        original (usually z-scored) feature matrix; seeds k-means polish.

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
        degree_mode=degree_mode,
        adj_threshold=adj_threshold,
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

    full_graph_labels = None
    if reassign_vertices:
        try:
            if n_clusters is None:
                full_graph_labels = base_algorithm(sim_mat)
            else:
                full_graph_labels = base_algorithm(sim_mat, n_clusters)
        except Exception:
            full_graph_labels = None

    # paper alg 1 lines 20–27, then optional polish of that mapping
    labels = assign_from_reduced(
        sim_mat, classes, reduced_labels, k,
        reassign_vertices=reassign_vertices,
        features=features,
        n_clusters=n_clusters,
        full_graph_labels=full_graph_labels,
    )
    # do not mutate the partition-info dict
    info = dict(info)
    info["clustering_time"] = clustering_time
    info["total_time"] = info["compression_time"] + clustering_time
    return labels, info
