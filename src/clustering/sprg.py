"""SPRG: spectral clustering on a clustering-random-forest affinity.

SPRG is the algorithm of Zhu, Loy & Gong, "Constructing robust affinity graphs
for spectral clustering" (CVPR 2014) — reference **[20]** of the main paper
(also cited as "SPRG" in Hou et al., PR 2023 [25] of the main paper). The main
paper describes it as "a variant of SPC" that "learns the similarity by
combining subtle similarity in discriminative feature subspaces" and does not
involve the parameter σ.

Method (all equation numbers from Zhu–Loy–Gong):

1. **Clustering forest** (Sec. 3.1): an ensemble of ``Tclust`` binary decision
   trees, each trained on a training set ``Xt ⊂ X`` drawn randomly from the
   data, unsupervised via the pseudo two-class algorithm (Liu–Xia–Yu): a
   synthetic sample set drawn uniformly in the feature bounding box is labelled
   class 1, the real samples class 0, and a classification tree (Gini
   information gain, Eq. 3) is learned. At each split node the optimal
   parameter ``ϑ*`` (Eq. 2) is searched greedily over ``mtry`` randomly
   selected features (``mtry = √d``) and the ``|S|−1`` mid-point thresholds of
   the arriving samples. Splitting stops when the number of arriving real
   samples is ≤ ``φ``.

2. **Structure-aware affinity** (Sec. 3.2, Eq. 7): for a sample pair
   ``(xi, xj)`` channelled through a tree, let ``λ`` be the number of internal
   tree nodes their paths share (the root is not counted) and ``M =
   max(|Pi|, |Pj|) − 1`` the number of non-root nodes on the longer path
   (internal nodes + leaf). The tree-level similarity is

       a^t_ij = ( Σ_{κ=1..λ} w_κ ) / ( Σ_{κ=1..M} w_κ )

   with node-weight variants:
   - ``"adpt"`` (ClustRF-Strct-Adpt, Eq. 12–13, best in the paper's
     experiments — default): ``w_κ = 1/|S_κ|`` for internal nodes and
     ``w = 1/|Λ_b̂|`` for the leaf of the longer path;
   - ``"unfm"`` (ClustRF-Strct-Unfm, Eq. 11): all weights 1, i.e.
     ``a^t_ij = λ / (max(|Pi|, |Pj|) − 1)``;
   - ``"bi"`` (ClustRF-Bi, Eq. 9–10): 1 iff the two samples share a leaf.

3. **Forest consensus** (Eq. 8): ``A = (1/Tclust) Σ_t A^t``.

4. **Clustering**: spectral clustering (the paper's SPC — unnormalized
   Laplacian, k smallest eigenvectors, k-means on the rows) on ``A`` with the
   ground-truth number of clusters ``k``.

Paper-silent implementation choices (documented in IMPLEMENTATION_PROOF.md):
``Tclust = 1000`` (paper Sec. 4), ``mtry = √d`` (paper Sec. 4), the per-tree
training set is a bootstrap sample, ``φ = 5``, the "adpt" variant as default,
and a fixed seed (the paper averages over 5 trials; we fix one for
reproducibility).
"""

import numpy as np
import scipy.sparse

from .. import config
from .spectral import spc


# --------------------------------------------------------------------------- #
# Clustering decision tree (pseudo two-class, Gini gain)
# --------------------------------------------------------------------------- #
def _best_split(values, labels):
    """Greedy split search on one feature: Gini-gain-best mid-point threshold.

    ``values``/``labels`` are the arriving samples' feature values and pseudo
    class labels (0 = real, 1 = synthetic). Returns ``(threshold, gain)`` with
    ``threshold = None`` if no valid split exists.
    """
    order = np.argsort(values, kind="stable")
    v = values[order]
    y = labels[order]
    m = v.size
    if m < 2 or v[0] == v[-1]:
        return None, -np.inf

    n1 = y.sum()
    p1 = n1 / m
    gini_s = 2.0 * p1 * (1.0 - p1)

    cs1 = np.cumsum(y)
    k = np.arange(1, m)                       # split after position k-1: |L| = k
    # valid only where consecutive values differ (mid-points, |S|-1 candidates)
    valid = v[:-1] < v[1:]

    p1_l = cs1[:-1] / k
    p1_r = (n1 - cs1[:-1]) / (m - k)
    gini_l = 2.0 * p1_l * (1.0 - p1_l)
    gini_r = 2.0 * p1_r * (1.0 - p1_r)
    gain = np.where(valid, gini_s - (k / m) * gini_l - ((m - k) / m) * gini_r, -np.inf)

    best = np.argmax(gain)
    if not np.isfinite(gain[best]):
        return None, -np.inf
    threshold = (v[best] + v[best + 1]) / 2.0
    return threshold, gain[best]


class _Tree:
    """One clustering decision tree; stores nodes and per-sample paths.

    Nodes are stored in parallel arrays; node 0 is the root. ``paths[i]`` is the
    list of internal-node ids (root excluded) traversed by sample i, and
    ``leaf_size[i]`` the number of real training samples in i's leaf.
    """

    def __init__(self, feature, threshold, left, right, n_real, paths, leaf_ids, leaf_sizes):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_real = n_real
        self.paths = paths
        self.leaf_ids = leaf_ids
        self.leaf_sizes = leaf_sizes


def _train_tree(X, rng, mtry, min_samples_leaf):
    """Train one clustering tree on a bootstrap sample of ``X``.

    Pseudo two-class construction: the bootstrap real samples are labelled 0
    and an equally sized synthetic set, uniform in the bootstrap's per-feature
    bounding box, is labelled 1. A Gini-gain classification tree is grown on
    the mixture; ``n_real[u]`` records the number of real training samples
    arriving at node ``u`` (the ``|S_κ|`` of Eq. 12). The tree is grown with an
    explicit stack so deep trees cannot hit Python's recursion limit.
    """
    n, d = X.shape
    boot = rng.integers(0, n, size=n)
    Xb = X[boot]
    lo, hi = Xb.min(0), Xb.max(0)
    Xsyn = rng.uniform(lo, hi, size=(n, d))
    feats = np.vstack([Xb, Xsyn])             # real first, synthetic last
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    feature, threshold, left, right, n_real = [], [], [], [], []

    def new_node(arriving):
        node = len(feature)
        feature.append(0), threshold.append(0.0), left.append(-1), right.append(-1)
        n_real.append(int((arriving < n).sum()))
        return node

    def try_split(node, idx):
        """Find the Gini-best split for node ``node``; wire children if found."""
        if n_real[node] <= min_samples_leaf:
            return
        feats_idx = rng.choice(d, size=min(mtry, d), replace=False)
        sub = feats[idx]
        best_gain, best_f, best_t = -np.inf, None, None
        for f in feats_idx:
            t, g = _best_split(sub[:, f], labels[idx])
            if t is not None and g > best_gain:
                best_gain, best_f, best_t = g, f, t
        if best_f is None:
            return
        mask = sub[:, best_f] < best_t
        if mask.all() or not mask.any():
            return
        feature[node], threshold[node] = best_f, best_t
        for is_left, m_ in ((True, mask), (False, ~mask)):
            child = new_node(idx[m_])
            if is_left:
                left[node] = child
            else:
                right[node] = child
            stack.append((child, idx[m_]))

    root = new_node(np.arange(2 * n))
    stack = [(root, np.arange(2 * n))]
    while stack:
        node, idx = stack.pop()
        try_split(node, idx)

    # route every sample of X through the tree
    paths, leaf_ids, leaf_sizes = [], [], []
    for i in range(n):
        node, path = 0, []
        while left[node] != -1 or right[node] != -1:
            path.append(node)
            node = left[node] if X[i, feature[node]] < threshold[node] else right[node]
        paths.append(path)
        leaf_ids.append(node)
        leaf_sizes.append(n_real[node])       # |Λ|: real training samples in the leaf

    return _Tree(feature, threshold, left, right, n_real, paths, leaf_ids, leaf_sizes)


# --------------------------------------------------------------------------- #
# Forest affinity (Eq. 7-13)
# --------------------------------------------------------------------------- #
def _accumulate_tree_affinity(A, tree, n, variant, block=1024):
    """Add one tree's affinity matrix (Eq. 7) into ``A`` in place.

    The numerator ``(P Pᵀ)`` is evaluated row-block by row-block so no full
    n×n temporary is materialised (USPS has n = 11000; a dense temporary would
    be ~1 GB per tree).
    """
    if variant == "bi":
        leaf = np.asarray(tree.leaf_ids)
        A += leaf[:, None] == leaf[None, :]
        return

    depth = np.array([len(p) for p in tree.paths], dtype=float)
    n_nodes = len(tree.feature)
    w = np.ones(n_nodes) if variant == "unfm" else np.array(
        [1.0 / nr if nr > 0 else 0.0 for nr in tree.n_real]
    )

    # P[i, u] = w_u for the internal nodes u (root excluded) on i's path;
    # numerator(i, j) = Σ w_u over the internal nodes shared by i and j
    rows = np.concatenate([np.full(len(p), i, dtype=int) for i, p in enumerate(tree.paths)])
    cols = np.concatenate([np.asarray(p, dtype=int) for p in tree.paths])
    if cols.size == 0:                         # single-leaf tree: zero numerator
        return
    P = scipy.sparse.csr_matrix((w[cols], (rows, cols)), shape=(n, n_nodes))
    path_weight = np.asarray(P.sum(axis=1)).ravel()

    if variant == "unfm":
        denom = depth + 1.0                    # internal nodes + leaf (weight 1)
    else:  # adpt: Σ 1/|S_κ| over i's internal nodes + 1/|Λ(i)|
        denom = path_weight + np.array(
            [1.0 / ls if ls > 0 else 0.0 for ls in tree.leaf_sizes]
        )
    denom = np.where(denom > 0, denom, 1e-12)

    # b̂ = argmax_{b∈{i,j}} |P_b| (Eq. 14): the longer path; ties -> i
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        numerator = np.asarray((P[i0:i1] @ P.T).todense())
        denom_mat = np.where(
            depth[i0:i1, None] >= depth[None, :], denom[i0:i1, None], denom[None, :]
        )
        A[i0:i1] += numerator / denom_mat


def forest_affinity(X, n_trees=None, variant=None, mtry=None,
                    min_samples_leaf=None, random_state=314, verbose=False):
    """Clustering-random-forest pairwise affinity (Zhu–Loy–Gong, Eq. 8).

    Parameters
    ----------
    X : np.ndarray (n, d)
        Feature matrix.
    n_trees : int
        Forest size ``Tclust`` (default ``config.SPRG_TREES`` = 1000, the paper's
        setting; reduce for smoke/demo runs).
    variant : str
        Node-weighting variant: ``"adpt"`` (default, ``config.SPRG_VARIANT``),
        ``"unfm"`` or ``"bi"``.
    mtry : int or None
        Number of candidate features per split (paper: ``√d``).
    min_samples_leaf : int
        Splitting stops when ≤ ``φ`` real samples arrive at a node (the paper
        selects ``φ`` by cross-validation; ``config.SPRG_MIN_LEAF`` = 5 is our
        documented default).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n_trees is None:
        n_trees = config.SPRG_TREES
    if variant is None:
        variant = config.SPRG_VARIANT
    if min_samples_leaf is None:
        min_samples_leaf = config.SPRG_MIN_LEAF
    if mtry is None:
        mtry = max(1, int(round(np.sqrt(d))))
    rng = np.random.default_rng(random_state)

    A = np.zeros((n, n))
    for t in range(n_trees):
        tree = _train_tree(X, rng, mtry, min_samples_leaf)
        _accumulate_tree_affinity(A, tree, n, variant)
        if verbose and (t + 1) % max(1, n_trees // 10) == 0:
            print(f"  [forest] tree {t + 1}/{n_trees}")
    A /= n_trees
    return (A + A.T) / 2.0


# --------------------------------------------------------------------------- #
# SPRG entry points
# --------------------------------------------------------------------------- #
def sprg_similarity(X, n_clusters=None, **kwargs):
    """SPRG-learned affinity matrix (the graph G for SPRG).

    The forest does not depend on the number of clusters; ``n_clusters`` is
    accepted for signature compatibility and ignored.
    """
    return forest_affinity(X, **kwargs)


def sprg(X, n_clusters, **kwargs):
    """SPRG: forest affinity + spectral clustering with ``n_clusters`` clusters."""
    A = forest_affinity(X, **kwargs)
    return spc(A, n_clusters)


def sprg_on_graph(sim_mat, n_clusters, min_samples_leaf=None, **kwargs):
    """SPRG on a graph whose only representation is a similarity matrix.

    Used for Reg-SPRG (Algorithm 1 line 19: "perform graph-based clustering on
    R"): the reduced graph R has no feature vectors, so each vertex is
    represented by its similarity profile (its row of R) and the clustering
    forest is grown on those k-dimensional profiles. ``min_samples_leaf``
    defaults to 1 here because R typically has far fewer vertices than a
    dataset has points (with the dataset-level φ = 5 a small R would yield a
    single-leaf forest and a zero affinity).
    """
    R = np.asarray(sim_mat, dtype=float)
    if min_samples_leaf is None:
        min_samples_leaf = 1
    return sprg(R, n_clusters, min_samples_leaf=min_samples_leaf, **kwargs)
