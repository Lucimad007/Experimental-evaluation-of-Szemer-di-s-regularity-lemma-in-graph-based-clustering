"""dispatch helpers: map a base-algorithm name to a callable.

each callable has signature ``f(sim_mat, n_clusters=none) -> labels``. algorithms
that determine the cluster count automatically (apc, dset) ignore ``n_clusters``;
spc and sprg require it.

see ``spec/base_algorithms.md``. hou et al., pr 171 (2026) §2 and alg 1 line 19.
"""

# arrays (dset wants a dense float graph)
import numpy as np

# paper-silent defaults live in config, not inline
from . import config
# the four paper algorithms
from .clustering import apc, dominant_sets, forest_affinity, spc, sprg, sprg_on_graph


def make_base_algorithm(name, X=None, clustering_variant=None,
                        preference_quantile=None, weight_threshold=None,
                        leftover_frac=None):
    """return a callable ``f(sim_mat, n_clusters=none) -> labels`` for ``name``.

    this callable is the **base algorithm run on the reduced graph r** (a k×k
    similarity matrix) inside algorithm 1 (line 19: "perform graph-based
    clustering on r"). spc, apc and dset consume r directly. sprg learns its
    similarity from features; since the vertices of r have no feature vectors,
    each vertex is represented by its similarity profile (its row of r) and the
    sprg forest is grown on those profiles — sprg proper, not a substitute.

    ``clustering_variant`` selects the implemented form of the algorithm (see
    ``config.clustering_variant_grid``). ``preference_quantile`` / ``weight_threshold``
    / ``leftover_frac`` override the decoded apc / dset variant when given explicitly.
    """
    parsed = config.parse_clustering_variant(name, clustering_variant)
    if preference_quantile is None:
        if "preference_quantile" in parsed:
            preference_quantile = parsed["preference_quantile"]
        else:
            preference_quantile = config.APC_PREFERENCE_QUANTILE
    # DSet "fiorucci" encodes cutoff as None (= 1/(1.5 n)). do not replace that
    # with the paper-cell 0.0001.
    if weight_threshold is None:
        if "weight_threshold" in parsed:
            weight_threshold = parsed["weight_threshold"]
        elif name == "DSet":
            weight_threshold = config.DSET_WEIGHT_THRESHOLD
    if leftover_frac is None and "leftover_frac" in parsed:
        leftover_frac = parsed["leftover_frac"]

    # paper §2.1
    if name == "SPC":
        v = parsed["spc_variant"]
        return lambda sim_mat, n_clusters=None: spc(sim_mat, n_clusters, variant=v)
    # paper §2.2
    if name == "APC":
        return lambda sim_mat, n_clusters=None: apc(
            sim_mat, preference_quantile=preference_quantile
        )
    # paper §2.3
    if name == "DSet":
        return lambda sim_mat, n_clusters=None: dominant_sets(
            np.asarray(sim_mat, dtype=float),
            weight_threshold=weight_threshold,
            leftover_frac=leftover_frac,
        )
    # paper sprg / ref [20], run on r's rows (alg 1 line 19)
    if name == "SPRG":
        forest = parsed["sprg_variant"]
        spectral = parsed["spc_variant"]
        def _sprg_on_reduced(sim_mat, n_clusters=None):
            if n_clusters is None:
                raise ValueError("SPRG requires the number of clusters")
            return sprg_on_graph(
                np.asarray(sim_mat, dtype=float), n_clusters,
                variant=forest, spc_variant=spectral,
            )
        return _sprg_on_reduced
    raise ValueError(f"unknown base algorithm: {name}")


def original_graph(name, X, sigma=None, n_clusters=None, clustering_variant=None,
                   knn=None, metric="euclidean", **kwargs):
    """return the original graph ``g`` (similarity matrix) for ``name``.

    for spc/apc/dset this is the gaussian similarity with parameter ``sigma``.
    for sprg this is the clustering-forest affinity (zhu–loy–gong); it is
    learned from ``x`` alone — ``sigma`` and ``n_clusters`` are not used.
    ``clustering_variant`` selects the sprg forest weighting (``adpt``/``unfm``/``bi``).
    """
    from .enhanced.similarity import gaussian_similarity

    # paper: sprg does not involve σ
    if name == "SPRG":
        parsed = config.parse_clustering_variant("SPRG", clustering_variant)
        return forest_affinity(X, variant=parsed["sprg_variant"], **kwargs)
    # paper §4: s = exp(−d/(d̄·σ)); metric is the paper-silent graph axis
    return gaussian_similarity(X, sigma, knn=knn, metric=metric)


def run_original(name, sim_mat, n_clusters, X=None, clustering_variant=None, **kwargs):
    """run an original (non-enhanced) base algorithm on the full similarity matrix.

    for sprg the original algorithm learns its own similarity from ``x`` and
    does not use the supplied ``sim_mat``; ``x`` must therefore be provided.
    """
    if name == "SPRG":
        if X is None:
            raise ValueError("SPRG requires the feature matrix X")
        parsed = config.parse_clustering_variant("SPRG", clustering_variant)
        return sprg(
            X, n_clusters,
            variant=parsed["sprg_variant"],
            spc_variant=parsed["spc_variant"],
            **kwargs,
        )
    fn = make_base_algorithm(name, X=X, clustering_variant=clustering_variant)
    # spc needs the ground-truth k
    if name in ("SPC",):
        return fn(sim_mat, n_clusters)
    # apc / dset choose k automatically
    return fn(sim_mat)
