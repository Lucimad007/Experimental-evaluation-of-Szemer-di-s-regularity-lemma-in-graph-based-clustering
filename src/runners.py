"""Dispatch helpers: map a base-algorithm name to a callable.

Each callable has signature ``f(sim_mat, n_clusters=None) -> labels``. Algorithms
that determine the cluster count automatically (APC, DSet) ignore ``n_clusters``;
SPC and SPRG require it.

See ``spec/base_algorithms.md``.
"""

import numpy as np

from .clustering import apc, dominant_sets, forest_affinity, spc, sprg, sprg_on_graph


def make_base_algorithm(name, X=None, preference_quantile=50, weight_threshold=None):
    """Return a callable ``f(sim_mat, n_clusters=None) -> labels`` for ``name``.

    This callable is the **base algorithm run on the reduced graph R** (a k×k
    similarity matrix) inside Algorithm 1 (line 19: "perform graph-based
    clustering on R"). SPC, APC and DSet consume R directly. SPRG learns its
    similarity from features; since the vertices of R have no feature vectors,
    each vertex is represented by its similarity profile (its row of R) and the
    SPRG forest is grown on those profiles — SPRG proper, not a substitute.

    ``preference_quantile`` is APC-only (Frey–Dueck shared preference as a
    percentile of positive similarities; 50 = median). ``weight_threshold`` is
    DSet-only (``None`` → ``1/(1.5 n)``; ``"rel95"`` → 95% of max weight).
    """
    if name == "SPC":
        return lambda sim_mat, n_clusters=None: spc(sim_mat, n_clusters)
    if name == "APC":
        return lambda sim_mat, n_clusters=None: apc(
            sim_mat, preference_quantile=preference_quantile
        )
    if name == "DSet":
        return lambda sim_mat, n_clusters=None: dominant_sets(
            np.asarray(sim_mat, dtype=float), weight_threshold=weight_threshold
        )
    if name == "SPRG":
        def _sprg_on_reduced(sim_mat, n_clusters=None):
            if n_clusters is None:
                raise ValueError("SPRG requires the number of clusters")
            return sprg_on_graph(np.asarray(sim_mat, dtype=float), n_clusters)
        return _sprg_on_reduced
    raise ValueError(f"unknown base algorithm: {name}")


def original_graph(name, X, sigma=None, n_clusters=None, **kwargs):
    """Return the original graph ``G`` (similarity matrix) for ``name``.

    For SPC/APC/DSet this is the Gaussian similarity with parameter ``sigma``.
    For SPRG this is the clustering-forest affinity (Zhu–Loy–Gong); it is
    learned from ``X`` alone — ``sigma`` and ``n_clusters`` are not used.
    """
    from .enhanced.similarity import gaussian_similarity

    if name == "SPRG":
        return forest_affinity(X, **kwargs)
    return gaussian_similarity(X, sigma)


def run_original(name, sim_mat, n_clusters, X=None, **kwargs):
    """Run an original (non-enhanced) base algorithm on the full similarity matrix.

    For SPRG the original algorithm learns its own similarity from ``X`` and
    does NOT use the supplied ``sim_mat``; ``X`` must therefore be provided.
    """
    if name == "SPRG":
        if X is None:
            raise ValueError("SPRG requires the feature matrix X")
        return sprg(X, n_clusters, **kwargs)
    fn = make_base_algorithm(name, X=X)
    if name in ("SPC",):
        return fn(sim_mat, n_clusters)
    return fn(sim_mat)
