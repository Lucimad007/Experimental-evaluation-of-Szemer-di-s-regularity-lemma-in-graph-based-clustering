"""Dispatch helpers: map a base-algorithm name to a callable.

Each callable has signature ``f(sim_mat, n_clusters=None) -> labels``. Algorithms
that determine the cluster count automatically (APC, DSet) ignore ``n_clusters``;
SPC and SPRG require it.

See ``spec/base_algorithms.md``.
"""

import numpy as np

from .clustering import apc, dominant_sets, spc, sprg, sprg_similarity


def make_base_algorithm(name, X=None):
    """Return a callable ``f(sim_mat, n_clusters=None) -> labels`` for ``name``.

    This callable is the **base algorithm run on the reduced graph R** (a k×k
    similarity matrix) inside Algorithm 1. For SPC/APC/DSet it is the algorithm
    itself; for SPRG it is SPC on R, because SPRG learns its similarity from
    features and the reduced graph has no features — SPC on the reduced similarity
    is the standard spectral step SPRG would perform on a learned matrix.

    ``X`` (feature matrix) is required for SPRG only to confirm it was provided.
    """
    if name == "SPC":
        return lambda sim_mat, n_clusters=None: spc(sim_mat, n_clusters)
    if name == "APC":
        return lambda sim_mat, n_clusters=None: apc(sim_mat)
    if name == "DSet":
        return lambda sim_mat, n_clusters=None: dominant_sets(np.asarray(sim_mat, dtype=float))
    if name == "SPRG":
        if X is None:
            raise ValueError("SPRG requires the feature matrix X")
        # On the reduced graph R (a similarity matrix, no features) run SPC.
        return lambda sim_mat, n_clusters=None: spc(np.asarray(sim_mat, dtype=float), n_clusters)
    raise ValueError(f"unknown base algorithm: {name}")


def original_graph(name, X, sigma=None):
    """Return the original graph ``G`` (similarity matrix) for ``name``.

    For SPC/APC/DSet this is the Gaussian similarity with parameter ``sigma``.
    For SPRG this is the SPRG-learned similarity (no ``sigma``); ``sigma`` is
    ignored.
    """
    from .enhanced.similarity import gaussian_similarity

    if name == "SPRG":
        return sprg_similarity(X)
    return gaussian_similarity(X, sigma)


def run_original(name, sim_mat, n_clusters, X=None):
    """Run an original (non-enhanced) base algorithm on the full similarity matrix.

    For SPRG the original algorithm learns its own similarity from ``X`` and does
    NOT use the supplied ``sim_mat``; ``X`` must therefore be provided.
    """
    if name == "SPRG":
        if X is None:
            raise ValueError("SPRG requires the feature matrix X")
        return sprg(X, n_clusters)
    fn = make_base_algorithm(name, X=X)
    if name in ("SPC",):
        return fn(sim_mat, n_clusters)
    return fn(sim_mat)
