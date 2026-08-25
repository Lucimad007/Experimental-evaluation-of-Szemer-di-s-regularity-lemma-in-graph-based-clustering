"""Affinity propagation clustering (APC) (§2.2).

Frey & Dueck message-passing algorithm. Takes the pairwise similarity matrix and
identifies exemplars automatically (no ``k`` required). Uses scikit-learn's
``AffinityPropagation``.

Preference (Frey & Dueck 2007, ref [10]): a shared value that "can be varied to
produce different numbers of clusters. The shared value could be the median of
the input similarities (resulting in a moderate number of clusters) or their
minimum (resulting in a small number of clusters)." Default is the median of
positive off-diagonal similarities (``preference_quantile=50``). Higher
quantiles yield more exemplars — necessary on a reduced graph R whose entries
are nearly uniform pair-densities, where the median under-clusters.
"""

import warnings

import numpy as np
from sklearn.cluster import AffinityPropagation


def _shared_preference(S, preference_quantile):
    """Frey–Dueck shared preference: a percentile of positive similarities."""
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    pos = off[off > 0]
    values = pos if pos.size else off
    if values.size == 0:
        return 0.0
    q = float(preference_quantile)
    if q <= 0:
        return float(np.min(values))
    if q >= 100:
        return float(np.max(values))
    return float(np.percentile(values, q))


def apc(
    sim_mat,
    random_state=314,
    max_iter=500,
    convergence_iter=15,
    damping=0.5,
    preference_quantile=50,
):
    """Cluster ``sim_mat`` (n×n similarity) with affinity propagation.

    Returns integer labels in ``{0, ..., k-1}`` where ``k`` is determined
    automatically by the algorithm.

    ``preference_quantile`` is the Frey–Dueck shared preference expressed as a
    percentile of positive off-diagonal similarities (50 = median, 0 = minimum).
    """
    S = np.asarray(sim_mat, dtype=float)
    S = (S + S.T) / 2.0
    pref = _shared_preference(S, preference_quantile)
    preferences = np.full(S.shape[0], pref)

    # degenerate inputs (e.g. an all-zero reduced graph) make sklearn emit a
    # UserWarning per call; the fallback behaviour (single arbitrary exemplar)
    # is fine for our purposes, so silence only that warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All samples have mutually equal similarities"
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")
        model = AffinityPropagation(
            affinity="precomputed",
            preference=preferences,
            random_state=random_state,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            damping=damping,
            copy=True,
        )
        labels = model.fit_predict(S)
    # sklearn may return -1 for non-converged points; remap to a valid label
    labels = np.asarray(labels, dtype=int)
    if (labels < 0).any():
        labels[labels < 0] = labels.max() + 1 if labels.max() >= 0 else 0
    return labels
