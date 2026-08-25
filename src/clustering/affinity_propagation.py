"""affinity propagation clustering (apc) (§2.2 of hou et al., pr 171 (2026)).

frey & dueck [10] message-passing algorithm. takes the pairwise similarity
matrix and identifies exemplars automatically (no ``k`` required). uses
scikit-learn's ``affinityPropagation``.

preference (frey & dueck 2007, ref [10]): a shared value that "can be varied to
produce different numbers of clusters. the shared value could be the median of
the input similarities (resulting in a moderate number of clusters) or their
minimum (resulting in a small number of clusters)." default is the median of
positive off-diagonal similarities (``preference_quantile=50``). higher
quantiles yield more exemplars — necessary on a reduced graph r whose entries
are nearly uniform pair-densities, where the median under-clusters.
"""

# silence sklearn's equal-similarity warning
import warnings

# arrays
import numpy as np
# frey–dueck message passing
from sklearn.cluster import AffinityPropagation


def _shared_preference(S, preference_quantile):
    """frey–dueck shared preference: a percentile of positive similarities."""
    n = S.shape[0]
    # drop the diagonal
    off = S[~np.eye(n, dtype=bool)]
    # paper [10]: typically the median of the input similarities
    pos = off[off > 0]
    values = pos if pos.size else off
    if values.size == 0:
        return 0.0
    q = float(preference_quantile)
    # 0 → minimum (few clusters)
    if q <= 0:
        return float(np.min(values))
    # 100 → maximum (many clusters)
    if q >= 100:
        return float(np.max(values))
    # 50 = median (paper-silent default)
    return float(np.percentile(values, q))


def apc(
    sim_mat,
    random_state=314,
    max_iter=500,
    convergence_iter=15,
    damping=0.5,
    preference_quantile=50,
):
    """cluster ``sim_mat`` (n×n similarity) with affinity propagation.

    returns integer labels in ``{0, ..., k-1}`` where ``k`` is determined
    automatically by the algorithm.

    ``preference_quantile`` is the frey–dueck shared preference expressed as a
    percentile of positive off-diagonal similarities (50 = median, 0 = minimum).
    """
    S = np.asarray(sim_mat, dtype=float)
    # undirected
    S = (S + S.T) / 2.0
    # shared preference (same value for every vertex)
    pref = _shared_preference(S, preference_quantile)
    preferences = np.full(S.shape[0], pref)

    # degenerate inputs (e.g. an all-zero reduced graph) make sklearn emit a
    # userwarning per call; the fallback behaviour (single arbitrary exemplar)
    # is fine for our purposes, so silence only that warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All samples have mutually equal similarities"
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster")
        # precomputed affinity = the similarity matrix itself
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
