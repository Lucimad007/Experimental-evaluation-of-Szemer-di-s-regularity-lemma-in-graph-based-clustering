"""affinity propagation clustering (apc) (§2.2 of hou et al., pr 171 (2026)).

hou et al. §2.2 is a description (pairwise similarity in, messages, automatic
k). the updates are frey & dueck [10]. this module calls sklearn's
``AffinityPropagation`` (the standard [10] implementation) with the knobs
[10] actually names:

- shared preference = median of the input similarities (or their minimum);
  computed on off-diagonal entries, i.e. pairwise s(i,j) for i≠j.
- stop when exemplar decisions are unchanged for 10 iterations ([10] p.973).
- damping 0.5.
"""

# silence sklearn's equal-similarity warning
import warnings

# arrays
import numpy as np
# frey–dueck message passing
from sklearn.cluster import AffinityPropagation


def _shared_preference(S, preference_quantile):
    """frey–dueck shared preference: median / min of the input similarities.

    [10]: "the shared value could be the median of the input similarities
    (resulting in a moderate number of clusters) or their minimum (resulting
    in a small number of clusters)." input similarities are the pairwise
    entries s(i,j) for i≠j (the diagonal is preference, not an observed
    similarity). zeros count: a sparse r whose median is 0 is a small-k
    preference, as [10] wrote.
    """
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    if off.size == 0:
        return 0.0
    q = float(preference_quantile)
    if q <= 0:
        return float(np.min(off))
    if q >= 100:
        return float(np.max(off))
    return float(np.percentile(off, q))


def apc(
    sim_mat,
    random_state=314,
    max_iter=500,
    convergence_iter=10,
    damping=0.5,
    preference_quantile=50,
):
    """cluster ``sim_mat`` (n×n similarity) with affinity propagation.

    returns integer labels in ``{0, ..., k-1}`` where ``k`` is determined
    automatically by the algorithm.

    ``preference_quantile`` is the frey–dueck shared preference as a
    percentile of off-diagonal similarities (50 = median, 0 = minimum).
    ``convergence_iter`` defaults to 10 ([10] p.973).
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
