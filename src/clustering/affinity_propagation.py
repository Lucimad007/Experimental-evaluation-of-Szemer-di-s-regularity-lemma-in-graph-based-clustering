"""Affinity propagation clustering (APC) (§2.2).

Frey & Dueck message-passing algorithm. Takes the pairwise similarity matrix and
identifies exemplars automatically (no ``k`` required). Uses scikit-learn's
``AffinityPropagation`` with preferences set to the median similarity, which is a
standard robust default.
"""

import numpy as np
from sklearn.cluster import AffinityPropagation


def apc(sim_mat, random_state=314, max_iter=500, convergence_iter=15, damping=0.5):
    """Cluster ``sim_mat`` (n×n similarity) with affinity propagation.

    Returns integer labels in ``{0, ..., k-1}`` where ``k`` is determined
    automatically by the algorithm.
    """
    S = np.asarray(sim_mat, dtype=float)
    S = (S + S.T) / 2.0
    preferences = np.full(S.shape[0], np.median(S[S > 0]) if np.any(S > 0) else 0.0)

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
