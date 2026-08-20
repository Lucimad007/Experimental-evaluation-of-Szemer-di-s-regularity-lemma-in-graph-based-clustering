"""Evaluation metrics (§4): NMI, ACC, ARI, RI.

See ``spec/metrics.md``. NMI/ARI/RI wrap scikit-learn; ACC is the best-permutation
clustering accuracy via the Hungarian algorithm on the confusion matrix.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    rand_score,
)


def nmi(labels_true, labels_pred):
    return float(normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic"))


def ari(labels_true, labels_pred):
    return float(adjusted_rand_score(labels_true, labels_pred))


def ri(labels_true, labels_pred):
    return float(rand_score(labels_true, labels_pred))


def acc(labels_true, labels_pred):
    """Best-permutation clustering accuracy in [0, 1]."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.size == 0:
        return 0.0
    # build confusion matrix
    unique_t, t_idx = np.unique(labels_true, return_inverse=True)
    unique_p, p_idx = np.unique(labels_pred, return_inverse=True)
    K = max(unique_t.size, unique_p.size)
    cm = np.zeros((K, K) if False else (unique_t.size, unique_p.size), dtype=np.int64)
    for t, p in zip(t_idx, p_idx):
        cm[t, p] += 1
    # maximize the diagonal via Hungarian on -cm
    rows, cols = linear_sum_assignment(-cm)
    correct = cm[rows, cols].sum()
    return float(correct) / labels_true.size


def evaluate(labels_true, labels_pred):
    """Return all four metrics as a dict."""
    return {
        "nmi": nmi(labels_true, labels_pred),
        "acc": acc(labels_true, labels_pred),
        "ari": ari(labels_true, labels_pred),
        "ri": ri(labels_true, labels_pred),
    }
