"""evaluation metrics (§4 of hou et al., pr 171 (2026)): nmi, acc, ari, ri.

see ``spec/metrics.md``. nmi/ari/ri wrap scikit-learn; acc is the
best-permutation clustering accuracy via the hungarian algorithm on the
confusion matrix. nmi is the paper's main metric.
"""

# confusion matrix
import numpy as np
# hungarian matching for acc
from scipy.optimize import linear_sum_assignment
# paper §4 metrics
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    rand_score,
)


def nmi(labels_true, labels_pred):
    # paper's main metric (figs. 2–11, tables 2–5)
    return float(normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic"))


def ari(labels_true, labels_pred):
    # paper tables 4
    return float(adjusted_rand_score(labels_true, labels_pred))


def ri(labels_true, labels_pred):
    # paper tables 5
    return float(rand_score(labels_true, labels_pred))


def acc(labels_true, labels_pred):
    """best-permutation clustering accuracy in [0, 1] (paper tables 3)."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    if labels_true.size == 0:
        return 0.0
    # build confusion matrix
    unique_t, t_idx = np.unique(labels_true, return_inverse=True)
    unique_p, p_idx = np.unique(labels_pred, return_inverse=True)
    cm = np.zeros((unique_t.size, unique_p.size), dtype=np.int64)
    for t, p in zip(t_idx, p_idx):
        cm[t, p] += 1
    # maximize the diagonal via hungarian on −cm
    rows, cols = linear_sum_assignment(-cm)
    correct = cm[rows, cols].sum()
    return float(correct) / labels_true.size


def evaluate(labels_true, labels_pred):
    """return all four paper metrics as a dict."""
    return {
        "nmi": nmi(labels_true, labels_pred),
        "acc": acc(labels_true, labels_pred),
        "ari": ari(labels_true, labels_pred),
        "ri": ri(labels_true, labels_pred),
    }
