"""[28] algorithm 2 two-part v0 guard: no new classes, even distribution."""

import numpy as np
import pytest

from src.szemeredi.refinement_step import apply_v0_guard


def _state(n, epsilon, k, c0_size, class_size):
    classes = np.zeros(n, dtype=float)
    offset = c0_size
    for lab in range(1, k + 1):
        classes[offset:offset + class_size] = lab
        offset += class_size
    assert offset == n
    st = type("S", (), {})()
    st.N = n
    st.epsilon = epsilon
    st.k = k
    st.classes = classes
    st.classes_cardinality = class_size
    return st


def test_small_c0_stays_in_v0():
    n, k, class_size, c0 = 100, 4, 22, 12
    # εn = 20; |c0|=12 ≤ 20
    st = _state(n, epsilon=0.2, k=k, c0_size=c0, class_size=class_size)
    labels_before = set(np.unique(st.classes[st.classes > 0]))
    c0_idx = np.where(st.classes == 0)[0]
    apply_v0_guard(st, c0_idx)
    assert int(np.sum(st.classes == 0)) == c0
    assert set(np.unique(st.classes[st.classes > 0])) == labels_before


def test_large_c0_distributed_without_new_labels():
    n, k, class_size, c0 = 100, 4, 15, 40
    # εn = 20; |c0|=40 > 20 and > k=4 → 10 vertices per class, leftover 0
    st = _state(n, epsilon=0.2, k=k, c0_size=c0, class_size=class_size)
    labels_before = set(np.unique(st.classes[st.classes > 0]))
    max_before = int(st.classes.max())
    c0_idx = np.where(st.classes == 0)[0]
    apply_v0_guard(st, c0_idx)
    leftover = int(np.sum(st.classes == 0))
    assert leftover == c0 % k
    assert leftover <= 0.2 * n
    labels_after = set(np.unique(st.classes[st.classes > 0]))
    assert labels_after == labels_before
    assert int(st.classes.max()) == max_before
    sizes = [int(np.sum(st.classes == lab)) for lab in labels_before]
    assert len(set(sizes)) == 1
    assert sizes[0] == class_size + c0 // k


def test_c0_too_big_and_not_more_than_p_raises():
    n, k, class_size, c0 = 40, 8, 4, 8
    # εn = 4; |c0|=8 > 4 but |c0| == |P| so the second guard fails
    st = _state(n, epsilon=0.1, k=k, c0_size=c0, class_size=class_size)
    with pytest.raises(RuntimeError, match="V0 exceeded"):
        apply_v0_guard(st, np.where(st.classes == 0)[0])
