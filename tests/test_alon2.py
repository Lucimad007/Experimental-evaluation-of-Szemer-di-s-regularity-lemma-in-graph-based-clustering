"""alon2: vs degrees, 1/8 both-direction trigger, complete bipartite."""

import numpy as np

from src.szemeredi.classes_pair import ClassesPair
from src.szemeredi.conditions import alon2


class _Eps:
    def __init__(self, epsilon):
        self.epsilon = epsilon


def _two_class_graph(block):
    n = block.shape[0]
    n_all = 2 * n
    adj = np.zeros((n_all, n_all), dtype=float)
    adj[:n, n:] = block
    adj[n:, :n] = block.T
    classes = np.zeros(n_all, dtype=float)
    classes[:n] = 1
    classes[n:] = 2
    return adj, classes


def _circulant_regular(n, degree):
    block = np.zeros((n, n), dtype=float)
    for i in range(n):
        cols = (np.arange(degree) + i) % n
        block[i, cols] = 1.0
    return block


def _add_edges(block, high_cols, delta):
    """raise high_cols' degrees by ``delta`` without lowering any other column."""
    block = block.copy()
    n = block.shape[0]
    for h in high_cols:
        added = 0
        for i in range(n):
            if added >= delta:
                break
            if block[i, h] == 0.0:
                block[i, h] = 1.0
                added += 1
        if added < delta:
            raise RuntimeError("could not add enough edges")
    return block


def _cut_edges(block, low_cols, delta):
    """lower low_cols' degrees by ``delta``."""
    block = block.copy()
    n = block.shape[0]
    for low in low_cols:
        removed = 0
        for i in range(n):
            if removed >= delta:
                break
            if block[i, low] == 1.0:
                block[i, low] = 0.0
                removed += 1
        if removed < delta:
            raise RuntimeError("could not cut enough edges")
    return block


def test_complete_bipartite_does_not_fire():
    n = 40
    eps = 0.5
    block = np.ones((n, n), dtype=float)
    adj, classes = _two_class_graph(block)
    pair = ClassesPair(adj, classes, r=1, s=2, epsilon=eps)
    is_irregular, certs, _ = alon2(_Eps(eps), pair)
    assert is_irregular is False
    assert certs == [[], []]


def test_mixup_certificate_is_deviant_vs():
    """deviant vertices live only in vs; the cert must be those vs ids, not a vr mask."""
    n = 400
    eps = 0.8
    # ε⁴ n ≈ 163.8; regular degree 200 so delta 180 puts high cols above the threshold
    block = _circulant_regular(n, n // 2)
    n_high = 25
    high_cols = list(range(n_high))
    block = _add_edges(block, high_cols, delta=180)
    adj, classes = _two_class_graph(block)
    pair = ClassesPair(adj, classes, r=1, s=2, epsilon=eps)
    is_irregular, certs, _ = alon2(_Eps(eps), pair)
    assert is_irregular is True
    vr_ids = set(pair.index_map[0])
    vs_ids = set(pair.index_map[1])
    assert set(certs[0]) == vr_ids
    cert_s = set(certs[1])
    assert cert_s.issubset(vs_ids)
    expected = set(n + c for c in high_cols)
    assert cert_s == expected


def test_one_sided_below_one_eighth_does_not_fire():
    """0.07 ε⁴ n one-sided: old 1/16 would fire, theorem trigger must not."""
    n = 400
    eps = 0.8
    eps4n = (eps ** 4.0) * n
    n_high = int(0.07 * eps4n)  # ~11; 1/16 ≈ 10.2, 1/8 ≈ 20.5
    assert n_high > (1.0 / 16.0) * eps4n
    assert n_high <= (1.0 / 8.0) * eps4n
    block = _circulant_regular(n, n // 2)
    high_cols = list(range(n_high))
    block = _add_edges(block, high_cols, delta=180)
    adj, classes = _two_class_graph(block)
    pair = ClassesPair(adj, classes, r=1, s=2, epsilon=eps)
    is_irregular, certs, _ = alon2(_Eps(eps), pair)
    assert is_irregular is False
    assert certs == [[], []]


def test_both_directions_above_one_eighth_fires_on_larger_side():
    n = 400
    eps = 0.8
    eps4n = (eps ** 4.0) * n
    n_high = int(0.09 * eps4n) + 1  # ~15
    n_low = int(0.09 * eps4n) + 1
    assert n_high + n_low > (1.0 / 8.0) * eps4n
    block = _circulant_regular(n, n // 2)
    high_cols = list(range(n_high))
    low_cols = list(range(n - n_low, n))
    block = _add_edges(block, high_cols, delta=180)
    block = _cut_edges(block, low_cols, delta=180)
    adj, classes = _two_class_graph(block)
    pair = ClassesPair(adj, classes, r=1, s=2, epsilon=eps)
    is_irregular, certs, _ = alon2(_Eps(eps), pair)
    assert is_irregular is True
    # equal counts → too-high side is the certificate
    expected = set(n + c for c in high_cols)
    assert set(certs[1]) == expected
