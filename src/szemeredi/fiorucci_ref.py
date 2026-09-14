"""Call Marco Fiorucci's dense_graph_reducer (Hou's cited lemma code).

Clone (gitignored):
  git clone --depth 1 https://github.com/MarcoFiorucci/dense_graph_reducer.git .reference/dense_graph_reducer

This is Alon + Fiorucci degree refinement as published in that repo — not Hou's
clustering wrapper, and not our stop_rule / adj_threshold / spectral extras.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_REF = _ROOT / ".reference" / "dense_graph_reducer"


def reference_root() -> Path:
    return _REF


def is_available() -> bool:
    return (_REF / "graph_reducer" / "szemeredi_lemma_builder.py").is_file()


def _import_builder():
    if not is_available():
        raise FileNotFoundError(
            f"Fiorucci dense_graph_reducer not found at {_REF}. "
            "Clone: git clone --depth 1 "
            "https://github.com/MarcoFiorucci/dense_graph_reducer.git "
            ".reference/dense_graph_reducer"
        )
    path = str(_REF)
    if path not in sys.path:
        sys.path.insert(0, path)
    from graph_reducer.szemeredi_lemma_builder import (
        generate_szemeredi_reg_lemma_implementation,
    )
    return generate_szemeredi_reg_lemma_implementation


class FiorucciRegularityLemma:
    """Same surface as our SzemerediRegularityLemma for algorithm1.build_reduced_graph."""

    def __init__(
        self,
        sim_mat,
        epsilon,
        is_weighted,
        drop_edges_between_irregular_pairs,
        kind="alon",
        random_initialization=False,
        random_refinement=False,
        **_ignored,
    ):
        factory = _import_builder()
        self._impl = factory(
            kind,
            np.asarray(sim_mat, dtype=float),
            float(epsilon),
            bool(is_weighted),
            bool(random_initialization),
            bool(random_refinement),
            bool(drop_edges_between_irregular_pairs),
        )

    def run(self, b=2, compression_rate=0.05, verbose=False, stop_rule=None, **_):
        self._impl.run(
            b=int(b),
            compression_rate=float(compression_rate),
            verbose=bool(verbose),
        )
        impl = self._impl
        self.k = int(impl.k)
        self.classes = np.asarray(impl.classes, dtype=int)
        self.reduced_sim_mat = np.asarray(impl.reduced_sim_mat, dtype=float)
        self.classes_cardinality = int(impl.classes_cardinality)
        self.index_vec = list(getattr(impl, "index_vec", []) or [])
        return self.reduced_sim_mat
