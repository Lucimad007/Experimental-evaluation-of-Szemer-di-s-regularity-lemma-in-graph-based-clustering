"""Szemerédi regularity lemma: regularity partitioning and reduced graph.

Implements the Alon et al. (1994) regularity-partitioning algorithm with the
practical modifications described in §3.2 of the paper:

1. limit the number of irregular pairs containing each class to at most one,
2. degree-based greedy certificates (Fiorucci et al. 2020),
3. terminate when the class size is sufficiently small (compression ratio ϵ).

See ``spec/regularity_partitioning.md`` and ``spec/reduced_graph.md``.
"""

from .regularity_lemma import SzemerediRegularityLemma
from .builder import build_regularity_lemma

__all__ = ["SzemerediRegularityLemma", "build_regularity_lemma"]
