"""szemerédi regularity lemma: regularity partitioning and reduced graph.

implements the alon et al. (1994) [30] regularity-partitioning algorithm with
the practical modifications described in §3.2 of hou et al., pr 171 (2026):

1. limit the number of irregular pairs containing each class to at most one,
2. degree-based greedy certificates (fiorucci et al. 2020 [28]),
3. terminate when the class size is sufficiently small (compression ratio ϵ).

see ``spec/regularity_partitioning.md`` and ``spec/reduced_graph.md``.
"""

# partitioning driver + d₀ helper
from .regularity_lemma import (
    SzemerediRegularityLemma,
    apply_density_threshold,
    unweighted_adjacency,
)
# alon vs frieze–kannan factory
from .builder import build_regularity_lemma
from .fiorucci_ref import FiorucciRegularityLemma, is_available as fiorucci_available

# public api
__all__ = [
    "SzemerediRegularityLemma",
    "build_regularity_lemma",
    "FiorucciRegularityLemma",
    "fiorucci_available",
    "apply_density_threshold",
    "unweighted_adjacency",
]
