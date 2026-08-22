"""Graph-based clustering algorithms used in the experiments (§2).

Four representative algorithms:

- ``spectral``  : SPC (unnormalized Laplacian, §2.1 of the paper)
- ``sprg``      : SPRG (Zhu–Loy–Gong clustering-forest affinity + SPC, ref [20])
- ``dominant_set`` : DSet (dominant sets via replicator dynamics)
- ``affinity_propagation`` : APC (Frey–Dueck message passing)

See ``spec/base_algorithms.md``.
"""

from .spectral import spc
from .sprg import forest_affinity, sprg, sprg_on_graph, sprg_similarity
from .dominant_set import dominant_sets
from .affinity_propagation import apc

__all__ = [
    "spc",
    "forest_affinity",
    "sprg",
    "sprg_on_graph",
    "sprg_similarity",
    "dominant_sets",
    "apc",
]

