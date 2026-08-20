"""Graph-based clustering algorithms used in the experiments (§2).

Four representative algorithms, each taking a pairwise similarity matrix as input:

- ``spectral``  : SPC (Ng–Jordan–Weiss) and SPRG (learned similarity + SPC)
- ``dominant_set`` : DSet (dominant sets via replicator dynamics)
- ``affinity_propagation`` : APC (Frey–Dueck message passing)

See ``spec/base_algorithms.md``.
"""

from .spectral import spc, sprg, sprg_similarity
from .dominant_set import dominant_sets
from .affinity_propagation import apc

__all__ = ["spc", "sprg", "sprg_similarity", "dominant_sets", "apc"]
