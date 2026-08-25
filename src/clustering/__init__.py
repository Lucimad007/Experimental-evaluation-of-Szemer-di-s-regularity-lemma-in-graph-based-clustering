"""graph-based clustering algorithms used in the experiments (§2 of hou et al.).

four representative algorithms:

- ``spectral``  : spc (unnormalized laplacian, §2.1 of the paper)
- ``sprg``      : sprg (zhu–loy–gong clustering-forest affinity + spc, ref [20])
- ``dominant_set`` : dset (dominant sets via replicator dynamics)
- ``affinity_propagation`` : apc (frey–dueck message passing)

see ``spec/base_algorithms.md``.
"""

# §2.1
from .spectral import spc
# §2 / ref [20]
from .sprg import forest_affinity, sprg, sprg_on_graph, sprg_similarity
# §2.3
from .dominant_set import dominant_sets
# §2.2
from .affinity_propagation import apc

# public api
__all__ = [
    "spc",
    "forest_affinity",
    "sprg",
    "sprg_on_graph",
    "sprg_similarity",
    "dominant_sets",
    "apc",
]
