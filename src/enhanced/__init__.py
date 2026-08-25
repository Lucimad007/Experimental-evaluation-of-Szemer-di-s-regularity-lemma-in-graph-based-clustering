"""enhanced graph-based clustering: algorithm 1 wrapper + k-means partition baseline.

see ``spec/algorithm1.md`` and ``spec/experiments.md``.
hou et al., pattern recognition 171 (2026) 112205, §3.4 and §4.2 fig. 11.
"""

# algorithm 1 lines 1–27
from .algorithm1 import assign_from_reduced, build_reduced_graph, enhance_clustering
# fig. 11 vertex-sampling baseline
from .kmeans_partition import kmeans_partition_clustering

# public api
__all__ = [
    "enhance_clustering",
    "kmeans_partition_clustering",
    "build_reduced_graph",
    "assign_from_reduced",
]
