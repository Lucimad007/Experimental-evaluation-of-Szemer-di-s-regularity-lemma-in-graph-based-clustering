"""Configuration: parameter grids and dataset metadata (§4.1, Table 1).

See ``spec/parameters.md`` and ``spec/datasets.md``.
"""

# Regularity-partitioning parameter grids (§4.1)
EPSILON_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)          # ε
COMPRESSION_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2)          # ϵ
B_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024)  # b

# Recommended narrow ranges (§4.1 conclusions)
EPSILON_RECOMMENDED = (0.1, 0.2)
COMPRESSION_RECOMMENDED = (0.02, 0.03, 0.04, 0.05, 0.1)
B_RECOMMENDED = (2, 3, 4, 5, 6, 7, 8, 9, 10, 16)

# Similarity σ grid (§4)
SIGMA_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

# Base algorithms
BASE_ALGORITHMS = ("SPC", "APC", "DSet", "SPRG")

# SPRG (Zhu–Loy–Gong CVPR 2014, ref [20]) settings. The paper uses Tclust=1000
# trees and mtry=√d; the node-weighting variant "adpt" (ClustRF-Strct-Adpt,
# the paper's best) and φ=5 are documented paper-silent choices. Reduce
# SPRG_TREES for faster smoke/demo runs.
SPRG_TREES = 1000
SPRG_VARIANT = "adpt"
SPRG_MIN_LEAF = 5

# Datasets (Table 1): name -> (NP, ND, NC)
DATASETS = {
    "Thyroid":      (215,   5,   3),
    "Wine":         (178,   13,  3),
    "Glass":        (214,   9,   6),
    "Leaves":       (1600,  64,  100),
    "Seeds":        (210,   7,   3),
    "Segment":      (2310,  19,  7),
    "Libras":       (360,   90,  15),
    "Ecoli":        (336,   7,   8),
    "Appendicitis": (106,   7,   2),
    "SCC":          (600,   60,  6),
    "USPS":         (11000, 256, 10),
    "Rice":         (3810,  7,   2),
    "Raisin":       (900,   7,   2),
    "Spambase":     (4601,  57,  2),
    "Sonar":        (208,   60,  2),
    "Banknote":     (1372,  4,   2),
    "Landsat":      (6435,  36,  6),
    "Landmine":     (338,   3,   5),
    "Dutchnumeral": (2000,  649, 10),
    "Spectf":       (267,   44,  2),
}
