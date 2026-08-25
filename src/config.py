"""configuration: parameter grids and dataset metadata (§4.1, table 1).

see ``spec/parameters.md`` and ``spec/datasets.md``.

all experiment search values below are quoted from hou et al., pattern
recognition 171 (2026) 112205, §4.1. reproduction uses only these grids —
no extra knobs.
"""

# ---------------------------------------------------------------------------
# paper §4.1 — values actually tested ("we test …")
# ---------------------------------------------------------------------------
# "in our experiments we test ε from 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5 and 0.6."
EPSILON_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)

# "the testing values of ϵ include 0.01, 0.02, 0.03, 0.04, 0.05, 0.1 and 0.2."
COMPRESSION_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2)

# "we test b with both small and large values, including 2 to 10, 16, 32, 64,
#  128, 256, 512 and 1024, with the constraint that b < |g|."
B_GRID = (2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 256, 512, 1024)

# "σ is selected from 0.1, 0.2, 0.5, 1, 2, 5 and 10."  sprg does not use σ.
SIGMA_GRID = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

# ---------------------------------------------------------------------------
# paper §4.1 — recommended / "selected parameters" (fig. 6, exp 2/3)
# ---------------------------------------------------------------------------
# "we recommend to use small ε, i.e., 0.1 to 0.2"
# tested values inside that closed interval:
EPSILON_RECOMMENDED = (0.1, 0.15, 0.2)

# "we recommend to set ϵ between 0.02 to 0.1"
COMPRESSION_RECOMMENDED = (0.02, 0.03, 0.04, 0.05, 0.1)

# "b ≤ 16 seems suitable for different datasets"
B_RECOMMENDED = (2, 3, 4, 5, 6, 7, 8, 9, 10, 16)

# exp 2b (fig. 11) does not name (ε, ϵ, b). these are a single point inside
# the recommended ranges, used so regularity and k-means share one operating
# point ("keep all the other parts unchanged").
EXP2B_EPSILON = 0.15
EXP2B_B = 4
EXP2B_COMPRESSION = 0.05

# paper-silent defaults used by the base algorithms (not searched):
# apc: frey–dueck median preference; dset: 1/(1.5 n); d₀: keep every eq. 3 weight.
APC_PREFERENCE_QUANTILE = 50
DSET_WEIGHT_THRESHOLD = None
DENSITY_THRESHOLD = 0

# paper §2: four representative algorithms
BASE_ALGORITHMS = ("SPC", "APC", "DSet", "SPRG")

# sprg (zhu–loy–gong cvpr 2014, ref [20]) settings. the paper uses tclust=1000
# trees and mtry=√d; the node-weighting variant "adpt" (clustrf-strct-adpt,
# the paper's best) and φ=5 are documented paper-silent choices. reduce
# sprg_trees for faster smoke/demo runs.
SPRG_TREES = 1000
SPRG_VARIANT = "adpt"
SPRG_MIN_LEAF = 5

# datasets (table 1): name -> (np, nd, nc)
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


def b_values(n, b_grid=None):
    """paper constraint: ``b < |g|``."""
    return [b for b in (b_grid or B_GRID) if b < n]


def in_grid(value, grid, tol=1e-12):
    """membership that survives csv round-trips (0.1 vs 0.10000000000000001)."""
    v = float(value)
    return any(abs(v - float(x)) < tol for x in grid)
