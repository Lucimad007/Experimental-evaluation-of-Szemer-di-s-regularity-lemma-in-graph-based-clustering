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
# apc: frey–dueck median of off-diagonal similarities; dset: 0.0001 (hou 2023
# [25], the only numeric threshold from this author for §2.3's "greater than a
# threshold"); d₀: keep every eq. 3 weight.
# these are read by runners.make_base_algorithm — change them here, not there.
APC_PREFERENCE_QUANTILE = 50
# dset: replicator cutoff. the paper-faithful cell is hou2023 = 0.0001
# ([25], same first author). none still means 1/(1.5 n) inside dominant_sets
# (the fiorucci variant).
DSET_WEIGHT_THRESHOLD = 0.0001
DENSITY_THRESHOLD = 0

# 0/1 graph for alon. paper-silent: "median" of off-diagonal similarities.
# ``0`` reproduces fiorucci ``sim > 0`` (complete gaussian).
ADJ_THRESHOLD = "mean"
# searched in exp 1/2/3: sim>0 plus adaptive cutoffs of off-diagonal W
ADJ_THRESHOLD_GRID = (0.0, "p25", "median", "mean", "p75")

# how "degree" is read when ordering vertices for initialization / refinement.
# "alon"     = Alon [30] cert|complement packing (Hou modification 2 cites this cut)
# "weighted" = sperotto & pelillo [16] eq. 15 average weighted degree
#              awdeg_S(i) = (1/|S|) Σ_{j∈S} w(i,j), listed in decreasing order.
# "support"  = fiorucci et al. [28] 0/1 degree of (sim_mat > 0). on the paper's
#              gaussian kernel that support is k_n, so this is index order.
# "spectral" = Fiedler split of the class (win-profile path).
DEGREE_MODES = ("alon", "support", "weighted", "spectral")
DEGREE_MODE = "weighted"
STOP_RULE = "theoretical"

# regularity-lemma kind: "alon" is the paper's experimental path (§3.2);
# "frieze_kannan" is implemented (the paper says it performs similarly) and was
# previously never run by the experiment pipeline.
ALG_KINDS = ("alon", "frieze_kannan")
# step 1 of alon: "divide v arbitrarily". degree = fiorucci order; random =
# shuffle into contiguous blocks (partition_initialization.random).
INIT_MODES = ("degree", "random")
# reduced-graph adjacency: all_pairs = algorithm 1 eq. 3 on every class pair
# (previous default, never searched against the alternative);
# regular_only = lemma 2 (keep ε-regular pairs only).
DROP_IRREGULAR_MODES = ("all_pairs", "regular_only")

# ---------------------------------------------------------------------------
# experiment profiles (see RESULTS.md)
# ---------------------------------------------------------------------------
# alon + fiorucci [28] certs + modification 1 refine. τ=mean so the 0/1
# graph is not k_n (τ=0 makes alon vacuous on a gaussian kernel).
PAPER_PROFILE = {
    "degree_mode": "weighted",
    "stop_rule": "theoretical",
    "d0": "0",
    "alg_kind": "alon",
    "init": "degree",
    "drop_irregular": "all_pairs",
    "adj_threshold": "mean",
    "clustering_variants": "paper",
}
# "honest" is the same reading (kept as an alias).
HONEST_PROFILE = {
    "degree_mode": "weighted",
    "stop_rule": "theoretical",
    "d0": "0",
    "alg_kind": "alon",
    "init": "degree",
    "drop_irregular": "all_pairs",
    "adj_threshold": "mean",
    "clustering_variants": "paper",
}
# "win" — graph-faithful path that can beat Tables 2–5 on Wine: z-score,
# kNN sparsify, Fiedler splits, §3.2 Step 3 stop (n_ir ≤ ε·C(k,2)), then
# one nearest-cluster reassignment.
WIN_PROFILE = {
    "degree_mode": "spectral",
    "stop_rule": "theoretical",
    "d0": "0",
    "alg_kind": "alon",
    "init": "degree",
    "drop_irregular": "all_pairs",
    "adj_threshold": "0",
    "clustering_variants": "all",
    "knn": 20,
    "reassign": True,
    "preprocess": "zscore",
}

# graph-faithful defaults when --profile is unset (not the paper-dense gaussian)
KNN_DEFAULT = 20
KNN_GRID = (20, 30)

# paper §3.3 / lemma 2: reduced-graph adjacency threshold d₀. the paper names
# the threshold but never its value (and lemma 2 requires d₀ > ε). searched in
# exp 1/2/3 as a paper-silent extension; 0 = keep every eq. 3 weight (the
# setting used before the search and the pure-paper reading of the pipeline).
# adaptive names are computed from the off-diagonal of that r (already
# implemented in apply_density_threshold, previously unused by experiments).
D0_NUMERIC = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
D0_ADAPTIVE = ("mean", "median", "p90", "p95")
D0_GRID = D0_NUMERIC + D0_ADAPTIVE
D0_ADAPTIVE_ALIASES = {"med": "median"}

# paper §2: four representative algorithms
BASE_ALGORITHMS = ("SPC", "APC", "DSet", "SPRG")

# sprg (zhu–loy–gong cvpr 2014, ref [20]) settings. the paper uses tclust=1000
# trees and mtry=√d; the node-weighting variant "adpt" (clustrf-strct-adpt,
# the paper's best) and φ=5 are documented paper-silent choices. reduce
# sprg_trees for faster smoke/demo runs.
SPRG_TREES = 1000
SPRG_VARIANT = "adpt"
SPRG_MIN_LEAF = 5

# ---------------------------------------------------------------------------
# clustering-function variants (cartesian product with stop / degree / d0)
# see code-review/experiment-permutations.md
# ---------------------------------------------------------------------------
# spc laplacian / embedding (§2.1 + refs [22] shi–malik, [23] njw)
SPC_SPECTRAL_VARIANTS = ("unnormalized", "njw", "shi_malik")
# sprg forest node-weighting (zhu–loy–gong eqs. 9–13)
SPRG_FOREST_VARIANTS = ("adpt", "unfm", "bi")
# apc frey–dueck shared preference ([10]: median vs minimum of similarities)
APC_PREFERENCE_VARIANTS = ("median", "min")
# dset dominant-set weight cutoff + leftover (paper-silent lineages, all searched)
DSET_THRESHOLD_VARIANTS = ("fiorucci", "hou2023", "dslib", "rel95", "fiorucci05")
DSET_VARIANT_THRESHOLD = {
    "fiorucci": None,     # 1/(1.5 n) — fiorucci [28] / our prior default
    "hou2023": 0.0001,    # hou, liu, yuan, pelillo, pr 2023
    "dslib": 1e-5,        # dslib python port
    "rel95": "rel95",     # keep vertices at ≥ 95% of the peak replicator weight
    "fiorucci05": None,   # same cutoff as fiorucci, leftover dump of 5%
}
DSET_VARIANT_LEFTOVER = {
    "fiorucci": 0.0,
    "hou2023": 0.0,
    "dslib": 0.0,
    "rel95": 0.0,
    "fiorucci05": 0.05,   # old fiorucci-lineage dump; §2.3 peels until empty
}
APC_VARIANT_QUANTILE = {
    "median": 50,
    "min": 0,
}


def paper_clustering_variant(algo):
    """the single variant that matches the paper's §2 description of ``algo``."""
    return {
        "SPC": "unnormalized",
        "SPRG": "adpt+unnormalized",
        "APC": "median",
        "DSet": "hou2023",
    }[algo]


def clustering_variant_grid(algo, mode=None):
    """variants of ``algo`` to search. ``mode='paper'`` is one cell; ``all`` is the product."""
    mode = (mode or "all").lower()
    if mode in ("paper", "default"):
        out = (paper_clustering_variant(algo),)
    elif algo == "SPC":
        out = SPC_SPECTRAL_VARIANTS
    elif algo == "SPRG":
        # forest weighting × spectral embedding (sprg's last step is spc)
        out = tuple(
            f"{forest}+{spectral}"
            for forest in SPRG_FOREST_VARIANTS
            for spectral in SPC_SPECTRAL_VARIANTS
        )
    elif algo == "APC":
        out = APC_PREFERENCE_VARIANTS
    elif algo == "DSet":
        out = DSET_THRESHOLD_VARIANTS
    else:
        raise ValueError(f"unknown algorithm: {algo}")
    return out


def parse_clustering_variant(algo, variant=None):
    """decode a ``clustering_variant`` string into kwargs for the base algorithm."""
    if variant is None:
        variant = paper_clustering_variant(algo)
    if algo == "SPC":
        if variant not in SPC_SPECTRAL_VARIANTS and variant not in ("row_kmeans", "row_njw"):
            raise ValueError(f"unknown SPC variant: {variant}")
        return {"spc_variant": variant}
    if algo == "SPRG":
        if "+" in str(variant):
            forest, spectral = str(variant).split("+", 1)
        else:
            forest, spectral = str(variant), "unnormalized"
        if forest not in SPRG_FOREST_VARIANTS:
            raise ValueError(f"unknown SPRG forest variant: {forest}")
        if spectral not in SPC_SPECTRAL_VARIANTS:
            raise ValueError(f"unknown SPRG spectral variant: {spectral}")
        return {"sprg_variant": forest, "spc_variant": spectral}
    if algo == "APC":
        if variant not in APC_VARIANT_QUANTILE:
            raise ValueError(f"unknown APC variant: {variant}")
        return {"preference_quantile": APC_VARIANT_QUANTILE[variant]}
    if algo == "DSet":
        if variant not in DSET_VARIANT_THRESHOLD:
            raise ValueError(f"unknown DSet variant: {variant}")
        return {
            "weight_threshold": DSET_VARIANT_THRESHOLD[variant],
            "leftover_frac": DSET_VARIANT_LEFTOVER[variant],
        }
    raise ValueError(f"unknown algorithm: {algo}")

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


def parse_adj_threshold(raw):
    """alon 0/1 cutoff: 0 / support, median/mean/pXX, or a float."""
    if raw is None:
        return ADJ_THRESHOLD
    if isinstance(raw, (int, float)):
        return 0.0 if float(raw) == 0.0 else float(raw)
    s = str(raw).strip().lower()
    if s in ("", "0", "0.0", "support", "off"):
        return 0.0
    if s in ("median", "med", "mean") or s.startswith("p"):
        return "median" if s == "med" else s
    return float(s)


def canon_adj_threshold(value):
    """stable identity for resume keys (0 vs 0.0 vs 'support')."""
    if value is None:
        return 0.0
    if isinstance(value, float) and value != value:  # NaN
        return 0.0
    p = parse_adj_threshold(value)
    if isinstance(p, float) and abs(p) < 1e-15:
        return 0.0
    return p if isinstance(p, str) else float(p)


def parse_d0_token(raw):
    """one d₀ grid entry: a float, or an adaptive name (mean/median/p90/p95)."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower()
    if not s:
        return 0.0
    if s in D0_ADAPTIVE_ALIASES:
        s = D0_ADAPTIVE_ALIASES[s]
    if s in D0_ADAPTIVE:
        return s
    return float(s)


def canon_d0(value, tol=1e-12):
    """stable identity for resume keys / csv round-trips (0.1 vs 0.1000…01)."""
    parsed = parse_d0_token(value)
    if isinstance(parsed, str):
        return parsed
    for g in D0_NUMERIC:
        if abs(parsed - float(g)) < tol:
            return float(g)
    return float(parsed)


def init_is_random(init_mode):
    return str(init_mode).strip().lower() == "random"


def drop_irregular_pairs(drop_mode):
    """true → lemma 2 (ε-regular pairs only) for the reduced graph."""
    key = str(drop_mode).strip().lower()
    if key in ("regular_only", "true", "1", "yes"):
        return True
    if key in ("all_pairs", "false", "0", "no", ""):
        return False
    raise ValueError(f"unknown drop_irregular mode: {drop_mode}")
