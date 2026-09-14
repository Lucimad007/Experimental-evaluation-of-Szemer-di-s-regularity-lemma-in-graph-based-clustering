"""wire alon vs frieze–kannan into szemerediRegularityLemma."""

from . import conditions, partition_initialization, refinement_step
from .regularity_lemma import SzemerediRegularityLemma


def build_regularity_lemma(
    kind,
    sim_mat,
    epsilon,
    is_weighted,
    random_initialization,
    random_refinement,
    drop_edges_between_irregular_pairs,
    density_threshold=0,
    degree_mode="weighted",
    adj_threshold="mean",
):
    """return a configured szemerediRegularityLemma.

    kind: "alon", "frieze_kannan", or "fiorucci" (upstream dense_graph_reducer).
    epsilon: paper ε. density_threshold: paper d₀ (§3.3, unnamed; 0 = keep all).
    degree_mode: vertex order for initialization (and leftover packing).
    "spectral" = Fiedler order for those steps. refinement is always alon
    [30] step 4 with hou modification 1. alon conditions run on the 0/1
    support; eq. 3 is used only to weight r.
    """
    if kind == "fiorucci":
        from .fiorucci_ref import FiorucciRegularityLemma

        return FiorucciRegularityLemma(
            sim_mat,
            epsilon,
            is_weighted,
            drop_edges_between_irregular_pairs,
            kind="alon",
            random_initialization=random_initialization,
            random_refinement=random_refinement,
        )

    alg = SzemerediRegularityLemma(
        sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,
        density_threshold=density_threshold, degree_mode=degree_mode,
        adj_threshold=adj_threshold,
    )

    # paper: "dividing v arbitrarily" — degree order is the fiorucci default.
    # both modes subdivide alg.degrees, which the constructor set per degree_mode
    alg.partition_initialization = (
        partition_initialization.random if random_initialization else partition_initialization.degree_based
    )
    # hou §3.2: alon step 4, ≤1 irregular partner → cert vs complement.
    alg.refinement_step = refinement_step.alon_based

    if kind == "alon":
        # hou §3.2 mod 2: certificates from fiorucci [28] (alon3), not alon2.
        # alon1 is the cheap sparse-regular test; alon2 is [30] only if [28]
        # does not decide.
        alg.conditions = [conditions.alon1, conditions.alon3, conditions.alon2]
    elif kind == "frieze_kannan":
        alg.conditions = [conditions.frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg
