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
):
    """return a configured szemerediRegularityLemma.

    kind: "alon" (paper: "we use the one by alon et al.") or "frieze_kannan".
    epsilon: paper ε. density_threshold: paper d₀ (§3.3, unnamed; 0 = keep all).
    """
    alg = SzemerediRegularityLemma(
        sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,
        density_threshold=density_threshold,
    )

    # paper: "dividing v arbitrarily" — degree order is the fiorucci default
    alg.partition_initialization = (
        partition_initialization.random if random_initialization else partition_initialization.degree_based
    )
    # randomized refinement is not implemented
    alg.refinement_step = refinement_step.degree_based

    if kind == "alon":
        # paper: three alon tests; experiments use this
        alg.conditions = [conditions.alon1, conditions.alon2, conditions.alon3]
    elif kind == "frieze_kannan":
        alg.conditions = [conditions.frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg
