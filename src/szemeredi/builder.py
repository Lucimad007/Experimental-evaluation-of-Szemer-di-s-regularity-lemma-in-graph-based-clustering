"""Builder for the regularity-lemma implementation (Alon vs Frieze–Kannan)."""

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
    """Construct a configured ``SzemerediRegularityLemma`` instance.

    Parameters
    ----------
    kind : str
        ``"alon"`` (Alon et al., 3 conditions) or ``"frieze_kannan"``.
    sim_mat : np.ndarray
        Similarity (or adjacency) matrix of the graph.
    epsilon : float
        Regularity parameter ``ε``.
    is_weighted : bool
        Whether the graph is edge-weighted (use Eq. 3 densities).
    random_initialization : bool
        Random vs degree-based initial partition.
    random_refinement : bool
        Random vs degree-based refinement (only degree-based is implemented here).
    drop_edges_between_irregular_pairs : bool
        If True, the reduced matrix keeps only regular pairs; otherwise it is
        fully connected (all pairs contribute their density).
    density_threshold : float or str
        ``d₀`` of §3.3: zero Eq. 3 weights at or below this value (see
        ``apply_density_threshold``). Default 0 keeps every pair density.
    """
    alg = SzemerediRegularityLemma(
        sim_mat, epsilon, is_weighted, drop_edges_between_irregular_pairs,
        density_threshold=density_threshold,
    )

    alg.partition_initialization = (
        partition_initialization.random if random_initialization else partition_initialization.degree_based
    )
    alg.refinement_step = refinement_step.degree_based  # randomized refinement not implemented

    if kind == "alon":
        # paper: "we use the one by Alon et al. in our experiments"
        alg.conditions = [conditions.alon1, conditions.alon2, conditions.alon3]
    elif kind == "frieze_kannan":
        alg.conditions = [conditions.frieze_kannan]
    else:
        raise ValueError("Could not find the specified graph summarization method")

    return alg
