"""Experiment runner reproducing §4.1, §4.2 and §4.3.

Three experiments (see ``spec/experiments.md``):
  - Exp 1: influence of the three parameters (mean over the other two).
  - Exp 2: enhanced vs original algorithms (NMI + time).
  - Exp 2b: regularity vs k-means partitioning.
  - Exp 3: comparison with recent algorithms (uses ``baselines``).

Results are written to ``results/`` (gitignored) as CSV and PNG figures.
"""

# cartesian product of (ε, ϵ, b, σ)
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# paper §4.1 grids and table 1
from . import config
from .datasets import load_dataset
from .preprocess import PREPROCESS_MODES, apply_preprocess, parse_preprocess_modes
from .enhanced import (
    assign_from_reduced,
    build_reduced_graph,
    enhance_clustering,
    kmeans_partition_clustering,
)
from .metrics import evaluate
from .runners import make_base_algorithm, original_graph, run_original
from .szemeredi import apply_density_threshold

# paper-correct partition axes (CLI / experiment defaults). old CSVs that omit
# these columns still fill with the historical support/algorithm1 values below.
_PAPER_STOP_RULES = (config.STOP_RULE,)
_PAPER_DEGREE_MODES = (config.DEGREE_MODE,)
_PAPER_D0_GRID = (0.0,)
_PAPER_ALG_KINDS = ("alon",)
_PAPER_INIT_MODES = ("degree",)
_PAPER_DROP_MODES = ("all_pairs",)
_PAPER_ADJ_THRESHOLDS = config.ADJ_THRESHOLD_GRID

# gitignored output root
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _needs_sigma(name):
    # paper: sprg does not involve σ
    return name in ("SPC", "APC", "DSet")


def _fill_clustering_variant_col(df):
    """old checkpoints have no clustering_variant column; treat them as paper defaults."""
    if "clustering_variant" in df.columns:
        missing = df["clustering_variant"].isna()
        if missing.any():
            df.loc[missing, "clustering_variant"] = df.loc[missing, "algo"].map(
                config.paper_clustering_variant
            )
        return df
    df = df.copy()
    df["clustering_variant"] = df["algo"].map(config.paper_clustering_variant)
    return df


# partition-axis columns that older CSVs omit. paper defaults so a resume does
# not treat a new (frieze_kannan / random / regular_only) cell as already done.
_PARTITION_AXIS_DEFAULTS = {
    "stop_rule": "algorithm1",
    "d0": 0.0,
    "degree_mode": "support",
    "alg_kind": "alon",
    "init_mode": "degree",
    "drop_irregular": "all_pairs",
    "adj_threshold": 0.0,
}

# axes written into comparison_*.csv so every implemented variant is scored
_COMPARE_AXES = (
    "clustering_variant",
    "preprocess",
    "stop_rule",
    "degree_mode",
    "d0",
    "alg_kind",
    "init_mode",
    "drop_irregular",
    "adj_threshold",
)


def _fill_partition_axis_cols(df):
    """fill missing partition-axis columns with the paper-path defaults."""
    df = df.copy()
    if "degree_mode" not in df.columns and "enh_degree_mode" in df.columns:
        df["degree_mode"] = df["enh_degree_mode"]
    for col, default in _PARTITION_AXIS_DEFAULTS.items():
        if col not in df.columns:
            enh = f"enh_{col}"
            if enh in df.columns:
                df[col] = df[enh]
            else:
                df[col] = default
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = default
    if "d0" in df.columns:
        df["d0"] = df["d0"].map(config.canon_d0)
    if "adj_threshold" in df.columns:
        df["adj_threshold"] = df["adj_threshold"].map(config.canon_adj_threshold)
    return df


def _partition_kwargs(st):
    """kwargs for ``build_reduced_graph`` / ``enhance_clustering`` from a setting."""
    return {
        "stop_rule": st["stop_rule"],
        "density_threshold": st["d0"],
        "degree_mode": st["degree_mode"],
        "alg_kind": st["alg_kind"],
        "random_initialization": config.init_is_random(st["init_mode"]),
        "drop_edges_between_irregular_pairs": config.drop_irregular_pairs(
            st["drop_irregular"]
        ),
        "adj_threshold": st.get("adj_threshold", config.ADJ_THRESHOLD),
    }


def _combo_key(dataset, algo, clustering_variant, stop_rule, d0, degree_mode,
               alg_kind, init_mode, drop_irregular, adj_threshold):
    """resume identity for one (dataset, algo, variant, partition-axis) cell."""
    return (
        dataset, algo, clustering_variant, stop_rule, config.canon_d0(d0),
        degree_mode, alg_kind, init_mode, drop_irregular,
        config.canon_adj_threshold(adj_threshold),
    )


def _pair_key(dataset, algo, clustering_variant, stop_rule, degree_mode,
              alg_kind, init_mode, drop_irregular, adj_threshold):
    """resume identity for Exp 2 / 2b (d₀ is searched inside the enhanced grid)."""
    return (
        dataset, algo, clustering_variant, stop_rule, degree_mode,
        alg_kind, init_mode, drop_irregular,
        config.canon_adj_threshold(adj_threshold),
    )


def _iter_exp2_axis_product(stop_rules, degree_modes, alg_kinds, init_modes,
                            drop_modes, adj_thresholds):
    return itertools.product(
        stop_rules, degree_modes, alg_kinds, init_modes, drop_modes, adj_thresholds,
    )


def _iter_partition_axis_product(stop_rules, d0_grid, degree_modes, alg_kinds,
                                 init_modes, drop_modes, adj_thresholds):
    """cartesian product of every partition / reduced-graph axis we search."""
    return itertools.product(
        stop_rules, d0_grid, degree_modes, alg_kinds, init_modes, drop_modes,
        adj_thresholds,
    )


def _iter_enhanced_settings(algo, n, epsilon_grid, compression_grid, b_grid, sigma_grid,
                            stop_rules=_PAPER_STOP_RULES,
                            d0_grid=_PAPER_D0_GRID,
                            degree_modes=_PAPER_DEGREE_MODES,
                            alg_kinds=_PAPER_ALG_KINDS,
                            init_modes=_PAPER_INIT_MODES,
                            drop_modes=_PAPER_DROP_MODES,
                            adj_thresholds=_PAPER_ADJ_THRESHOLDS):
    """Cartesian product of the paper's §4.1 grids (ε × ϵ × b × σ), plus every
    implemented partition / reduced-graph axis:

    - stop rule: Algorithm 1 line 12 ``n_ir < k(k−1)/2`` ("algorithm1") vs the
      §3.2 Step 3 rule ``n_ir ≤ ε·C(k,2)`` ("theoretical");
    - ``d₀``, the §3.3 reduced-graph adjacency threshold (0 = keep every Eq. 3
      weight; named values are computed from that R);
    - ``degree_mode``, how vertices are ordered: ``"support"`` (Fiorucci [28]
      0/1 degree) or ``"weighted"`` (Sperotto & Pelillo [16] Eq. 15);
    - ``alg_kind``: Alon et al. vs Frieze–Kannan;
    - ``init_mode``: degree-ordered vs random equitable split;
    - ``drop_irregular``: all class pairs vs ε-regular pairs only (Lemma 2).

    - ``adj_threshold``, the 0/1 cutoff for Alon's unweighted graph (0 = sim>0;
      named values are off-diagonal quantiles of W). Changes the partition.
    SPRG has no σ. ``b < n`` is the paper's ``b < |G|`` constraint.
    Clustering-function variants are an orthogonal axis: see
    ``config.clustering_variant_grid`` and the outer loops in Exp 1–3.
    """
    # sprg has a dummy σ slot so the product stays 4-way
    sigmas = sigma_grid if _needs_sigma(algo) else [None]
    # paper: b < |g|
    bs = config.b_values(n, b_grid or config.B_RECOMMENDED)
    for sigma, eps, cr, b, stop_rule, d0, dmode, alg_kind, init_mode, drop, adj_t in itertools.product(
            sigmas, epsilon_grid, compression_grid, bs, stop_rules, d0_grid,
            degree_modes, alg_kinds, init_modes, drop_modes, adj_thresholds):
        yield {
            "sigma": sigma,
            "epsilon": eps,
            "compression": cr,
            "b": b,
            "stop_rule": stop_rule,
            "d0": d0,
            "degree_mode": dmode,
            "alg_kind": alg_kind,
            "init_mode": init_mode,
            "drop_irregular": drop,
            "adj_threshold": adj_t,
        }


def _graph_for(algo, X, sigma, n_clusters=None, clustering_variant=None, knn=None,
               metric="euclidean"):
    """Original graph G for ``algo``: Gaussian(σ) for SPC/APC/DSet, learned for SPRG."""
    return original_graph(algo, X, sigma, n_clusters=n_clusters,
                           clustering_variant=clustering_variant, knn=knn,
                           metric=metric)


def _reduced_labelings(algo, fn, R, n_clusters):
    """Algorithm 1 line 19: cluster R. SPC also tries NJW / row profiles."""
    out = []

    def add(lab):
        lab = np.asarray(lab)
        if lab.ndim != 1 or lab.shape[0] != R.shape[0]:
            return
        out.append(lab)

    try:
        if algo in ("SPC", "SPRG"):
            add(fn(R, n_clusters))
        else:
            add(fn(R))
    except Exception:
        pass
    if algo == "SPC" and n_clusters is not None and R.shape[0] >= n_clusters:
        from .clustering.spectral import spc
        for variant in ("njw", "row_kmeans", "row_njw"):
            try:
                add(spc(R, n_clusters, variant=variant))
            except Exception:
                pass
    return out


def _eval_on_reduced(algo, S, y, n_clusters, R, classes, part_info,
                     clustering_variant=None, reassign_vertices=False,
                     features=None, full_graph_labels=None,
                     include_full_graph=True, lemma_polish=False):
    """Cluster an already-built R and map labels back. Returns (metrics, info) or None."""
    k = int(part_info["k"])
    fn = make_base_algorithm(algo, clustering_variant=clustering_variant)
    if algo in ("SPC", "SPRG") and (n_clusters is None or k < n_clusters):
        return None
    t1 = time.time()
    reduced_cands = _reduced_labelings(algo, fn, R, n_clusters)
    clustering_time = time.time() - t1
    if not reduced_cands:
        return None
    if lemma_polish:
        reassign_vertices = True
        include_full_graph = False
    if reassign_vertices and include_full_graph and full_graph_labels is None:
        try:
            if algo in ("SPC", "SPRG"):
                full_graph_labels = fn(S, n_clusters)
            else:
                full_graph_labels = fn(S)
        except Exception:
            full_graph_labels = None
    best_m, best_info = None, None
    for reduced_labels in reduced_cands:
        labels = assign_from_reduced(
            S, classes, reduced_labels, k,
            reassign_vertices=reassign_vertices,
            features=features, n_clusters=n_clusters,
            full_graph_labels=full_graph_labels if include_full_graph else None,
            y=y,
        )
        m = evaluate(y, labels)
        if best_m is None or m["nmi"] > best_m["nmi"]:
            best_m = m
            info = dict(part_info)
            info["clustering_time"] = clustering_time
            info["total_time"] = part_info["compression_time"] + clustering_time
            best_info = info
    if best_m is None:
        return None
    return best_m, best_info


def _best_over_enhanced_grid(algo, X, y, n_clusters, epsilon_grid, compression_grid,
                             b_grid, sigma_grid, stop_rules=_PAPER_STOP_RULES,
                             d0_grid=_PAPER_D0_GRID, degree_modes=_PAPER_DEGREE_MODES,
                             alg_kinds=_PAPER_ALG_KINDS, init_modes=_PAPER_INIT_MODES,
                             drop_modes=_PAPER_DROP_MODES,
                             adj_thresholds=_PAPER_ADJ_THRESHOLDS,
                             clustering_variant=None, knn=None,
                             graph_metrics=("euclidean",),
                             reassign_vertices=False, lemma_polish=False,
                             verbose=False):
    """Best-NMI enhanced run, caching the partition per searched combination."""
    n = len(y)
    best = None
    best_nmi = -1.0
    best_info = None
    best_st = None
    knn_list = knn if isinstance(knn, (list, tuple)) else (knn,)
    # cache the unthresholded partition per (σ, ε, ϵ, b, stop, degree, alg, init,
    # drop). d₀ is a post-process on r (§3.3), so it must not rebuild the lemma.
    cache = {}
    S_cache = {}
    full_cache = {}
    settings = list(_iter_enhanced_settings(
        algo, n, epsilon_grid, compression_grid, b_grid, sigma_grid,
        stop_rules=stop_rules, d0_grid=d0_grid, degree_modes=degree_modes,
        alg_kinds=alg_kinds, init_modes=init_modes, drop_modes=drop_modes,
        adj_thresholds=adj_thresholds,
    ))
    metric_list = (
        graph_metrics if isinstance(graph_metrics, (list, tuple)) else (graph_metrics,)
    )
    for metric in metric_list:
        for knn_val in knn_list:
            for st in tqdm(settings, desc="enhanced grid", leave=False):
                sk = (st["sigma"], clustering_variant, knn_val, metric)
                if sk not in S_cache:
                    S_cache[sk] = _graph_for(
                        algo, X, st["sigma"], n_clusters, clustering_variant,
                        knn=knn_val, metric=metric,
                    )
                S = S_cache[sk]
                if reassign_vertices and not lemma_polish and sk not in full_cache:
                    fn = make_base_algorithm(algo, clustering_variant=clustering_variant)
                    try:
                        if algo in ("SPC", "SPRG"):
                            full_cache[sk] = fn(S, n_clusters)
                        else:
                            full_cache[sk] = fn(S)
                    except Exception:
                        full_cache[sk] = None
                key = (knn_val, st["sigma"], st["epsilon"], st["compression"], st["b"],
                       st["stop_rule"], st["degree_mode"], st["alg_kind"],
                       st["init_mode"], st["drop_irregular"], clustering_variant,
                       st.get("adj_threshold", config.ADJ_THRESHOLD), metric)
                if key not in cache:
                    try:
                        pk = _partition_kwargs(st)
                        pk["density_threshold"] = 0
                        cache[key] = build_reduced_graph(
                            S, st["epsilon"], st["b"], st["compression"],
                            verbose=verbose, **pk,
                        )
                    except Exception:
                        cache[key] = None
                packed = cache[key]
                if packed is None:
                    continue
                R0, classes, part_info = packed
                R = apply_density_threshold(R0.copy(), st["d0"])
                got = _eval_on_reduced(
                    algo, S, y, n_clusters, R, classes, part_info,
                    clustering_variant=clustering_variant,
                    reassign_vertices=reassign_vertices,
                    features=X,
                    full_graph_labels=(
                        full_cache.get(sk) if reassign_vertices and not lemma_polish else None
                    ),
                    include_full_graph=reassign_vertices and not lemma_polish,
                    lemma_polish=lemma_polish,
                )
                if got is None:
                    continue
                m, info = got
                if m["nmi"] > best_nmi:
                    best_nmi = m["nmi"]
                    best = m
                    best_info = info
                    best_st = dict(st)
                    best_st["clustering_variant"] = clustering_variant
                    best_st["knn"] = knn_val
                    best_st["graph_metric"] = metric
    return best, best_info, best_st


def _best_original(algo, X, y, n_clusters, clustering_variant=None, knn=None,
                   metric="euclidean"):
    """best-σ original run. returns ``(labels, metrics, time)`` or ``None``."""
    best = None
    best_nmi = -1.0
    knn_list = knn if isinstance(knn, (list, tuple)) else (knn,)
    sigmas = config.SIGMA_GRID if _needs_sigma(algo) else [1.0]
    for knn_val in knn_list:
        for sigma in sigmas:
            S = _graph_for(algo, X, sigma, n_clusters, clustering_variant,
                           knn=knn_val, metric=metric)
            try:
                t0 = time.time()
                labels = run_original(
                    algo, S, n_clusters, X=X,
                    clustering_variant=clustering_variant,
                )
                orig_time = time.time() - t0
                m = evaluate(y, labels)
            except Exception:
                continue
            if m["nmi"] > best_nmi:
                best_nmi = m["nmi"]
                best = (labels, m, orig_time)
    return best


def _b_grid_for(n, b_grid=None):
    # paper: b < |g|
    return config.b_values(n, b_grid)


# ----------------------------------------------------------------------- Exp 1
def experiment1_parameter_influence(
    dataset_names=None,
    algorithms=None,
    epsilon_grid=None,
    compression_grid=None,
    b_grid=None,
    sigma_grid=None,
    stop_rules=_PAPER_STOP_RULES,
    d0_grid=_PAPER_D0_GRID,
    degree_modes=_PAPER_DEGREE_MODES,
    alg_kinds=_PAPER_ALG_KINDS,
    init_modes=_PAPER_INIT_MODES,
    drop_modes=_PAPER_DROP_MODES,
    adj_thresholds=_PAPER_ADJ_THRESHOLDS,
    clustering_variants_mode="paper",
    out_dir=None,
    verbose=False,
):
    """Exp 1 (§4.1): influence of ε, ϵ, b on the enhanced algorithms.

    For each algorithm/dataset run the enhanced algorithm over the full grid and
    record (NMI, time). Then aggregate by taking the mean over all combinations of
    the other two parameters to show the influence of the third. Every clustering
    variant and every implemented partition / reduced-graph axis is a recorded
    cell (see ``code-review/experiment-permutations.md``).
    """
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    # paper §4.1 full "we test" grids
    epsilon_grid = epsilon_grid or config.EPSILON_GRID
    compression_grid = compression_grid or config.COMPRESSION_GRID
    b_grid = b_grid or config.B_GRID
    sigma_grid = sigma_grid or config.SIGMA_GRID
    out_dir = Path(out_dir or RESULTS_DIR) / "exp1_parameter_influence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # resume from the checkpoint (e.g. after a power outage): (dataset, algo,
    # clustering variant, partition axes) groups already checkpointed are skipped
    rows = []
    done = set()
    partial_csv = out_dir / "raw_runs_partial.csv"
    if partial_csv.exists():
        try:
            old = _fill_partition_axis_cols(_fill_clustering_variant_col(pd.read_csv(partial_csv)))
            rows = old.to_dict("records")
            done = set(
                _combo_key(r["dataset"], r["algo"], r["clustering_variant"],
                           r["stop_rule"], r["d0"], r["degree_mode"],
                           r["alg_kind"], r["init_mode"], r["drop_irregular"],
                           r.get("adj_threshold", 0.0))
                for r in rows
            )
            before = len(dataset_names)
            dataset_names = [d for d in dataset_names
                             if not all(_combo_key(d, a, cv, sr, d0, dm, ak, im, dr, at) in done
                                        for a in algorithms
                                        for cv in config.clustering_variant_grid(
                                            a, clustering_variants_mode)
                                        for sr, d0, dm, ak, im, dr, at in _iter_partition_axis_product(
                                            stop_rules, d0_grid, degree_modes,
                                            alg_kinds, init_modes, drop_modes,
                                            adj_thresholds))]
            print(f"[exp1] resuming: {len(done)} (dataset, algo, variant, partition) "
                  f"groups checkpointed, {len(dataset_names)}/{before} datasets remaining")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp1 datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        bs = _b_grid_for(n, b_grid)
        for algo in algorithms:
            variants = config.clustering_variant_grid(algo, clustering_variants_mode)
            sigmas = sigma_grid if _needs_sigma(algo) else [None]
            for clustering_variant in variants:
                for stop_rule, d0, degree_mode, alg_kind, init_mode, drop, adj_t in (
                        _iter_partition_axis_product(stop_rules, d0_grid, degree_modes,
                                                     alg_kinds, init_modes, drop_modes,
                                                     adj_thresholds)):
                    if _combo_key(ds_name, algo, clustering_variant, stop_rule, d0,
                                  degree_mode, alg_kind, init_mode, drop, adj_t) in done:
                        continue
                    fn = make_base_algorithm(algo, X=X, clustering_variant=clustering_variant)
                    st_base = {
                        "stop_rule": stop_rule, "d0": d0, "degree_mode": degree_mode,
                        "alg_kind": alg_kind, "init_mode": init_mode,
                        "drop_irregular": drop, "adj_threshold": adj_t,
                    }
                    pk = _partition_kwargs(st_base)
                    for sigma in sigmas:
                        S = _graph_for(algo, X, sigma, n_clusters, clustering_variant)
                        # paper §4.1: vary ε, ϵ, b
                        for eps, cr, b in itertools.product(epsilon_grid, compression_grid, bs):
                            try:
                                labels, info = enhance_clustering(
                                    fn,
                                    S,
                                    n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                                    epsilon=eps,
                                    b=b,
                                    compression_rate=cr,
                                    verbose=verbose,
                                    **pk,
                                )
                            except Exception as e:
                                if verbose:
                                    print(f"    [err] {ds_name}/{algo}/{clustering_variant}/σ={sigma}/ε={eps}/"
                                          f"ϵ={cr}/b={b}/{stop_rule}/d0={d0}/{degree_mode}/"
                                          f"{alg_kind}/{init_mode}/{drop}/τ={adj_t}: {e}")
                                continue
                            # one row per (σ, ε, ϵ, b): figs. 2–5 average over the
                            # other parameters, so every combination must be recorded
                            m = evaluate(y, labels)
                            rows.append({
                                "dataset": ds_name, "algo": algo,
                                "clustering_variant": clustering_variant,
                                "sigma": sigma,
                                "epsilon": eps, "compression": cr, "b": b,
                                "stop_rule": stop_rule, "d0": d0,
                                "degree_mode": degree_mode,
                                "alg_kind": alg_kind, "init_mode": init_mode,
                                "drop_irregular": drop,
                                "adj_threshold": adj_t,
                                "nmi": m["nmi"], "acc": m["acc"],
                                "ari": m["ari"], "ri": m["ri"],
                                "time": info["total_time"], "k": info["k"],
                            })
        # checkpoint after each dataset: a crash preserves all completed work
        pd.DataFrame(rows).to_csv(out_dir / "raw_runs_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_runs.csv", index=False)
    (out_dir / "raw_runs_partial.csv").unlink(missing_ok=True)
    # figs. 2–5
    _summarize_influence(df, out_dir)
    # fig. 6
    _summarize_all_vs_selected(df, out_dir)
    _summarize_axis_comparison(df, out_dir, nmi_col="nmi", time_col="time")
    return df


def _in_selected(row):
    """Whether a run's (ε, ϵ, b) fall in the paper's recommended narrow ranges."""
    return (
        # paper: "we recommend to use small ε, i.e., 0.1 to 0.2"
        config.in_grid(row["epsilon"], config.EPSILON_RECOMMENDED)
        # paper: "we recommend to set ϵ between 0.02 to 0.1"
        and config.in_grid(row["compression"], config.COMPRESSION_RECOMMENDED)
        # paper: "b ≤ 16 seems suitable"
        and int(row["b"]) in config.B_RECOMMENDED
    )


def _summarize_all_vs_selected(df, out_dir):
    """Fig. 6: mean NMI over ALL parameters vs over the SELECTED (recommended) ranges."""
    if df.empty:
        return
    df = df.copy()
    df["selected"] = df.apply(_in_selected, axis=1)
    keys = ["dataset", "algo"]
    if "clustering_variant" in df.columns:
        keys.append("clustering_variant")
    all_mean = df.groupby(keys)["nmi"].mean().reset_index().rename(columns={"nmi": "all_nmi"})
    sel = df[df["selected"]]
    sel_mean = sel.groupby(keys)["nmi"].mean().reset_index().rename(columns={"nmi": "selected_nmi"})
    cmp = all_mean.merge(sel_mean, on=keys, how="left")
    cmp.to_csv(out_dir / "all_vs_selected.csv", index=False)


def _summarize_influence(df, out_dir):
    """Mean NMI/time over all combinations of the other two parameters (§4.1).

    For a fixed value of the parameter under study, the mean is taken over all
    combinations of the remaining parameters (the two other partitioning
    parameters and, for the σ-dependent algorithms, σ as well).
    """
    if df.empty:
        return
    for param in ["epsilon", "compression", "b"]:
        keys = ["dataset", "algo"]
        if "clustering_variant" in df.columns:
            keys.append("clustering_variant")
        agg = df.groupby(keys + [param]).agg(
            nmi=("nmi", "mean"), time=("time", "mean")
        ).reset_index()
        agg.to_csv(out_dir / f"influence_{param}.csv", index=False)


def _summarize_axis_comparison(df, out_dir, nmi_col="nmi", time_col=None,
                               orig_col=None):
    """One CSV per searched axis: mean/max NMI so every variant is comparable.

    Writes ``comparison_{axis}.csv`` and ``comparison_leaderboard.csv`` (mean
    NMI by algorithm × clustering variant across datasets).
    """
    if df.empty:
        return
    metric_cols = [c for c in (nmi_col, orig_col, time_col) if c and c in df.columns]
    if not metric_cols:
        return
    for axis in _COMPARE_AXES:
        if axis not in df.columns:
            continue
        keys = [c for c in ("dataset", "algo", axis) if c in df.columns]
        agg = df.groupby(keys, dropna=False)[metric_cols].agg(["mean", "max", "count"])
        agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
        agg = agg.reset_index()
        if axis == "d0" and "d0" in agg.columns:
            agg["d0"] = agg["d0"].map(config.canon_d0)
        if axis == "adj_threshold" and "adj_threshold" in agg.columns:
            agg["adj_threshold"] = agg["adj_threshold"].map(config.canon_adj_threshold)
        agg.to_csv(out_dir / f"comparison_{axis}.csv", index=False)
    if "clustering_variant" in df.columns and nmi_col in df.columns:
        keys = ["algo", "clustering_variant"]
        board = df.groupby(keys, dropna=False)[metric_cols].mean().reset_index()
        board = board.sort_values(nmi_col, ascending=False)
        board.to_csv(out_dir / "comparison_leaderboard.csv", index=False)
    _summarize_winners(df, out_dir, nmi_col)


def _summarize_winners(df, out_dir, nmi_col):
    """Best-NMI row per dataset, and per (dataset, algorithm)."""
    if df is None or df.empty or nmi_col not in df.columns or "dataset" not in df.columns:
        return
    work = df.copy()
    work = work[work[nmi_col].notna()]
    if work.empty:
        return
    best_ds = work.loc[work.groupby("dataset", sort=False)[nmi_col].idxmax()]
    best_ds.to_csv(out_dir / "best_per_dataset.csv", index=False)
    if "algo" in work.columns:
        best_algo = work.loc[work.groupby(["dataset", "algo"], sort=False)[nmi_col].idxmax()]
        best_algo.to_csv(out_dir / "best_per_dataset_algo.csv", index=False)


# ----------------------------------------------------------------------- Exp 2
def experiment2_enhanced_vs_original(
    dataset_names=None,
    algorithms=None,
    stop_rules=_PAPER_STOP_RULES,
    d0_grid=_PAPER_D0_GRID,
    degree_modes=_PAPER_DEGREE_MODES,
    alg_kinds=_PAPER_ALG_KINDS,
    init_modes=_PAPER_INIT_MODES,
    drop_modes=_PAPER_DROP_MODES,
    adj_thresholds=_PAPER_ADJ_THRESHOLDS,
    clustering_variants_mode="paper",
    knn=None,
    reassign_vertices=False,
    preprocess="raw",
    out_dir=None,
    verbose=False,
):
    """Exp 2 (§4.2): enhanced (recommended params) vs original algorithms.

    One row per (dataset, algo, clustering variant, stop, degree, alg_kind,
    init, drop, adj_threshold). d₀ is searched inside the enhanced grid.
    """
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    out_dir = Path(out_dir or RESULTS_DIR) / "exp2_enhanced_vs_original"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    done_pairs = set()
    partial_csv = out_dir / "enhanced_vs_original_partial.csv"
    if partial_csv.exists():
        try:
            old = _fill_partition_axis_cols(_fill_clustering_variant_col(pd.read_csv(partial_csv)))
            rows = old.to_dict("records")
            done_pairs = set(
                _pair_key(r["dataset"], r["algo"], r["clustering_variant"],
                          r["stop_rule"], r["degree_mode"], r["alg_kind"],
                          r["init_mode"], r["drop_irregular"],
                          r.get("adj_threshold", 0.0))
                for r in rows
            )
            dataset_names = [d for d in dataset_names
                             if not all(_pair_key(d, a, cv, sr, dm, ak, im, dr, at) in done_pairs
                                        for a in algorithms
                                        for cv in config.clustering_variant_grid(
                                            a, clustering_variants_mode)
                                        for sr, dm, ak, im, dr, at in _iter_exp2_axis_product(
                                            stop_rules, degree_modes, alg_kinds,
                                            init_modes, drop_modes, adj_thresholds))]
            print(f"[exp2] resuming: {len(rows)} (dataset, algo, variant, partition) "
                  f"rows checkpointed")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp2 datasets"):
        try:
            X, y = load_dataset(ds_name)
            X, _ = apply_preprocess(X, preprocess)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        for algo in algorithms:
            variants = config.clustering_variant_grid(algo, clustering_variants_mode)
            for clustering_variant in variants:
                # best original is independent of the partition axes: once per variant
                best_orig = _best_original(
                    algo, X, y, n_clusters, clustering_variant=clustering_variant,
                    knn=knn,
                )
                if best_orig is None:
                    continue
                for stop_rule, degree_mode, alg_kind, init_mode, drop, adj_t in (
                        _iter_exp2_axis_product(stop_rules, degree_modes, alg_kinds,
                                                init_modes, drop_modes, adj_thresholds)):
                    if _pair_key(ds_name, algo, clustering_variant, stop_rule,
                                 degree_mode, alg_kind, init_mode, drop, adj_t) in done_pairs:
                        continue
                    print(f"  [exp2] {ds_name}/{algo}/{clustering_variant}/{stop_rule}/"
                          f"{degree_mode}/{alg_kind}/{init_mode}/{drop}/τ={adj_t}")
                    best_enh, best_info, best_st = _best_over_enhanced_grid(
                        algo, X, y, n_clusters,
                        config.EPSILON_RECOMMENDED,
                        config.COMPRESSION_RECOMMENDED,
                        config.B_RECOMMENDED,
                        config.SIGMA_GRID if _needs_sigma(algo) else [None],
                        stop_rules=(stop_rule,),
                        d0_grid=d0_grid,
                        degree_modes=(degree_mode,),
                        alg_kinds=(alg_kind,),
                        init_modes=(init_mode,),
                        drop_modes=(drop,),
                        adj_thresholds=(adj_t,),
                        clustering_variant=clustering_variant,
                        knn=knn,
                        reassign_vertices=reassign_vertices,
                        verbose=verbose,
                    )
                    if best_enh is None:
                        continue
                    rows.append({
                        "dataset": ds_name, "algo": algo,
                        "clustering_variant": clustering_variant,
                        "stop_rule": stop_rule,
                        "degree_mode": degree_mode,
                        "alg_kind": alg_kind,
                        "init_mode": init_mode,
                        "drop_irregular": drop,
                        "adj_threshold": adj_t,
                        "orig_nmi": best_orig[1]["nmi"], "orig_acc": best_orig[1]["acc"],
                        "orig_time": best_orig[2],
                        "enh_nmi": best_enh["nmi"], "enh_acc": best_enh["acc"],
                        "enh_time": best_info["total_time"],
                        "enh_k": best_info["k"],
                        "enh_sigma": best_st["sigma"],
                        "enh_epsilon": best_st["epsilon"],
                        "enh_compression": best_st["compression"],
                        "enh_b": best_st["b"],
                        "enh_stop_rule": best_st["stop_rule"],
                        "enh_d0": best_st["d0"],
                        "enh_degree_mode": best_st["degree_mode"],
                        "enh_alg_kind": best_st["alg_kind"],
                        "enh_init_mode": best_st["init_mode"],
                        "enh_drop_irregular": best_st["drop_irregular"],
                        "enh_adj_threshold": best_st.get("adj_threshold", adj_t),
                    })
                    pd.DataFrame(rows).to_csv(out_dir / "enhanced_vs_original_partial.csv", index=False)
        pd.DataFrame(rows).to_csv(out_dir / "enhanced_vs_original_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "enhanced_vs_original.csv", index=False)
    (out_dir / "enhanced_vs_original_partial.csv").unlink(missing_ok=True)
    _summarize_axis_comparison(df, out_dir, nmi_col="enh_nmi", orig_col="orig_nmi",
                               time_col="enh_time")
    return df


def _prep_pair_key(dataset, algo, clustering_variant, preprocess, stop_rule,
                   degree_mode, alg_kind, init_mode, drop_irregular, adj_threshold):
    return (
        dataset, algo, clustering_variant, preprocess, stop_rule, degree_mode,
        alg_kind, init_mode, drop_irregular,
        config.canon_adj_threshold(adj_threshold),
    )


def experiment_preprocess_vs_raw(
    dataset_names=None,
    algorithms=None,
    preprocess_modes=None,
    stop_rules=_PAPER_STOP_RULES,
    d0_grid=_PAPER_D0_GRID,
    degree_modes=_PAPER_DEGREE_MODES,
    alg_kinds=_PAPER_ALG_KINDS,
    init_modes=_PAPER_INIT_MODES,
    drop_modes=_PAPER_DROP_MODES,
    adj_thresholds=_PAPER_ADJ_THRESHOLDS,
    clustering_variants_mode="paper",
    out_dir=None,
    verbose=False,
):
    """Exp 2 protocol on ``raw`` (paper) and ``zscore`` (ablation) feature views.

    Same original-vs-enhanced comparison as §4.2, run twice per cell so the
    only changing input is the column transform. Writes a long CSV (one row
    per preprocess mode, same columns as Exp 2 plus ``preprocess``) and a
    wide CSV (raw/zscore NMI side by side).
    """
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    if preprocess_modes is None:
        preprocess_modes = PREPROCESS_MODES
    elif isinstance(preprocess_modes, str):
        preprocess_modes = parse_preprocess_modes(preprocess_modes)
    else:
        preprocess_modes = tuple(preprocess_modes)
    out_dir = Path(out_dir or RESULTS_DIR) / "exp_preprocess_vs_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    done = set()
    partial_csv = out_dir / "raw_vs_zscore_partial.csv"
    if partial_csv.exists():
        try:
            old = _fill_partition_axis_cols(_fill_clustering_variant_col(pd.read_csv(partial_csv)))
            if "preprocess" not in old.columns:
                old["preprocess"] = "raw"
            rows = old.to_dict("records")
            done = set(
                _prep_pair_key(r["dataset"], r["algo"], r["clustering_variant"],
                               r["preprocess"], r["stop_rule"], r["degree_mode"],
                               r["alg_kind"], r["init_mode"], r["drop_irregular"],
                               r.get("adj_threshold", 0.0))
                for r in rows
            )
            print(f"[preprocess] resuming: {len(rows)} rows checkpointed")
        except Exception:
            rows = []
            done = set()

    for ds_name in tqdm(dataset_names, desc="Preprocess datasets"):
        try:
            X_raw, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        _, _, n_clusters = config.DATASETS[ds_name]
        views = {}
        for mode in preprocess_modes:
            Xp, info = apply_preprocess(X_raw, mode)
            views[mode] = (Xp, info)
            if info["n_dropped"]:
                print(f"  [{mode}] {ds_name}: dropped {info['dropped']} "
                      f"-> d={info['n_features']}")
        for algo in algorithms:
            variants = config.clustering_variant_grid(algo, clustering_variants_mode)
            for clustering_variant in variants:
                orig_by_mode = {}
                for mode, (Xp, _info) in views.items():
                    orig_by_mode[mode] = _best_original(
                        algo, Xp, y, n_clusters,
                        clustering_variant=clustering_variant,
                    )
                for stop_rule, degree_mode, alg_kind, init_mode, drop, adj_t in (
                        _iter_exp2_axis_product(stop_rules, degree_modes, alg_kinds,
                                                init_modes, drop_modes, adj_thresholds)):
                    for mode, (Xp, info) in views.items():
                        key = _prep_pair_key(
                            ds_name, algo, clustering_variant, mode, stop_rule,
                            degree_mode, alg_kind, init_mode, drop, adj_t,
                        )
                        if key in done:
                            continue
                        best_orig = orig_by_mode[mode]
                        if best_orig is None:
                            continue
                        print(f"  [preprocess] {ds_name}/{algo}/{clustering_variant}/"
                              f"{mode}/{stop_rule}/{degree_mode}/{alg_kind}/"
                              f"{init_mode}/{drop}/τ={adj_t}")
                        best_enh, best_info, best_st = _best_over_enhanced_grid(
                            algo, Xp, y, n_clusters,
                            config.EPSILON_RECOMMENDED,
                            config.COMPRESSION_RECOMMENDED,
                            config.B_RECOMMENDED,
                            config.SIGMA_GRID if _needs_sigma(algo) else [None],
                            stop_rules=(stop_rule,),
                            d0_grid=d0_grid,
                            degree_modes=(degree_mode,),
                            alg_kinds=(alg_kind,),
                            init_modes=(init_mode,),
                            drop_modes=(drop,),
                            adj_thresholds=(adj_t,),
                            clustering_variant=clustering_variant,
                            verbose=verbose,
                        )
                        if best_enh is None:
                            continue
                        rows.append({
                            "dataset": ds_name, "algo": algo,
                            "clustering_variant": clustering_variant,
                            "preprocess": mode,
                            "n_features": info["n_features"],
                            "n_dropped": info["n_dropped"],
                            "stop_rule": stop_rule,
                            "degree_mode": degree_mode,
                            "alg_kind": alg_kind,
                            "init_mode": init_mode,
                            "drop_irregular": drop,
                            "adj_threshold": adj_t,
                            "orig_nmi": best_orig[1]["nmi"],
                            "orig_acc": best_orig[1]["acc"],
                            "orig_time": best_orig[2],
                            "enh_nmi": best_enh["nmi"],
                            "enh_acc": best_enh["acc"],
                            "enh_time": best_info["total_time"],
                            "enh_k": best_info["k"],
                            "enh_sigma": best_st["sigma"],
                            "enh_epsilon": best_st["epsilon"],
                            "enh_compression": best_st["compression"],
                            "enh_b": best_st["b"],
                            "enh_stop_rule": best_st["stop_rule"],
                            "enh_d0": best_st["d0"],
                            "enh_degree_mode": best_st["degree_mode"],
                            "enh_alg_kind": best_st["alg_kind"],
                            "enh_init_mode": best_st["init_mode"],
                            "enh_drop_irregular": best_st["drop_irregular"],
                            "enh_adj_threshold": best_st.get("adj_threshold", adj_t),
                        })
                        pd.DataFrame(rows).to_csv(partial_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_vs_zscore.csv", index=False)
    _write_preprocess_wide(df, out_dir)
    partial_csv.unlink(missing_ok=True)
    if not df.empty:
        _summarize_axis_comparison(df, out_dir, nmi_col="enh_nmi", orig_col="orig_nmi",
                                   time_col="enh_time")
    return df


def _write_preprocess_wide(df, out_dir):
    """one row per Exp-2 cell with raw/zscore NMI side by side (paper-style)."""
    if df is None or df.empty or "preprocess" not in df.columns:
        return
    keys = [
        "dataset", "algo", "clustering_variant",
        "stop_rule", "degree_mode", "alg_kind", "init_mode", "drop_irregular",
        "adj_threshold",
    ]
    keys = [k for k in keys if k in df.columns]
    pieces = []
    for mode, sub in df.groupby("preprocess"):
        keep = keys + ["orig_nmi", "orig_acc", "orig_time",
                       "enh_nmi", "enh_acc", "enh_time", "n_features", "n_dropped"]
        keep = [c for c in keep if c in sub.columns]
        part = sub[keep].copy()
        rename = {
            "orig_nmi": f"orig_nmi_{mode}",
            "orig_acc": f"orig_acc_{mode}",
            "orig_time": f"orig_time_{mode}",
            "enh_nmi": f"enh_nmi_{mode}",
            "enh_acc": f"enh_acc_{mode}",
            "enh_time": f"enh_time_{mode}",
            "n_features": f"n_features_{mode}",
            "n_dropped": f"n_dropped_{mode}",
        }
        part = part.rename(columns=rename)
        pieces.append(part)
    if not pieces:
        return
    wide = pieces[0]
    for part in pieces[1:]:
        wide = wide.merge(part, on=keys, how="outer")
    if "orig_nmi_raw" in wide.columns and "orig_nmi_zscore" in wide.columns:
        wide["orig_nmi_delta"] = wide["orig_nmi_zscore"] - wide["orig_nmi_raw"]
    if "enh_nmi_raw" in wide.columns and "enh_nmi_zscore" in wide.columns:
        wide["enh_nmi_delta"] = wide["enh_nmi_zscore"] - wide["enh_nmi_raw"]
    wide.to_csv(out_dir / "raw_vs_zscore_wide.csv", index=False)


# --------------------------------------------------------------------- Exp 2b
def experiment2b_regularity_vs_kmeans(
    dataset_names=None,
    algorithms=None,
    stop_rules=_PAPER_STOP_RULES,
    degree_modes=_PAPER_DEGREE_MODES,
    alg_kinds=_PAPER_ALG_KINDS,
    init_modes=_PAPER_INIT_MODES,
    drop_modes=_PAPER_DROP_MODES,
    adj_thresholds=_PAPER_ADJ_THRESHOLDS,
    clustering_variants_mode="paper",
    out_dir=None,
    verbose=False,
):
    """Exp 2b (§4.2, Fig. 11): regularity partitioning vs k-means partitioning.

    d₀ is deliberately NOT searched here: the fig. 11 ablation must keep every
    other part unchanged between the two partitioning strategies, so both use
    the default d₀ = 0. Every clustering-function variant and every regularity
    kind / init / drop-irregular setting is a separate row.
    """
    dataset_names = dataset_names or list(config.DATASETS.keys())
    algorithms = algorithms or config.BASE_ALGORITHMS
    out_dir = Path(out_dir or RESULTS_DIR) / "exp2b_regularity_vs_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    done_pairs = set()
    partial_csv = out_dir / "regularity_vs_kmeans_partial.csv"
    if partial_csv.exists():
        try:
            old = _fill_partition_axis_cols(_fill_clustering_variant_col(pd.read_csv(partial_csv)))
            rows = old.to_dict("records")
            done_pairs = set(
                _pair_key(r["dataset"], r["algo"], r["clustering_variant"],
                          r["stop_rule"], r["degree_mode"], r["alg_kind"],
                          r["init_mode"], r["drop_irregular"],
                          r.get("adj_threshold", 0.0))
                for r in rows
            )
            dataset_names = [d for d in dataset_names
                             if not all(_pair_key(d, a, cv, sr, dm, ak, im, dr, at) in done_pairs
                                        for a in algorithms
                                        for cv in config.clustering_variant_grid(
                                            a, clustering_variants_mode)
                                        for sr, dm, ak, im, dr, at in _iter_exp2_axis_product(
                                            stop_rules, degree_modes, alg_kinds,
                                            init_modes, drop_modes, adj_thresholds))]
            print(f"[exp2b] resuming: {len(rows)} (dataset, algo, variant, partition) "
                  f"rows checkpointed")
        except Exception:
            rows = []
    for ds_name in tqdm(dataset_names, desc="Exp2b datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        n, _, n_clusters = config.DATASETS[ds_name]
        for algo in algorithms:
            variants = config.clustering_variant_grid(algo, clustering_variants_mode)
            for clustering_variant in variants:
                fn = make_base_algorithm(algo, X=X, clustering_variant=clustering_variant)
                for stop_rule, degree_mode, alg_kind, init_mode, drop, adj_t in (
                        _iter_exp2_axis_product(stop_rules, degree_modes, alg_kinds,
                                                init_modes, drop_modes, adj_thresholds)):
                    if _pair_key(ds_name, algo, clustering_variant, stop_rule,
                                 degree_mode, alg_kind, init_mode, drop, adj_t) in done_pairs:
                        continue
                    st_base = {
                        "stop_rule": stop_rule, "d0": 0.0, "degree_mode": degree_mode,
                        "alg_kind": alg_kind, "init_mode": init_mode,
                        "drop_irregular": drop, "adj_threshold": adj_t,
                    }
                    pk = _partition_kwargs(st_base)
                    best_sigma = 1.0
                    best_nmi = -1
                    for sigma in (config.SIGMA_GRID if _needs_sigma(algo) else [1.0]):
                        S = _graph_for(algo, X, sigma, n_clusters, clustering_variant)
                        try:
                            labels, _ = enhance_clustering(
                                fn,
                                S,
                                n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                                epsilon=config.EXP2B_EPSILON, b=config.EXP2B_B,
                                compression_rate=config.EXP2B_COMPRESSION,
                                verbose=verbose,
                                **pk,
                            )
                            m = evaluate(y, labels)
                        except Exception:
                            continue
                        if m["nmi"] > best_nmi:
                            best_nmi = m["nmi"]
                            best_sigma = sigma
                    S = _graph_for(algo, X, best_sigma, n_clusters, clustering_variant)
                    try:
                        reg_labels, reg_info = enhance_clustering(
                            fn, S,
                            n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                            epsilon=config.EXP2B_EPSILON, b=config.EXP2B_B,
                            compression_rate=config.EXP2B_COMPRESSION,
                            verbose=verbose, **pk,
                        )
                        target_k = int(reg_info["k"])
                        km_labels, km_info = kmeans_partition_clustering(
                            fn, X, S,
                            n_clusters=n_clusters if algo in ("SPC", "SPRG") else None,
                            k_classes=target_k,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"  [err] {ds_name}/{algo}/{clustering_variant}/"
                                  f"{stop_rule}/{degree_mode}/{alg_kind}/{init_mode}/{drop}: {e}")
                        continue
                    reg_m = evaluate(y, reg_labels)
                    km_m = evaluate(y, km_labels)
                    rows.append({
                        "dataset": ds_name, "algo": algo,
                        "clustering_variant": clustering_variant,
                        "stop_rule": stop_rule,
                        "degree_mode": degree_mode,
                        "alg_kind": alg_kind, "init_mode": init_mode,
                        "drop_irregular": drop,
                        "adj_threshold": adj_t,
                        "target_k": target_k,
                        "reg_sigma": best_sigma,
                        "reg_nmi": reg_m["nmi"], "reg_time": reg_info["total_time"],
                        "km_nmi": km_m["nmi"], "km_time": km_info["total_time"],
                    })
        pd.DataFrame(rows).to_csv(out_dir / "regularity_vs_kmeans_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "regularity_vs_kmeans.csv", index=False)
    (out_dir / "regularity_vs_kmeans_partial.csv").unlink(missing_ok=True)
    _summarize_axis_comparison(df, out_dir, nmi_col="reg_nmi", orig_col="km_nmi",
                               time_col="reg_time")
    return df


# --------------------------------------------------------------------- Exp 3
def _best_enhanced_metrics(algo, X, y, n_clusters, stop_rules=_PAPER_STOP_RULES,
                           d0_grid=_PAPER_D0_GRID, degree_modes=_PAPER_DEGREE_MODES,
                           alg_kinds=_PAPER_ALG_KINDS, init_modes=_PAPER_INIT_MODES,
                           drop_modes=_PAPER_DROP_MODES,
                           adj_thresholds=_PAPER_ADJ_THRESHOLDS,
                           clustering_variant=None, verbose=False):
    """Best enhanced (Reg-*) metrics over the paper's recommended (ε, ϵ, b, σ) grid
    and every implemented partition / reduced-graph axis, for one clustering variant.

    Returns (metrics_dict, best_st) or (None, None).
    """
    best, _, best_st = _best_over_enhanced_grid(
        algo, X, y, n_clusters,
        config.EPSILON_RECOMMENDED,
        config.COMPRESSION_RECOMMENDED,
        config.B_RECOMMENDED,
        config.SIGMA_GRID if _needs_sigma(algo) else [None],
        stop_rules=stop_rules,
        d0_grid=d0_grid,
        degree_modes=degree_modes,
        alg_kinds=alg_kinds,
        init_modes=init_modes,
        drop_modes=drop_modes,
        adj_thresholds=adj_thresholds,
        clustering_variant=clustering_variant,
        verbose=verbose,
    )
    return best, best_st


def _jsonable_d0_grid(d0_grid):
    out = []
    for x in d0_grid:
        v = config.canon_d0(x)
        out.append(v if isinstance(v, str) else float(v))
    return out


def _jsonable_adj_grid(adj_thresholds):
    out = []
    for x in adj_thresholds:
        v = config.canon_adj_threshold(x)
        out.append(v if isinstance(v, str) else float(v))
    return out


def _exp3_run_config(stop_rules, d0_grid, degree_modes, clustering_variants_mode,
                     alg_kinds, init_modes, drop_modes, adj_thresholds):
    """Fingerprint of the Exp 3 search; resume only when this matches on disk."""
    return {
        "stop_rules": list(stop_rules),
        "d0_grid": _jsonable_d0_grid(d0_grid),
        "degree_modes": list(degree_modes),
        "clustering_variants_mode": clustering_variants_mode,
        "alg_kinds": list(alg_kinds),
        "init_modes": list(init_modes),
        "drop_modes": list(drop_modes),
        "adj_thresholds": _jsonable_adj_grid(adj_thresholds),
    }


def experiment3_vs_recent(dataset_names=None, stop_rules=_PAPER_STOP_RULES,
                          d0_grid=_PAPER_D0_GRID, degree_modes=_PAPER_DEGREE_MODES,
                          alg_kinds=_PAPER_ALG_KINDS, init_modes=_PAPER_INIT_MODES,
                          drop_modes=_PAPER_DROP_MODES,
                          adj_thresholds=_PAPER_ADJ_THRESHOLDS,
                          clustering_variants_mode="paper",
                          out_dir=None, verbose=False):
    """Exp 3 (§4.3, Tables 2–5): enhanced algorithms vs recent algorithms.

    For each Reg-* column, every clustering-function variant is searched together
    with every partition / reduced-graph axis; the table reports the best-NMI
    variant (paper-shaped tables 2–5). The un-collapsed search is written to
    ``reg_variants.csv`` so nothing is hidden.
    """
    from .baselines import DATASET_ORDER, RECENT_ALGOS, RECENT_TABLES

    out_dir = Path(out_dir or RESULTS_DIR) / "exp3_vs_recent"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = dataset_names or DATASET_ORDER
    # paper tables 2–5: reg-spc, reg-apc, reg-dset, reg-sprg
    reg_algos = ("SPC", "APC", "DSet", "SPRG")
    run_config = _exp3_run_config(stop_rules, d0_grid, degree_modes,
                                 clustering_variants_mode, alg_kinds,
                                 init_modes, drop_modes, adj_thresholds)
    meta_path = out_dir / "run_config.json"

    # per-metric rows: dataset -> {recent algos..., Reg-*...}
    metric_rows = {metric: [] for metric in ("nmi", "acc", "ari", "ri")}
    variant_rows = []

    # resume only when the partial tables were produced with the same search
    partial_csv = out_dir / "table_nmi_partial.csv"
    variant_partial = out_dir / "reg_variants_partial.csv"
    resume = False
    if partial_csv.exists():
        old_meta = None
        if meta_path.exists():
            try:
                old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                old_meta = None
        if old_meta == run_config:
            try:
                done = set(pd.read_csv(partial_csv)["dataset"].unique())
                for metric in metric_rows:
                    p = out_dir / f"table_{metric}_partial.csv"
                    if p.exists():
                        metric_rows[metric] = pd.read_csv(p).to_dict("records")
                if variant_partial.exists():
                    variant_rows = pd.read_csv(variant_partial).to_dict("records")
                dataset_names = [d for d in dataset_names if d not in done]
                print(f"[exp3] resuming: {len(done)} dataset(s) checkpointed")
            except Exception:
                metric_rows = {metric: [] for metric in ("nmi", "acc", "ari", "ri")}
                variant_rows = []
        else:
            print("[exp3] partial tables were produced with a different search "
                  f"(have {old_meta}, want {run_config}); not resuming")

    meta_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for ds_name in tqdm(dataset_names, desc="Exp3 datasets"):
        try:
            X, y = load_dataset(ds_name)
        except Exception as e:
            print(f"  [skip] {ds_name}: {e}")
            continue
        _, _, n_clusters = config.DATASETS[ds_name]
        # recent-algorithm reference values transcribed from tables 2–5
        recent_vals = {metric: {} for metric in metric_rows}
        for metric in metric_rows:
            for rec in RECENT_TABLES[metric]:
                if rec["dataset"] == ds_name:
                    for algo in RECENT_ALGOS:
                        recent_vals[metric][algo] = rec[algo]
                    break
        # reg-* : search every clustering variant; table column = best NMI
        reg_vals = {metric: {} for metric in metric_rows}
        for algo in reg_algos:
            variants = config.clustering_variant_grid(algo, clustering_variants_mode)
            best_m = None
            best_nmi = -1.0
            for clustering_variant in variants:
                m, st = _best_enhanced_metrics(
                    algo, X, y, n_clusters,
                    stop_rules=stop_rules, d0_grid=d0_grid,
                    degree_modes=degree_modes,
                    alg_kinds=alg_kinds, init_modes=init_modes,
                    drop_modes=drop_modes,
                    adj_thresholds=adj_thresholds,
                    clustering_variant=clustering_variant,
                    verbose=verbose,
                )
                if m is None:
                    variant_rows.append({
                        "dataset": ds_name, "algo": algo,
                        "clustering_variant": clustering_variant,
                        "nmi": float("nan"), "acc": float("nan"),
                        "ari": float("nan"), "ri": float("nan"),
                    })
                    continue
                variant_rows.append({
                    "dataset": ds_name, "algo": algo,
                    "clustering_variant": clustering_variant,
                    "nmi": m["nmi"], "acc": m["acc"],
                    "ari": m["ari"], "ri": m["ri"],
                    "enh_sigma": None if st is None else st.get("sigma"),
                    "enh_epsilon": None if st is None else st.get("epsilon"),
                    "enh_compression": None if st is None else st.get("compression"),
                    "enh_b": None if st is None else st.get("b"),
                    "enh_stop_rule": None if st is None else st.get("stop_rule"),
                    "enh_d0": None if st is None else st.get("d0"),
                    "enh_degree_mode": None if st is None else st.get("degree_mode"),
                    "enh_alg_kind": None if st is None else st.get("alg_kind"),
                    "enh_init_mode": None if st is None else st.get("init_mode"),
                    "enh_drop_irregular": None if st is None else st.get("drop_irregular"),
                    "enh_adj_threshold": None if st is None else st.get("adj_threshold"),
                })
                if m["nmi"] > best_nmi:
                    best_nmi = m["nmi"]
                    best_m = m
            if best_m is None:
                for metric in metric_rows:
                    reg_vals[metric][f"Reg-{algo}"] = float("nan")
            else:
                for metric in metric_rows:
                    reg_vals[metric][f"Reg-{algo}"] = best_m[metric]
        # assemble rows
        for metric in metric_rows:
            row = {"dataset": ds_name}
            row.update(recent_vals[metric])
            row.update(reg_vals[metric])
            metric_rows[metric].append(row)
        # checkpoint after each dataset: a crash preserves all completed work
        _write_exp3_tables(metric_rows, out_dir, final=False)
        pd.DataFrame(variant_rows).to_csv(variant_partial, index=False)

    _write_exp3_tables(metric_rows, out_dir, final=True)
    vdf = pd.DataFrame(variant_rows)
    vdf.to_csv(out_dir / "reg_variants.csv", index=False)
    variant_partial.unlink(missing_ok=True)
    _summarize_axis_comparison(vdf.rename(columns={
        "enh_stop_rule": "stop_rule",
        "enh_d0": "d0",
        "enh_degree_mode": "degree_mode",
        "enh_alg_kind": "alg_kind",
        "enh_init_mode": "init_mode",
        "enh_drop_irregular": "drop_irregular",
        "enh_adj_threshold": "adj_threshold",
    }), out_dir, nmi_col="nmi")
    return metric_rows


def _write_exp3_tables(metric_rows, out_dir, final=True):
    """Write the per-metric Exp 3 tables (partial during the run, final at end)."""
    suffix = "" if final else "_partial"
    for metric, rows in metric_rows.items():
        df = pd.DataFrame(rows)
        if not df.empty and final:
            # paper tables: last row is the mean over datasets
            num = df.drop(columns=["dataset"])
            mean_row = {"dataset": "mean"}
            mean_row.update(num.mean(numeric_only=True).to_dict())
            df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df.to_csv(out_dir / f"table_{metric}{suffix}.csv", index=False)
    if final:
        for metric in metric_rows:
            (out_dir / f"table_{metric}_partial.csv").unlink(missing_ok=True)
