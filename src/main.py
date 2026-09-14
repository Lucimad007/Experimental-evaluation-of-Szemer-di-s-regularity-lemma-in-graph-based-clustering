"""command-line entry point for the experiments.

usage:
    python -m src.main exp1            # parameter influence (§4.1, figs. 2-5)
    python -m src.main exp2            # enhanced vs original (§4.2, figs. 7-10)
    python -m src.main exp2b           # regularity vs kmeans partitioning (fig. 11)
    python -m src.main exp3            # vs recent algorithms (§4.3, tables 2-5)
    python -m src.main eda             # dataset profile (const/dups/scale)
    python -m src.main preprocess      # exp2 protocol on raw vs z-score features
    python -m src.main smoke           # quick smoke test on one small dataset
    python -m src.main all            # run exp1, exp2, exp2b, exp3 in sequence
    python -m src.main invariance      # re-run exp2 winners on shuffled rows

options:
    --datasets Thyroid,Wine           # restrict to a subset of datasets
    --algorithms SPC,APC              # restrict to a subset of base algorithms
    --out results                     # output directory (default: results/)
    --d0 0                            # reduced-graph threshold (0 = keep Eq. 3)
    --verbose                         # print per-run details
"""

import argparse

from . import config
from .experiments import (
    experiment1_parameter_influence,
    experiment2_enhanced_vs_original,
    experiment2b_regularity_vs_kmeans,
    experiment3_vs_recent,
    experiment_preprocess_vs_raw,
)
from . import plotting


def _split(arg):
    # comma-separated cli lists
    return [s.strip() for s in arg.split(",") if s.strip()]


def stop_rules_for(args):
    # partition-loop stop rule(s): both Algorithm 1 line 12 and the §3.2 Step 3 rule
    return ("algorithm1", "theoretical") if args.stop_rule in ("both", "all") else (args.stop_rule,)


def degree_modes_for(args):
    # full search = every implemented refine cut
    return config.DEGREE_MODES if args.degree_mode in ("both", "all") else (args.degree_mode,)


def alg_kinds_for(args):
    return config.ALG_KINDS if args.alg_kind in ("both", "all") else (args.alg_kind,)


def init_modes_for(args):
    return config.INIT_MODES if args.init in ("both", "all") else (args.init,)


def drop_modes_for(args):
    return config.DROP_IRREGULAR_MODES if args.drop_irregular in ("both", "all") else (args.drop_irregular,)


def clustering_variants_mode_for(args):
    return getattr(args, "clustering_variants", None) or "paper"


def adj_thresholds_for(args):
    # Alon 0/1 cutoff: default = full grid (0, p25, median, mean, p75).
    # --adj-threshold grid|all, a comma list, or a single token.
    raw = getattr(args, "adj_threshold", None)
    if raw is None:
        return config.ADJ_THRESHOLD_GRID
    if isinstance(raw, (int, float)):
        return (config.canon_adj_threshold(raw),)
    s = str(raw).strip().lower()
    if s in ("grid", "all", "both"):
        return config.ADJ_THRESHOLD_GRID
    return tuple(
        config.canon_adj_threshold(x) for x in str(raw).split(",") if x.strip()
    )


def d0_grid_for(args):
    # "grid" = the paper-silent d0 search in config.D0_GRID; else comma-separated
    # floats and/or adaptive names (mean, median, p90, p95). 0 keeps every eq. 3
    # weight (the pure-paper reading of §3.3).
    raw = args.d0.strip().lower()
    if raw in ("grid", "all"):
        return config.D0_GRID
    return tuple(config.parse_d0_token(x) for x in args.d0.split(",") if x.strip())


_CLI_AXIS_DEFAULTS = {
    "degree_mode": "spectral",
    "stop_rule": "theoretical",
    "alg_kind": "alon",
    "init": "degree",
    "drop_irregular": "all_pairs",
    "d0": "0",
    "clustering_variants": "all",
    "adj_threshold": "0",
}


def apply_profile(args):
    """Fill unset CLI axes from --profile, else from the high-NMI defaults."""
    chosen = dict(_CLI_AXIS_DEFAULTS)
    profile = getattr(args, "profile", None)
    extra = {}
    if profile == "paper":
        chosen.update(config.PAPER_PROFILE)
    elif profile == "honest":
        chosen.update(config.HONEST_PROFILE)
    elif profile == "win" or profile is None:
        if profile == "win":
            chosen.update(config.WIN_PROFILE)
        extra = {
            "knn": config.WIN_PROFILE.get("knn", config.KNN_DEFAULT),
            "reassign": True,
            "preprocess": "zscore",
        }

    if args.degree_mode is None:
        args.degree_mode = chosen["degree_mode"]
    if args.stop_rule is None:
        args.stop_rule = chosen["stop_rule"]
    if getattr(args, "alg_kind", None) is None:
        args.alg_kind = chosen["alg_kind"]
    if getattr(args, "init", None) is None:
        args.init = chosen["init"]
    if getattr(args, "drop_irregular", None) is None:
        args.drop_irregular = chosen["drop_irregular"]
    if args.d0 is None:
        args.d0 = chosen["d0"]
    if getattr(args, "adj_threshold", None) is None and "adj_threshold" in chosen:
        args.adj_threshold = chosen["adj_threshold"]
    raw_adj = getattr(args, "adj_threshold", None)
    if raw_adj is not None:
        token = str(raw_adj).strip().lower()
        if token not in ("grid", "all", "both") and "," not in str(raw_adj):
            config.ADJ_THRESHOLD = config.parse_adj_threshold(raw_adj)
    if getattr(args, "clustering_variants", None) is None:
        args.clustering_variants = chosen.get("clustering_variants", "paper")
    if extra.get("knn") is not None and getattr(args, "knn", None) is None:
        args.knn = extra["knn"]
    if extra.get("reassign") and not getattr(args, "no_reassign", False):
        if not getattr(args, "reassign", False):
            args.reassign = True
    if extra.get("preprocess") and getattr(args, "preprocess", None) is None:
        args.preprocess = extra["preprocess"]
    if getattr(args, "preprocess", None) is None:
        args.preprocess = "raw"


def run_smoke(args):
    # no network, synthetic blobs
    from .smoke_test import main as _smoke_main
    _smoke_main()


def run_exp1(args):
    # paper §4.1 figs. 2–6
    df = experiment1_parameter_influence(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        stop_rules=stop_rules_for(args),
        degree_modes=degree_modes_for(args),
        d0_grid=d0_grid_for(args),
        alg_kinds=alg_kinds_for(args),
        init_modes=init_modes_for(args),
        drop_modes=drop_modes_for(args),
        adj_thresholds=adj_thresholds_for(args),
        clustering_variants_mode=clustering_variants_mode_for(args),
        out_dir=args.out,
        verbose=args.verbose,
    )
    exp1_dir = (args.out or "results") + "/exp1_parameter_influence"
    plotting.plot_parameter_influence(exp1_dir)
    plotting.plot_all_vs_selected(exp1_dir)
    plotting.plot_axis_comparisons(exp1_dir)
    print(f"[exp1] wrote {len(df)} rows")


def run_exp2(args):
    # paper §4.2 figs. 7–10
    df = experiment2_enhanced_vs_original(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        stop_rules=stop_rules_for(args),
        degree_modes=degree_modes_for(args),
        d0_grid=d0_grid_for(args),
        alg_kinds=alg_kinds_for(args),
        init_modes=init_modes_for(args),
        drop_modes=drop_modes_for(args),
        adj_thresholds=adj_thresholds_for(args),
        clustering_variants_mode=clustering_variants_mode_for(args),
        knn=getattr(args, "knn", None),
        reassign_vertices=bool(getattr(args, "reassign", False))
        and not bool(getattr(args, "no_reassign", False)),
        preprocess=getattr(args, "preprocess", "raw") or "raw",
        out_dir=args.out,
        verbose=args.verbose,
    )
    exp2_dir = (args.out or "results") + "/exp2_enhanced_vs_original"
    plotting.plot_enhanced_vs_original(exp2_dir + "/enhanced_vs_original.csv")
    plotting.plot_axis_comparisons(exp2_dir)
    print(f"[exp2] wrote {len(df)} rows")


def run_exp2b(args):
    # paper §4.2 fig. 11
    df = experiment2b_regularity_vs_kmeans(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        stop_rules=stop_rules_for(args),
        degree_modes=degree_modes_for(args),
        alg_kinds=alg_kinds_for(args),
        init_modes=init_modes_for(args),
        drop_modes=drop_modes_for(args),
        adj_thresholds=adj_thresholds_for(args),
        clustering_variants_mode=clustering_variants_mode_for(args),
        out_dir=args.out,
        verbose=args.verbose,
    )
    exp2b_dir = (args.out or "results") + "/exp2b_regularity_vs_kmeans"
    plotting.plot_regularity_vs_kmeans(exp2b_dir + "/regularity_vs_kmeans.csv")
    plotting.plot_axis_comparisons(exp2b_dir)
    print(f"[exp2b] wrote {len(df)} rows")


def run_exp3(args):
    # paper §4.3 tables 2–5
    tables = experiment3_vs_recent(
        dataset_names=_split(args.datasets) if args.datasets else None,
        stop_rules=stop_rules_for(args),
        degree_modes=degree_modes_for(args),
        d0_grid=d0_grid_for(args),
        alg_kinds=alg_kinds_for(args),
        init_modes=init_modes_for(args),
        drop_modes=drop_modes_for(args),
        adj_thresholds=adj_thresholds_for(args),
        clustering_variants_mode=clustering_variants_mode_for(args),
        out_dir=args.out,
        verbose=args.verbose,
    )
    exp3_dir = (args.out or "results") + "/exp3_vs_recent"
    plotting.plot_vs_recent(exp3_dir)
    plotting.plot_axis_comparisons(exp3_dir)
    print(f"[exp3] wrote {len(tables)} metric tables (recent + Reg-*)")


def run_eda(args):
    from .eda import original_preview, profile_all

    names = _split(args.datasets) if args.datasets else None
    df = profile_all(dataset_names=names, out_dir=args.out)
    print(f"[eda] wrote {len(df)} dataset rows")
    prev = original_preview(
        dataset_names=names,
        algorithms=_split(args.algorithms) if args.algorithms else ("SPC",),
        out_dir=args.out,
    )
    print(f"[eda] original NMI preview: {len(prev)} rows (raw vs zscore, no regularity)")


def run_preprocess(args):
    df = experiment_preprocess_vs_raw(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        stop_rules=stop_rules_for(args),
        degree_modes=degree_modes_for(args),
        d0_grid=d0_grid_for(args),
        alg_kinds=alg_kinds_for(args),
        init_modes=init_modes_for(args),
        drop_modes=drop_modes_for(args),
        adj_thresholds=adj_thresholds_for(args),
        clustering_variants_mode=clustering_variants_mode_for(args),
        out_dir=args.out,
        verbose=args.verbose,
    )
    prep_dir = (args.out or "results") + "/exp_preprocess_vs_raw"
    plotting.plot_raw_vs_zscore(prep_dir + "/raw_vs_zscore.csv")
    plotting.plot_axis_comparisons(prep_dir)
    print(f"[preprocess] wrote {len(df)} rows")


def run_invariance(args):
    # re-evaluate each exp2 winning config on a row-permuted copy of x
    from .invariance_check import check_exp2_csv

    csv_path = (args.out or "results") + "/exp2_enhanced_vs_original/enhanced_vs_original.csv"
    check_exp2_csv(csv_path)


def _force_utf8_console():
    # Windows consoles default to cp1252, which cannot encode ε / ϵ / σ used in
    # verbose output (e.g. experiments.py's per-run [err] line); without this
    # those runs die with UnicodeEncodeError mid-experiment
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _force_utf8_console()
    p = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["smoke", "exp1", "exp2", "exp2b", "exp3",
                                       "eda", "preprocess", "all", "invariance"])
    p.add_argument("--datasets", default=None, help="comma-separated dataset names")
    p.add_argument("--algorithms", default=None, help="comma-separated algorithm names")
    p.add_argument("--stop-rule", choices=["both", "all", "algorithm1", "theoretical"],
                   default=None,
                   help="partition-loop stop rule(s): default 'both' = theoretical and "
                        "algorithm1; a single name pins one rule")
    p.add_argument("--degree-mode", choices=["both", "all", "alon", "support", "weighted", "spectral"],
                   default=None,
                   help="vertex order for init / leftover packing. 'spectral' also "
                        "uses Fiedler class-halving (default). other names keep Alon+mod1")
    p.add_argument("--alg-kind", choices=["both", "all", "alon", "frieze_kannan", "fiorucci"],
                   default=None, dest="alg_kind",
                   help="regularity lemma: alon (this repo), frieze_kannan, or "
                        "fiorucci (cloned dense_graph_reducer). "
                        "'both'/'all' still search alon + frieze_kannan only")
    p.add_argument("--init", choices=["both", "all", "degree", "random"],
                   default=None,
                   help="partition initialization: default 'both' = degree and random")
    p.add_argument("--drop-irregular", choices=["both", "all", "all_pairs", "regular_only"],
                   default=None, dest="drop_irregular",
                   help="reduced-graph adjacency: default 'both' = all_pairs and regular_only")
    p.add_argument("--d0", default=None,
                   help="reduced-graph d0: default 'grid' = config.D0_GRID; "
                        "0 = keep every Eq. 3 weight; or comma-separated floats/names")
    p.add_argument("--adj-threshold", default=None, dest="adj_threshold",
                   help="0/1 cutoff for the Alon graph: default searches "
                        "0,p25,median,mean,p75; 'grid'/'all' same; a comma list "
                        "or a single token (0 = sim>0 complete Gaussian)")
    p.add_argument("--knn", type=int, default=None,
                   help="symmetrized kNN sparsify of the gaussian (default 20 unless "
                        "--profile paper)")
    p.add_argument("--reassign", action="store_true",
                   help="after Algorithm 1 mapping, reassign every vertex to the "
                        "cluster with highest mean similarity (on by default)")
    p.add_argument("--no-reassign", action="store_true", dest="no_reassign",
                   help="keep Algorithm 1 class-constant labels (paper mapping)")
    p.add_argument("--preprocess", choices=["raw", "zscore"], default=None,
                   help="feature view for exp2: default zscore; 'raw' is the paper cell")
    p.add_argument("--clustering-variants", choices=["all", "paper"], default=None,
                   dest="clustering_variants",
                   help="'all' (default) = every implemented variant; "
                        "'paper' = unnormalized SPC, adpt+unnormalized SPRG, "
                        "median APC, Hou-2023 DSet")
    p.add_argument("--profile", choices=["paper", "honest", "win"], default=None,
                   help="'paper' pins the Tables 2–5 path (no permutation search); "
                        "'honest' = median Alon cutoff; "
                        "'win' = z-score + kNN + Fiedler splits + reassignment")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    apply_profile(args)

    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "invariance":
        run_invariance(args)
    elif args.command == "eda":
        run_eda(args)
    elif args.command == "preprocess":
        run_preprocess(args)
    elif args.command in ("exp1", "all"):
        run_exp1(args)
    if args.command in ("exp2", "all"):
        run_exp2(args)
    if args.command in ("exp2b", "all"):
        run_exp2b(args)
    if args.command in ("exp3", "all"):
        run_exp3(args)


if __name__ == "__main__":
    main()
