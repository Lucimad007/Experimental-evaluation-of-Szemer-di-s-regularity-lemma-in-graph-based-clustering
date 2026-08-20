"""Command-line entry point for the experiments.

Usage:
    python -m src.main exp1            # parameter influence (§4.1, Figs. 2-5)
    python -m src.main exp2            # enhanced vs original (§4.2, Figs. 7-10)
    python -m src.main exp2b           # regularity vs kmeans partitioning (Fig. 11)
    python -m src.main exp3            # vs recent algorithms (§4.3, Tables 2-5)
    python -m src.main smoke           # quick smoke test on one small dataset
    python -m src.main all            # run exp1, exp2, exp2b, exp3 in sequence

Options:
    --datasets Thyroid,Wine           # restrict to a subset of datasets
    --algorithms SPC,APC              # restrict to a subset of base algorithms
    --out results                     # output directory (default: results/)
    --verbose                         # print per-run details
"""

import argparse
import sys

from . import config
from .datasets import load_dataset
from .enhanced import enhance_clustering
from .enhanced.similarity import gaussian_similarity
from .experiments import (
    experiment1_parameter_influence,
    experiment2_enhanced_vs_original,
    experiment2b_regularity_vs_kmeans,
    experiment3_vs_recent,
)
from .metrics import evaluate
from .runners import make_base_algorithm
from . import plotting


def _split(arg):
    return [s.strip() for s in arg.split(",") if s.strip()]


def run_smoke(args):
    from .smoke_test import main as _smoke_main
    _smoke_main()


def run_exp1(args):
    df = experiment1_parameter_influence(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        out_dir=args.out,
        verbose=args.verbose,
    )
    plotting.plot_parameter_influence(df.attrs.get("out_dir") or (args.out or "results") + "/exp1_parameter_influence")
    print(f"[exp1] wrote {len(df)} rows")


def run_exp2(args):
    df = experiment2_enhanced_vs_original(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        out_dir=args.out,
        verbose=args.verbose,
    )
    plotting.plot_enhanced_vs_original((args.out or "results") + "/exp2_enhanced_vs_original/enhanced_vs_original.csv")
    print(f"[exp2] wrote {len(df)} rows")


def run_exp2b(args):
    df = experiment2b_regularity_vs_kmeans(
        dataset_names=_split(args.datasets) if args.datasets else None,
        algorithms=_split(args.algorithms) if args.algorithms else None,
        out_dir=args.out,
        verbose=args.verbose,
    )
    plotting.plot_regularity_vs_kmeans((args.out or "results") + "/exp2b_regularity_vs_kmeans/regularity_vs_kmeans.csv")
    print(f"[exp2b] wrote {len(df)} rows")


def run_exp3(args):
    tables = experiment3_vs_recent(
        dataset_names=_split(args.datasets) if args.datasets else None,
        out_dir=args.out,
        verbose=args.verbose,
    )
    plotting.plot_vs_recent((args.out or "results") + "/exp3_vs_recent")
    print(f"[exp3] wrote {len(tables)} metric tables (recent + Reg-*)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["smoke", "exp1", "exp2", "exp2b", "exp3", "all"])
    p.add_argument("--datasets", default=None, help="comma-separated dataset names")
    p.add_argument("--algorithms", default=None, help="comma-separated algorithm names")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.command == "smoke":
        run_smoke(args)
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
