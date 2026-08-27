"""Merge SPRG-only re-runs into existing experiment outputs.

The long experiment runs launched before the SPRG affinity fixes (commit
"Fix two SPRG affinity bugs...") computed Reg-SPRG with the buggy affinity;
their SPC/APC/DSet rows are unaffected. This utility merges freshly computed
SPRG-only rows into the existing CSVs and regenerates the summaries/figures.

Workflow per experiment:
  1. python -m src.main exp2 --algorithms SPRG --out results/_sprg_fix
     (same for exp2b, exp3; for exp1 also --datasets ...)
  2. python -m src.merge_sprg exp2  (or exp2b / exp3 / exp1)
"""

import argparse
from pathlib import Path

import pandas as pd

from . import plotting
from .experiments import _summarize_all_vs_selected, _summarize_influence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SPRG_DIR = RESULTS / "_sprg_fix"
REG_COLS = ["Reg-SPC", "Reg-APC", "Reg-DSet", "Reg-SPRG"]


def merge_exp2():
    main_csv = RESULTS / "exp2_enhanced_vs_original" / "enhanced_vs_original.csv"
    new_csv = SPRG_DIR / "exp2_enhanced_vs_original" / "enhanced_vs_original.csv"
    if not main_csv.exists() or not new_csv.exists():
        print("merge_exp2: missing CSVs — run both the original and SPRG-only exp2 first")
        return
    old = pd.read_csv(main_csv)
    new = pd.read_csv(new_csv)
    merged = pd.concat([old[old["algo"] != "SPRG"], new[new["algo"] == "SPRG"]])
    merged = merged.sort_values(["dataset", "algo"]).reset_index(drop=True)
    merged.to_csv(main_csv, index=False)
    plotting.plot_enhanced_vs_original(str(main_csv))
    print(f"exp2 merged: {len(merged)} rows")


def merge_exp2b():
    main_csv = RESULTS / "exp2b_regularity_vs_kmeans" / "regularity_vs_kmeans.csv"
    new_csv = SPRG_DIR / "exp2b_regularity_vs_kmeans" / "regularity_vs_kmeans.csv"
    if not main_csv.exists() or not new_csv.exists():
        print("merge_exp2b: missing CSVs")
        return
    old = pd.read_csv(main_csv)
    new = pd.read_csv(new_csv)
    merged = pd.concat([old[old["algo"] != "SPRG"], new[new["algo"] == "SPRG"]])
    merged = merged.sort_values(["dataset", "algo"]).reset_index(drop=True)
    merged.to_csv(main_csv, index=False)
    plotting.plot_regularity_vs_kmeans(str(main_csv))
    print(f"exp2b merged: {len(merged)} rows")


def merge_exp3():
    main_dir = RESULTS / "exp3_vs_recent"
    new_dir = SPRG_DIR / "exp3_vs_recent"
    for metric in ("nmi", "acc", "ari", "ri"):
        main_csv = main_dir / f"table_{metric}.csv"
        new_csv = new_dir / f"table_{metric}.csv"
        if not main_csv.exists() or not new_csv.exists():
            print(f"merge_exp3: missing table_{metric}.csv")
            continue
        old = pd.read_csv(main_csv)
        new = pd.read_csv(new_csv)
        old = old[old["dataset"] != "mean"].set_index("dataset")
        new = new[new["dataset"] != "mean"].set_index("dataset")
        for ds in new.index:
            if ds in old.index and pd.notna(new.loc[ds, "Reg-SPRG"]):
                old.loc[ds, "Reg-SPRG"] = new.loc[ds, "Reg-SPRG"]
        old = old.reset_index()
        num = old.drop(columns=["dataset"])
        mean_row = {"dataset": "mean"}
        mean_row.update(num.mean(numeric_only=True).to_dict())
        out = pd.concat([old, pd.DataFrame([mean_row])], ignore_index=True)
        out.to_csv(main_csv, index=False)
    plotting.plot_vs_recent(main_dir)
    print("exp3 merged: Reg-SPRG column updated + mean rows recomputed")


def merge_exp1():
    main_dir = RESULTS / "exp1_parameter_influence"
    new_dir = SPRG_DIR / "exp1_parameter_influence"
    main_csv = main_dir / "raw_runs.csv"
    new_csv = new_dir / "raw_runs.csv"
    if not main_csv.exists() or not new_csv.exists():
        print("merge_exp1: missing raw_runs.csv (new run may still be in progress "
              "— its checkpoint is raw_runs_partial.csv)")
        return
    old = pd.read_csv(main_csv)
    new = pd.read_csv(new_csv)
    # a (dataset, algo, sigma, epsilon, compression, b[, stop_rule]) run is unique;
    # replace old SPRG rows with the fresh ones, keeping any dataset the new run lacks
    key = ["dataset", "algo", "sigma", "epsilon", "compression", "b"]
    if "stop_rule" in old.columns and "stop_rule" in new.columns:
        key.append("stop_rule")
    if "d0" in old.columns and "d0" in new.columns:
        key.append("d0")
    new_keys = set(map(tuple, new[key].to_numpy().tolist()))
    old_kept = old[~((old["algo"] == "SPRG") & old[key].apply(tuple, axis=1).isin(new_keys))]
    merged = pd.concat([old_kept, new]).sort_values(key).reset_index(drop=True)
    merged.to_csv(main_csv, index=False)
    _summarize_influence(merged, main_dir)
    _summarize_all_vs_selected(merged, main_dir)
    plotting.plot_parameter_influence(main_dir)
    plotting.plot_all_vs_selected(main_dir)
    print(f"exp1 merged: {len(merged)} rows ({len(new)} fresh SPRG rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["exp1", "exp2", "exp2b", "exp3"])
    args = ap.parse_args()
    {"exp1": merge_exp1, "exp2": merge_exp2, "exp2b": merge_exp2b, "exp3": merge_exp3}[args.experiment]()


if __name__ == "__main__":
    main()
