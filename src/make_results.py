"""Build RESULTS.md: computed results vs the paper's reported values.

Compares the Reg-* columns produced by experiment 3 (computed here) against
the paper's own per-dataset values (Tables 2-5, transcribed in
``src/paper_reference.py``), plus the enhanced-vs-original win counts of
experiment 2 and the regularity-vs-k-means ablation of experiment 2b.

Usage:  python -m src.make_results [--out RESULTS.md]
"""

import argparse
from pathlib import Path

import pandas as pd

from . import paper_reference as pr

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PAPER_MEANS = {  # stated in the paper's Tables 2-5 discussion (Sec. 4.3)
    "nmi": {"Reg-SPC": 0.76, "Reg-APC": 0.72, "Reg-DSet": 0.73, "Reg-SPRG": 0.76},
    "acc": {"Reg-SPC": 0.86, "Reg-APC": 0.82, "Reg-DSet": 0.82, "Reg-SPRG": 0.85},
    "ari": {"Reg-SPC": 0.73, "Reg-APC": 0.68, "Reg-DSet": 0.71, "Reg-SPRG": 0.74},
    "ri": {"Reg-SPC": 0.91, "Reg-APC": 0.91, "Reg-DSet": 0.91, "Reg-SPRG": 0.91},
}
REG_COLS = ["Reg-SPC", "Reg-APC", "Reg-DSet", "Reg-SPRG"]


def _exp3_section():
    lines = ["## Exp 3 — Reg-* vs the paper's Tables 2–5", ""]
    any_table = False
    for metric in ("nmi", "acc", "ari", "ri"):
        csv = RESULTS / "exp3_vs_recent" / f"table_{metric}.csv"
        if not csv.exists():
            lines.append(f"_{metric.upper()}: table_{metric}.csv not found — run `python -m src.main exp3`._")
            lines.append("")
            continue
        any_table = True
        df = pd.read_csv(csv)
        df = df[df["dataset"] != "mean"].set_index("dataset")
        paper_tbl = getattr(pr, metric.upper())
        rows = []
        for ds in pr.DATASET_ORDER:
            if ds not in df.index:
                continue
            ours = {c: df.loc[ds, c] for c in REG_COLS}
            paper = {c: paper_tbl[ds][8 + i] for i, c in enumerate(REG_COLS)}
            rows.append((ds, ours, paper))
        lines.append(f"### {metric.upper()} — ours vs paper (per dataset)")
        lines.append("")
        header = "| dataset | " + " | ".join(
            f"{c} ours/paper" for c in REG_COLS) + " | mean |Δ| |"
        lines.append(header)
        lines.append("|" + "---|" * (len(REG_COLS) + 2))
        deltas_all = {c: [] for c in REG_COLS}
        for ds, ours, paper in rows:
            cells = []
            row_deltas = []
            for c in REG_COLS:
                o = ours[c] if pd.notna(ours[c]) else float("nan")
                cells.append(f"{o:.2f} / {paper[c]:.2f}")
                if pd.notna(o):
                    deltas_all[c].append(abs(o - paper[c]))
                    row_deltas.append(abs(o - paper[c]))
            md = sum(row_deltas) / max(1, len(row_deltas)) if row_deltas else float("nan")
            lines.append(f"| {ds} | " + " | ".join(cells) + f" | {md:.2f} |")
        means = {c: (sum(deltas_all[c]) / len(deltas_all[c]) if deltas_all[c] else float("nan"))
                 for c in REG_COLS}
        lines.append("| **mean |Δ| per column** | " + " | ".join(
            f"**{means[c]:.2f}**" for c in REG_COLS) + " | |")
        # overall means of our computed values vs the paper's stated means
        our_means = df[REG_COLS].mean(numeric_only=True)
        pm = PAPER_MEANS[metric]
        lines.append("")
        lines.append(
            "Overall means (ours / paper): "
            + ", ".join(f"{c} {our_means[c]:.2f}/{pm[c]:.2f}" for c in REG_COLS)
        )
        lines.append("")
    return lines, any_table


def _exp2_section():
    csv = RESULTS / "exp2_enhanced_vs_original" / "enhanced_vs_original.csv"
    if not csv.exists():
        return ["## Exp 2 — enhanced vs original", "", "_not run yet_", ""]
    df = pd.read_csv(csv)
    wins = (df["enh_nmi"] > df["orig_nmi"]).sum()
    per_algo = df.groupby("algo").apply(
        lambda g: pd.Series({
            "wins": int((g["enh_nmi"] > g["orig_nmi"]).sum()),
            "total": len(g),
            "orig_mean": g["orig_nmi"].mean(),
            "enh_mean": g["enh_nmi"].mean(),
        }), include_groups=False)
    lines = [
        "## Exp 2 — enhanced vs original (Figs 7–10)", "",
        f"Enhanced NMI > original on **{wins}/{len(df)}** runs "
        f"(paper: enhanced SPC better on 19/20 datasets).", "",
        "| algo | wins/total | mean orig NMI | mean enh NMI |",
        "|---|---|---|---|",
    ]
    for algo, r in per_algo.iterrows():
        lines.append(f"| {algo} | {int(r['wins'])}/{int(r['total'])} | "
                     f"{r['orig_mean']:.3f} | {r['enh_mean']:.3f} |")
    lines.append("")
    lines.extend(_axis_table(
        RESULTS / "exp2_enhanced_vs_original",
        "### Exp 2 by clustering variant",
        nmi_col="enh_nmi", orig_col="orig_nmi",
    ))
    lines.extend(_axis_table(
        RESULTS / "exp2_enhanced_vs_original",
        "### Exp 2 by regularity kind / init / drop-irregular",
        nmi_col="enh_nmi",
        axes=("alg_kind", "init_mode", "drop_irregular", "stop_rule", "degree_mode"),
    ))
    return lines


def _exp2b_section():
    csv = RESULTS / "exp2b_regularity_vs_kmeans" / "regularity_vs_kmeans.csv"
    if not csv.exists():
        return ["## Exp 2b — regularity vs k-means partitioning (Fig. 11)", "", "_not run yet_", ""]
    df = pd.read_csv(csv)
    wins = (df["reg_nmi"] > df["km_nmi"]).sum()
    ties = (df["reg_nmi"] == df["km_nmi"]).sum()
    lines = [
        "## Exp 2b — regularity vs k-means partitioning (Fig. 11)", "",
        f"Regularity partitioning beats k-means partitioning on **{wins}/{len(df)}** runs "
        f"({ties} ties); paper Fig. 11: regularity wins on most datasets.", "",
        f"- mean NMI: regularity {df['reg_nmi'].mean():.3f} vs k-means {df['km_nmi'].mean():.3f}",
        "",
    ]
    lines.extend(_axis_table(
        RESULTS / "exp2b_regularity_vs_kmeans",
        "### Exp 2b by clustering variant",
        nmi_col="reg_nmi", orig_col="km_nmi",
    ))
    return lines


def _nmi_mean_col(df):
    for name in ("nmi_mean", "enh_nmi_mean", "reg_nmi_mean", "nmi", "enh_nmi", "reg_nmi"):
        if name in df.columns:
            return name
    return None


def _axis_table(exp_dir, heading, nmi_col="nmi", orig_col=None,
                axes=("clustering_variant",)):
    """Render comparison_{axis}.csv as markdown tables."""
    lines = [heading, ""]
    any_table = False
    for axis in axes:
        csv = Path(exp_dir) / f"comparison_{axis}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if df.empty or axis not in df.columns:
            continue
        col = _nmi_mean_col(df)
        if col is None:
            continue
        any_table = True
        orig = None
        if orig_col:
            for cand in (f"{orig_col}_mean", orig_col):
                if cand in df.columns:
                    orig = cand
                    break
        if "algo" in df.columns:
            keys = ["algo", axis]
        else:
            keys = [axis]
        g = df.groupby(keys, dropna=False)[col].mean().reset_index()
        lines.append(f"**{axis}**")
        lines.append("")
        if orig and "algo" in df.columns:
            o = df.groupby(keys, dropna=False)[orig].mean().reset_index()
            g = g.merge(o, on=keys)
            lines.append("| algo | " + axis + " | mean NMI | mean orig/kmeans |")
            lines.append("|---|---|---|---|")
            for _, r in g.iterrows():
                lines.append(
                    f"| {r['algo']} | {r[axis]} | {r[col]:.3f} | {r[orig]:.3f} |"
                )
        elif "algo" in df.columns:
            lines.append("| algo | " + axis + " | mean NMI |")
            lines.append("|---|---|---|")
            for _, r in g.iterrows():
                lines.append(f"| {r['algo']} | {r[axis]} | {r[col]:.3f} |")
        else:
            lines.append("| " + axis + " | mean NMI |")
            lines.append("|---|---|")
            for _, r in g.iterrows():
                lines.append(f"| {r[axis]} | {r[col]:.3f} |")
        lines.append("")
    if not any_table:
        return []
    return lines


def _exp3_variants_section():
    csv = RESULTS / "exp3_vs_recent" / "reg_variants.csv"
    board = RESULTS / "exp3_vs_recent" / "comparison_leaderboard.csv"
    if not csv.exists():
        return ["## Exp 3 — per clustering variant", "",
                "_reg_variants.csv not found — run exp3._", ""]
    lines = ["## Exp 3 — per clustering variant (un-collapsed search)", ""]
    df = pd.read_csv(csv)
    if df.empty:
        lines.append("_empty_")
        lines.append("")
        return lines
    if board.exists():
        lines.extend(_axis_table(
            RESULTS / "exp3_vs_recent",
            "Mean NMI by clustering variant (over datasets)",
            nmi_col="nmi",
        ))
    else:
        g = df.groupby(["algo", "clustering_variant"], dropna=False)["nmi"].mean().reset_index()
        lines.append("| algo | clustering_variant | mean NMI |")
        lines.append("|---|---|---|")
        for _, r in g.sort_values("nmi", ascending=False).iterrows():
            lines.append(f"| {r['algo']} | {r['clustering_variant']} | {r['nmi']:.3f} |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "RESULTS.md"))
    args = ap.parse_args()

    exp3_lines, _ = _exp3_section()
    lines = [
        "# Results — reproduction vs paper", "",
        "Computed with the paper-faithful implementation (see "
        "`IMPLEMENTATION_PROOF.md`); 'ours/paper' columns show our computed "
        "value against the paper's own reported value for the same cell.", "",
        *exp3_lines,
        *_exp3_variants_section(),
        *_exp2_section(),
        *_exp2b_section(),
        "## Exp 1 — parameter influence (Figs 2–6)", "",
        "See `results/exp1_parameter_influence/` (`influence_*.csv`, "
        "`influence_{ALGO}.png` with NMI + running-time panels, "
        "`all_vs_selected.csv` / `.png` for the Fig. 6 comparison, "
        "`comparison_*.csv` for every clustering / partition axis).", "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
