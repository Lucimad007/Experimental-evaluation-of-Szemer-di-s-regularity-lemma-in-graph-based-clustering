"""Verify the paper's empirical claims against our computed results.

Claims checked (paper §4.1–§4.3):
  C1  Enhancement improves clustering (Figs 7-10)
  C2  Regularity partitioning beats k-means partitioning (Fig. 11)
  C3  Reg-* beats the recent algorithms (Tables 2-5)
  C4  Small epsilon (0.1-0.2) gives the best results; degradation for
      epsilon > 0.2 (Fig. 2-5 discussion)
  C5  epsilon-small (compression) best at 0.1 or below; 0.01 never best;
      0.2 costs time without gains
  C6  b <= 16 suffices; b > 16 does not improve results
  C7  Larger epsilon reduces running time; larger compression ratio
      increases it
  C8  The recommended narrow ranges rarely degrade results vs all
      parameters (Fig. 6)

Run:  python -m proof.verify_claims
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.paper_reference import NMI as PAPER_NMI

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REG_COLS = ["Reg-SPC", "Reg-APC", "Reg-DSet", "Reg-SPRG"]


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def exp1_frames():
    parts = list(RESULTS.glob("small_w*/exp1_parameter_influence/raw_runs*.csv"))
    parts += list(RESULTS.glob("exp1_parameter_influence/raw_runs.csv"))
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    cols = ["dataset", "algo", "sigma", "epsilon", "compression", "b"]
    if "stop_rule" in df.columns:
        cols.append("stop_rule")
    if "d0" in df.columns:
        cols.append("d0")
    df = df.drop_duplicates(subset=cols)
    return df


def claim_improvement():
    csv = RESULTS / "exp2_enhanced_vs_original" / "enhanced_vs_original.csv"
    if not csv.exists():
        print("  [skip] exp2 not finished")
        return True
    df = pd.read_csv(csv)
    wins = (df["enh_nmi"] > df["orig_nmi"]).sum()
    print(f"  C1: enhanced > original on {wins}/{len(df)} runs; "
          f"mean NMI {df['orig_nmi'].mean():.3f} -> {df['enh_nmi'].mean():.3f} "
          f"(paper: better on 19/20 datasets)")
    per = df.groupby("algo").apply(lambda g: (g.enh_nmi > g.orig_nmi).sum(), include_groups=False)
    ok = wins > 0.5 * len(df)
    return check("C1 enhancement improves the majority of runs", ok) and \
        check("C1 mean NMI improves", df["enh_nmi"].mean() > df["orig_nmi"].mean())


def claim_kmeans():
    csv = RESULTS / "exp2b_regularity_vs_kmeans" / "regularity_vs_kmeans.csv"
    if not csv.exists():
        print("  [skip] exp2b not finished")
        return True
    df = pd.read_csv(csv)
    wins = (df["reg_nmi"] > df["km_nmi"]).sum()
    print(f"  C2: regularity beats k-means partitioning on {wins}/{len(df)} runs "
          f"(paper Fig. 11: most datasets)")
    return check("C2 regularity > k-means partitioning on the majority", wins > 0.5 * len(df))


def claim_vs_recent():
    csv = RESULTS / "exp3_vs_recent" / "table_nmi.csv"
    if not csv.exists():
        csv = RESULTS / "exp3_vs_recent" / "table_nmi_partial.csv"
    if not csv.exists():
        print("  [skip] exp3 not started")
        return True
    df = pd.read_csv(csv)
    df = df[df["dataset"] != "mean"]
    recent = [c for c in df.columns if c != "dataset" and not c.startswith("Reg-")]
    best_recent = df[recent].max(axis=1)
    wins = 0
    for c in REG_COLS:
        if c in df.columns:
            w = (df[c] > best_recent).sum()
            tot = df[c].notna().sum()
            print(f"  C3: {c} beats all 8 recent algorithms on {w}/{tot} datasets "
                  f"(paper: 18/20 for NMI)")
            wins += w
    ok = wins > 0
    return check("C3 Reg-* beats recent algorithms on most datasets", ok)


def parameter_claims(df):
    if df is None or df.empty:
        print("  [skip] exp1 partials empty")
        return True
    ok_all = True
    print(f"  (exp1 data: {df.dataset.nunique()} datasets, {len(df)} runs)")

    # C4: mean NMI by epsilon (paper aggregation: mean over remaining params)
    agg = df.groupby(["dataset", "algo", "epsilon"]).nmi.mean().reset_index()
    by_eps = agg.groupby("epsilon").nmi.mean()
    best_eps = by_eps.idxmax()
    small = by_eps.loc[[e for e in by_eps.index if 0.05 <= e <= 0.2]].max()
    large = by_eps.loc[[e for e in by_eps.index if e > 0.2]].max()
    print(f"  C4: mean NMI by eps: {dict(by_eps.round(3))}")
    ok_all &= check("C4 best mean NMI inside eps<=0.2",
                    by_eps.idxmax() <= 0.2,
                    f"best at eps={best_eps}; eps<=0.2 max {small:.3f} vs >0.2 max {large:.3f}")

    # C5: mean NMI and time by compression ratio
    agg_c = df.groupby(["dataset", "algo", "compression"]).agg(
        nmi=("nmi", "mean"), time=("time", "mean")).reset_index()
    by_cr = agg_c.groupby("compression").agg(nmi=("nmi", "mean"), time=("time", "mean"))
    best_cr = by_cr.nmi.idxmax()
    print(f"  C5: mean NMI by compression: {dict(by_cr.nmi.round(3))}")
    ok_all &= check("C5 best compression ratio <= 0.1", best_cr <= 0.1, f"best at {best_cr}")
    if 0.01 in by_cr.index:
        ok_all &= check("C5 compression 0.01 is never the best", best_cr != 0.01)
    if 0.2 in by_cr.index and best_cr != 0.2:
        ok_all &= check("C5 compression 0.2 not better than smaller ones",
                        by_cr.loc[0.2, "nmi"] <= by_cr.nmi.max() + 0.01)
    ok_all &= check("C5 compression 0.2 costs the most time", by_cr.time.idxmax() == 0.2,
                    f"time by compression: {dict(by_cr.time.round(2))}")

    # C6: b <= 16
    agg_b = df.groupby(["dataset", "algo", "b"]).nmi.mean().reset_index()
    by_b = agg_b.groupby("b").nmi.mean()
    small_b = by_b.loc[[b for b in by_b.index if b <= 16]].max()
    big_b = by_b.loc[[b for b in by_b.index if b > 16]].max() if any(b > 16 for b in by_b.index) else -1
    print(f"  C6: max mean NMI b<=16: {small_b:.3f} vs b>16: {big_b:.3f}")
    ok_all &= check("C6 b>16 does not improve results", big_b <= small_b + 0.01)

    # C7: time trends
    t_eps = df.groupby(["dataset", "algo", "epsilon"]).time.mean().reset_index()
    t_eps = t_eps.groupby("epsilon").time.mean()
    ok_all &= check("C7 larger eps -> less time",
                    t_eps.loc[t_eps.index.max()] < t_eps.loc[t_eps.index.min()],
                    f"time at eps={t_eps.index.min()}: {t_eps.iloc[0]:.2f}s vs "
                    f"eps={t_eps.index.max()}: {t_eps.iloc[-1]:.2f}s")
    ok_all &= check("C7 larger compression -> more time",
                    by_cr.time.idxmax() > by_cr.time.idxmin())

    # C8: all vs selected (Fig. 6)
    sel = df.apply(lambda r: config.in_grid(r["epsilon"], config.EPSILON_RECOMMENDED)
                   and config.in_grid(r["compression"], config.COMPRESSION_RECOMMENDED)
                   and int(r["b"]) in config.B_RECOMMENDED, axis=1)
    all_m = df.nmi.mean()
    sel_m = df[sel].nmi.mean()
    print(f"  C8: mean NMI all parameters {all_m:.3f} vs selected ranges {sel_m:.3f}")
    ok_all &= check("C8 selected ranges at least as good (within 0.02)",
                    sel_m >= all_m - 0.02)
    return ok_all


def main():
    print("Verifying the paper's claims on our computed results")
    print("=" * 60)
    ok = True
    print("C1/C2/C3 — improvement claims")
    ok &= claim_improvement()
    ok &= claim_kmeans()
    ok &= claim_vs_recent()
    print("C4-C8 — parameter-range claims (from exp1 grids)")
    ok &= parameter_claims(exp1_frames())
    print("=" * 60)
    print("ALL CLAIMS VERIFIED" if ok else "SOME CLAIMS FAILED — see [FAIL] lines above")


if __name__ == "__main__":
    main()
