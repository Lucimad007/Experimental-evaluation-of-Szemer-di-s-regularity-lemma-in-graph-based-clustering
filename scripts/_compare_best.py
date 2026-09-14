"""Compare best.csv against the previous all_small_first.csv (same cells)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "results/lemma_claim/all_small_first.csv"
BEST = ROOT / "results/lemma_claim/best.csv"


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    old = {}
    for row in read(OLD):
        if row.get("lemma"):
            old[(row["dataset"], row["clustering"])] = float(row["lemma"])
    rows = []
    for row in read(BEST):
        key = (row["dataset"], row["clustering"])
        o = old.get(key)
        n = float(row["nmi"])
        rows.append({
            "dataset": row["dataset"], "clustering": row["clustering"],
            "old": o, "new": n, "delta": (n - o) if o is not None else None,
            "hou": row.get("hou"), "source": row["source"],
        })
    rows.sort(key=lambda r: (r["dataset"], r["clustering"]))
    print(f"{'dataset':14s} {'cl':4s} {'old':>6s} {'new':>6s} {'delta':>7s} "
          f"{'hou':>5s}  source")
    for r in rows:
        d = f"{r['delta']:+.3f}" if r["delta"] is not None else "  new"
        print(f"{r['dataset']:14s} {r['clustering']:4s} "
              f"{r['old'] if r['old'] is not None else float('nan'):6.3f} "
              f"{r['new']:6.3f} {d:>7s} {r['hou'] or '':>5s}  {r['source']}")
    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    gains = [d for d in deltas if d > 1e-9]
    print(f"\ncells={len(rows)}  improved={len(gains)}/{len(deltas)}  "
          f"mean_delta={sum(deltas)/len(deltas):+.4f}  "
          f"max={max(gains) if gains else 0:+.4f}")
    wins = sum(1 for r in rows if r["hou"] and r["new"] > float(r["hou"]))
    print(f"beats paper: {wins}/{len(rows)}")


if __name__ == "__main__":
    main()
