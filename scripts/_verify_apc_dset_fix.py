"""Narrow-grid check that Reg-APC / Reg-DSet now reach the paper's Leaves numbers."""
from src.datasets import load_dataset
from src.experiments import _best_over_enhanced_grid, _graph_for
from src.metrics import evaluate
from src.runners import run_original

PAPER = {
    "Leaves": {"APC": (0.75, 0.51), "DSet": (0.79, 0.56), "SPC": (0.82, 0.60)},
    "Ecoli": {"APC": (0.77, 0.90), "DSet": (0.71, 0.80), "SPC": (0.78, 0.89)},
}

SIGMAS = (0.2, 0.5, 1.0)
EPS = (0.1, 0.2)
CR = (0.05, 0.1)
B = (8, 16)


def main():
    for ds in ("Ecoli", "Leaves"):
        X, y = load_dataset(ds)
        nc = len(set(y))
        print("==== %s (paper Table 2) ====" % ds, flush=True)
        for algo in ("SPC", "APC", "DSet"):
            best_o, best_os = None, None
            for sigma in SIGMAS:
                S = _graph_for(algo, X, sigma, nc)
                lab = run_original(algo, S, nc if algo == "SPC" else None, X=X)
                m = evaluate(y, lab)
                if best_o is None or m["nmi"] > best_o["nmi"]:
                    best_o, best_os = m, sigma
            m, info, st = _best_over_enhanced_grid(
                algo, X, y, nc, EPS, CR, B, SIGMAS
            )
            pn, pa = PAPER[ds][algo]
            print(
                "%-4s orig NMI/ACC=%.3f/%.3f (sigma=%s)  enh=%.3f/%.3f  paper=%.2f/%.2f  "
                "k=%s  sig=%s eps=%s cr=%s b=%s d0=%s pref=%s dset=%s"
                % (
                    algo,
                    best_o["nmi"], best_o["acc"], best_os,
                    m["nmi"] if m else float("nan"),
                    m["acc"] if m else float("nan"),
                    pn, pa,
                    info["k"] if info else None,
                    None if st is None else st["sigma"],
                    None if st is None else st["epsilon"],
                    None if st is None else st["compression"],
                    None if st is None else st["b"],
                    None if st is None else st.get("density_threshold"),
                    None if st is None else st.get("preference_quantile"),
                    None if st is None else st.get("weight_threshold"),
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
