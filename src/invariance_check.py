"""Re-run Exp 2 winning configs on a row-permuted copy of X.

A graph algorithm must return essentially the same NMI after permuting rows
(and the matching labels). The old support-mode partitioner fails this because
it cuts classes by vertex index. Weighted ([16] Eq. 15) must pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .datasets import load_dataset
from .enhanced import enhance_clustering
from .metrics import evaluate
from .runners import make_base_algorithm, original_graph


def check_exp2_csv(csv_path, seed=0, nmi_tol=0.05):
    """Re-evaluate each Exp 2 winner on file-order X and a shuffled copy.

    Writes ``invariance.csv`` next to ``csv_path``. ``nmi_tol`` is loose on
    purpose: spectral k-means can move a little under eigenvector sign flips;
    the failure mode we care about is ~0.8 vs ~0.08.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    rows = []
    n_ok = 0
    for rec in df.to_dict("records"):
        ds_name = rec["dataset"]
        algo = rec["algo"]
        X, y = load_dataset(ds_name)
        n_clusters = config.DATASETS[ds_name][2]
        perm = rng.permutation(len(y))
        Xp, yp = X[perm], y[perm]
        sigma = rec.get("enh_sigma", 1.0)
        if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)):
            sigma = 1.0
        stop_rule = rec.get("enh_stop_rule", rec.get("stop_rule", "algorithm1"))
        degree_mode = rec.get("degree_mode", rec.get("enh_degree_mode", "weighted"))
        d0 = rec.get("enh_d0", rec.get("d0", 0.0))
        if d0 is None or (isinstance(d0, float) and np.isnan(d0)):
            d0 = 0.0
        clustering_variant = rec.get("clustering_variant")
        if clustering_variant is None or (isinstance(clustering_variant, float)
                                            and np.isnan(clustering_variant)):
            clustering_variant = config.paper_clustering_variant(algo)
        alg_kind = rec.get("enh_alg_kind", rec.get("alg_kind", "alon"))
        if alg_kind is None or (isinstance(alg_kind, float) and np.isnan(alg_kind)):
            alg_kind = "alon"
        init_mode = rec.get("enh_init_mode", rec.get("init_mode", "degree"))
        if init_mode is None or (isinstance(init_mode, float) and np.isnan(init_mode)):
            init_mode = "degree"
        drop = rec.get("enh_drop_irregular", rec.get("drop_irregular", "all_pairs"))
        if drop is None or (isinstance(drop, float) and np.isnan(drop)):
            drop = "all_pairs"
        adj_t = rec.get("enh_adj_threshold", rec.get("adj_threshold", config.ADJ_THRESHOLD))
        if adj_t is None or (isinstance(adj_t, float) and np.isnan(adj_t)):
            adj_t = config.ADJ_THRESHOLD
        kwargs = dict(
            epsilon=float(rec["enh_epsilon"]),
            b=int(rec["enh_b"]),
            compression_rate=float(rec["enh_compression"]),
            stop_rule=stop_rule,
            density_threshold=d0,
            degree_mode=degree_mode,
            alg_kind=alg_kind,
            random_initialization=config.init_is_random(init_mode),
            drop_edges_between_irregular_pairs=config.drop_irregular_pairs(drop),
            adj_threshold=adj_t,
        )
        n_k = n_clusters if algo in ("SPC", "SPRG") else None
        S = original_graph(algo, X, sigma, n_clusters=n_clusters,
                            clustering_variant=clustering_variant)
        Sp = original_graph(algo, Xp, sigma, n_clusters=n_clusters,
                             clustering_variant=clustering_variant)
        labels, _ = enhance_clustering(
            make_base_algorithm(algo, X=X, clustering_variant=clustering_variant),
            S, n_clusters=n_k, **kwargs
        )
        labels_p, _ = enhance_clustering(
            make_base_algorithm(algo, X=Xp, clustering_variant=clustering_variant),
            Sp, n_clusters=n_k, **kwargs
        )
        m = evaluate(y, labels)
        mp = evaluate(yp, labels_p)
        diff = abs(m["nmi"] - mp["nmi"])
        ok = diff <= nmi_tol
        n_ok += int(ok)
        rows.append({
            "dataset": ds_name,
            "algo": algo,
            "clustering_variant": clustering_variant,
            "stop_rule": rec.get("stop_rule", stop_rule),
            "degree_mode": degree_mode,
            "alg_kind": alg_kind,
            "init_mode": init_mode,
            "drop_irregular": drop,
            "file_order_nmi": m["nmi"],
            "shuffled_nmi": mp["nmi"],
            "csv_enh_nmi": rec["enh_nmi"],
            "abs_diff": diff,
            "ok": ok,
        })
        flag = "ok" if ok else "FAIL"
        print(f"  [{flag}] {ds_name}/{algo}/{clustering_variant}/{stop_rule}/"
              f"{degree_mode}/{alg_kind}/{init_mode}/{drop}: "
              f"file={m['nmi']:.3f} shuffled={mp['nmi']:.3f} "
              f"csv={rec['enh_nmi']:.3f} |diff|={diff:.3f}")
    out = csv_path.parent / "invariance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[invariance] {n_ok}/{len(rows)} configs passed (tol={nmi_tol}); wrote {out}")
    return n_ok == len(rows) and len(rows) > 0
