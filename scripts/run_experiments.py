"""
Run baselines, the plain GCN, ablations and HR-GNN over several seeds.

    python scripts/run_experiments.py                         # everything, 5 seeds
    python scripts/run_experiments.py --variants gcn hrgnn --seeds 3
    python scripts/run_experiments.py --dev                   # fast debug run (not for reporting)
    python scripts/run_experiments.py --csv data/raw/synthetic.csv --seeds 1 --epochs 20
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import run_baseline
from src.config import BASELINES, VARIANTS, DataConfig, TrainConfig
from src.data import describe_split, prepare
from src.train import run_gnn

SHOW = ["ap", "auc", "precision", "recall", "f1", "false_alerts_per_1000", "recall@0.5%"]


def main(a):
    dcfg = DataConfig(csv_path=a.csv, dev_normal_frac=0.1 if a.dev else None)
    tcfg = TrainConfig(device=a.device)
    if a.epochs:
        tcfg = replace(tcfg, epochs=a.epochs)
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    prep = prepare(dcfg)
    print("[split]", json.dumps(describe_split(prep), indent=1))

    names = a.variants or (BASELINES + list(VARIANTS))
    rows = []
    for name in names:
        for seed in range(a.seeds):
            print(f"== {name} | seed {seed}")
            if name in BASELINES:
                res, arrays = run_baseline(prep, name, seed)
            else:
                res, arrays = run_gnn(prep, name, VARIANTS[name], tcfg, seed)
            (out / "runs" / f"{name}_s{seed}.json").write_text(json.dumps(res, indent=1))
            np.savez_compressed(out / "runs" / f"{name}_s{seed}.npz", **arrays)
            rows.append({"model": name, "seed": seed, **res["test"],
                         "train_seconds": res.get("train_seconds", np.nan)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "results_long.csv", index=False)

    g = df.groupby("model", sort=False)
    summary = pd.DataFrame({m: g[m].mean().round(4).astype(str) + " ± " + g[m].std().fillna(0).round(4).astype(str)
                            for m in SHOW})
    summary.to_csv(out / "results_summary.csv")
    print("\n=== TEST RESULTS (mean ± std over seeds) ===")
    print(summary.to_string())

    # significance: HR-GNN vs plain GCN, paired over seeds
    if {"gcn", "hrgnn"} <= set(df["model"]) and a.seeds >= 3:
        x = df[df.model == "hrgnn"].sort_values("seed")["ap"].values
        y = df[df.model == "gcn"].sort_values("seed")["ap"].values
        t, p = stats.ttest_rel(x, y)
        print(f"\nPaired t-test on AP, hrgnn vs gcn: t={t:.3f}, p={p:.4f} (n={len(x)} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", help="subset of baselines/variants (default: all)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--csv", default=None, help="local CSV instead of Hugging Face")
    ap.add_argument("--dev", action="store_true", help="10%% of normal rows, for debugging only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs")
    main(ap.parse_args())
