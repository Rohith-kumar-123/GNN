"""
Generate a SMALL fake dataset with the same columns as AMLSim HI-Small.
Only for testing that the pipeline runs end to end (no internet / download needed).
Do NOT report results from it.

    python scripts/make_synthetic_data.py --out data/raw/synthetic.csv
    python scripts/run_experiments.py --csv data/raw/synthetic.csv --variants gcn hrgnn --seeds 1
"""
import argparse
import numpy as np
import pandas as pd

FORMATS = ["Cheque", "Credit Card", "ACH", "Reinvestment", "Wire", "Cash", "Bitcoin"]


def main(a):
    rng = np.random.default_rng(a.seed)
    n_acc, n_days = a.accounts, 10
    ids = np.array([f"80{i:07X}" for i in range(n_acc)])
    bank = rng.integers(1, 50, n_acc)
    mules = rng.choice(n_acc, size=int(0.03 * n_acc), replace=False)   # laundering accounts
    is_mule = np.zeros(n_acc, bool)
    is_mule[mules] = True
    mule_start = {m: rng.integers(2, 8) for m in mules}                # appear late = low persistence

    rows = []
    # normal traffic: steady daily activity, small repeat-counterparty sets
    partners = {i: rng.integers(0, n_acc, 5) for i in range(n_acc)}
    for _ in range(a.rows):
        s = rng.integers(0, n_acc)
        if is_mule[s]:
            continue
        d = partners[s][rng.integers(0, 5)] if rng.random() < 0.8 else rng.integers(0, n_acc)
        t = rng.integers(0, n_days * 1440)
        rows.append((t, s, d, float(np.exp(rng.normal(5, 1.5))), rng.choice(FORMATS[:4]), 0))

    # laundering: bursts of large-ish transfers between mules, after they "wake up"
    for m in mules:
        for _ in range(rng.integers(5, 15)):
            d = rng.choice(mules)
            t = mule_start[m] * 1440 + rng.integers(0, 1440 * 2)
            if t < n_days * 1440:
                rows.append((t, m, d, float(np.exp(rng.normal(8, 0.8))), rng.choice(["Cheque", "ACH"]), 1))
        for _ in range(rng.integers(0, 4)):   # a few normal-looking transactions too
            t = mule_start[m] * 1440 + rng.integers(0, 1440 * 2)
            rows.append((t, m, rng.integers(0, n_acc), float(np.exp(rng.normal(5, 1.5))), "Credit Card", 0))

    t, s, d, amt, fmt, y = map(np.array, zip(*rows))
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(t.astype(int), unit="m")
    out = pd.DataFrame({
        "Timestamp": ts.strftime("%Y/%m/%d %H:%M"),
        "From Bank": bank[s], "Account": ids[s], "To Bank": bank[d], "Account.1": ids[d],
        "Amount Received": amt.round(2), "Receiving Currency": "US Dollar",
        "Amount Paid": amt.round(2), "Payment Currency": "US Dollar",
        "Payment Format": fmt, "Is Laundering": y.astype(int),
    })
    out = out.sample(frac=1, random_state=0)   # unsorted, like the real file
    # write the original duplicate header name "Account" for the receiver column
    out.columns = ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
                   "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out):,} rows, fraud rate {y.mean():.3%} -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw/synthetic.csv")
    p.add_argument("--rows", type=int, default=60000)
    p.add_argument("--accounts", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
