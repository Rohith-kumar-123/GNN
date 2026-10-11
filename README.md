# HR-GNN: Historical- and Relational-aware Graph Learning for Fraud Detection

Detecting laundering transactions in the **AMLSim HI-Small** dataset
([Hugging Face](https://huggingface.co/datasets/bbfizp/AMLSim-HI-Small)) with a graph neural network that knows
**how long an account has been active** and **how trustworthy each transaction link is**.

**Research question:** does adding account history (active-day persistence) and learned edge reliability to a GNN
beat a plain GCN, while keeping false alerts under control?

---

## The idea in four steps

| Module | What it does | File |
|---|---|---|
| **A. History + persistence** | Per account: active days, active-day ratio, streak, recency, past amounts, past counterparties, burstiness. A small gate turns the persistence features into a trust score `p_v` in [0,1]. | `src/features.py`, `src/model.py` |
| **B. Edge reliability** | For every transaction `u -> v`, learn `r_uv in [0,1]` from both accounts' history, the amount, and the edge's age. | `src/model.py` |
| **C. Weighted message passing** | Neighbours speak louder if they are persistent (`p_u`), recent (`exp(-lambda*dt)`) and reliable (`r_uv`). Optional second channel keeps the *unreliable* neighbourhood as a signal. | `src/model.py` |
| **D. Edge classifier** | Predicts fraud per transaction from both account embeddings, the transaction features and `r_uv`. | `src/model.py` |

Loss = focal loss + `beta` * reliability loss (weak supervision from **training** labels) + `gamma` * entropy of `r`.

## Project layout

```
hr-gnn-fraud/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py      all settings + the list of model variants (baseline, ablations, full model)
│   ├── data.py        load, sort by time, chronological split, caching
│   ├── features.py    persistence / history features and node snapshots (leak-free)
│   ├── graph.py       builds the per-phase graphs as tensors
│   ├── model.py       HR-GNN (one class; switches turn modules on/off)
│   ├── train.py       training loop, early stopping, threshold selection
│   ├── baselines.py   logistic regression and XGBoost (no graph)
│   └── metrics.py     AP, AUC, precision, recall, F1, alert rate, recall@alert-budget
└── scripts/
    ├── eda.py                  data overview + daily plot
    ├── run_experiments.py      runs everything, several seeds, significance test
    ├── make_report.py          alert-budget curves, reliability histogram, cold-start table
    └── make_synthetic_data.py  tiny fake dataset to test the pipeline offline
```

## How to run

```bash
pip install -r requirements.txt

python scripts/eda.py                            # 1. look at the data (downloads from Hugging Face)
python scripts/run_experiments.py --dev --seeds 1 --epochs 15   # 2. fast debug run
python scripts/run_experiments.py                # 3. full run: all models x 5 seeds
python scripts/make_report.py                    # 4. figures and tables
```

No internet? Test the pipeline on fake data first:

```bash
python scripts/make_synthetic_data.py
python scripts/run_experiments.py --csv data/raw/synthetic.csv --seeds 1 --epochs 15
```

Results go to `outputs/` (`results_summary.csv` is the main table; per-run JSON and predictions are in `outputs/runs/`).

## Models compared

| Name | Purpose |
|---|---|
| `lr_tx`, `xgb_tx` | Non-graph reference, transaction features only |
| `xgb_tx_hist` | Non-graph, with history features: shows how much of the gain is "just history" |
| `gcn` | **Plain GCN baseline** (structure-only node inputs, uniform edge weights) |
| `gcn_hist` | Isolates Module A |
| `gcn_reliability` | Isolates Module B |
| `hrgnn_no_decay`, `hrgnn_no_persistence`, `hrgnn_single_channel` | Ablations |
| `hrgnn` | Full proposed model |

## Evaluation protocol (no leakage)

* Rows are sorted by timestamp and split **60 / 20 / 20 chronologically** (train / val / test).
* Per-transaction history uses **only earlier rows**; persistence is counted in **whole earlier days**.
* The graph used to score a phase contains only transactions up to the end of that phase, and **no labels** are used at inference.
* Standardisation statistics come from the training rows only.
* The decision threshold (max F1) is chosen on **validation** and frozen for test.
* Metrics: **AP (primary)**, ROC-AUC, precision, recall, F1, false alerts per 1,000 transactions, recall at 0.1 / 0.5 / 1 % alert budget.
* Run 5 seeds; `run_experiments.py` prints mean ± std and a paired t-test (HR-GNN vs GCN).

## Hypotheses

* **H1** History/persistence features improve AP over plain GCN (`gcn` vs `gcn_hist`).
* **H2** Learned reliability improves AP further and lowers false alerts at equal recall (`gcn_hist` vs `hrgnn`).
* **H3** Gains are largest for low-history accounts and later time windows (`cold_start.csv`, per-day analysis).

## Design decisions and honest limitations

* **Node snapshot vs per-edge history.** The GNN uses one account snapshot per phase (as of the end of that phase).
  For *training* edges this snapshot includes later days of the training window, so it is not strictly "as of the edge".
  This never touches validation/test or labels, but a stricter version would use rolling snapshots. The per-edge
  history features (used by the reliability module and the classifier) *are* strictly as-of.
* **Left censoring.** The dataset starts on day 0, so every account looks "new" at the start. Persistence is therefore
  most informative later in the window. Check `eda.py` for the real span of the data.
* **Drift.** Count-style features grow over time; node snapshots are z-scored per phase to reduce this, edge features are not.
* **Reliability supervision.** "Suspicious" edges = edges touching any account that appeared in a training-period fraud
  transaction. This is weak and noisy by design, so `beta_rel` is small. The entropy term makes `r` decisive;
  tune or ablate it.
* **Plain GCN.** Implemented in the same framework with mean-normalised aggregation and a self transform
  (not PyG's symmetric normalisation). Swap in `torch_geometric.nn.GCNConv` if your paper needs the textbook layer.
* **Memory.** Full HI-Small is ~5M edges and is run full-batch. Use a GPU with 16-24 GB, or lower `hidden` in `config.py`.
  `--dev` keeps only 10 % of normal rows, which changes the fraud rate, so never report dev numbers.
* **Synthetic data.** HI-Small is simulated, so conclusions may not transfer to real banks. Validate on another
  benchmark (e.g. Elliptic) as future work.
* **GPU determinism.** `index_add_` on CUDA is not bit-exact, so seeds reduce but do not eliminate run-to-run variation.

## Suggested extensions

GRU/temporal-attention history encoder, contrastive loss across time windows, probability calibration (isotonic/Platt),
Optuna search on validation AP, neighbour sampling for larger datasets.
