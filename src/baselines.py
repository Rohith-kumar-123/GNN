"""
Non-graph reference models, trained on the same chronological split.

    lr_tx        logistic regression on transaction features only
    xgb_tx       XGBoost on transaction features only
    xgb_tx_hist  XGBoost on transaction + history features (shows what history gives
                 WITHOUT any graph, so the extra value of the GNN can be measured fairly)
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from .metrics import best_f1_threshold, evaluate


def run_baseline(prep, name: str, seed: int):
    y = prep.df["y"].values
    tr, va, te = (slice(*prep.bounds(p)) for p in ["train", "val", "test"])

    X = prep.edge_tx if name in ("lr_tx", "xgb_tx") else np.concatenate([prep.edge_tx, prep.edge_hist], axis=1)

    if name == "lr_tx":
        model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed)
        model.fit(X[tr], y[tr])
        predict = lambda s: model.predict_proba(X[s])[:, 1]
    else:
        pos = max(int(y[tr].sum()), 1)
        model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=min((len(y[tr]) - pos) / pos, 100.0),   # capped class weight
            eval_metric="aucpr", early_stopping_rounds=30, tree_method="hist",
            random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        predict = lambda s: model.predict_proba(X[s])[:, 1]

    v_score, t_score = predict(va), predict(te)
    thr = best_f1_threshold(y[va], v_score)
    result = {"model": name, "seed": seed,
              "val": evaluate(y[va], v_score, thr), "test": evaluate(y[te], t_score, thr)}
    return result, {"y": y[te], "score": t_score}
