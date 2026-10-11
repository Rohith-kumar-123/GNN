"""
Train and evaluate one GNN variant for one seed.

Loss = focal(edge classification)
       + beta  * reliability loss  (weak supervision, training labels only)
       + gamma * entropy of r      (makes reliability decisive instead of "mushy 0.5")
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from .config import ModelConfig, TrainConfig
from .graph import make_phases
from .metrics import best_f1_threshold, evaluate
from .model import HRGNN, focal_loss
from . import features as FE


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def score_phase(model, ph):
    """Fraud scores (and reliability) for the target edges of a phase."""
    model.eval()
    idx = torch.arange(ph.start, ph.end, device=ph.src.device)
    logits, r, p = model(ph, idx)
    score = torch.sigmoid(logits).cpu().numpy()
    r_t = r[idx].cpu().numpy() if r is not None else None
    return score, r_t, ph.y[idx].cpu().numpy()


def sample_batch(y_train: torch.Tensor, neg_ratio: int, device):
    """All fraud edges + `neg_ratio` random normal edges per fraud edge."""
    pos = torch.nonzero(y_train == 1).squeeze(-1)
    neg = torch.nonzero(y_train == 0).squeeze(-1)
    if neg_ratio > 0:
        k = min(len(neg), neg_ratio * max(len(pos), 1))
        neg = neg[torch.randperm(len(neg), device=device)[:k]]
    return torch.cat([pos, neg])


def run_gnn(prep, name: str, mcfg: ModelConfig, tcfg: TrainConfig, seed: int, verbose=True):
    set_seed(seed)
    device = get_device(tcfg.device)
    phases, weak = make_phases(prep, mcfg, device)
    tr, va, te = phases["train"], phases["val"], phases["test"]

    n_persist = len(FE.PERSIST_COLS)
    model = HRGNN(mcfg, tr.node_x.size(1), n_persist, tr.edge_x.size(1)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        idx = sample_batch(tr.y, tcfg.neg_ratio, device)
        logits, r, _ = model(tr, idx)
        loss = focal_loss(logits, tr.y[idx], tcfg.focal_gamma, tcfg.focal_alpha)

        if r is not None:
            # weak supervision: edges next to known-fraud accounts should get LOW reliability
            target = 1.0 - weak
            w = torch.where(weak == 1, 0.5 / weak.mean().clamp_min(1e-6),
                            0.5 / (1 - weak.mean()).clamp_min(1e-6))   # balance both groups
            rel_loss = F.binary_cross_entropy(r.clamp(1e-6, 1 - 1e-6), target, weight=w)
            entropy = -(r * torch.log(r + 1e-6) + (1 - r) * torch.log(1 - r + 1e-6)).mean()
            loss = loss + tcfg.beta_rel * rel_loss + tcfg.gamma_entropy * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # validation (graph = train + val edges, targets = val edges)
        v_score, _, v_y = score_phase(model, va)
        v_ap = average_precision_score(v_y, v_score) if v_y.sum() else 0.0
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{name} s{seed}] epoch {epoch:3d}  loss {loss.item():.4f}  val AP {v_ap:.4f}")

        if v_ap > best_ap:
            best_ap, best_epoch, bad = v_ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.patience:
                break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    v_score, _, v_y = score_phase(model, va)
    t_score, t_r, t_y = score_phase(model, te)
    thr = best_f1_threshold(v_y, v_score)               # chosen on validation, frozen for test
    result = {
        "model": name, "seed": seed, "best_epoch": best_epoch, "n_params": n_params,
        "train_seconds": train_time,
        "val": evaluate(v_y, v_score, thr),
        "test": evaluate(t_y, t_score, thr),
    }
    if mcfg.use_time_decay:
        result["lambda_per_day"] = float(F.softplus(model.raw_lambda).item())
    arrays = {"y": t_y, "score": t_score}
    if t_r is not None:
        arrays["r"] = t_r
    return result, arrays
