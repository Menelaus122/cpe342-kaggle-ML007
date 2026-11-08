"""Task 1: tabular MLP (trained from scratch) as an extra, differently-biased stack member.

Inputs: raw features, quantile-transformed to N(0,1) on the training fold, NaN -> 0 plus a missing indicator.
5 folds (the notebook's), AdamW, cosine schedule, early stopping on validation log-loss.
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, t1_score, OUT  # noqa: E402
from src.utils import load_preds  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
A, At = tr[raw].to_numpy(float), te[raw].to_numpy(float)


def prep(qt, M):
    Z = qt.transform(M)
    miss = np.isnan(M).astype(np.float32)
    return np.hstack([np.nan_to_num(Z, nan=0.0), miss]).astype(np.float32)


def make(d):
    return nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
                         nn.Linear(256, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
                         nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.1),
                         nn.Linear(128, 1)).to(dev)


def predict(m, Z):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([m(torch.from_numpy(Z[i:i + 8192]).to(dev)) for i in range(0, len(Z), 8192)])).cpu().numpy().ravel()


oof, test = np.zeros(len(y)), np.zeros(len(At))
for f, (a, b) in enumerate(folds):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(A[a])
    Za, Zb, Zt = prep(qt, A[a]), prep(qt, A[b]), prep(qt, At)
    m = make(Za.shape[1])
    epochs, bs = 40, 512
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(a) // bs + 1))
    Xa, ya = torch.from_numpy(Za).to(dev), torch.from_numpy(y[a].astype(np.float32)).to(dev)
    best, best_state, bad = 9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(a), device=dev)
        for i in range(0, len(a), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(m(Xa[idx]).ravel(), ya[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ll = log_loss(y[b], np.clip(predict(m, Zb), 1e-6, 1 - 1e-6))
        if ll < best - 1e-5:
            best, best_state, bad = ll, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    m.load_state_dict(best_state)
    oof[b] = predict(m, Zb)
    test += predict(m, Zt) / len(folds)
    print(f"fold {f}: AUC {roc_auc_score(y[b], oof[b]):.4f}, log-loss {best:.4f}, epochs {ep + 1}", flush=True)
np.savez(OUT / "t1_mlp.npz", oof=oof, test=test)
base = [load_preds(n) for n in T1_MODELS]
s0, thr0, so0, _ = t1_score([d["oof"] for d in base], [d["test"] for d in base], y, folds)
s1, thr1, so1, _ = t1_score([d["oof"] for d in base] + [oof], [d["test"] for d in base] + [test], y, folds)
print(f"MLP alone: AUC {roc_auc_score(y, oof):.4f} (LightGBM {roc_auc_score(y, base[0]['oof']):.4f})")
print(f"stack F2 {s0:.4f} -> with MLP {s1:.4f} ({s1 - s0:+.4f}); AUC {roc_auc_score(y, so0):.4f} -> {roc_auc_score(y, so1):.4f}")
