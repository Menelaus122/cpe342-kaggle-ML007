"""Round 26, Task 2: an MLP with periodic (PLR) embeddings of the numeric features as a fifth stack member.

    python dn_r26_t2_plr.py

Round 23's plain MLP on quantile-normalised inputs scored 0.7494 alone and +0.0000 in the stack. Periodic embeddings (Gorishniy et
al., NeurIPS 2022, "On Embeddings for Numerical Features in Tabular Deep Learning") are the usual fix for MLPs being too smooth in
each feature: every numeric feature x_j is mapped to [sin(2 pi c_jk x_j), cos(2 pi c_jk x_j)], k = 1..K, with learned frequencies
c_jk (initialised N(0, sigma^2)), then a per-feature linear layer + ReLU, and the concatenation goes to the MLP.
Settings fixed before the run (nothing is tuned on the scores below): K = 24, sigma = 0.3, embedding size 16, MLP 256-128, dropout
0.25, AdamW lr 1e-3 (one-cycle), weight decay 1e-5, label smoothing 0.05, 30 epochs, batch 512, 3 seeds averaged.
Inputs: the pipeline's numeric features (task2.build), quantile-normalised inside each fold (fitted on the fold's training rows);
missing cells become 0 and a missing flag goes straight to the MLP. Nominal columns are left out (Cramer's V <= 0.008, round 23).
Scored alone (argmax) and as a fifth member of the pipeline stack (LogReg stacker C = 0.5, class multipliers chosen nested).
Acceptance (round 25): mean nested gain > 0 and >= 3/5 folds; the CI is reported, not used as a gate.

"CatBoost feature combinations": the pipeline's CatBoost already gets the seven nominal columns as cat_features with the default
max_ctr_complexity = 4, i.e. combinations of up to four categorical features are already built; those columns carry no signal
(round 23), so there is nothing further to run.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402

OUT = Path("experiments/claude/outputs/r26")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
train, test = task2.load()
X, y, Xt = task2.build(train, test)
folds = make_folds(y)
fold_of = np.zeros(len(y), int)
for f, (_, va) in enumerate(folds):
    fold_of[va] = f
num = [c for c in X.columns if str(X[c].dtype) != "category"]
Xn = X[num].to_numpy(float)
mf1 = lambda yy, p: f1_score(yy, p, average="macro")
dev = "cuda"
K, SIGMA, D = 24, 0.3, 16


class PLR(nn.Module):
    def __init__(self, n_num, n_flag):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_num, K) * SIGMA)
        self.w = nn.Parameter(torch.randn(n_num, 2 * K, D) * (2 * K) ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_num, D))
        self.n_num = n_num
        self.mlp = nn.Sequential(nn.Linear(n_num * D + n_flag, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 4))

    def forward(self, x):
        v, flag = x[:, :self.n_num], x[:, self.n_num:]
        a = 2 * torch.pi * self.c[None] * v[..., None]
        e = torch.relu(torch.einsum("nfk,fkd->nfd", torch.cat([a.sin(), a.cos()], -1), self.w) + self.b)
        return self.mlp(torch.cat([e.flatten(1), flag], 1))


def prep(tr, va):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=0).fit(Xn[tr])
    a, b = qt.transform(Xn[tr]), qt.transform(Xn[va])
    ma, mb = np.isnan(a), np.isnan(b)
    keep = ma.mean(0) > 0
    return np.hstack([np.nan_to_num(a), ma[:, keep]]).astype(np.float32), np.hstack([np.nan_to_num(b), mb[:, keep]]).astype(np.float32)


def fit_predict(a, ya, b, seed, ep=30, bs=512):
    torch.manual_seed(seed)
    net = PLR(Xn.shape[1], a.shape[1] - Xn.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ep * (len(a) // bs + 1))
    A, Y = torch.tensor(a, device=dev), torch.tensor(ya, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(ep):
        net.train()
        for idx in torch.randperm(len(A), generator=g, device=dev).split(bs):
            if len(idx) < 2:
                continue
            loss = nn.functional.cross_entropy(net(A[idx]), Y[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        return torch.cat([net(c).softmax(1) for c in torch.tensor(b, device=dev).split(4096)]).cpu().numpy()


f_out = OUT / "t2_plr_mlp.npz"
if f_out.exists():
    oof = np.load(f_out)["oof"]
else:
    oof = np.zeros((len(y), 4))
    log(f"{len(y)} rows, {len(num)} numeric features, K {K}, sigma {SIGMA}")
    for f, (tr, va) in enumerate(folds):
        a, b = prep(tr, va)
        ps = [fit_predict(a, y[tr], b, s) for s in range(3)]
        oof[va] = np.mean(ps, 0)
        log(f"fold {f}: seeds alone {' '.join(f'{mf1(y[va], p.argmax(1)):.4f}' for p in ps)}, averaged {mf1(y[va], oof[va].argmax(1)):.4f}")
    np.savez(f_out, oof=oof)

M = {k: np.load(f"outputs/preds/t2_{k}.npz")["oof"] for k in ("lgb", "xgb", "cat", "lr")}


def stack(members):
    S = np.column_stack([np.log(np.clip(m, 1e-6, 1)) for m in members])
    out = np.zeros((len(y), 4))
    for tr, va in folds:
        out[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    return out


def nested(P):
    out = np.zeros(len(y), int)
    for tr, va in folds:
        out[va] = apply_class_weights(P[va], optimize_class_weights(y[tr], P[tr])[0])
    return out


ll = lambda P: -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
base_p = stack(list(M.values()))
base = nested(base_p)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
r23 = Path("experiments/claude/outputs/r23_t2/mlp.npz")
print(f"pipeline stack: nested macro-F1 {mf1(y, base):.4f}, log-loss {ll(base_p):.5f}")
print(f"alone (argmax): LightGBM {mf1(y, M['lgb'].argmax(1)):.4f} | PLR-MLP {mf1(y, oof.argmax(1)):.4f} (log-loss {ll(oof):.4f})"
      + (f" | round 23 plain MLP {mf1(y, np.load(r23)['oof'].argmax(1)):.4f}" if r23.exists() else ""))
for name, members in [("stack + PLR-MLP", list(M.values()) + [oof]), ("stack, PLR-MLP replaces LogReg", [M["lgb"], M["xgb"], M["cat"], oof])]:
    sp = stack(members)
    pred = nested(sp)
    d = [mf1(y[i], pred[i]) - mf1(y[i], base[i]) for i in boot]
    pf = [mf1(y[fold_of == f], pred[fold_of == f]) - mf1(y[fold_of == f], base[fold_of == f]) for f in range(5)]
    print(f"  {name:32s} {mf1(y, pred):.4f}  delta {mf1(y, pred) - mf1(y, base):+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]  "
          f"folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)})  log-loss {ll(sp):.5f}  labels changed {np.mean(pred != base):.3%}")
