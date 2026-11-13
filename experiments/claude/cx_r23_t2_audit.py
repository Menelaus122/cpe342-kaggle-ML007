"""Round 23, Task 2: does anything non-linear, geometric or calibration-based move past the stack (nested 0.7564)? Training rows only.

    python cx_r23_t2_audit.py

  A. Is there within-class dependence that linear correlation misses? For Casual, Grinder and Social separately: binned mutual
     information (20x20 quantile grid) of every pair of continuous columns, against the same statistic after shuffling each column
     within the class (marginals kept, dependence destroyed).
  B. Is naive Bayes merely miscalibrated? Temperature on the KDE naive Bayes log-posterior (fitted nested), log-loss vs the stack.
  C. LightGBM (pipeline parameters) with the brief's extra ratios and row-wise group summaries (z-score mean / max / min of
     playtime-combat, social and spending groups, and their differences).
  D. Non-tree models on quantile-normalised numeric features: a torch MLP (3 seeds, dropout, label smoothing, cosine schedule),
     Nystroem RBF features + logistic regression, distance-weighted KNN with k = 15 / 50 / 150.
  E. Class multipliers with a quadratic pull towards 1: max macro-F1 - lambda * sum (w_c - 1)^2, lambda in {0.05, 0.1, 0.2}.
Every candidate is scored alone (argmax) and as a fifth stack member (pipeline LogReg stacker C = 0.5, multipliers chosen nested).
Pass rule (as in round 21): nested gain of the stack with the member, paired-bootstrap 95% CI above 0 and better in >= 4/5 folds.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize, minimize_scalar
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds, run_cv  # noqa: E402

OUT = Path("experiments/claude/outputs/r23_t2")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
NAMES = ["Casual", "Grinder", "Social", "Whale"]
train, test = task2.load()
X, y, Xt = task2.build(train, test)
folds = make_folds(y)
fold_of = np.zeros(len(y), int)
for f, (_, va) in enumerate(folds):
    fold_of[va] = f
cat = [c for c in X.columns if str(X[c].dtype) == "category"]
num = [c for c in X.columns if c not in cat]
mf1 = lambda yy, p: f1_score(yy, p, average="macro")

# ---------------------------------------------------------------- pipeline stack (reference)
M = {k: np.load(f"outputs/preds/t2_{k}.npz")["oof"] for k in ("lgb", "xgb", "cat", "lr")}


def stack(members):
    S = np.column_stack([np.log(np.clip(m, 1e-6, 1)) for m in members])
    out = np.zeros((len(y), 4))
    for tr, va in folds:
        out[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    return out


def nested(P, lam=None):
    out = np.zeros(len(y), int)
    for tr, va in folds:
        if lam is None:
            w = optimize_class_weights(y[tr], P[tr])[0]
        else:
            obj = lambda w: -(mf1(y[tr], apply_class_weights(P[tr], np.abs(w))) - lam * np.sum((np.abs(w) - 1) ** 2))
            w = np.abs(minimize(obj, np.ones(4), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6}).x)
        out[va] = apply_class_weights(P[va], w)
    return out


base_p = stack(list(M.values()))
base = nested(base_p)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
ll = lambda P: -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
log(f"pipeline stack: nested macro-F1 {mf1(y, base):.4f}, log-loss {ll(base_p):.5f}")


def compare(name, pred):
    d = [mf1(y[i], pred[i]) - mf1(y[i], base[i]) for i in boot]
    pf = [mf1(y[fold_of == f], pred[fold_of == f]) - mf1(y[fold_of == f], base[fold_of == f]) for f in range(5)]
    ok = np.quantile(d, .025) > 0 and sum(x > 0 for x in pf) >= 4
    print(f"    {name:40s} {mf1(y, pred):.4f} ({mf1(y, pred) - mf1(y, base):+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}], "
          f"{sum(x > 0 for x in pf)}/5) {'PASS' if ok else ''}", flush=True)


def member(name, oof):
    np.savez(OUT / f"{name}.npz", oof=oof)
    print(f"  {name}: alone argmax {mf1(y, oof.argmax(1)):.4f}, log-loss {ll(oof):.4f}", flush=True)
    sp = stack(list(M.values()) + [oof])
    compare(f"stack + {name} (log-loss {ll(sp):.5f})", nested(sp))


# ---------------------------------------------------------------- A. non-linear within-class dependence
log("A. within-class mutual information, real vs column-shuffled")
cont = [c for c in train.columns if c not in ("player_id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS)
        and train[c].nunique() > 50]
G = 20


def mi_pairs(R):
    Q = np.full(R.shape, -1, int)
    for j in range(R.shape[1]):
        ok = ~np.isnan(R[:, j])
        Q[ok, j] = np.minimum((pd.Series(R[ok, j]).rank(pct=True, method="first").values * G).astype(int), G - 1)
    out = []
    for i in range(R.shape[1]):
        for j in range(i + 1, R.shape[1]):
            ok = (Q[:, i] >= 0) & (Q[:, j] >= 0)
            c = np.bincount(Q[ok, i] * G + Q[ok, j], minlength=G * G).reshape(G, G).astype(float) / ok.sum()
            pi, pj = c.sum(1, keepdims=True), c.sum(0, keepdims=True)
            nz = c > 0
            out.append(((c[nz] * np.log(c[nz] / (pi @ pj)[nz])).sum(), i, j))
    return out


for k in range(3):
    R = train.loc[y == k, cont].to_numpy(float)
    real = mi_pairs(R)
    null = max(m for m, _, _ in mi_pairs(np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])))
    real.sort(reverse=True)
    above = [(round(m, 4), cont[i], cont[j]) for m, i, j in real if m > null]
    print(f"  {NAMES[k]:7s}: {len(real)} pairs; max MI {real[0][0]:.4f} vs shuffled max {null:.4f}; pairs above: {len(above)} {above[:4]}")

# ---------------------------------------------------------------- B. is naive Bayes just miscalibrated?
log("B. temperature-scaled KDE naive Bayes")
kde = np.load("experiments/claude/outputs/r21_t2/kde.npz")["oof"]
logp = np.log(np.clip(kde, 1e-300, 1))
cal = np.zeros_like(kde)
for tr, va in folds:
    def nll(t, rows):
        z = logp[rows] / t
        z = z - z.max(1, keepdims=True)
        return -(z[np.arange(len(rows)), y[rows]] - np.log(np.exp(z).sum(1))).mean()
    t = minimize_scalar(lambda t: nll(t, tr), bounds=(0.3, 5), method="bounded").x
    z = logp[va] / t
    cal[va] = np.exp(z - z.max(1, keepdims=True))
    cal[va] /= cal[va].sum(1, keepdims=True)
print(f"  KDE naive Bayes log-loss {ll(kde):.4f} -> {ll(cal):.4f} with a nested temperature (stack {ll(base_p):.4f})")

# ---------------------------------------------------------------- C. extra ratios and row-wise group summaries in LightGBM
log("C. LightGBM with the brief's ratios and group summaries")
GROUPS = {"combat": ["play_frequency", "avg_session_duration", "total_playtime_hours", "login_streak", "tournament_entries",
                     "competitive_rank", "win_rate_ranked", "ranked_participation_rate", "speed_of_progression", "avg_match_length",
                     "peak_concurrent_hours", "achievement_completion_rate"],
          "social": ["friend_count", "team_play_percentage", "chat_activity_score", "friend_invites_sent", "gifts_sent_received",
                     "item_type_preference_social"],
          "spend": ["total_spending_thb", "avg_monthly_spending", "spending_frequency", "rare_items_count", "collection_progress",
                    "vip_tier", "item_type_preference_cosmetic"]}


def extra(D):
    D = D.copy()
    e = 1e-5
    D["spend_total_per_month"] = D.total_spending_thb / (D.avg_monthly_spending + e)
    D["social_per_play"] = (D.friend_count + D.team_play_percentage) / (D.play_frequency * D.avg_session_duration + e)
    D["win_x_freq"] = D.win_rate_ranked * D.play_frequency
    for g, cols in GROUPS.items():
        Z = pd.concat([(D[c] - X[c].mean()) / X[c].std() for c in cols], axis=1)     # training-set statistics, label-free
        D[f"{g}_zmean"], D[f"{g}_zmax"], D[f"{g}_zmin"] = Z.mean(1), Z.max(1), Z.min(1)
    D["social_minus_combat"] = D.social_zmean - D.combat_zmean
    D["spend_minus_combat"] = D.spend_zmean - D.combat_zmean
    D["spend_minus_social"] = D.spend_zmean - D.social_zmean
    return D


import json  # noqa: E402
P2 = json.loads(Path("outputs/params/task2_lgb.json").read_text())
Xe, Xte = extra(X), extra(Xt)
r = run_cv("lgb", "multiclass", P2, Xe, y, Xte, folds, n_classes=4, verbose=False, name="T2 lgb+extra")
print(f"  pipeline LightGBM alone {mf1(y, M['lgb'].argmax(1)):.4f} -> with extras {mf1(y, r['oof'].argmax(1)):.4f}")
compare("stack with LightGBM replaced by lgb+extra", nested(stack([r["oof"], M["xgb"], M["cat"], M["lr"]])))
member("lgb_extra", r["oof"])

# ---------------------------------------------------------------- D. non-tree models
log("D. MLP / Nystroem-RBF / KNN on quantile-normalised numeric features")
Xn = Xe[num + [c for c in Xe.columns if c not in X.columns]].to_numpy(float)


def prep(tr, va):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=0).fit(Xn[tr])
    a, b = qt.transform(Xn[tr]), qt.transform(Xn[va])
    ma, mb = np.isnan(a), np.isnan(b)
    keep = ma.mean(0) > 0
    return (np.hstack([np.nan_to_num(a), ma[:, keep]]).astype(np.float32), np.hstack([np.nan_to_num(b), mb[:, keep]]).astype(np.float32))


def mlp(a, ya, b, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(a.shape[1], 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25),
                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 4))
    ep = 30
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=ep * (len(a) // 512 + 1))
    A, Y = torch.tensor(a), torch.tensor(ya)
    g = torch.Generator().manual_seed(seed)
    for _ in range(ep):
        net.train()
        for idx in torch.randperm(len(A), generator=g).split(512):
            loss = nn.functional.cross_entropy(net(A[idx]), Y[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(b)).softmax(1).numpy()


oof_mlp, oof_nys = np.zeros((len(y), 4)), np.zeros((len(y), 4))
oof_knn = {k: np.zeros((len(y), 4)) for k in (15, 50, 150)}
for f, (tr, va) in enumerate(folds):
    a, b = prep(tr, va)
    oof_mlp[va] = np.mean([mlp(a, y[tr], b, s) for s in range(3)], 0)
    ny = Nystroem(kernel="rbf", gamma=1.0 / a.shape[1], n_components=1500, random_state=0).fit(a)
    oof_nys[va] = LogisticRegression(C=1.0, max_iter=2000).fit(ny.transform(a), y[tr]).predict_proba(ny.transform(b))
    dist, ind = NearestNeighbors(n_neighbors=150, n_jobs=-1).fit(a).kneighbors(b)
    for k in oof_knn:
        w = 1.0 / (dist[:, :k] + 1e-6)
        P = np.zeros((len(va), 4))
        for c in range(4):
            P[:, c] = (w * (y[tr][ind[:, :k]] == c)).sum(1)
        oof_knn[k][va] = (P + 0.5) / (P.sum(1, keepdims=True) + 2.0)
    log(f"  fold {f} done")
member("mlp", oof_mlp)
member("nystroem_rbf", oof_nys)
for k, o in oof_knn.items():
    member(f"knn{k}", o)

# ---------------------------------------------------------------- E. regularised multipliers
log("E. multipliers with a quadratic pull towards 1")
for lam in [0.05, 0.1, 0.2]:
    compare(f"stack, multipliers lambda {lam}", nested(base_p, lam))
log("done")
