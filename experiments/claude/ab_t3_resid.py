"""Task 3: hurdle stage 2 on the RATIO target r = log1p(y) - log1p(base) (base = historical_spending, filled),
instead of log1p(y). The generator looks multiplicative (y ~ hist * segment factor * noise, aa_t3_target_structure),
which trees approximate only piecewise when they have to model log1p(y) directly. Same folds as the notebook.
Also: how well can the 5.8% missing `segment` be recovered from the other features (train rows only)?"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import common  # noqa
from common import t3_blend_score, OUT
from src import task3
from src.metrics import nmae
from src.modeling import make_folds, fit_predict
from t3_fill import fill

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, Xt = task3.add_features(trf), task3.add_features(tef)
z = np.log1p(y)
pos = y > 0
h = dict(np.load(OUT / "t3_identity_hurdle.npz"))
pool = np.load(OUT / "t3_identity_pool_all.npz")
names, P, Pt, gate = list(pool["names"]), pool["P_oof"], pool["P_test"], pool["gate_oof"]

REG = {"objective": "regression_l1", "learning_rate": 0.05}
res = {}
for bname in ("historical_spending", "prev_month_spending"):
    b, bt = np.log1p(trf[bname].values), np.log1p(tef[bname].values)
    okb, okbt = np.isfinite(b), np.isfinite(bt)
    mu, mu_t = h["mu_oof"].copy(), h["mu_test"].copy()          # fallback where the base is missing
    mu_t_acc = np.zeros(len(Xt))
    for a, v in folds:
        ap, vp = a[pos[a] & okb[a]], v[pos[v] & okb[v]]
        vb = v[okb[v]]
        _, (rv, rt), _ = fit_predict("lgb", "regression", REG, X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp],
                                     [X.iloc[vb], Xt])
        mu[vb] = b[vb] + rv
        mu_t_acc += rt / len(folds)
    mu_t[okbt] = bt[okbt] + mu_t_acc[okbt]
    m = pos
    print(f"base {bname}: stage-2 MAE on log1p(y), positive rows: old {np.mean(np.abs(z[m] - h['mu_oof'][m])):.4f} "
          f"-> ratio target {np.mean(np.abs(z[m] - mu[m])):.4f}  (base missing on {int((~okb).sum())} train rows)", flush=True)
    res[bname] = (mu, mu_t)


def members(mu, mu_t, tag):
    resid = z[pos] - mu[pos]
    sigma0 = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    smear = float(np.mean(np.exp(resid)))
    tm = task3.tune_mixture(y, h["p_oof"], mu, sigma0)
    th = task3.tune_hard_hurdle(y, h["p_oof"], mu)
    sh = lambda p: np.clip(p + tm["shift"], 0, 1)
    return {f"{tag} mixture-median": (task3.mixture_median(sh(h["p_oof"]), mu, tm["sigma"]),
                                      task3.mixture_median(sh(h["p_test"]), mu_t, tm["sigma"])),
            f"{tag} hard gate": (task3.hard_hurdle(h["p_oof"], mu, th["tau"]), task3.hard_hurdle(h["p_test"], mu_t, th["tau"])),
            f"{tag} E[y]": (task3.expected_value(h["p_oof"], mu, smear), task3.expected_value(h["p_test"], mu_t, smear))}


s0, pred0 = t3_blend_score(y, folds, P, gate)
print(f"\nblend, current pool (v6): {s0:.5f}")
extra = {}
for bname, (mu, mu_t) in res.items():
    mem = members(mu, mu_t, f"ratio[{bname[:4]}]")
    for k, (o, t) in mem.items():
        print(f"  {k:32s} alone {nmae(y, o):.5f}")
    extra.update(mem)
    Pn = np.column_stack([P] + [o for o, _ in mem.values()])
    s, _ = t3_blend_score(y, folds, Pn, gate)
    print(f"  pool + ratio[{bname[:4]}] members: {s:.5f} ({s - s0:+.5f})", flush=True)
Pn = np.column_stack([P] + [o for o, _ in extra.values()])
Ptn = np.column_stack([Pt] + [t for _, t in extra.values()])
s_all, pred_all = t3_blend_score(y, folds, Pn, gate)
print(f"  pool + both: {s_all:.5f} ({s_all - s0:+.5f})")
np.savez(OUT / "t3_ratio_members.npz", P_oof=Pn, P_test=Ptn, names=np.array(names + list(extra)),
         **{f"mu_{k}": v[0] for k, v in res.items()}, **{f"mut_{k}": v[1] for k, v in res.items()})

# whale-only model on top (same recipe as v6)
seg = tr.segment.fillna(-1).values
w = seg == 3
WP = dict(objective="regression_l1", learning_rate=0.02, num_leaves=15, min_child_samples=50, colsample_bytree=0.8)
oof_w = np.full(len(y), np.nan)
for a, b in folds:
    a, b = a[w[a]], b[w[b]]
    _, (pv,), _ = fit_predict("lgb", "regression", WP, X.iloc[a], z[a], X.iloc[b], z[b], [X.iloc[b]])
    oof_w[b] = np.clip(np.expm1(pv), 0, None)
for nm, pr in (("v6 pool", pred0), ("pool + ratio members", pred_all)):
    q = pr.copy(); q[w] = 0.5 * oof_w[w] + 0.5 * pr[w]
    print(f"  {nm:22s} + whale 50/50: {nmae(y, q):.5f}")

# segment recoverability (train rows with a known segment, features without segment)
from sklearn.metrics import accuracy_score
known = ~np.isnan(tr.segment.values)
Xs = X.drop(columns=["segment"])
sk = tr.segment.values
acc = []
for a, b in folds[:2]:
    a, b = a[known[a]], b[known[b]]
    _, (pp,), _ = fit_predict("lgb", "multiclass", {"learning_rate": 0.1}, Xs.iloc[a], sk[a].astype(int), Xs.iloc[b],
                              sk[b].astype(int), [Xs.iloc[b]])
    acc.append(accuracy_score(sk[b], pp.argmax(1)))
print(f"\nsegment recoverable from other features: accuracy {np.mean(acc):.3f} (2 folds)")
