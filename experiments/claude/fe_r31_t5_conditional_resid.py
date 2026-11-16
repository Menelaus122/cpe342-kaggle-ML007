"""Round 31, Task 5 item 2: feature-wise conditional prediction residuals (label-free; Task 5 has only test.csv, by design).

    python fe_r31_t5_conditional_resid.py

Attribute-wise learning for outlier scoring (ALSO; Paulheim & Meusel, Machine Learning 2015): predict each attribute from the
others and score an account by how far its values are from what its own context predicts. A group whose period-4 values are not
extreme in any column but inconsistent with its own context (e.g. purchases and transfers far above what its own period-4
activity, logins and session time imply) is invisible to marginal gaps but leaves large conditional residuals.
On U (the 23,741 accounts the pipeline does not flag), for each of the 24 continuous behaviours g: target = normal score of g_4
(over U); inputs = periods 1-3 of g, all four periods of the other behaviours, the static columns and the yes/no flags; LightGBM,
Huber loss, 5-fold out of fold (4 threads). Standardised residual r_g = (y - y_hat) / MAD. Account scores: S2 = sum_g r_g^2
(clipped at 10 per behaviour), Smax = max_g |r_g|, and the economy residual (purchase_amount_mean_4 + currency_transfers_4 +
purchase_frequency_4, signed).
Null: a Gaussian copula with U's marginals and rank correlations (no non-linear structure, no planted group), same models. As the
copula also removes the real non-linear relations, tails are compared in MAD units and through the self-calibrated gap statistic
(lambda) rather than raw counts alone.
Power check: 150 unflagged accounts copied with purchase_amount_mean_4 and currency_transfers_4 raised to their own 85-95th
percentile of U (not extreme marginally) while every other column is kept; how many reach the top 1% of S2 / economy residual.
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)

df = task5.load()
X, _ = task5.build_features(df)
_, gm = task5.gap_clusters(df, X)
gap = gm.any(axis=1).values
pocket, _ = task5.pocket_detector(X, gap, ["accuracy_std_level", "kda_std_level"], n_perm=300)
u = ~(gap | pocket)
BEH = [g for g in task5.temporal_groups(df) if g not in ("login_lat", "login_lon", "device_count", "guild_changes", "refund_count")]
COLS = [c for c in df.columns if c != "player_id"]
D0 = df.loc[u, COLS].reset_index(drop=True)
ECON = ["purchase_amount_mean", "currency_transfers", "purchase_frequency"]
PRM = dict(objective="huber", alpha=1.0, learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=40,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=4, verbose=-1)
log(f"U {len(D0)} accounts, {len(BEH)} behaviours, {len(COLS)} columns")


def nscore(x):
    r = pd.Series(x).rank(method="average", na_option="keep")
    return np.asarray(norm.ppf(((r - 0.5) / r.notna().sum()).to_numpy(float)))


def residuals(D, seed=0):
    R = np.zeros((len(D), len(BEH)))
    kf = KFold(5, shuffle=True, random_state=seed)
    for j, g in enumerate(BEH):
        y = nscore(D[f"{g}_4"].to_numpy(float))
        feats = [c for c in D.columns if c != f"{g}_4"]
        ok = ~np.isnan(y)
        pred = np.full(len(D), np.nan)
        for a, b in kf.split(D):
            a = a[ok[a]]
            m = lgb.LGBMRegressor(**PRM, random_state=seed).fit(D.iloc[a][feats], y[a])
            pred[b] = m.predict(D.iloc[b][feats])
        r = y - pred
        mad = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826
        R[:, j] = np.nan_to_num(r / mad)
    return R


def summarise(R):
    S2 = np.minimum(R ** 2, 100).sum(1)
    Smax = np.abs(R).max(1)
    econ = R[:, [BEH.index(g) for g in ECON]].sum(1)
    return {"S2": S2, "Smax": Smax, "economy+": econ, "economy-": -econ}


def lam_gap(x, tail=0.01, k=200):
    s = np.sort(x)
    nn = len(s)
    g = np.diff(s)
    idx = np.arange(int((1 - tail) * nn), nn - 1)
    best = 0.0
    for i in idx[np.argsort(-g[idx])[:3]]:
        if nn - 1 - i < 10:
            continue
        bulk = s[max(0, i - k):i + 1]
        best = max(best, (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[i])
    return best


def copula(D, rng):
    Zs = np.column_stack([np.nan_to_num(nscore(D[c].to_numpy(float))) for c in D.columns])
    C = np.corrcoef(Zs, rowvar=False)
    G = rng.multivariate_normal(np.zeros(len(D.columns)), C, size=len(D), method="eigh")
    out = {}
    for j, c in enumerate(D.columns):
        v = D[c].to_numpy(float)
        vals = np.sort(v[~np.isnan(v)])
        q = norm.cdf(G[:, j])
        x = vals[np.minimum((q * len(vals)).astype(int), len(vals) - 1)]
        x[np.random.default_rng(j).random(len(x)) < np.isnan(v).mean()] = np.nan      # keep the missing share
        out[c] = x
    return pd.DataFrame(out)


Rr = residuals(D0)
log("real residuals done")
Rn = residuals(copula(D0, np.random.default_rng(0)))
log("copula-null residuals done")
sr, sn = summarise(Rr), summarise(Rn)
for k in sr:
    line = f"{k:10s}"
    for q in (0.99, 0.999):
        th = np.quantile(sn[k], q)
        line += f" | beyond null {q:.1%}: real {int(np.sum(sr[k] > th))} (null {int(np.sum(sn[k] > th))})"
    line += f" | upper-tail gap lambda real {lam_gap(sr[k]):.1f}, null {lam_gap(sn[k]):.1f}"
    log(line)
ex = []
for j, g in enumerate(BEH):
    ex.append((g, int(np.sum(np.abs(Rr[:, j]) > 6)), int(np.sum(np.abs(Rn[:, j]) > 6)), lam_gap(np.abs(Rr[:, j])), lam_gap(np.abs(Rn[:, j]))))
ex.sort(key=lambda t: -t[3])
log("per behaviour, |r| > 6 MAD (real / null) and upper-tail gap lambda of |r| (real / null), largest real lambda first: "
    + "; ".join(f"{g} {a}/{b}, {la:.1f}/{lb:.1f}" for g, a, b, la, lb in ex[:8]))
top = np.argsort(-sr["S2"])[:200]
log("top 200 by S2: behaviours with the largest mean |r|: "
    + ", ".join(f"{BEH[j]} {v:.2f}" for j, v in sorted(enumerate(np.abs(Rr[top]).mean(0)), key=lambda t: -t[1])[:6])
    + f"; mean pairwise cosine of their residual vectors {((lambda W: (W @ W.T).sum() - len(W))(Rr[top] / np.linalg.norm(Rr[top], axis=1, keepdims=True))) / (200 * 199):+.3f} "
      f"(null top 200: {((lambda W: (W @ W.T).sum() - len(W))(Rn[np.argsort(-sn['S2'])[:200]] / np.linalg.norm(Rn[np.argsort(-sn['S2'])[:200]], axis=1, keepdims=True))) / (200 * 199):+.3f})")

# power check
rng = np.random.default_rng(5)
src = rng.choice(len(D0), 150, replace=False)
add = D0.iloc[src].copy()
for g in ("purchase_amount_mean", "currency_transfers"):
    v = D0[f"{g}_4"].dropna().to_numpy()
    add[f"{g}_4"] = np.quantile(v, rng.uniform(0.85, 0.95, 150))
Di = pd.concat([D0, add], ignore_index=True)
si = summarise(residuals(Di))
for k in ("S2", "economy+"):
    th = np.quantile(si[k][:len(D0)], 0.99)
    log(f"power: planted context-inconsistent purchases / transfers (85-95th percentile, not extreme): {int(np.sum(si[k][len(D0):] > th))}/150 "
        f"in the top 1% of {k}; upper-tail gap lambda {lam_gap(si[k]):.1f}")
log("done")
