"""Round 31, Task 5 item 4: trajectory shape and change points inside each account (label-free; Task 5 has only test.csv).

    python fg_r31_t5_trajectory.py

Every account has a state vector z_t in R^24 per period t = 1..4 (the 24 continuous behaviours; device_count, guild_changes and
refund_count have 4-6 values and are left out). Each behaviour is turned into normal scores over all 4n values pooled (so a change
is measured on a common scale), the population median of each period is subtracted (common time trends), and a missing period
takes the account's own mean (no change). Statistics per account, on U = the 23,741 accounts the pipeline does not flag:
  accel4   ||z4 - z3||^2 - (||z3 - z2||^2 + ||z2 - z1||^2) / 2        (sudden change in period 4 after a steady 1-3)
  break_t  CUSUM-type break at t = 2, 3, 4: (n1 n2 / 4) sum_g (mean after - mean before)^2 / s_g^2, s_g^2 = median within-account
           variance of behaviour g; the account's break period = argmax_t
  maha_t   robust Mahalanobis distance of the break vector (mean after - mean before) at t = 3 and t = 4 (MCD covariance)
  cos      mean cosine of consecutive change vectors (dz_2, dz_3), (dz_3, dz_4): +1 = steady drift (gradual botting), -1 = ping-pong
Null: the order of each account's four state vectors permuted at random (one permutation for all behaviours of the account; the
per-period centring redone), 10 draws. Under "no planted trajectory" the periods are exchangeable, so every statistic has the same
distribution. Reported: accounts above the null's 99 / 99.9 / 99.99% quantiles (real vs null), the break-period split of the
accounts beyond the null's 99.9% (a period-3 or period-4 group would tilt it), the largest upper-tail density gap (lambda), and the
direction coherence of the top 200 (mean pairwise cosine of their break vectors; a planted group shares a direction).
Power check: 150 unflagged accounts copied with a shift of d within-account SDs in 6 behaviours from period 3 (or only period 4),
d in {1, 1.5, 2}: how many of them pass the null's 99.9% quantile and how the excess changes.
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.covariance import MinCovDet  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
rng = np.random.default_rng(0)

df = task5.load()
X, _ = task5.build_features(df)
_, gm = task5.gap_clusters(df, X)
gap = gm.any(axis=1).values
pocket, _ = task5.pocket_detector(X, gap, ["accuracy_std_level", "kda_std_level"], n_perm=300)
u = ~(gap | pocket)
BEH = [g for g in task5.temporal_groups(df) if g not in ("login_lat", "login_lon", "device_count", "guild_changes", "refund_count")]
V = np.stack([df.loc[u, [f"{g}_{i}" for i in range(1, 5)]].to_numpy(float) for g in BEH], axis=2)      # n x 4 x 24
n = len(V)
log(f"U {n} accounts, {len(BEH)} behaviours")


def normal_scores(V):
    Z = np.empty_like(V)
    for j in range(V.shape[2]):
        v = V[:, :, j].ravel()
        ok = ~np.isnan(v)
        r = np.full(v.shape, np.nan)
        r[ok] = (pd.Series(v[ok]).rank(method="average").values - 0.5) / ok.sum()
        Z[:, :, j] = norm.ppf(r).reshape(V.shape[:2])
    m = np.nanmean(Z, axis=1, keepdims=True)
    Z = np.where(np.isnan(Z), np.broadcast_to(m, Z.shape), Z)
    return np.nan_to_num(Z)


def centre(Z):
    return Z - np.median(Z, axis=0, keepdims=True)


Z0 = centre(normal_scores(V))
S2 = np.median(Z0.var(axis=1), axis=0)                        # typical within-account variance per behaviour
cov_cache = {}


def stats(Z, fit=False):
    d = np.diff(Z, axis=1)                                     # n x 3 x 24
    sq = (d ** 2).sum(2)
    out = {"accel4": sq[:, 2] - 0.5 * (sq[:, 1] + sq[:, 0])}
    br = []
    for t in (2, 3, 4):
        a, b = Z[:, :t - 1].mean(1), Z[:, t - 1:].mean(1)
        n1, n2 = t - 1, 5 - t
        br.append((n1 * n2 / 4) * (((b - a) ** 2) / S2).sum(1))
        if t in (3, 4):
            vec = b - a
            if fit:
                cov_cache[t] = MinCovDet(random_state=0, support_fraction=0.9).fit(vec[np.random.default_rng(1).choice(len(vec), 6000, replace=False)])
            out[f"maha{t}"] = cov_cache[t].mahalanobis(vec)
            out[f"vec{t}"] = vec
    br = np.stack(br, 1)
    out["break_max"] = br.max(1)
    out["break_at"] = br.argmax(1) + 2
    for t in (3, 4):
        out[f"break{t}"] = br[:, t - 2]
    nrm = np.linalg.norm(d, axis=2) + 1e-9
    out["cos"] = 0.5 * ((d[:, 0] * d[:, 1]).sum(1) / (nrm[:, 0] * nrm[:, 1]) + (d[:, 1] * d[:, 2]).sum(1) / (nrm[:, 1] * nrm[:, 2]))
    out["neg_cos"] = -out["cos"]
    return out


def permute_periods(V, r):
    P = np.argsort(r.random((len(V), 4)), axis=1)
    return np.take_along_axis(V, P[:, :, None], axis=1)


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


real = stats(Z0, fit=True)
VZ = normal_scores(V)                                            # normal scores before centring, permuted per draw
nulls = [stats(centre(permute_periods(VZ, np.random.default_rng(10 + i)))) for i in range(10)]
log("statistics on the real data and 10 period-permuted nulls done")
KEYS = ["accel4", "break_max", "break3", "break4", "maha3", "maha4", "cos", "neg_cos"]
for key in KEYS:
    nv = np.concatenate([d[key] for d in nulls])
    line = f"{key:10s}"
    for q in (0.99, 0.999, 0.9999):
        th = np.quantile(nv, q)
        line += f" | >{q:.2%} null q: real {int(np.sum(real[key] > th))} vs null {np.mean([np.sum(d[key] > th) for d in nulls]):.1f}"
    line += f" | upper-tail gap lambda real {lam_gap(real[key]):.1f}, null max {max(lam_gap(d[key]) for d in nulls):.1f}"
    log(line)
th = np.quantile(np.concatenate([d["break_max"] for d in nulls]), 0.999)
sel = real["break_max"] > th
log(f"break period of the accounts beyond the null's 99.9% of break_max: real {np.bincount(real['break_at'][sel], minlength=5)[2:].tolist()} "
    f"(t = 2, 3, 4), null mean {np.mean([np.bincount(d['break_at'][d['break_max'] > th], minlength=5)[2:] for d in nulls], 0).round(1).tolist()}")


def coherence(vec, score, top=200):
    i = np.argsort(-score)[:top]
    w = vec[i] / (np.linalg.norm(vec[i], axis=1, keepdims=True) + 1e-9)
    c = w @ w.T
    return (c.sum() - top) / (top * (top - 1))


for t in (3, 4):
    log(f"direction coherence of the top 200 by maha{t}: real {coherence(real[f'vec{t}'], real[f'maha{t}']):+.3f}, null "
        f"{np.mean([coherence(d[f'vec{t}'], d[f'maha{t}']) for d in nulls]):+.3f} (max {max(coherence(d[f'vec{t}'], d[f'maha{t}']) for d in nulls):+.3f})")

# power check
SDg = np.sqrt(np.median(np.nanvar(V, axis=1), axis=0))          # within-account SD per behaviour, raw units
pick6 = ["currency_transfers", "purchase_amount_mean", "login_count", "session_duration_mean", "skill_rating", "kda_mean"]
jj = [BEH.index(g) for g in pick6]
for start in (3, 4):
    for dlt in (1.0, 1.5, 2.0):
        rr = np.random.default_rng(int(dlt * 10) + start)
        src = rr.choice(n, 150, replace=False)
        add = V[src].copy()
        for j in jj:
            add[:, start - 1:, j] += dlt * SDg[j] * rr.choice([-1, 1])
        Vi = np.concatenate([V, add])
        Zi = centre(normal_scores(Vi))
        si = stats(Zi)
        nz = [stats(centre(permute_periods(normal_scores(Vi), np.random.default_rng(50 + i)))) for i in range(3)]
        key = f"maha{start}"
        thi = np.quantile(np.concatenate([d[key] for d in nz]), 0.999)
        tb = np.quantile(np.concatenate([d["break_max"] for d in nz]), 0.999)
        log(f"power: shift {dlt} SD in 6 behaviours from period {start}: planted beyond the null 99.9% of {key} "
            f"{int(np.sum(si[key][n:] > thi))}/150, of break_max {int(np.sum(si['break_max'][n:] > tb))}/150; total beyond ({key}) real "
            f"{int(np.sum(si[key] > thi))} vs null {np.mean([np.sum(d[key] > thi) for d in nz]):.1f}; coherence of the top 200 "
            f"{coherence(si[f'vec{start}'], si[key]):+.3f}")
log("done")
