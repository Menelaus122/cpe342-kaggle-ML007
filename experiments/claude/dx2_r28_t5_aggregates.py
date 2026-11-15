"""Round 28, Task 5, part 2: follow-ups of dx_r28_t5_unflagged_scan.py inside the unflagged accounts U (test features only, no labels).

    python dx2_r28_t5_aggregates.py

  1  trajectory pair pockets again, without pairs from the same behaviour and without the small-integer groups (part 1 counted
     mechanical same-group pairs)
  2  discrete columns: value combinations ACROSS groups only (within a group the periods share an account-level base count)
  3  aggregate statistics over all behaviours: a family that is only mildly unusual in each behaviour but in all of them at once
     is invisible to 1-D gaps and 2-D pockets. Per behaviour, a per-account statistic is turned into normal scores within U and
     averaged over the 24 continuous behaviours:
        volatility (signed)      log of the within-account SD over the four periods   -> "too steady" / "erratic in everything"
        odd period t (t = 1..4)  squared deviation of period t from the mean of the other three -> a moderate shift everywhere
        level (squared)          squared account mean                                 -> moderately extreme in many behaviours
     Null: every behaviour's column shuffled among U (marginals kept, cross-behaviour dependence destroyed). Reported: SD ratio
     real / null, accounts beyond the null's 0.1% / 99.9% quantiles (24 expected), the widest tail gap (lambda), and the
     rank correlation with behavior_stability and account_age_days.
  4  the "too typical" accounts of part 1 (Mahalanobis distance below the copula null): who are they?
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

df = task5.load()
X, _ = task5.build_features(df)
flag = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
U = ~flag
n_u = int(U.sum())
rng = np.random.default_rng(0)
DISC = ("device_count", "refund_count", "guild_changes")
GROUPS = [g for g in task5.temporal_groups(df) if g not in ("login_lat", "login_lon") and g not in DISC]
V = {g: df.loc[U, [f"{g}_{i}" for i in range(1, 5)]].to_numpy(float) for g in GROUPS}

# ---------------------------------------------------------------- 1. trajectory pair pockets, cross-behaviour pairs only
T = {}
for g in GROUPS:
    v = V[g]
    sd = np.nanstd(v)
    T[f"{g}|d12"], T[f"{g}|d23"], T[f"{g}|d34"] = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 3] - v[:, 2]
    T[f"{g}|curv2"], T[f"{g}|curv3"] = v[:, 0] - 2 * v[:, 1] + v[:, 2], v[:, 1] - 2 * v[:, 2] + v[:, 3]
    T[f"{g}|slope"] = (-1.5 * v[:, 0] - 0.5 * v[:, 1] + 0.5 * v[:, 2] + 1.5 * v[:, 3]) / 5
    T[f"{g}|range"] = np.nanmax(v, 1) - np.nanmin(v, 1)
    T[f"{g}|mid_vs_ends"] = (v[:, 1] + v[:, 2]) / 2 - (v[:, 0] + v[:, 3]) / 2
    T[f"{g}|p1_vs_rest"] = v[:, 0] - np.nanmean(v[:, 1:], 1)
T = pd.DataFrame(T)
cols = list(T.columns)
base = [c.split("|")[0].replace("_mean", "").replace("_std", "") for c in cols]
G = 20


def codes(R):
    Q = np.full(R.shape, -1, np.int16)
    for j in range(R.shape[1]):
        v = R[:, j]
        ok = ~np.isnan(v)
        r = pd.Series(v[ok] + 1e-9 * rng.standard_normal(ok.sum()) * (np.nanstd(v) + 1e-12)).rank(pct=True).to_numpy()   # random tie-break
        Q[ok, j] = np.minimum((r * G).astype(int), G - 1)
    return Q


def max_cells(R):
    Q = codes(R)
    out = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        if base[i] == base[j]:
            continue
        ok = (Q[:, i] >= 0) & (Q[:, j] >= 0)
        cnt = np.bincount(Q[ok, i] * G + Q[ok, j], minlength=G * G).reshape(G, G).astype(float)
        exp = np.outer(cnt.sum(1), cnt.sum(0)) / cnt.sum()
        z = (cnt - exp) / np.sqrt(np.maximum(exp, 1))
        k = int(np.argmax(z))
        out.append((z.flat[k], cols[i], cols[j], k // G, k % G, cnt.flat[k], exp.flat[k]))
    return sorted(out, key=lambda r: -r[0])


R = T.to_numpy(float)
real = max_cells(R)
noise = max(max_cells(np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])]))[0][0] for _ in range(3))
print(f"accounts {len(df)}; unflagged {n_u}")
print(f"\n== 1. trajectory pair pockets, cross-behaviour pairs only: {len(real)} pairs; noise level z {noise:.1f}; above it: {sum(r[0] > noise for r in real)}; top 6:")
for z, a, b, ia, ib, obs, exp in real[:6]:
    print(f"     z {z:5.1f}  {a} x {b}  cell ({ia}, {ib})  observed {obs:.0f} expected {exp:.1f}")

# ---------------------------------------------------------------- 2. discrete combinations across groups
disc = {c: df.loc[U, c].to_numpy(float) for c in task5.FLAG_COLS}
for g in DISC:
    for i in range(1, 5):
        disc[f"{g}_{i}"] = df.loc[U, f"{g}_{i}"].to_numpy(float)
    disc[f"{g}_sum"] = df.loc[U, [f"{g}_{i}" for i in range(1, 5)]].sum(axis=1, min_count=4).to_numpy(float)
D = pd.DataFrame(disc)
grp = {c: (c.rsplit("_", 1)[0] if c.rsplit("_", 1)[0] in DISC else c) for c in D.columns}
cd = {c: pd.factorize(D[c], sort=True)[0] for c in D.columns}


def combo_scan(cdx, order):
    out = []
    for combo in itertools.combinations(list(cdx), order):
        if len({grp[c] for c in combo}) < order:
            continue
        ok = np.all([cdx[c] >= 0 for c in combo], 0)
        sizes = [int(cdx[c].max()) + 1 for c in combo]
        key = np.zeros(ok.sum(), int)
        for c, s in zip(combo, sizes):
            key = key * s + cdx[c][ok]
        cnt = np.bincount(key, minlength=int(np.prod(sizes))).reshape(sizes).astype(float)
        exp = np.ones(sizes) * ok.sum()
        for ax, (c, s) in enumerate(zip(combo, sizes)):
            exp = exp * (np.bincount(cdx[c][ok], minlength=s) / ok.sum()).reshape([-1 if a == ax else 1 for a in range(order)])
        z = np.where(exp >= 5, (cnt - exp) / np.sqrt(np.maximum(exp, 1e-9)), 0)
        k = int(np.argmax(z))
        out.append((z.flat[k], combo, np.unravel_index(k, sizes), cnt.flat[k], exp.flat[k]))
    return sorted(out, key=lambda r: -r[0])


print("\n== 2. discrete value combinations across groups (4 flags, device / refund / guild counts per period and their sums)")
for order in (2, 3):
    real = combo_scan(cd, order)
    nz = max(combo_scan({c: rng.permutation(v) for c, v in cd.items()}, order)[0][0] for _ in range(5))
    print(f"  {order}-way: {len(real)} combinations; noise level z {nz:.1f}; above it: {sum(r[0] > nz for r in real)}; top 4:")
    for z, combo, cell, obs, exp in real[:4]:
        vals = [sorted(D[c].dropna().unique())[i] for c, i in zip(combo, cell)]
        print(f"     z {z:5.1f}  " + ", ".join(f"{c} = {v:g}" for c, v in zip(combo, vals)) + f"  observed {obs:.0f} expected {exp:.1f}  (excess {obs - exp:.0f})")

# ---------------------------------------------------------------- 3. aggregates over behaviours


def nscore(v):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    r = pd.Series(v[ok] + 1e-9 * rng.standard_normal(ok.sum()) * (np.nanstd(v[ok]) + 1e-12)).rank().to_numpy()
    out[ok] = norm.ppf((r - 0.5) / ok.sum())
    return out


def lam_tail(s, side, frac=0.05, k=200):
    s = np.sort(s[~np.isnan(s)])
    n = len(s)
    g = np.diff(s)
    idx = np.arange(int((1 - frac) * n), n - 1) if side == "high" else np.arange(0, int(frac * n))
    j = idx[int(np.argmax(g[idx]))]
    bulk = s[max(0, j - k):j + 1] if side == "high" else s[j + 1:j + 2 + k]
    return (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[j], (n - 1 - j) if side == "high" else (j + 1)


stats = {"volatility (signed)": ({g: np.log(np.nanstd(V[g], 1) + 1e-3 * np.nanstd(V[g])) for g in GROUPS}, False),
         "level (squared)": ({g: np.nanmean(V[g], 1) for g in GROUPS}, True)}
for t in range(4):
    stats[f"odd period {t + 1} (squared)"] = ({g: V[g][:, t] - np.nanmean(np.delete(V[g], t, axis=1), 1) for g in GROUPS}, True)
stab = df.loc[U, "behavior_stability"].to_numpy(float)
age = df.loc[U, "account_age_days"].to_numpy(float)
print("\n== 3. aggregates over the 24 continuous behaviours (normal scores within U, mean over behaviours; null = columns shuffled among U)")
print(f"  {'statistic':26s} | SD real / null | below null 0.1% | above null 99.9% (24 expected each) | gap lambda low (size) | gap lambda high (size) | rho with stability / age")
AGG = {}
for name, (per_group, squared) in stats.items():
    Z = np.column_stack([nscore(per_group[g]) for g in GROUPS])
    f = (lambda A: np.nanmean(A ** 2, 1)) if squared else (lambda A: np.nanmean(A, 1))
    a = f(Z)
    nulls = np.concatenate([f(np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])) for _ in range(20)])
    lo, hi = np.quantile(nulls, 0.001), np.quantile(nulls, 0.999)
    ll, lh = lam_tail(a, "low"), lam_tail(a, "high")
    ok = ~np.isnan(stab)
    ok2 = ~np.isnan(age)
    AGG[name] = a
    print(f"  {name:26s} | {np.nanstd(a) / nulls.std():14.3f} | {(a < lo).sum():15d} | {(a > hi).sum():16d} | {ll[0]:8.1f} ({ll[1]}) | {lh[0]:8.1f} ({lh[1]}) | "
          f"{spearmanr(a[ok], stab[ok])[0]:+.3f} / {spearmanr(a[ok2], age[ok2])[0]:+.3f}")
vol = AGG["volatility (signed)"]
Zv = np.column_stack([nscore(stats["volatility (signed)"][0][g]) for g in GROUPS])
C = pd.DataFrame(Zv).corr().to_numpy()
off = C[np.triu_indices_from(C, 1)]
print(f"  volatility: cross-behaviour correlations of the per-behaviour scores: mean {off.mean():+.4f}, max {off.max():+.3f}, min {off.min():+.3f} "
      f"(noise ~ {1 / np.sqrt(n_u):.4f}); top eigenvalue {np.linalg.eigvalsh(C)[-1]:.3f} (independence bound {(1 + np.sqrt(len(GROUPS) / n_u)) ** 2:.3f})")
print("  volatility aggregate, quantiles real vs N(0, 1/sqrt(24)): " + ", ".join(
    f"{q:g}: {np.nanquantile(vol, q):+.3f} / {norm.ppf(q) / np.sqrt(len(GROUPS)):+.3f}" for q in (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)))
# the same aggregate on ALL accounts, to see where the known families and the pocket sit
Va = {g: df[[f"{g}_{i}" for i in range(1, 5)]].to_numpy(float) for g in GROUPS}
vol_all = np.nanmean(np.column_stack([nscore(np.log(np.nanstd(Va[g], 1) + 1e-3 * np.nanstd(Va[g]))) for g in GROUPS]), 1)
gc_table, gc_masks = task5.gap_clusters(df, X)
fam = gc_masks.any(axis=1).to_numpy()
print(f"  volatility aggregate on all accounts: unflagged mean {vol_all[U].mean():+.3f} sd {vol_all[U].std():.3f}; gap families {vol_all[fam].mean():+.3f}; "
      f"pocket {vol_all[flag & ~fam].mean():+.3f}")

# ---------------------------------------------------------------- 4. the "too typical" accounts
print("\n== 4. accounts closer to the centre than a single Gaussian-copula population allows (part 1: 192 below the null's 0.1% quantile)")
feat = df.loc[U, [c for c in df.columns if c != "player_id" and c not in task5.FLAG_COLS and not c.startswith(("login_lat", "login_lon")) and df[c].nunique() > 50]]
Gs = np.nan_to_num(np.column_stack([nscore(feat[c].to_numpy(float)) for c in feat.columns]), nan=0.0)
Cg = np.corrcoef(Gs, rowvar=False)
w, Vv = np.linalg.eigh(Cg)
d2 = np.einsum("ij,jk,ik->i", Gs, (Vv / np.maximum(w, 1e-6)) @ Vv.T, Gs)
nmiss = feat.isna().sum(1).to_numpy()
low = d2 < np.quantile(d2, 192 / n_u)
print(f"  Mahalanobis d2: rho with the number of missing cells {spearmanr(d2, nmiss)[0]:+.3f}, with the volatility aggregate {spearmanr(d2, vol)[0]:+.3f}, "
      f"with behavior_stability {spearmanr(d2[~np.isnan(stab)], stab[~np.isnan(stab)])[0]:+.3f}")
print(f"  the 192 lowest: missing cells mean {nmiss[low].mean():.1f} (all unflagged {nmiss.mean():.1f}); volatility aggregate mean {vol[low].mean():+.3f} "
      f"(sd of all {np.nanstd(vol):.3f}); behavior_stability mean {np.nanmean(stab[low]):.3f} (all {np.nanmean(stab):.3f})")
ll = lam_tail(d2, "low")
print(f"  lower tail of d2: widest gap lambda {ll[0]:.1f} (size {ll[1]}); d2 quantiles 0.1% {np.quantile(d2, .001):.1f}, 1% {np.quantile(d2, .01):.1f}, 50% {np.median(d2):.1f}")
