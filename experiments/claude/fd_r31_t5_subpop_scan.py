"""Round 31, Task 5 item 1: scans inside yes/no-flag subpopulations and 3-way tail cubes (label-free; Task 5 has only test.csv).

    python fd_r31_t5_subpop_scan.py

All scans run on U = the 23,741 accounts the pipeline does NOT flag (2,148 flags = six gap families + the dependence pocket).
A group planted only among, say, VPN users is diluted in a scan of all of U; scanning inside the subpopulation removes the dilution.
Subpopulations of U: vpn_usage = 1, password_changed = 1, mass_item_sale = 1, suspicious_login_time = 1, a new device in period 4
(device_count_4 above the account's periods 1-3), more guild changes in period 4 than in any of periods 1-3.
Per subpopulation S (round 23: outside the families the flags are independent of the features, so under "nothing planted" S is a
random subset of U, and the null is a random subset of U of the same size):
  B1  1-D density gaps inside S (the pipeline's gap statistic: lambda = accounts expected inside the gap if the bulk continued),
      every continuous raw and engineered column, both 10% tails; null = 20 size-matched random subsets of U (per column and the
      maximum over columns);
  B2  2-D dependence pockets inside S against S's own column-shuffle null (task5.dependence_pockets; 20 x 20 grid, 10 x 10 below
      4,000 accounts; 3 shuffles);
  B3  S against the rest of U on a 10 x 10 quantile grid of every pair (cells over-full in S relative to U \\ S), null = 10
      size-matched random subsets.
C   3-way tail cubes on U: for features from three different behaviour categories (skill / combat, social / guild, economy / geo,
    login / session), level and period-4 deviation of each behaviour, all 8 corners of top / bottom deciles. z against independence
    (expected n p_a p_b p_c, about n / 1000) and against the Kirkwood superposition E = n p_ab p_ac p_bc / (p_a p_b p_c), which keeps every pairwise
    dependence, so only a genuine 3-way group stands out. Noise: the same maxima on column-shuffled data (independence z) and on a
    Gaussian copula with U's rank correlations (Kirkwood z).
A finding needs to exceed its noise level and to point at a group of at least ~20 accounts (F3 break-even: ~10% anomalies).
"""
import itertools
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm  # noqa: E402

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
flag = gap | pocket
u = ~flag
log(f"flags {flag.sum()} (gap {gap.sum()}, pocket {pocket.sum()}); unflagged U {u.sum()}")

dc = df[[f"device_count_{i}" for i in range(1, 5)]].values
SUB = {f: (df[f] == 1).values for f in task5.FLAG_COLS}
SUB["new device in period 4"] = dc[:, 3] > np.nanmax(dc[:, :3], axis=1)
gc = df[[f"guild_changes_{i}" for i in range(1, 5)]].values
SUB["guild change in period 4"] = gc[:, 3] > np.nanmax(gc[:, :3], axis=1)        # above the account's periods 1-3 (79% have > 0)
SUB = {k: v & u for k, v in SUB.items()}
log("subpopulations of U: " + ", ".join(f"{k} {v.sum()} ({v.sum() / u.sum():.1%})" for k, v in SUB.items()))

# columns
raw = df.drop(columns=["player_id", *task5.FLAG_COLS])
eng = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 50]]
C1 = pd.concat([raw[[c for c in raw.columns if raw[c].nunique() > 50]], eng], axis=1)
C1 = C1.loc[:, ~C1.columns.duplicated()]
V1 = C1.to_numpy(float)
cols2 = list(eng.columns)
log(f"B1 columns {V1.shape[1]}, B2/B3 columns {len(cols2)}")


def best_gaps(V, tail=0.10, n_gaps=3, min_size=10, max_share=0.3):
    """per column: largest lambda over both tails (pipeline statistic, bulk window scaled with n)."""
    out = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        s = np.sort(V[:, j][~np.isnan(V[:, j])])
        n = len(s)
        if n < 100:
            continue
        k = max(30, int(round(0.0077 * n)))
        g = np.diff(s)
        best = 0.0
        for side in ("low", "high"):
            idx = np.arange(int((1 - tail) * n), n - 1) if side == "high" else np.arange(0, int(tail * n))
            for i in idx[np.argsort(-g[idx])[:n_gaps]]:
                size = (n - 1 - i) if side == "high" else (i + 1)
                bulk = s[max(0, i - k):i + 1] if side == "high" else s[i + 1:i + 2 + k]
                grp = s[i + 1:] if side == "high" else s[:i + 1]
                if not (min_size <= size <= max_share * n) or np.unique(grp).size < 10:
                    continue
                best = max(best, (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[i])
        out[j] = best
    return out


Uidx = np.where(u)[0]
G10 = task5._quantile_codes(eng.loc[u].to_numpy(float), 10)          # deciles among U
pairs = [(i, j) for i in range(len(cols2)) for j in range(i + 1, len(cols2))
         if task5._base_feature(cols2[i]) != task5._base_feature(cols2[j])
         and frozenset((task5._base_feature(cols2[i]), task5._base_feature(cols2[j]))) not in task5.MECHANICAL_PAIRS]


def contrast(inS):
    """max over pairs of the most over-full 10 x 10 cell of S relative to U \\ S: (z, i, j, cell, obs, exp)."""
    best = (-np.inf,)
    nS, nR = inS.sum(), (~inS).sum()
    for i, j in pairs:
        ok = (G10[:, i] >= 0) & (G10[:, j] >= 0)
        code = G10[:, i] * 10 + G10[:, j]
        cs = np.bincount(code[ok & inS], minlength=100)
        cr = np.bincount(code[ok & ~inS], minlength=100)
        e = cr * (ok & inS).sum() / max((ok & ~inS).sum(), 1)
        z = (cs - e) / np.sqrt(np.maximum(e, 1))
        c = int(np.argmax(z))
        if z[c] > best[0]:
            best = (float(z[c]), cols2[i], cols2[j], (c // 10, c % 10), int(cs[c]), float(e[c]))
    return best


for name, S in SUB.items():
    nS = int(S.sum())
    log(f"\n=== {name}: {nS} accounts")
    if nS < 300:
        log("  fewer than 300 accounts: skipped")
        continue
    lamS = best_gaps(V1[S])
    lam0 = np.array([best_gaps(V1[rng.choice(Uidx, nS, replace=False)]) for _ in range(20)])
    top = np.argsort(-lamS)[:5]
    log(f"  B1 gaps: largest lambda in S " + ", ".join(f"{C1.columns[j]} {lamS[j]:.1f} (null max for that column {lam0[:, j].max():.1f})" for j in top)
        + f" | null: max over columns per draw, median {np.median(lam0.max(1)):.1f}, max {lam0.max():.1f}; columns where S beats its own "
          f"column's null maximum: {int(np.sum(lamS > lam0.max(0)))} of {len(lamS)}")
    G = 20 if nS >= 4000 else 10
    scan, noise = task5.dependence_pockets(X, S, G=G, n_null=3)
    log(f"  B2 pockets inside S (G={G}): noise {noise:.2f}; pairs above it {int((scan.z > noise).sum())}; top: "
        + "; ".join(f"{r.a} x {r.b} z {r.z:.2f} cell ({r.bin_a},{r.bin_b}) obs {r.observed:.0f} exp {r.expected:.1f}" for r in scan.head(3).itertuples()))
    inS = S[Uidx]
    real = contrast(inS)
    null = [contrast(rng.permutation(inS))[0] for _ in range(10)]
    log(f"  B3 S vs rest of U: max z {real[0]:.2f} ({real[1]} x {real[2]}, cell {real[3]}, obs {real[4]} vs exp {real[5]:.1f}); "
        f"null (size-matched random subsets) max z median {np.median(null):.2f}, max {np.max(null):.2f}")

# ------------------------------------------------------------------ C: 3-way tail cubes on U
CAT = {"skill/combat": ["skill_rating", "skill_trajectory", "kda_mean", "kda_std", "accuracy_mean", "accuracy_std", "reaction_time_mean",
                        "reaction_time_std", "performance_stability"],
       "social/guild": ["friend_adds", "friend_removes", "communication_frequency", "guild_changes"],
       "economy/geo": ["purchase_frequency", "purchase_amount_mean", "purchase_amount_std", "refund_count", "currency_transfers",
                       "location_changes"],
       "login/session": ["login_count", "login_time_variance", "session_duration_mean", "session_duration_std", "activity_level",
                         "play_pattern_consistency", "ip_hash_entropy", "device_count"]}
feat, cat = [], []
for c, gs in CAT.items():
    for g in gs:
        for suf in ("level", "dev4"):
            if f"{g}_{suf}" in X:
                feat.append(f"{g}_{suf}")
                cat.append(c)
for g in ("geo_hop_max", "geo_p4_from_home_km"):
    feat.append(g)
    cat.append("economy/geo")
M = X.loc[u, feat].to_numpy(float)
n = len(M)
log(f"\nC: {len(feat)} features in {len(CAT)} categories, {n} accounts")


def tails(A):
    R = pd.DataFrame(A).rank(pct=True).to_numpy()
    return [np.packbits(np.nan_to_num(R, nan=0.5) <= 0.1, axis=0), np.packbits(np.nan_to_num(R, nan=0.5) > 0.9, axis=0)]


POP = np.array([bin(i).count("1") for i in range(256)], np.int32)
cnt = lambda *bits: int(POP[np.bitwise_and.reduce(bits)].sum())
triples = [t for t in itertools.combinations(range(len(feat)), 3) if len({cat[t[0]], cat[t[1]], cat[t[2]]}) == 3]
log(f"C: {len(triples)} cross-category triples x 8 corners")


def cube_scan(A, top=10):
    T = tails(A)                                        # T[side] : packed bits (n/8, features)
    pair = {}
    res = []
    single = [[POP[T[sd][:, j]].sum() / n for j in range(T[0].shape[1])] for sd in (0, 1)]
    for a, b, c in triples:
        for sa, sb, sc in itertools.product((0, 1), repeat=3):
            pa, pb, pc = single[sa][a], single[sb][b], single[sc][c]
            if min(pa, pb, pc) < 0.02:                  # a tail swallowed by ties (count columns)
                continue
            xa, xb, xc = T[sa][:, a], T[sb][:, b], T[sc][:, c]
            o = cnt(xa, xb, xc)
            pab = pair.setdefault((a, sa, b, sb), cnt(xa, xb) / n)
            pac = pair.setdefault((a, sa, c, sc), cnt(xa, xc) / n)
            pbc = pair.setdefault((b, sb, c, sc), cnt(xb, xc) / n)
            e_ind = n * pa * pb * pc
            e_k = n * pab * pac * pbc / (pa * pb * pc)
            res.append((o, e_ind, e_k, a, b, c, sa, sb, sc))
    R = np.array([r[:3] for r in res])
    z_ind = (R[:, 0] - R[:, 1]) / np.sqrt(R[:, 1])
    z_k = (R[:, 0] - R[:, 2]) / np.sqrt(np.maximum(R[:, 2], 1))
    return z_ind, z_k, res


z_ind, z_k, res = cube_scan(M)
log("C real data done")
Ms = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
zs_ind, _, _ = cube_scan(Ms)
Rk = pd.DataFrame(M).rank(pct=True).to_numpy()
Zn = norm.ppf(np.clip(np.nan_to_num(Rk, nan=0.5), 0.5 / n, 1 - 0.5 / n))
Cc = np.corrcoef(Zn, rowvar=False)
Mg = rng.multivariate_normal(np.zeros(len(feat)), Cc, size=n, method="eigh")
_, zg_k, _ = cube_scan(Mg)
log(f"C noise: independence z max on shuffled data {zs_ind.max():.2f}; Kirkwood z max on the Gaussian copula {zg_k.max():.2f}")
side = ["low", "high"]
log(f"C real: independence z max {z_ind.max():.2f} (cells above the shuffled max: {int(np.sum(z_ind > zs_ind.max()))}); Kirkwood z max "
    f"{z_k.max():.2f} (cells above the copula max: {int(np.sum(z_k > zg_k.max()))})")
for i in np.argsort(-z_k)[:10]:
    o, ei, ek, a, b, c, sa, sb, sc = res[i]
    log(f"  Kirkwood z {z_k[i]:.2f} (indep. z {z_ind[i]:.1f}): {feat[a]} {side[sa]} x {feat[b]} {side[sb]} x {feat[c]} {side[sc]}: "
        f"obs {o}, expected {ek:.1f} (independence {ei:.1f}), excess {o - ek:.0f}")
log("done")
