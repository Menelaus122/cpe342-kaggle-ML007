"""Round 28, Task 5 stress test: would the scans see a SUBTLE version of the four threats of instruction slide 22?

    python eb_r28_t5_stress.py

Slide 22 names account theft ("smart attackers who wait 24 hours between actions"), account-selling rings, bot farms ("gradual
botting patterns") and VPN regional-pricing exploiters, and warns against flagging travellers and players who simply improve.
Task 5 has no labels and only test.csv, so the question "is anything left among the 23,741 unflagged accounts?" is answered in
two label-free ways (nothing reads the leaderboard):

A  Injection. n unflagged accounts are copied and given a planted pattern of strength s (in units of the behaviour's
   within-account SD over the four periods, so s = 2 is a change an account shows by chance in about 5% of periods), the scans
   are run on the modified data, and each scan's statistic is compared with its level on the untouched data:
     takeover   period 4 only: transfers, purchases, session and login-time behaviour shift, new device, password change, a move
     sold       from period 3 on: skill, K/D, accuracy up, reaction time down, friends churn, guild and device change
     bot        a linear drift over the four periods: timing and IP entropy steadier, logins and activity up (no single jump)
     vpn        VPN flag, location alternates home / away / home / away, purchases up in the away periods
   Scans: 1-D density gaps (raw + engineered + trajectory features), pair pockets, cross-behaviour aggregates (odd period, level,
   volatility, linear drift), Mahalanobis distance in normal-score space, and a ping-pong location statistic.
   Output: for every pattern, size and strength, which scans fire and how many of the planted accounts the firing scan isolates.
B  The real unflagged accounts, two patterns the earlier scans did not target:
     ping-pong travel   location returns (A, B, A) or alternates (A, B, A, B); null by exchangeability of the periods
     account rings      near-duplicate accounts in sub-spaces (location, timing, performance, economy), exact repeated values
                        across accounts, accounts created at the same moment
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

df0 = task5.load()
flag = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
U0 = df0.loc[~flag].reset_index(drop=True)
n_u = len(U0)
DISC = ("device_count", "refund_count", "guild_changes")
GROUPS = [g for g in task5.temporal_groups(df0) if g not in ("login_lat", "login_lon") and g not in DISC]
P = lambda g: [f"{g}_{i}" for i in range(1, 5)]
SW = {g: float(np.nanmedian(np.nanstd(U0[P(g)].to_numpy(float), axis=1, ddof=1))) for g in GROUPS}       # within-account SD
SP = {g: float(np.nanstd(U0[P(g)].to_numpy(float))) for g in GROUPS}
print(f"unflagged accounts {n_u}; within-account SD / population SD per behaviour: median {np.median([SW[g] / SP[g] for g in GROUPS]):.2f} "
      f"(min {min(SW[g] / SP[g] for g in GROUPS):.2f}, max {max(SW[g] / SP[g] for g in GROUPS):.2f})")
LO, HI = U0.drop(columns=["player_id"]).min(), U0.drop(columns=["player_id"]).max()
lat0, lon0 = U0[P("login_lat")].to_numpy(float), U0[P("login_lon")].to_numpy(float)
hav = task5.haversine
hop0 = np.column_stack([hav(lat0[:, i], lon0[:, i], lat0[:, i + 1], lon0[:, i + 1]) for i in range(3)])
ok_geo = (np.abs(lat0) <= 90).all(1) & (np.abs(lon0) <= 180).all(1)
HOP = float(np.nanmedian(hop0[ok_geo]))
print(f"hops between consecutive periods (valid coordinates): median {HOP:.1f} km, 99% {np.nanquantile(hop0[ok_geo], .99):.0f} km, max {np.nanmax(hop0[ok_geo]):.0f} km")


# ---------------------------------------------------------------- planted patterns
def shift(D, rows, g, periods, s, ramp=False):
    for t in periods:
        k = (t - 1) / 3 if ramp else 1.0
        D.loc[rows, f"{g}_{t}"] = D.loc[rows, f"{g}_{t}"] + s * k * SW[g]


def move(D, rows, periods, km, rng):
    ang = rng.uniform(0, 2 * np.pi, len(rows))
    for t in periods:
        D.loc[rows, f"login_lat_{t}"] = D.loc[rows, f"login_lat_{t}"] + km * np.cos(ang) / 111.0
        D.loc[rows, f"login_lon_{t}"] = D.loc[rows, f"login_lon_{t}"] + km * np.sin(ang) / (111.0 * np.cos(np.radians(D.loc[rows, f"login_lat_{t}"].clip(-80, 80))))


def plant(kind, n, s, seed):
    rng = np.random.default_rng(seed)
    D = U0.copy()
    rows = rng.choice(np.where(ok_geo)[0], n, replace=False)
    if kind == "takeover":
        for g, sg in (("currency_transfers", 1), ("purchase_frequency", 1), ("purchase_amount_mean", 1), ("play_pattern_consistency", -1),
                      ("login_time_variance", 1), ("session_duration_mean", -1)):
            shift(D, rows, g, [4], sg * s)
        D.loc[rows, "device_count_4"] = D.loc[rows, "device_count_4"] + 1
        D.loc[rows[rng.random(n) < 0.8], "password_changed"] = 1.0
        move(D, rows, [4], s * HOP, rng)
    elif kind == "sold":
        for g, sg in (("skill_rating", 1), ("kda_mean", 1), ("accuracy_mean", 1), ("reaction_time_mean", -1), ("friend_adds", 1)):
            shift(D, rows, g, [3, 4], sg * s)
        shift(D, rows, "friend_removes", [3], s)
        D.loc[rows, "guild_changes_3"] = D.loc[rows, "guild_changes_3"] + 1
        for t in (3, 4):
            D.loc[rows, f"device_count_{t}"] = D.loc[rows, f"device_count_{t}"] + 1
    elif kind == "bot":
        for g, sg in (("session_duration_std", -1), ("login_time_variance", -1), ("reaction_time_std", -1), ("ip_hash_entropy", -1),
                      ("play_pattern_consistency", 1), ("login_count", 1), ("activity_level", 1), ("communication_frequency", -1)):
            shift(D, rows, g, [1, 2, 3, 4], sg * s, ramp=True)
    elif kind == "vpn":
        D.loc[rows, "vpn_usage"] = 1.0
        move(D, rows, [2, 4], s * HOP, rng)                      # 84-335 km: inside the range of normal hops
        for g in ("purchase_frequency", "purchase_amount_mean"):
            shift(D, rows, g, [2, 4], s)
        shift(D, rows, "location_changes", [2, 3, 4], s)
    for c in D.columns:                                         # stay inside the support of the unflagged accounts
        if c != "player_id" and not c.startswith(("login_lat", "login_lon")):
            D.loc[rows, c] = D.loc[rows, c].clip(LO[c], HI[c])
    planted = np.zeros(n_u, bool)
    planted[rows] = True
    return D, planted


# ---------------------------------------------------------------- scans
def nscore(v, rng):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    r = pd.Series(v[ok] + 1e-9 * rng.standard_normal(ok.sum()) * (np.nanstd(v[ok]) + 1e-12)).rank().to_numpy()
    out[ok] = norm.ppf((r - 0.5) / ok.sum())
    return out


def trajectories(D):
    T = {}
    for g in GROUPS:
        v = D[P(g)].to_numpy(float)
        T[f"{g}_d34"], T[f"{g}_d23"], T[f"{g}_d12"] = v[:, 3] - v[:, 2], v[:, 2] - v[:, 1], v[:, 1] - v[:, 0]
        T[f"{g}_slope"] = (-1.5 * v[:, 0] - 0.5 * v[:, 1] + 0.5 * v[:, 2] + 1.5 * v[:, 3]) / 5
        T[f"{g}_step3"] = (v[:, 2] + v[:, 3]) / 2 - (v[:, 0] + v[:, 1]) / 2
        T[f"{g}_alt"] = (v[:, 1] + v[:, 3]) / 2 - (v[:, 0] + v[:, 2]) / 2
        T[f"{g}_range"] = np.nanmax(v, 1) - np.nanmin(v, 1)
    return pd.DataFrame(T)


CONT = {}


def gap_best(frame, key, tail=0.10, k=200, min_size=15, max_share=0.05):
    best = (0.0, "", None)
    if key not in CONT:                                         # fixed on the first (untouched) call
        CONT[key] = [c for c in frame.columns if np.unique(frame[c].to_numpy(float)[np.isfinite(frame[c].to_numpy(float))]).size >= 50]
    for c in CONT[key]:
        x = frame[c].to_numpy(float)
        v = x[np.isfinite(x)]
        s = np.sort(v)
        n = len(s)
        g = np.diff(s)
        for side in ("low", "high"):
            idx = np.arange(int((1 - tail) * n), n - 1) if side == "high" else np.arange(0, int(tail * n))
            for j in idx[np.argsort(-g[idx])[:3]]:
                size = (n - 1 - j) if side == "high" else (j + 1)
                if not (min_size <= size <= max_share * n):
                    continue
                bulk = s[max(0, j - k):j + 1] if side == "high" else s[j + 1:j + 2 + k]
                lam = (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[j]
                if lam > best[0]:
                    cut = (s[j] + s[j + 1]) / 2
                    best = (lam, f"{c} {side}", (x > cut) if side == "high" else (x < cut))
    return best


def aggregates(D, rng):
    out = {}
    V = {g: D[P(g)].to_numpy(float) for g in GROUPS}
    per = {"odd period 4": {g: V[g][:, 3] - np.nanmean(V[g][:, :3], 1) for g in GROUPS},
           "step at period 3": {g: (V[g][:, 2] + V[g][:, 3]) / 2 - (V[g][:, 0] + V[g][:, 1]) / 2 for g in GROUPS},
           "linear drift": {g: -1.5 * V[g][:, 0] - 0.5 * V[g][:, 1] + 0.5 * V[g][:, 2] + 1.5 * V[g][:, 3] for g in GROUPS},
           "alternation": {g: (V[g][:, 1] + V[g][:, 3]) / 2 - (V[g][:, 0] + V[g][:, 2]) / 2 for g in GROUPS},
           "volatility": {g: np.log(np.nanstd(V[g], 1) + 1e-3 * SP[g]) for g in GROUPS}}
    for name, d in per.items():
        Z = np.column_stack([nscore(d[g], rng) for g in GROUPS])
        out[name] = np.nanmean(Z ** 2, 1)
    return out


def mahal(D, rng):
    cols = [c for g in GROUPS for c in P(g)] + ["account_age_days", "behavior_stability"]
    G = np.nan_to_num(np.column_stack([nscore(D[c].to_numpy(float), rng) for c in cols]), nan=0.0)
    w, Vv = np.linalg.eigh(np.corrcoef(G, rowvar=False))
    return np.einsum("ij,jk,ik->i", G, (Vv / np.maximum(w, 1e-6)) @ Vv.T, G)


def pingpong(D):
    la, lo = D[P("login_lat")].to_numpy(float), D[P("login_lon")].to_numpy(float)
    d = lambda i, j: hav(la[:, i], lo[:, i], la[:, j], lo[:, j])
    consec = (d(0, 1) + d(1, 2) + d(2, 3)) / 3
    back = (d(0, 2) + d(1, 3)) / 2
    return consec - back                                     # large when the account keeps returning (A, B, A, B)


def run_scans(D, rng, with_pockets=True):
    Xf, _ = task5.build_features(D)
    raw = D[[c for c in D.columns if c not in ("player_id", *task5.FLAG_COLS) and not c.startswith(("login_lat", "login_lon"))]]
    res = {}
    res["gap"] = max((gap_best(raw, "raw"), gap_best(Xf, "eng"), gap_best(trajectories(D), "traj")), key=lambda r: r[0])
    if with_pockets:
        tab = task5._max_cell_z(Xf[CONT["eng"]].to_numpy(float), CONT["eng"], 20)
        res["pocket"] = (float(tab.z.iloc[0]), f"{tab.a.iloc[0]} x {tab.b.iloc[0]}")
    res["agg"] = aggregates(D, rng)
    res["mahal"] = mahal(D, rng)
    res["pp"] = pingpong(D)
    return res


rng = np.random.default_rng(0)
clean = run_scans(U0, rng)
THR = {k: np.quantile(v, 0.999) for k, v in clean["agg"].items()}
THR["mahal"] = np.quantile(clean["mahal"], 0.999)
THR["pp"] = np.nanquantile(clean["pp"], 0.999)
print(f"\nuntouched data: largest gap lambda {clean['gap'][0]:.1f} ({clean['gap'][1]}); largest pocket z {clean['pocket'][0]:.1f} ({clean['pocket'][1]}); "
      f"every tail count below is 24 by construction")
print("a scan FIRES when: gap lambda >= 50 | pocket z >= 10.9 (the pipeline's noise level) | a tail count >= 45 (24 expected, SD 4.9)")
print("\n== A. injection: pattern, n, strength -> gap lambda [planted found / group size] | pocket z | accounts above the untouched 99.9% level, with the planted among them in ()")
print(f"{'pattern':9s} {'n':>4s} {'s':>3s} | {'gap':>18s} | {'pocket':>6s} | {'odd p4':>10s} {'step p3':>10s} {'drift':>10s} {'altern.':>10s} {'volat.':>10s} {'Mahal.':>10s} {'ping-pong':>10s} | fires")
summary = {}
for kind in ("takeover", "sold", "bot", "vpn"):
    for n in (30, 100, 200):
        for s in (1, 2, 3, 4):
            D, planted = plant(kind, n, s, seed=1000 * n + s)
            r = run_scans(D, np.random.default_rng(1), with_pockets=(n == 200))
            lam, name, grp = r["gap"]
            fires = []
            cells = []
            for k in ("odd period 4", "step at period 3", "linear drift", "alternation", "volatility"):
                above = r["agg"][k] > THR[k]
                cells.append(f"{above.sum():4d} ({(above & planted).sum():3d})")
                if above.sum() >= 45:
                    fires.append(k)
            for k in ("mahal", "pp"):
                above = np.nan_to_num(r[k], nan=-np.inf) > THR[k]
                cells.append(f"{above.sum():4d} ({(above & planted).sum():3d})")
                if above.sum() >= 45:
                    fires.append("Mahalanobis" if k == "mahal" else "ping-pong")
            if lam >= 50:
                fires.insert(0, "gap")
            pz = f"{r['pocket'][0]:6.1f}" if "pocket" in r else "     -"
            if "pocket" in r and r["pocket"][0] >= 10.9:
                fires.append("pocket")
            gtxt = f"{lam:6.1f} [{(grp & planted).sum() if grp is not None else 0:3d}/{grp.sum() if grp is not None else 0:4d}]"
            summary[(kind, n, s)] = fires
            print(f"{kind:9s} {n:4d} {s:3d} | {gtxt:>18s} | {pz} | " + " ".join(f"{c:>10s}" for c in cells) + " | " + (", ".join(fires) if fires else "nothing"), flush=True)
print("\nsmallest strength at which some scan fires:")
for kind in ("takeover", "sold", "bot", "vpn"):
    print(f"  {kind:9s} " + "; ".join(f"n = {n}: " + (f"s = {min(s for s in (1, 2, 3, 4) if summary[(kind, n, s)])}" if any(summary[(kind, n, s)] for s in (1, 2, 3, 4)) else "not seen up to s = 4")
                                  for n in (30, 100, 200)))

# ---------------------------------------------------------------- B. real data
print("\n== B. the real unflagged accounts")
la, lo = lat0[ok_geo], lon0[ok_geo]
n_g = len(la)
d = lambda i, j: hav(la[:, i], lo[:, i], la[:, j], lo[:, j])
D6 = {(i, j): d(i, j) for i in range(4) for j in range(i + 1, 4)}
# B1. ping-pong. Under exchangeable periods the three ways to split the six distances into "consecutive" and "return" pairs give
# identically distributed statistics; only the true order (1, 2, 3, 4) is a ping-pong statistic.
pairings = {"true order 1-2-3-4 (return pairs 1-3, 2-4)": ((0, 2), (1, 3)),
            "order 1-3-2-4 (return pairs 1-2, 3-4)": ((0, 1), (2, 3)),
            "order 1-2-4-3 (return pairs 1-4, 2-3)": ((0, 3), (1, 2))}
allsum = sum(D6.values())
stat = {}
for name, back in pairings.items():
    b = D6[back[0]] + D6[back[1]]
    stat[name] = (allsum - b) / 4 - b / 2                      # mean of the four non-return distances minus mean return distance
thr = np.quantile(np.concatenate([stat[k] for k in list(pairings)[1:]]), [0.99, 0.999])
print(f"  ping-pong travel (A, B, A, B) among {n_g} accounts with valid coordinates: accounts above the 99% / 99.9% level of the two re-ordered versions "
      f"(expected {n_g / 100:.0f} / {n_g / 1000:.0f})")
for name in pairings:
    print(f"    {name:45s} {(stat[name] > thr[0]).sum():4d} / {(stat[name] > thr[1]).sum():3d}   max {stat[name].max():+.0f} km, median {np.median(stat[name]):+.1f} km")
tri = np.column_stack([D6[(0, 1)], D6[(1, 2)], D6[(0, 2)]])
top = tri.max(1) > np.quantile(tri.max(1), 0.9)                # symmetric conditioning: the largest of the three distances
print(f"    (A, B, A) over periods 1-3: the return distance 1-3 is the smallest of the three in {np.mean(tri.argmin(1) == 2):.4f} of accounts (1/3 expected), "
      f"{np.mean(tri[top].argmin(1) == 2):.4f} among the 10% with the largest triangle (binomial SD {np.sqrt(2 / 9 / top.sum()):.4f})")

# B2. accounts that travel together: all four period locations within r km of another account's. Null: the other account's periods
# rotated by one (same homes, same period noise, co-movement destroyed).
from scipy.spatial import cKDTree

km = np.column_stack([la * 111.0, lo * 111.0 * np.cos(np.radians(np.nanmean(la, 1, keepdims=True)))])       # 8 numbers per account, km
tree = cKDTree(km)
rot = np.column_stack([np.roll(km[:, :4], 1, axis=1), np.roll(km[:, 4:], 1, axis=1)])
print("  accounts moving together (every period within r km, Chebyshev over the 8 coordinates): pairs real / pairs after rotating the partner's periods")
for r in (2, 5, 10, 20, 40):
    real_pairs = len(tree.query_pairs(r, p=np.inf))
    cnt = tree.query_ball_point(rot, r, p=np.inf, return_length=True)
    self_hit = (np.abs(rot - km).max(1) <= r).sum()
    print(f"    r = {r:2d} km: {real_pairs:6d} / {(cnt.sum() - self_hit) / 2:8.1f}")
home = np.column_stack([np.nanmean(la, 1), np.nanmean(lo, 1)])
cells = pd.Series(list(zip(np.round(home[:, 0], 0), np.round(home[:, 1], 0)))).value_counts()
print(f"    homes (mean position) fall in {len(cells)} one-degree cells; the 5 fullest hold {cells.iloc[:5].sum() / n_g:.1%} of the accounts; "
      f"latitude {np.quantile(home[:, 0], .01):.1f} to {np.quantile(home[:, 0], .99):.1f}, longitude {np.quantile(home[:, 1], .01):.1f} to {np.quantile(home[:, 1], .99):.1f}")

# B3. rings by co-movement: two accounts run by one operator change in the same periods. Deviation of every behaviour from the
# account's own mean, in within-account SDs: 24 behaviours x 4 periods. Nearest neighbour by correlation of these 96 numbers.
# Null: the behaviours' blocks are shuffled between accounts independently (every account keeps realistic blocks, alignment destroyed).
dev = []
for g in GROUPS:
    v = U0[P(g)].to_numpy(float)
    dev.append(np.nan_to_num((v - np.nanmean(v, 1, keepdims=True)) / SW[g], nan=0.0))


def top_similarity(blocks):
    R = np.hstack(blocks)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    best = np.zeros(len(R))
    for i in range(0, len(R), 2000):
        S = R[i:i + 2000] @ R.T
        S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -1
        best[i:i + 2000] = S.max(1)
    return best


real = top_similarity(dev)
rs = np.random.default_rng(9)
nulls = np.concatenate([top_similarity([b[rs.permutation(len(b))] for b in dev]) for _ in range(2)])
print("  co-movement rings: correlation of the period-deviation profile (96 numbers) with the most similar other account, real vs block-shuffled null")
print("    quantiles 50% / 99% / 99.9% / max: real " + " / ".join(f"{np.quantile(real, q):.3f}" for q in (0.5, 0.99, 0.999, 1.0))
      + "; null " + " / ".join(f"{np.quantile(nulls, q):.3f}" for q in (0.5, 0.99, 0.999, 1.0))
      + f"; accounts above the null's 99.9% level: {(real > np.quantile(nulls, 0.999)).sum()} (expected {n_u / 1000:.0f}); above the null's maximum: {(real > nulls.max()).sum()}")
# power of this test: plant rings of 5 accounts sharing one deviation profile plus individual noise of the same size
for n_rings in (5, 20):
    blocks = [b.copy() for b in dev]
    rp = np.random.default_rng(3)
    members = rp.choice(n_u, n_rings * 5, replace=False).reshape(n_rings, 5)
    for ring in members:
        for b in blocks:
            b[ring] = b[ring[0]] * 0.7 + b[ring] * 0.7           # shared profile + own noise (correlation about 0.5 between members)
    t = top_similarity(blocks)
    print(f"    power check, {n_rings} planted rings of 5 (pairwise profile correlation about 0.5): accounts above the null's maximum {(t > nulls.max()).sum()}, "
          f"of which planted {(t[members.ravel()] > nulls.max()).sum()} of {members.size}")

# B4. exact repeats across accounts
print("  exact repeated values across accounts (columns with > 1,000 distinct values): accounts sharing a value with another account")
rep = []
for c in [c for c in U0.columns if c != "player_id" and U0[c].nunique() > 1000]:
    v = U0[c].dropna()
    rep.append((c, int(v.duplicated(keep=False).sum()), int((v == 0).sum())))
rep = pd.DataFrame(rep, columns=["column", "sharing", "of_which_zero"])
rep["sharing_nonzero"] = rep.sharing - rep.of_which_zero
print(f"    columns with any shared non-zero value: {(rep.sharing_nonzero > 0).sum()} of {len(rep)}; largest: "
      + ", ".join(f"{r.column} {r.sharing_nonzero}" for r in rep.sort_values("sharing_nonzero", ascending=False).head(4).itertuples()))
key = pd.Series([" ".join(f"{x:.3f}" for x in row) for row in np.column_stack([la, lo])])
print(f"    accounts whose four (lat, lon) pairs equal another account's to 3 decimals: {int(key.duplicated(keep=False).sum())}")
age = U0["account_age_days"].dropna()
print(f"    account_age_days: {int(age.duplicated(keep=False).sum())} accounts share an exact value (largest group {age.value_counts().iloc[0]}); {age.nunique()} distinct of {len(age)}")
