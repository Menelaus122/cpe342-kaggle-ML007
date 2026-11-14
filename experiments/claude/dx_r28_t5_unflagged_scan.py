"""Round 28, Task 5: a fresh label-free scan of the accounts v19 does NOT flag (re-released test.csv).

    python dx_r28_t5_unflagged_scan.py

Question: is there an anomaly family in the re-released Task 5 data that the six gap families + the pocket (2,148 flags) miss?
Task 5 has no training set and no labels; like the pipeline, everything here is fitted on the unlabeled test features. Nothing
reads the leaderboard, and no upload variants are made.

Everything is computed inside U = the unflagged accounts, so the known families cannot mask a smaller group:
  1  gap scan (1-D density gaps, lambda = accounts expected inside the gap if the bulk continued) on raw columns, the pipeline's
     engineered features, and new trajectory features (period differences, ratios, curvature, slope, min / max, zero counts)
  2  pair pockets (20x20 quantile grid, noise level from column shuffles) on the engineered and the trajectory features
  3  exact repeats across periods and drops to zero, against a shuffle null
  4  discrete columns (4 yes/no flags, device_count, refund_count, guild_changes per period, missing-cell count): every pair and
     triple of values, observed vs expected under independence, noise level from shuffles
  5  multivariate detectors in normal-score space (Mahalanobis, k-NN distance in both tails, Isolation Forest, LOF, autoencoder,
     top principal components) against a Gaussian-copula null with the same marginals and correlations (a single population);
     power check: the same comparison after adding the 339 pocket accounts back
A finding counts only if it is above its own noise level; group sizes are reported so the F3 arithmetic can be done.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

df = task5.load()
X, _ = task5.build_features(df)
X = X.copy()
flag = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
U = ~flag
n_u = int(U.sum())
print(f"accounts {len(df)}; flagged {flag.sum()} ({flag.mean():.2%}); unflagged {n_u}")
GROUPS = [g for g in task5.temporal_groups(df) if g not in ("login_lat", "login_lon")]
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- trajectory features
T = {}
for g in GROUPS:
    v = df[[f"{g}_{i}" for i in range(1, 5)]].to_numpy(float)
    sd = np.nanstd(v)
    T[f"{g}_d12"], T[f"{g}_d23"], T[f"{g}_d34"] = v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 3] - v[:, 2]
    T[f"{g}_curv2"], T[f"{g}_curv3"] = v[:, 0] - 2 * v[:, 1] + v[:, 2], v[:, 1] - 2 * v[:, 2] + v[:, 3]
    T[f"{g}_slope"] = (-1.5 * v[:, 0] - 0.5 * v[:, 1] + 0.5 * v[:, 2] + 1.5 * v[:, 3]) / 5
    T[f"{g}_min"], T[f"{g}_max"] = np.nanmin(v, 1), np.nanmax(v, 1)
    T[f"{g}_range"] = T[f"{g}_max"] - T[f"{g}_min"]
    T[f"{g}_ratio4"] = v[:, 3] / (np.abs(np.nanmean(v[:, :3], 1)) + 0.1 * sd)
    T[f"{g}_ratio1"] = v[:, 0] / (np.abs(np.nanmean(v[:, 1:], 1)) + 0.1 * sd)
    T[f"{g}_p2p3_vs_p1p4"] = (v[:, 1] + v[:, 2]) / 2 - (v[:, 0] + v[:, 3]) / 2
    T[f"{g}_absdiff_sum"] = np.abs(np.diff(v, axis=1)).sum(1)
T = pd.DataFrame(T, index=df.index)


# ---------------------------------------------------------------- 1. gap scan inside U
def best_gaps(frame, tail=0.10, k=200, min_size=15, max_share=0.05, min_distinct=10, n_gaps=5):
    rows = []
    for c in frame.columns:
        v = frame[c].to_numpy(float)
        v = v[~np.isnan(v) & np.isfinite(v)]
        if np.unique(v).size < 50:
            continue
        s = np.sort(v)
        n = len(s)
        g = np.diff(s)
        for side in ("low", "high"):
            idx = np.arange(int((1 - tail) * n), n - 1) if side == "high" else np.arange(0, int(tail * n))
            best = None
            for j in idx[np.argsort(-g[idx])[:n_gaps]]:
                size = (n - 1 - j) if side == "high" else (j + 1)
                bulk = s[max(0, j - k):j + 1] if side == "high" else s[j + 1:j + 2 + k]
                lam = (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[j]
                group = s[j + 1:] if side == "high" else s[:j + 1]
                if min_size <= size <= max_share * n and np.unique(group).size >= min_distinct and (best is None or lam > best[0]):
                    best = (lam, size, (s[j] + s[j + 1]) / 2)
            if best:
                rows.append((c, side, best[0], best[1], best[2]))
    return pd.DataFrame(rows, columns=["column", "side", "lambda", "size", "cut"]).sort_values("lambda", ascending=False, ignore_index=True)


print("\n== 1. gap scan inside U (pipeline accepts lambda >= 50; under a smooth tail the largest of ~1e6 gaps is about 14)")
raw_cols = [c for c in df.columns if c not in ("player_id", *task5.FLAG_COLS) and not c.startswith(("login_lat", "login_lon"))]
for name, frame in (("raw columns", df.loc[U, raw_cols]), ("engineered features", X.loc[U]), ("trajectory features", T.loc[U])):
    tab = best_gaps(frame)
    print(f"  {name}: {len(tab)} column tails tested; lambda >= 50: {(tab['lambda'] >= 50).sum()}; >= 25: {(tab['lambda'] >= 25).sum()}; top 6:")
    for r in tab.head(6).itertuples():
        print(f"     {r.column:38s} {r.side:4s} lambda {r._3:7.1f} size {r.size:5d} cut {r.cut:.4g}")
# reference: the same scan on ALL accounts recovers the known families
tab = best_gaps(df[raw_cols])
print(f"  reference, raw columns of ALL accounts: lambda >= 50 in {(tab['lambda'] >= 50).sum()} tails (largest {tab['lambda'].max():.0f})")

# ---------------------------------------------------------------- 2. pair pockets inside U
print("\n== 2. pair pockets inside U (20x20 quantile cells; z of the most over-full cell vs the largest z after column shuffles)")
scan, noise = task5.dependence_pockets(X, U)
print(f"  engineered features: {len(scan)} pairs; noise level {noise:.1f}; pairs above it: {(scan.z > noise).sum()}; top 5:")
for r in scan.head(5).itertuples():
    print(f"     z {r.z:5.1f}  {r.a} x {r.b}  cell ({r.bin_a}, {r.bin_b})  observed {r.observed:.0f} expected {r.expected:.1f}")
Tc = T[[c for c in T.columns if c.endswith(("_d34", "_slope", "_curv3", "_range", "_ratio4", "_p2p3_vs_p1p4"))]]
scan_t, noise_t = task5.dependence_pockets(Tc, U)
print(f"  trajectory features (d34, slope, curvature, range, ratio4, mid-vs-ends): {len(scan_t)} pairs; noise level {noise_t:.1f}; "
      f"pairs above it: {(scan_t.z > noise_t).sum()}; top 5:")
for r in scan_t.head(5).itertuples():
    print(f"     z {r.z:5.1f}  {r.a} x {r.b}  cell ({r.bin_a}, {r.bin_b})  observed {r.observed:.0f} expected {r.expected:.1f}")

# ---------------------------------------------------------------- 3. exact repeats and zeros across periods
print("\n== 3. exact repeats across periods and zero periods, inside U (null: each period column shuffled among U)")


def repeat_stats(V):
    eq = np.zeros(len(V), int)
    for a, b in itertools.combinations(range(4), 2):
        eq += (V[:, a] == V[:, b]) & ~np.isnan(V[:, a])
    return eq


rep_total, rep_null = np.zeros(n_u, int), np.zeros(n_u, int)
zero_total, drop4 = np.zeros(n_u, int), np.zeros(n_u, int)
rows = []
for g in GROUPS:
    V = df.loc[U, [f"{g}_{i}" for i in range(1, 5)]].to_numpy(float)
    if np.unique(V[~np.isnan(V)]).size < 50:
        continue                                                # small-integer counts repeat by construction: section 4
    eq = repeat_stats(V)
    Vn = np.column_stack([rng.permutation(V[:, j]) for j in range(4)])
    eqn = repeat_stats(Vn)
    rep_total += eq
    rep_null += eqn
    z = (V == 0)
    zero_total += z.sum(1)
    d4 = z[:, 3] & (np.nan_to_num(V[:, :3], nan=0).min(1) > 0)
    drop4 += d4
    rows.append((g, (eq > 0).sum(), (eqn > 0).sum(), z.any(1).sum(), d4.sum(), (z.sum(1) >= 2).sum()))
tab = pd.DataFrame(rows, columns=["group", "accounts_with_repeat", "same_shuffled", "accounts_with_zero", "zero_only_in_p4", "zero_in_2plus"])
print(tab.sort_values("accounts_with_repeat", ascending=False).head(8).to_string(index=False))
print(f"  accounts by number of repeated pairs over all continuous groups: real {np.bincount(rep_total)[:8].tolist()}; shuffled {np.bincount(rep_null)[:8].tolist()}")
print(f"  accounts by number of zero cells: {np.bincount(zero_total)[:8].tolist()}; accounts whose only zero is period 4, in >= 2 groups: {(drop4 >= 2).sum()}")

# ---------------------------------------------------------------- 4. discrete columns
print("\n== 4. discrete columns inside U: value combinations vs independence (z = (obs - exp) / sqrt(exp), cells with exp >= 5)")
disc = {c: df.loc[U, c].to_numpy(float) for c in task5.FLAG_COLS}
for g in ("device_count", "refund_count", "guild_changes"):
    for i in range(1, 5):
        disc[f"{g}_{i}"] = df.loc[U, f"{g}_{i}"].to_numpy(float)
nm = df.loc[U].isna().sum(1).to_numpy()
disc["n_missing_ge10"] = (nm >= 10).astype(float)
D = pd.DataFrame(disc)
print("  value shares, unflagged vs flagged (columns where a value's share differs by > 3x and it has >= 30 flagged accounts are marked):")
for c in D.columns[:-1]:
    a, b = df.loc[U, c].value_counts(normalize=True), df.loc[flag, c].value_counts(normalize=True)
    for val in sorted(set(a.index) | set(b.index)):
        pa, pb = a.get(val, 0), b.get(val, 0)
        if pb * flag.sum() >= 30 and (pb > 3 * pa or pa > 3 * pb):
            print(f"     {c} = {val:g}: unflagged {pa:.4f}, flagged {pb:.4f}")
print(f"  negative counts (-1) inside U: " + ", ".join(f"{c} {int((D[c] == -1).sum())}" for c in D.columns if (D[c] == -1).any()))
codes = {c: pd.factorize(D[c], sort=True)[0] for c in D.columns}                 # NaN -> -1


def combo_scan(cd, order):
    out = []
    cols = list(cd)
    for combo in itertools.combinations(cols, order):
        ok = np.all([cd[c] >= 0 for c in combo], 0)
        if ok.sum() == 0:
            continue
        sizes = [cd[c].max() + 1 for c in combo]
        key = np.zeros(ok.sum(), int)
        for c, s in zip(combo, sizes):
            key = key * s + cd[c][ok]
        cnt = np.bincount(key, minlength=int(np.prod(sizes))).reshape(sizes).astype(float)
        exp = np.ones(sizes) * ok.sum()
        for ax, (c, s) in enumerate(zip(combo, sizes)):
            p = np.bincount(cd[c][ok], minlength=s) / ok.sum()
            exp = exp * p.reshape([-1 if a == ax else 1 for a in range(order)])
        z = np.where(exp >= 5, (cnt - exp) / np.sqrt(np.maximum(exp, 1e-9)), 0)
        k = int(np.argmax(z))
        out.append((z.flat[k], combo, np.unravel_index(k, sizes), cnt.flat[k], exp.flat[k]))
    return sorted(out, key=lambda r: -r[0])


for order in (2, 3):
    real = combo_scan(codes, order)
    nz = max(combo_scan({c: rng.permutation(v) for c, v in codes.items()}, order)[0][0] for _ in range(5))
    above = [r for r in real if r[0] > nz]
    print(f"  {order}-way: {len(real)} combinations; noise level z {nz:.1f}; above it: {len(above)}; top 5:")
    for z, combo, cell, obs, exp in real[:5]:
        vals = [sorted(D[c].dropna().unique())[i] for c, i in zip(combo, cell)]
        print(f"     z {z:5.1f}  " + ", ".join(f"{c} = {v:g}" for c, v in zip(combo, vals)) + f"  observed {obs:.0f} expected {exp:.1f}")
p_miss = df.loc[U].drop(columns=["player_id"]).isna().to_numpy().mean()
ncol = df.shape[1] - 1
print(f"  missing cells per account: mean {nm.mean():.2f} of {ncol} (binomial sd {np.sqrt(ncol * p_miss * (1 - p_miss)):.2f}, observed sd {nm.std():.2f}); "
      f"max {nm.max()}; accounts with >= 15 missing: {(nm >= 15).sum()} (binomial expects {n_u * (1 - __import__('scipy.stats').stats.binom.cdf(14, ncol, p_miss)):.1f})")

# ---------------------------------------------------------------- 5. multivariate detectors vs a Gaussian-copula null
print("\n== 5. multivariate structure inside U, normal-score space, against a Gaussian-copula null (same marginals and correlations)")
feat = pd.concat([df[[c for c in raw_cols if df[c].nunique() > 50]], np.log1p(X[["geo_hop_12", "geo_hop_23", "geo_hop_34"]])], axis=1)


def normal_scores(frame):
    G = np.zeros(frame.shape)
    for j, c in enumerate(frame.columns):
        v = frame[c].to_numpy(float)
        ok = ~np.isnan(v)
        r = pd.Series(v[ok]).rank(method="average").to_numpy()
        G[ok, j] = norm.ppf((r - 0.5) / ok.sum())
    return G, frame.isna().to_numpy()


def detectors(G, seed=0):
    C = np.corrcoef(G, rowvar=False)
    w, V = np.linalg.eigh(C)
    Ci = (V / np.maximum(w, 1e-6)) @ V.T
    out = {"Mahalanobis d2": np.einsum("ij,jk,ik->i", G, Ci, G)}
    nn = NearestNeighbors(n_neighbors=101, algorithm="brute").fit(G)
    dist = nn.kneighbors(G)[0]
    for k in (5, 20, 100):
        out[f"kNN distance k={k}"] = dist[:, k]
    out["Isolation Forest"] = -IsolationForest(n_estimators=300, random_state=seed, n_jobs=8).fit(G).score_samples(G)
    out["LOF k=20"] = -LocalOutlierFactor(n_neighbors=20).fit(G).negative_outlier_factor_
    out["LOF k=100"] = -LocalOutlierFactor(n_neighbors=100).fit(G).negative_outlier_factor_
    P = G @ V[:, ::-1][:, :5]
    for i in range(5):
        out[f"PC{i + 1} |score|"] = np.abs(P[:, i]) / np.sqrt(w[::-1][i])
    try:
        out["autoencoder error"] = task5.score_autoencoder(G.astype(np.float32), epochs=40, seed=seed)
    except Exception as e:                                                # noqa: BLE001
        print("  (autoencoder skipped:", str(e)[:80] + ")")
    return out, w[::-1]


def compare(mask, label, n_null=3):
    G, miss = normal_scores(feat.loc[mask])
    C = pd.DataFrame(np.where(miss, np.nan, G)).corr().to_numpy()                 # pairwise complete
    w, V = np.linalg.eigh(C)
    L = V * np.sqrt(np.clip(w, 1e-8, None))
    real, ev = detectors(G)
    nulls = []
    for s in range(n_null):
        Gn = np.random.default_rng(100 + s).standard_normal(G.shape) @ L.T
        Gn = Gn / Gn.std(0)
        Gn[miss] = 0
        nulls.append(detectors(Gn, seed=s + 1)[0])
    n = len(G)
    mp = (1 + np.sqrt(G.shape[1] / n)) ** 2
    print(f"  {label}: {n} accounts x {G.shape[1]} features; top eigenvalues {np.round(ev[:6], 2).tolist()} (independence bound {mp:.2f})")
    print(f"  {'score':22s} | above null 99.9% (expect {n / 1000:.0f}) | above every null value | below null 0.1% | below every null value | top-tail gap lambda")
    for k in real:
        nv = np.concatenate([d[k] for d in nulls])
        hi, lo = np.quantile(nv, 0.999), np.quantile(nv, 0.001)
        s = np.sort(real[k])
        g = np.diff(s[-int(0.05 * n):])
        j = int(np.argmax(g))
        bulk = s[-int(0.05 * n):][max(0, j - 200):j + 1]
        lam = (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[j] if len(bulk) > 1 else 0
        print(f"  {k:22s} | {(real[k] > hi).sum():6d} | {(real[k] > nv.max()).sum():6d} | {(real[k] < lo).sum():6d} | {(real[k] < nv.min()).sum():6d} | "
              f"{lam:6.1f} (size {int(0.05 * n) - 1 - j})")
    return real


real_u = compare(U, "unflagged accounts")
gc_table, gc_masks = task5.gap_clusters(df, X)
pocket = flag & ~gc_masks.any(axis=1).to_numpy()
print(f"\n  power check: the same comparison with the pocket accounts added back ({pocket.sum()} accounts, {pocket.sum() / (n_u + pocket.sum()):.2%})")
compare(U | pocket, "unflagged + pocket")
fam6 = gc_masks[[c for c in gc_masks.columns if "ip_hash_entropy" in c]].any(axis=1).to_numpy() & flag
print(f"\n  power check: with the single-IP bot family added back ({fam6.sum()} accounts)")
compare(U | fam6, "unflagged + single-IP family")
