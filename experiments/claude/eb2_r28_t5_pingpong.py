"""Round 28, Task 5: follow-up of the ping-pong count in eb_r28_t5_stress.py (197 accounts above the 99% level, 159 expected).

    python eb2_r28_t5_pingpong.py

If the excess were VPN regional-pricing exploiters, the accounts at the top of the ping-pong statistic would also show the rest of
the story: the VPN flag, purchases that rise in the "away" periods, more location changes. Test features only, no labels.
  1  the excess by level (95% ... 99.9%) against both re-ordered versions, with a bootstrap interval
  2  the top accounts of the true order vs the top accounts of the re-ordered versions: VPN / password flags, alternation of
     purchases and location changes in phase with the travel, number of missing coordinates
  3  the same statistic on ALL accounts with valid coordinates (flagged included): where the known families sit
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task5  # noqa: E402

df = task5.load()
flag = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
P = lambda g: [f"{g}_{i}" for i in range(1, 5)]
la, lo = df[P("login_lat")].to_numpy(float), df[P("login_lon")].to_numpy(float)
ok = (np.abs(la) <= 90).all(1) & (np.abs(lo) <= 180).all(1)
hav = task5.haversine
D6 = {(i, j): hav(la[:, i], lo[:, i], la[:, j], lo[:, j]) for i in range(4) for j in range(i + 1, 4)}
allsum = sum(D6.values())
ORD = {"true 1-2-3-4": ((0, 2), (1, 3)), "re-ordered 1-3-2-4": ((0, 1), (2, 3)), "re-ordered 1-2-4-3": ((0, 3), (1, 2))}
stat = {k: (allsum - D6[b[0]] - D6[b[1]]) / 4 - (D6[b[0]] + D6[b[1]]) / 2 for k, b in ORD.items()}
U = ok & ~flag
n = int(U.sum())
print(f"unflagged accounts with four valid coordinates: {n}")
rng = np.random.default_rng(0)
S = np.column_stack([stat[k][U] for k in ORD])
print("1. accounts above the level set by the two re-ordered versions: true order | re-ordered | expected; excess [bootstrap 95%]")
for q in (0.95, 0.98, 0.99, 0.995, 0.999):
    def excess(M):
        thr = np.quantile(M[:, 1:], q)
        return (M[:, 0] > thr).sum(), (M[:, 1] > thr).sum(), (M[:, 2] > thr).sum()
    a, b, c = excess(S)
    bs = []
    for _ in range(400):
        M = S[rng.integers(0, n, n)]
        x, y_, z = excess(M)
        bs.append(x - (y_ + z) / 2)
    print(f"   {q:.3f}: {a:4d} | {b:4d}, {c:4d} | {n * (1 - q):6.1f}; excess {a - (b + c) / 2:+6.1f} [{np.quantile(bs, .025):+.1f}, {np.quantile(bs, .975):+.1f}]")

alt = lambda g: (df[f"{g}_2"] + df[f"{g}_4"]).to_numpy(float) / 2 - (df[f"{g}_1"] + df[f"{g}_3"]).to_numpy(float) / 2      # away periods 2, 4 minus home periods 1, 3
thr = np.quantile(S[:, 1:], 0.99)
print("\n2. who is at the top (above the 99% level)? true order vs the re-ordered versions vs all unflagged")
groups = {"true order": U & (stat["true 1-2-3-4"] > thr), "re-ordered 1-3-2-4": U & (stat["re-ordered 1-3-2-4"] > thr),
          "re-ordered 1-2-4-3": U & (stat["re-ordered 1-2-4-3"] > thr), "all unflagged": U}
rows = []
for name, m in groups.items():
    r = {"group": name, "accounts": int(m.sum()), "vpn_usage": df.loc[m, "vpn_usage"].mean(), "password_changed": df.loc[m, "password_changed"].mean(),
         "suspicious_login_time": df.loc[m, "suspicious_login_time"].mean()}
    for g in ("purchase_frequency", "purchase_amount_mean", "location_changes", "currency_transfers", "device_count"):
        sd = np.nanstd(alt(g)[U])
        r[f"|alt| {g}"] = np.nanmean(np.abs(alt(g)[m])) / sd
    r["max hop km"] = np.nanmedian(np.maximum.reduce([D6[(0, 1)], D6[(1, 2)], D6[(2, 3)]])[m])
    rows.append(r)
print(pd.DataFrame(rows).set_index("group").round(3).T.to_string())
print("   (|alt| = mean absolute difference between periods 2, 4 and periods 1, 3, in SDs of that difference among the unflagged; the "all unflagged" column is the reference)")

print("\n3. the same statistic on all accounts with valid coordinates: share above the unflagged 99% level")
print(f"   unflagged {np.mean(stat['true 1-2-3-4'][U] > thr):.4f}; flagged {np.mean(stat['true 1-2-3-4'][ok & flag] > thr):.4f} "
      f"(re-ordered versions among the flagged: {np.mean(stat['re-ordered 1-3-2-4'][ok & flag] > thr):.4f}, {np.mean(stat['re-ordered 1-2-4-3'][ok & flag] > thr):.4f})")
