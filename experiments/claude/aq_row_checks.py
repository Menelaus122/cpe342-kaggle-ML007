"""Insight round 7: structure checks that were not run before (all on files we were given; no labels of the test set).

1. Row position: does a label (train) or a prediction (test) drift with the row's position in the csv file?
   (Earlier rounds checked player_id only, not the file order.)
2. Same row across tasks: the submission puts all five tasks on one row (ANS00001 ...). Is row i of Task 1 / 2 / 3 / 5
   the same player? Earlier rounds compared the same player_id across tasks, not the same row position.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2_contingency

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402

D = {1: "01_binary_classification", 2: "02_multiclass_classification", 3: "03_regression", 5: "05_anomaly_detection"}
T = {1: "is_cheater", 2: "segment", 3: "spending_30d"}
tr = {k: pd.read_csv(f"{d}/train.csv") for k, d in D.items() if k != 5}
te = {k: pd.read_csv(f"{d}/test.csv") for k, d in D.items()}
sub = pd.read_csv("submission_v8_try.csv")


def pos_check(name, v, kind):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    idx = np.arange(len(v))[ok]
    v = v[ok]
    rho, p = spearmanr(idx, v)
    dec = pd.Series(v).groupby(pd.qcut(idx, 10, labels=False)).mean().round(4).tolist()
    if kind == "cat":
        tab = pd.crosstab(pd.qcut(idx, 20, labels=False), v)
        p = chi2_contingency(tab)[1]
    print(f"  {name:<34} Spearman(row, value) {rho:+.4f}  p {p:.3f}  decile means {dec}")


print("1. ROW POSITION")
for k in (1, 2, 3):
    y = tr[k][T[k]].values
    pos_check(f"T{k} train label", y, "cat" if k != 3 else "num")
    if k == 1:
        pos_check("T1 train label missing?", np.isnan(y.astype(float)), "cat")
    if k == 3:
        pos_check("T3 train label = 0", (y == 0).astype(float), "cat")
for k in (1, 2, 3, 5):
    pos_check(f"T{k} test submission (v8)", sub[f"task{k}"].values, "cat" if k != 3 else "num")
print("  T5 test sorted by player_id?", bool((te[5].player_id.values == np.sort(te[5].player_id.values)).all()))
for k in (1, 2, 3):
    ids = tr[k].player_id.str[1:].astype(int).values
    print(f"  T{k} train sorted by player_id? {bool((np.diff(ids) > 0).all())}; Spearman(row, id) {spearmanr(np.arange(len(ids)), ids)[0]:+.4f}")

print("\n2. SAME ROW ACROSS TASKS (row i of one file vs row i of another)")
pairs = [((1, "account_age_days"), (2, "account_age_days")),
         ((2, "friend_count"), (3, "friend_count")),
         ((1, "friend_network_size"), (2, "friend_count")),
         ((2, "total_playtime_hours"), (3, "total_playtime_hours")),
         ((2, "days_since_last_login"), (3, "days_since_last_login")),
         ((2, "achievement_completion_rate"), (3, "achievement_completion_rate"))]
for split, S in (("train", tr), ("test", te)):
    for (a, ca), (b, cb) in pairs:
        if ca in S[a] and cb in S[b]:
            x, z = S[a][ca].values, S[b][cb].values
            ok = np.isfinite(x) & np.isfinite(z)
            print(f"  {split:<5} T{a}.{ca} ~ T{b}.{cb}: Spearman {spearmanr(x[ok], z[ok])[0]:+.4f} (n {ok.sum():,})")
# same row, sorted by id instead of file order
for (a, ca), (b, cb) in pairs[:2]:
    A = tr[a].assign(i=tr[a].player_id.str[1:].astype(int)).sort_values("i")[ca].values
    B = tr[b].assign(i=tr[b].player_id.str[1:].astype(int)).sort_values("i")[cb].values
    ok = np.isfinite(A) & np.isfinite(B)
    print(f"  train, k-th smallest id: T{a}.{ca} ~ T{b}.{cb}: Spearman {spearmanr(A[ok], B[ok])[0]:+.4f}")
# label agreement on the same row (train): T2 segment vs T3 segment column, T1 cheater vs T2 segment
seg3 = tr[3]["segment"] if "segment" in tr[3] else None
if seg3 is not None:
    ok = seg3.notna().values
    print(f"  train T2 segment == T3 segment column on the same row: {np.mean(tr[2].segment.values[ok] == seg3.values[ok]):.3f} (chance ~0.25-0.3)")
ok = tr[1].is_cheater.notna().values
print("  train T1 cheater rate by T2 segment on the same row:",
      pd.Series(tr[1].is_cheater.values[ok]).groupby(tr[2].segment.values[ok]).mean().round(4).to_dict())
