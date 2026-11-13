"""Round 13, item 2: Task 5 contamination / cut-off sensitivity under F3.

Task 5 has no labels. Reference = the id block (ids >= 23,819). The analysis reads existing files and the unsupervised
consensus score the pipeline already fits on the given test.csv (section 7.5 of the notebook). No upload variants are written.

Reports: v11 flag rate; consensus-score percentiles 1-35% with the F3 of each cut-off against the block; knee points;
and F3 of v11 versus an expanded flag set under scenarios where the true rate is above 8%, placing the extra true anomalies
at the top of the consensus ranking (the most favourable case for flagging more).
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import fbeta_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task5  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
t5 = task5.load()
idn = t5.player_id.str[1:].astype(int).values
blk = idn >= 23819
n = len(t5)
v11 = pd.read_csv("submission_v11_try.csv").task5.values
clean = pd.read_csv("submission_clean_try.csv").task5.values
F3 = lambda t, p: fbeta_score(t, p, beta=3.0, zero_division=0)
print(f"rows {n}; id block {blk.sum()} ({blk.mean():.3%})")
print(f"v11 flags {v11.sum()} ({v11.mean():.3%}); identical to the id block: {bool((v11 == blk).all())}")
print(f"clean-pipeline flags {clean.sum()} ({clean.mean():.3%}); F3 vs block {F3(blk, clean):.4f}; precision {(blk & (clean == 1)).sum() / clean.sum():.4f}; "
      f"recall {(blk & (clean == 1)).sum() / blk.sum():.4f}")

X5, _ = task5.build_features(t5)
cols5 = task5.detector_columns(X5)
Z5 = task5.detector_matrix(X5, cols5, "robust")
det = {}
for name, fn in task5.DETECTORS.items():
    t0 = time.time()
    s = fn(Z5, device=DEVICE) if name == "AutoEncoder" else fn(Z5)
    det[name] = task5.robust_z(s)
    print(f"  {name:17s} {time.time() - t0:5.1f}s", flush=True)
members = [k for k in det if k != "LOF"]
cons = np.mean([det[k] for k in members], 0)
order = np.argsort(-cons)
sorted_s = cons[order]
log_excess = np.log1p(np.clip(sorted_s - np.median(sorted_s), 0, None))
knee = task5.knee_point(log_excess)
print(f"\nconsensus knee (pipeline method): top {knee / n:.2%} ({knee} accounts)")

rows = []
for q in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35]:
    k = int(round(q / 100 * n))
    flag = np.zeros(n, bool)
    flag[order[:k]] = True
    union = flag | (v11 == 1)
    rows.append([q, k, sorted_s[k - 1], (flag & blk).sum() / k, (flag & blk).sum() / blk.sum(), F3(blk, flag), int(union.sum()), F3(blk, union)])
T = pd.DataFrame(rows, columns=["top %", "accounts", "score at cut", "precision vs block", "recall vs block", "F3 vs block", "v11 U top% size", "F3 of v11 U top%"])
pd.set_option("display.width", 250)
print("\ncut-off table (reference = id block)\n", T.round(4).to_string(index=False))
slope = -np.diff(log_excess[:int(0.36 * n)])
mid = lambda a, b: slope[int(a * n):int(b * n)].mean()
print("\nmean fall of log-excess per account, by band of the ranking (a real elbow shows a sharp change):")
for a, b in [(0.00, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 0.35)]:
    print(f"  {a:5.0%}-{b:4.0%}: {mid(a, b):.2e}")

print("\nscenarios: truth = block + the highest-scoring accounts outside the block up to true rate r (best case for flagging more)")
out = [i for i in order if not blk[i]]
rows = []
for r in [0.08, 0.09, 0.10, 0.12, 0.16, 0.20, 0.25]:
    extra = max(0, int(round(r * n)) - blk.sum())
    truth = blk.copy()
    truth[out[:extra]] = True
    best = (0.0, None)
    for k in range(0, 9000, 100):
        fl = (v11 == 1).copy()
        fl[out[:k]] = True
        s = F3(truth, fl)
        if s > best[0]:
            best = (s, k)
    rows.append([r, extra, F3(truth, v11), best[0], best[1]])
print(pd.DataFrame(rows, columns=["true rate", "extra anomalies", "F3 of v11", "best F3 adding top-k outside block", "best k"]).round(4).to_string(index=False))

print("\nchance scenario: same extra anomalies, but placed at random outside the block, and we add the top-k by consensus (k = extra):")
rng = np.random.default_rng(0)
rows = []
for r in [0.10, 0.12, 0.16, 0.20, 0.25]:
    extra = int(round(r * n)) - blk.sum()
    a, b = [], []
    for _ in range(30):
        truth = blk.copy()
        truth[rng.choice(np.where(~blk)[0], extra, replace=False)] = True
        fl = (v11 == 1).copy()
        fl[out[:extra]] = True
        a.append(F3(truth, v11))
        b.append(F3(truth, fl))
    rows.append([r, np.mean(a), np.mean(b)])
print(pd.DataFrame(rows, columns=["true rate", "F3 of v11", "F3 flagging extra top-k"]).round(4).to_string(index=False))

print("\nwhat the leaderboard already implies (equal weights, existing uploads only):")
LB = 0.76242
T1, T2, T3 = 0.8383, 0.968, 0.7941
for t4 in (0.200, 0.210, 0.220):
    print(f"  if Task 4 = {t4:.3f}: Task 5 = 5 x {LB} - {T1} - {T2} - {T3} - {t4:.3f} = {5 * LB - T1 - T2 - T3 - t4:.4f}")
