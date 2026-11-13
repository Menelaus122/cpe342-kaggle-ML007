"""Round 21, Task 5: is the 8.30% flag rate too conservative for F3? Tail of the score ensemble and detector fusion, test features only.

    python ct_r21_t5_tail.py

Task 5 has no labels. The v15 flags (2,148 = six gap-separated families + the dependence pocket) come from
outputs/submission_parts/task5.csv; detectors, families and the detector matrix are rebuilt with the pipeline's own functions.
  A. detector scale: robust-z quantiles of each detector, and each one's share of the fused score in the top tail
  B. fusion: robust-z mean (pipeline cross-check) vs rank mean vs soft-OR (log-sum-exp of robust z); overlap of their top-k sets
     with each other and with the v15 flags, k = 2,148 and 8 / 10 / 12 / 14% of accounts
  C. tail of the accounts outside the v15 flags: (1) the largest density gap among their top 3,000 fused scores, as lambda =
     expected accounts inside the gap (the families' own test, lambda >= 50 = isolated); (2) a permutation null: shuffle the rows of
     each behaviour block (columns built from the same raw feature move together) among these accounts, rescore, and count how
     many real accounts exceed each cut-off beyond what the null produces. excess / added = estimated share of anomalies among the
     next accounts one would add; F3 makes adding worthwhile only above about F*/10 (~0.1).
  D. the F3 arithmetic: what the public score allows for the true anomaly rate.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t5 = task5.load()
X5, _ = task5.build_features(t5)
cols5 = task5.detector_columns(X5)
Z5 = task5.detector_matrix(X5, cols5, "robust")
_, gc_masks = task5.gap_clusters(t5, X5)
gap5 = gc_masks.any(axis=1).values
final = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
pocket = final & ~gap5
n = len(final)
print(f"{n} accounts; v15 flags {final.sum()} = families {gap5.sum()} + pocket {pocket.sum()} ({final.mean():.2%}); "
      f"families inside flags: {(gap5 <= final).all()}")

DET = {k: f for k, f in task5.DETECTORS.items() if k != "LOF"}          # the pipeline's fusion excludes LOF
raw = {k: (f(Z5, device="cpu") if k == "AutoEncoder" else f(Z5)) for k, f in DET.items()}
Zd = pd.DataFrame({k: task5.robust_z(s) for k, s in raw.items()})

print("\n== A. detector scale (robust z = (score - median) / MAD)")
q = Zd.quantile([0.5, 0.9, 0.99, 0.999, 1.0]).T
q.columns = ["p50", "p90", "p99", "p99.9", "max"]
top = Zd.mean(axis=1).rank(ascending=False) <= final.sum()
q["share of fused score, top 2,148"] = Zd[top].clip(lower=0).mean() / Zd[top].clip(lower=0).mean().sum()
q["share among v15 flags"] = Zd[final].clip(lower=0).mean() / Zd[final].clip(lower=0).mean().sum()
print(q.round(2).to_string())

print("\n== B. fusion rules")
F = {"robust-z mean (pipeline)": Zd.mean(axis=1).values,
     "rank mean": Zd.rank(pct=True).mean(axis=1).values,
     "soft-OR logsumexp(z)": np.logaddexp.reduce(Zd.clip(-50, 50).values, axis=1)}
ks = {"2,148": int(final.sum()), "8%": int(0.08 * n), "10%": int(0.10 * n), "12%": int(0.12 * n), "14%": int(0.14 * n)}
topk = lambda s, k: np.argsort(-s)[:k]
for name, s in F.items():
    line = [f"{name:26s}"]
    for kn, k in ks.items():
        t = np.zeros(n, bool)
        t[topk(s, k)] = True
        line.append(f"top {kn:>5s}: v15 recall {(t & final).sum() / final.sum():.3f}, families {(t & gap5).sum() / gap5.sum():.3f}")
    print(" | ".join(line))
names = list(F)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = set(topk(F[names[i]], ks["2,148"])), set(topk(F[names[j]], ks["2,148"]))
        print(f"  Jaccard of top 2,148: {names[i]} vs {names[j]}: {len(a & b) / len(a | b):.3f}")

print("\n== C. tail outside the v15 flags")
rest = np.where(~final)[0]
for name, s in F.items():
    v = np.sort(s[rest])[::-1][:3200]
    gaps = v[:-1] - v[1:]
    lam = []
    for i in range(3000):
        width = v[i + 1] - v[min(i + 201, len(v) - 1)]
        dens = 200 / max(width, 1e-12)
        lam.append(gaps[i] * dens)
    lam = np.array(lam)
    i = int(np.argmax(lam))
    print(f"  {name:26s}: largest gap among the top 3,000 unflagged at rank {i + 1}, lambda = {lam[i]:.1f} expected accounts in it "
          f"(families need >= 50); next largest {np.sort(lam)[-2]:.1f}")

base_of = np.array([task5._base_feature(c) for c in cols5])
Zr = Z5[rest]
FAST = {k: DET[k] for k in ["ExtremeValue", "IsolationForest", "ECOD", "COPOD", "Mahalanobis(MCD)"]}


def fused(Z):
    return np.mean([task5.robust_z(f(Z)) for f in FAST.values()], axis=0)


real = np.sort(fused(Zr))[::-1]
rng = np.random.default_rng(0)
nulls = []
for p in range(3):
    Zs = Zr.copy()
    for b in np.unique(base_of):
        cols = np.where(base_of == b)[0]
        Zs[:, cols] = Zr[rng.permutation(len(Zr))][:, cols]
    nulls.append(np.sort(fused(Zs))[::-1])
    print(f"  null permutation {p + 1} scored", flush=True)
print("  accounts added beyond the v15 flags (fast 5-detector fusion, refitted on the unflagged accounts) vs block-shuffled null:")
print(f"  {'added':>6s} {'cut-off':>8s} {'null count above':>17s} {'est. anomalies':>15s} {'share of the added':>19s} {'share in this band':>19s}")
prev = (0, 0.0)
for add in [100, 200, 400, 600, 1000, 1500, 2000]:
    cut = real[add - 1]
    null_cnt = np.mean([(nl >= cut).sum() for nl in nulls])
    ex = add - null_cnt
    band = (ex - prev[1]) / (add - prev[0])
    print(f"  {add:6d} {cut:8.2f} {null_cnt:17.1f} {ex:15.1f} {ex / add:19.3f} {band:19.3f}")
    prev = (add, ex)

print("\n== D. F3 arithmetic (positive class, beta = 3)")
f3 = lambda P, Rc: 10 * P * Rc / (9 * P + Rc)
print("  if all 2,148 flags are true anomalies and the true rate is r, recall = 2,148 / (r n):")
for r in [0.083, 0.09, 0.10, 0.12, 0.14]:
    Rc = min(1.0, final.sum() / (r * n))
    print(f"    r = {r:.1%}: recall {Rc:.3f} -> F3 {f3(1.0, Rc):.3f}")
lb, t1, t2, t3 = 0.85032, 0.85, 0.757, 0.795
print(f"  public LB {lb} x 5 - (T1 ~{t1}, T2 ~{t2}, T3 ~{t3}) = T4 + T5 ~ {5 * lb - t1 - t2 - t3:.3f}; with T4 <= 0.99 (its OOF), "
      f"T5 >= {5 * lb - t1 - t2 - t3 - 0.99:.3f}")
