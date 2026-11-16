"""Round 30, Task 3: where does the absolute error sit, and can a residual model reach it? (training rows only, CPU, 2 threads)

    python ej_r30_t3_forensics.py

Input: the pipeline's out-of-fold prediction rebuilt nested in round 28 (outputs/r28/t3_oof_rebuilt.npz: final THB prediction and
P(y > 0)), the training frame with the identity repair (task3.fill_identities, row-wise, no other rows).
Part A: share of the total |y - yhat| by segment (Whale = 3), by identity-filled vs complete spending history, by size of y; for
Whales: how often y = 0 and whether it is predictable beyond P(y > 0) (5-fold CV AUC of a LightGBM on the features + P(y > 0)); how
often y exceeds historical_spending and whether that is predictable.
Part B: residual L1 boosting from the pipeline prediction: LightGBM, objective l1, init_score = final, inputs = repaired features +
final + P(y > 0) (fixed before the run: 200 trees, lr 0.02, 31 leaves, 300 rows per leaf, 3 seeds averaged). Nested on the
pipeline's Task 3 folds (fit on 4, predict the 5th). Variants: all rows; only Whale rows corrected.
Acceptance (user, 2026-11-16): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0005.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402

t0 = time.time()
tr = pd.read_csv(DATA_DIRS[3] / "train.csv")
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
z = np.load("experiments/claude/outputs/r28/t3_oof_rebuilt.npz")
final, p_oof = z["final"], z["p_oof"]
print(f"{len(y)} training rows; pipeline OOF NMAE {nmae(y, final):.5f} (round 28 rebuild); zero share y {np.mean(y == 0):.3f}, prediction {np.mean(final == 0):.3f}")
SIX = ["historical_spending", "total_transactions", "avg_transaction_value", "prev_month_spending", "purchases_on_discount", "discount_rate_used"]
filled = tr[SIX].isna().any(axis=1).values
rep, _ = task3.fill_identities(tr)
X = task3.add_features(rep).drop(columns=[c for c in (task3.TARGET, "player_id", "id") if c in rep.columns], errors="ignore")
X = X[[c for c in X.columns if c != task3.TARGET]]
seg = tr["segment"].fillna(-1).values
err = np.abs(y - final)
E = err.sum()


def table(name, groups):
    rows = []
    for g, m in groups:
        m = np.asarray(m, bool)
        rows.append([g, m.sum(), y[m].mean(), np.mean(y[m] == 0), np.mean(final[m] == 0), err[m].sum() / E, err[m].mean(),
                     1 / (1 + err[m].sum() / max(y[m].sum(), 1))])
    df = pd.DataFrame(rows, columns=[name, "rows", "mean y", "y = 0", "pred = 0", "share of |error|", "mean |error|", "NMAE in group"])
    print(df.set_index(name).round(4).to_string(), "\n")


print("\nA. error decomposition")
table("segment", [(f"{int(s)}" if s >= 0 else "missing", seg == s) for s in (-1, 0, 1, 2, 3)])
table("spending history", [("filled from identities", filled), ("complete", ~filled)])
qs = np.quantile(y[y > 0], [0.5, 0.9, 0.99])
table("y", [("0", y == 0), ("(0, median]", (y > 0) & (y <= qs[0])), ("(median, p90]", (y > qs[0]) & (y <= qs[1])),
            ("(p90, p99]", (y > qs[1]) & (y <= qs[2])), ("> p99", y > qs[2])])

W = seg == 3
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, n_jobs=2, verbose=-1)


def cv_auc(Xm, t):
    o = np.zeros(len(t))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, t):
        o[b] = lgb.LGBMClassifier(**P).fit(Xm.iloc[a], t[a]).predict_proba(Xm.iloc[b])[:, 1]
    return roc_auc_score(t, o)


hist = rep["historical_spending"].values
XW = X[W].reset_index(drop=True).assign(p_oof=p_oof[W], final=final[W])
t0w = (y[W] == 0).astype(int)
print(f"Whales: {W.sum()} rows, y = 0 in {t0w.mean():.3f}; predicted 0 in {np.mean(final[W] == 0):.3f}")
print(f"  'y = 0' from P(y > 0) alone: AUC {roc_auc_score(t0w, -p_oof[W]):.3f}; from features + P(y > 0): AUC {cv_auc(XW, t0w):.3f}")
fin = np.isfinite(hist[W])
ratio = y[W][fin] / np.maximum(hist[W][fin], 1e-6)
surge = (y[W][fin] > hist[W][fin]).astype(int)
print(f"  y > historical_spending in {surge.mean():.3f} of Whales (y / hist quantiles 50/90/99%: "
      + " / ".join(f"{q:.2f}" for q in np.quantile(ratio, [.5, .9, .99])) + ")")
if surge.sum() >= 50:
    print(f"  ({fin.sum()} Whales with a finite repaired history) surge from final / hist alone: AUC "
          f"{roc_auc_score(surge, final[W][fin] / np.maximum(hist[W][fin], 1e-6)):.3f}; from features + pipeline output: AUC "
          f"{cv_auc(XW[fin].reset_index(drop=True), surge):.3f}")
print(f"[{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ Part B: residual L1 boosting
XB = X.assign(p_oof=p_oof, final=final)
PB = dict(objective="l1", n_estimators=200, learning_rate=0.02, num_leaves=31, min_child_samples=300, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, n_jobs=2, verbose=-1)
res = np.zeros(len(y))
for i, (a, b) in enumerate(folds):
    out = np.zeros(len(b))
    for s in (0, 1, 2):
        m = lgb.LGBMRegressor(**PB, random_state=s).fit(XB.iloc[a], y[a], init_score=final[a])
        out += (final[b] + m.predict(XB.iloc[b], raw_score=True)) / 3
    res[b] = np.maximum(out, 0)
    print(f"  residual booster fold {i} done [{time.time() - t0:.0f}s]", flush=True)
print("\nB. residual L1 boosting, nested on the pipeline folds (NMAE, higher is better)")
for label, cand in [("all rows", res), ("Whale rows only", np.where(W, res, final))]:
    per = [nmae(y[b], cand[b]) - nmae(y[b], final[b]) for _, b in folds]
    d = nmae(y, cand) - nmae(y, final)
    print(f"  {label:16s} {nmae(y, cand):.5f} vs {nmae(y, final):.5f}: delta {d:+.5f}; folds +{sum(p > 0 for p in per)}/5 "
          f"({' '.join(f'{p:+.5f}' for p in per)}); rows changed by > 1 THB {np.mean(np.abs(cand - final) > 1):.2%}")
print(f"[{time.time() - t0:.0f}s]")
