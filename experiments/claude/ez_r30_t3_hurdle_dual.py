"""Round 30, Task 3 brief items 4-5 (training rows only, CPU).

    python ez_r30_t3_hurdle_dual.py

item 4  within-segment hurdle: slowdown features (prev / (hist + 1) and its distance below 0.30 / above 0.40, purchase frequency per
        account day, days since the last purchase / login, within-segment z-scores of the activity and recency columns) and one
        LightGBM P(y > 0) per imputed segment (the tuned stage-1 parameters), out of fold on the Task 3 folds. Blended with the 10-seed
        hurdle: p' = (1 - a) p + a p_seg, a in {0, .25, .5, .75} chosen on the training folds by log-loss; then the whole pipeline is
        rebuilt nested with p' (gate and every read-out use it).
item 5  dual-anchor ratio stage 2: target log1p(y) - (0.6 log1p(hist) + 0.4 log1p(prev)) on positive rows with both known (round 5:
        y ~ hist^0.59 prev^0.40); LightGBM (the tuned ratio parameters) and CatBoost (lr 0.08, depth 8) via task3.ratio_stage2_cv,
        one seed; added to the stage-2 pool, pipeline rebuilt nested.
Reference: the same nested rebuild with the pipeline's members. Acceptance (user): pooled NMAE delta > 0, >= 3/5 folds, no fold
below -0.0002.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import t3_nested as T  # noqa: E402  (imports common -> chdir to the repo root)
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import fit_predict  # noqa: E402
from src.utils import load_params  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
D = T.load_all()
y, X, folds, p = D["y"], D["X"].copy(), D["folds"], D["p"]
seg = X["segment"].values.astype(int)
log(f"{len(y)} rows loaded")
ref = T.nested_final(D, p, D["MU"], log)
log(f"reference (nested) NMAE {nmae(y, ref):.5f}")

# ---------------- item 4
H, Pm = X["historical_spending"].values, X["prev_month_spending"].values
F = X.copy()
F["prev_to_hist"] = Pm / (H + 1)
F["below_band"] = np.maximum(0, 0.30 - F["prev_to_hist"])
F["above_band"] = np.maximum(0, F["prev_to_hist"] - 0.40)
F["purchases_per_day"] = X["purchase_frequency"] / (X["account_age_days"] + 1)
for c in ["days_since_last_purchase", "days_since_last_login", "purchase_frequency", "sessions_per_week", "prev_month_spending",
          "total_transactions"]:
    g = F.groupby(seg)[c]
    F[f"{c}_z_in_segment"] = (F[c] - g.transform("median")) / (g.transform(lambda v: v.quantile(0.75) - v.quantile(0.25)) + 1e-9)
S1 = load_params("task3_stage1") or {"learning_rate": 0.05}
S1["learning_rate"] = S1["learning_rate"] / 2
S1["n_jobs"] = 4
p_seg = np.zeros(len(y))
for a, b in folds:
    for s_ in np.unique(seg):
        ta, tb = a[seg[a] == s_], b[seg[b] == s_]
        yy = (y > 0).astype(int)
        if yy[ta].min() == yy[ta].max():
            p_seg[tb] = yy[ta].mean()
            continue
        _, (pv,), _ = fit_predict("lgb", "binary", S1, F.iloc[ta], yy[ta], F.iloc[tb], yy[tb], [F.iloc[tb]])
        p_seg[tb] = pv
ll = lambda q, idx: -np.mean(np.where(y[idx] > 0, np.log(np.clip(q[idx], 1e-6, 1)), np.log(np.clip(1 - q[idx], 1e-6, 1))))
allr = np.arange(len(y))
log(f"item 4: log-loss P(y>0): 10-seed hurdle {ll(p, allr):.4f}, within-segment hurdle {ll(p_seg, allr):.4f}; by segment "
    + ", ".join(f"{s_}: {ll(p, np.where(seg == s_)[0]):.4f} -> {ll(p_seg, np.where(seg == s_)[0]):.4f}" for s_ in np.unique(seg)))
p_new, chosen = p.copy(), []
for a, b in folds:
    a_best = min((0.0, 0.25, 0.5, 0.75), key=lambda w: ll((1 - w) * p + w * p_seg, a))
    chosen.append(a_best)
    p_new[b] = (1 - a_best) * p[b] + a_best * p_seg[b]
log(f"item 4: blend weight chosen per fold {chosen}")
cand4 = T.nested_final(D, p_new, D["MU"], log) if any(chosen) else ref.copy()
print(T.compare(D, ref, cand4, "item 4 within-segment hurdle"), flush=True)
W = seg == 3
print(f"    Whale rows: mean |error| {np.abs(y - ref)[W].mean():,.0f} -> {np.abs(y - cand4)[W].mean():,.0f}; Whales with y = 0: "
      f"{np.abs(y - ref)[W & (y == 0)].mean():,.0f} -> {np.abs(y - cand4)[W & (y == 0)].mean():,.0f}", flush=True)

# ---------------- item 5
R2 = load_params("task3_ratio_stage2") or {"learning_rate": 0.05}
R2["learning_rate"] = R2["learning_rate"] / 2
X5 = X.copy()
X5["dual_base"] = np.expm1(0.6 * np.log1p(H) + 0.4 * np.log1p(Pm))
dummy = X5.head(5)
MU5 = dict(D["MU"])
for name, kind, prm in [("ratio[dual] LGB", "lgb", {**R2, "n_jobs": 4}), ("ratio[dual] Cat", "cat", {"learning_rate": 0.08, "depth": 8, "thread_count": 4})]:
    r = task3.ratio_stage2_cv(kind, prm, X5, y, dummy, folds, "dual_base", D["MU"]["Hurdle"], np.zeros(len(dummy)))
    MU5[name] = r["mu_oof"]
    pos = (y > 0) & np.isfinite(np.log1p(X5["dual_base"].values))
    log(f"item 5: {name}: median |log1p(y) - mu| on positive rows with a base {np.median(np.abs(np.log1p(y[pos]) - r['mu_oof'][pos])):.4f} "
        f"(ratio[hist] LGB tuned {np.median(np.abs(np.log1p(y[pos]) - D['MU']['ratio[hist] LGB tuned'][pos])):.4f})")
cand5 = T.nested_final(D, p, MU5, log)
print(T.compare(D, ref, cand5, "item 5 dual-anchor members added"), flush=True)
np.savez("experiments/claude/outputs/r30/t3_items45.npz", ref=ref, cand4=cand4, cand5=cand5, p_seg=p_seg,
         mu_dual_lgb=MU5["ratio[dual] LGB"], mu_dual_cat=MU5["ratio[dual] Cat"])
log("done")
