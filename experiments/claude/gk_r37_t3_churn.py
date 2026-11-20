"""Round 37, Task 3 (one last CPU round, about 30 minutes; training rows only): the "spent nothing this month" error and an
age-stratified blend.

    python gk_r37_t3_churn.py

Round 8: rows predicted to spend that spent 0 cause 33% of the absolute error (mostly whales and segment-2 players skipping a month;
within a segment AUC 0.57-0.63). The user's brief of 2026-11-17:
item 1  a stage-1 specialist for P(y > 0) trained only on rows with (imputed) segment >= 1, with slowdown features:
          prev_to_avg_monthly       = prev_month_spending / (historical_spending / max(account_age_days / 30, 1) + 1)
                                      (the data have no avg_monthly_spending; history per 30 days of account age stands in)
          recent_daily_spend_ratio  = (prev_month_spending / 30) / (historical_spending / (account_age_days + 1) + 1e-3)
          vip_freq_interaction      = vip_status x purchase_frequency   (the data's frequency column)
          premium_recency           = is_premium_member x days_since_last_purchase
        LightGBM with the pipeline's tuned stage-1 parameters (task3_stage1, learning rate halved as in the pipeline), early
        stopping as in the pipeline's hurdle (task3.hurdle_cv / fit_predict), 3 seeds, out of fold on the pipeline folds.
          1a  segment >= 1 rows take the specialist's P(y > 0), segment 0 rows keep the pipeline's 10-seed P(y > 0);
          1b  segment >= 1 rows take the mean log-odds of the specialist and the pipeline.
        The new P(y > 0) drives the whole nested rebuild (read-outs, exact L1 blend, gate, scale, cap: t3_nested.nested_final).
item 2  the exact L1 blend weights fitted separately for young and old accounts (account_age_days below / above the training
        folds' median) and shrunk to the global weights: w_g = 0.5 w_group + 0.5 w_global; gate shift, scale and cap as before.
Reference = the pipeline rebuilt fully nested (t3_nested.nested_final). Acceptance (user, Task 3): pooled delta > 0, >= 3/5 folds,
no fold below -0.0002.
"""
import os
import time

NT = int(os.environ.get("R37_THREADS", 4))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import t3_nested as T  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import fit_predict  # noqa: E402
from src.utils import load_params  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
D = T.load_all()
y, p, folds, raw = D["y"], D["p"], D["folds"], D["rep"]
seg = D["X"]["segment"].values.astype(int)
t = (y > 0).astype(int)
X = D["X"].select_dtypes("number").drop(columns=[c for c in (task3.TARGET, "player_id", "id") if c in D["X"].columns]).copy()
h, pv, age = (raw[c].values.astype(float) for c in ("historical_spending", "prev_month_spending", "account_age_days"))
X["prev_to_avg_monthly"] = pv / (h / np.maximum(age / 30, 1) + 1)
X["recent_daily_spend_ratio"] = (pv / 30) / (h / (age + 1) + 1e-3)
X["vip_freq_interaction"] = raw["vip_status"].map(lambda v: v if isinstance(v, (int, float)) else np.nan).astype(float).values \
    * raw["purchase_frequency"].values.astype(float) if pd.api.types.is_numeric_dtype(raw["vip_status"]) else np.nan
X["premium_recency"] = raw["is_premium_member"].astype(float).values * raw["days_since_last_purchase"].values.astype(float)
print(f"{len(y)} rows, {X.shape[1]} stage-1 features; segment >= 1: {np.mean(seg >= 1):.3f} of the rows, positive rate {t[seg >= 1].mean():.3f} "
      f"(segment 0: {t[seg == 0].mean():.3f}); vip_status dtype {raw['vip_status'].dtype}", flush=True)

ref = T.nested_final(D, p, D["MU"])
log(f"reference nested NMAE {nmae(y, ref):.5f}")
pos_pred = ref > 0
fz = pos_pred & (y == 0)
print(f"error shares (reference): predicted spend, spent 0: {np.abs(y - ref)[fz].sum() / np.abs(y - ref).sum():.3f} "
      f"(segment >= 1 part {np.abs(y - ref)[fz & (seg >= 1)].sum() / np.abs(y - ref).sum():.3f})", flush=True)

# item 1: the segment >= 1 specialist (skipped with ITEM2_ONLY=1)
if not os.environ.get("ITEM2_ONLY"):
    S1 = dict(load_params("task3_stage1") or {"learning_rate": 0.05})
    S1["learning_rate"] = S1.get("learning_rate", 0.05) / 2
    sp = np.zeros(len(y))
    for f, (a, b) in enumerate(folds):
        a1, b1 = a[seg[a] >= 1], b[seg[b] >= 1]
        ps = []
        for s in (0, 1, 2):
            _, (pb,), _ = fit_predict("lgb", "binary", {**S1, "random_state": s, "n_jobs": NT}, X.iloc[a1], t[a1], X.iloc[b1], t[b1], [X.iloc[b1]])
            ps.append(pb)
        sp[b1] = np.mean(ps, 0)
    m1 = seg >= 1
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    pa = np.where(m1, sp, p)
    pb = np.where(m1, 1 / (1 + np.exp(-(lg(sp) + lg(p)) / 2)), p)
    auc_seg = lambda q: [roc_auc_score(t[seg == s], q[seg == s]) for s in (1, 2, 3)]
    log(f"stage-1 AUC on segment >= 1 rows: pipeline {roc_auc_score(t[m1], p[m1]):.4f}, specialist {roc_auc_score(t[m1], sp[m1]):.4f}, mean "
        f"{roc_auc_score(t[m1], pb[m1]):.4f}; within segments 1 / 2 / 3: pipeline {np.round(auc_seg(p), 4)}, specialist {np.round(auc_seg(pa), 4)}; "
        f"log-loss on segment >= 1: pipeline {-np.mean(t[m1] * np.log(p[m1]) + (1 - t[m1]) * np.log(1 - p[m1])):.4f}, specialist "
        f"{-np.mean(t[m1] * np.log(np.clip(sp[m1], 1e-6, 1)) + (1 - t[m1]) * np.log(np.clip(1 - sp[m1], 1e-6, 1))):.4f}")
    for name, pn in (("1a specialist P(y>0) on segment >= 1", pa), ("1b mean of specialist and pipeline on segment >= 1", pb)):
        cand = T.nested_final(D, pn, D["MU"])
        fzc = (cand > 0) & (y == 0)
        print(T.compare(D, ref, cand, name) + f"  | error from predicted-spend-but-zero rows {np.abs(y - ref)[fz].sum() / y.sum():.4f} -> "
              f"{np.abs(y - cand)[fzc].sum() / y.sum():.4f} of sum y; rows crossing the gate {np.mean((ref > 0) != (cand > 0)):.3%}", flush=True)


# item 2: age-stratified exact L1 blend shrunk to the global weights
def nested_age(D, p, MU, lam=0.5):
    out = np.zeros(len(y))
    for i, (tr, va) in enumerate(folds):
        cols_tr, cols_va = [v[tr] for v in D["raw_pool"].values()], [v[va] for v in D["raw_pool"].values()]
        shift = 0.0
        for tag, mu in MU.items():
            ro = task3.readouts(y[tr], p[tr], mu[tr])
            if tag == "Hurdle":
                shift = ro["shift"]
            for k in ("mixture-median", "hard gate", "E[y] + smearing"):
                cols_tr.append(task3.apply_readout(k, p[tr], mu[tr], ro))
                cols_va.append(task3.apply_readout(k, p[va], mu[va], ro))
        Ptr, Pva = np.column_stack(cols_tr), np.column_stack(cols_va)
        wg = task3.l1_blend_weights_exact(y[tr], [Ptr[:, j] for j in range(Ptr.shape[1])])
        cut = np.nanmedian(age[tr])                       # account_age_days has missing cells: they join the older group
        btr, bva = np.zeros(len(tr)), np.zeros(len(va))
        young_tr, young_va = age[tr] <= cut, age[va] <= cut
        for grp_tr, grp_va in ((young_tr, young_va), (~young_tr, ~young_va)):
            w_ = task3.l1_blend_weights_exact(y[tr][grp_tr], [Ptr[grp_tr, j] for j in range(Ptr.shape[1])])
            w = lam * w_ + (1 - lam) * wg
            btr[grp_tr], bva[grp_va] = Ptr[grp_tr] @ w, Pva[grp_va] @ w
        gtr, gva = np.clip(p[tr] + shift, 0, 1) > 0.5, np.clip(p[va] + shift, 0, 1) > 0.5
        s = T.SCALES[int(np.argmin([np.abs(y[tr] - btr * s_ * gtr).sum() for s_ in T.SCALES]))]
        out[va] = np.minimum(bva * s * gva, y[tr].max())
    return out


cand = nested_age(D, p, D["MU"])
print(T.compare(D, ref, cand, "2 age-stratified L1 blend, lambda 0.5"), flush=True)
log("done")
