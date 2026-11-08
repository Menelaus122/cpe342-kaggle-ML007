"""Task 3: fill missing spending-history values from the exact identities, then retrain the LightGBM members.

Identities (v_identity_scan.py, R^2 = 1.00000 on all complete rows):
    historical_spending   = total_transactions * avg_transaction_value
    purchases_on_discount = total_transactions * discount_rate_used
    prev_month_spending  ~= 0.35 * historical_spending  (ratio 0.30-0.40)
Only feature values are filled (row by row, from the same row): no target information is involved.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import P3_lgb, t3_pool, t3_blend_score, OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, run_cv  # noqa: E402

from t3_fill import fill, RATIO  # noqa: E402

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
trf, s1 = fill(tr)
tef, s2 = fill(te)
print("values filled (train):", s1)
# sanity: the exact fill reproduces known values
ok = tr.historical_spending.notna() & (tr.total_transactions > 0) & tr.avg_transaction_value.notna()
print(f"exact identity check on known rows: max rel. error "
      f"{np.max(np.abs(tr.total_transactions[ok] * tr.avg_transaction_value[ok] / tr.historical_spending[ok] - 1)):.2e}")
X, Xt = task3.add_features(trf), task3.add_features(tef)

X_old, y_old, Xt_old, folds_old, m_old, gate_oof, gate_test = t3_pool()
assert np.array_equal(y_old, y)
new = {}
r = run_cv("lgb", "regression", P3_lgb, X, y, Xt, folds, verbose=False)
new["LGB-L1 raw THB"] = (np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None))
r = run_cv("lgb", "regression", P3_lgb, X, np.log1p(y), Xt, folds, verbose=False)
new["LGB-L1 log1p"] = (np.clip(np.expm1(r["oof"]), 0, None), np.clip(np.expm1(r["test"]), 0, None))
h = task3.hurdle_cv("lgb", X, y, Xt, folds, clf_params={"learning_rate": 0.05}, reg_params={"learning_rate": 0.05}, verbose=False)
np.savez(OUT / "t3_identity_hurdle.npz", **{k: v for k, v in h.items()})
tm = task3.tune_mixture(y, h["p_oof"], h["mu_oof"], float(h["sigma"]))
new["Hurdle mixture-median"] = (task3.mixture_median(np.clip(h["p_oof"] + tm["shift"], 0, 1), h["mu_oof"], tm["sigma"]),
                                task3.mixture_median(np.clip(h["p_test"] + tm["shift"], 0, 1), h["mu_test"], tm["sigma"]))
th = task3.tune_hard_hurdle(y, h["p_oof"], h["mu_oof"])
new["Hurdle hard gate"] = (task3.hard_hurdle(h["p_oof"], h["mu_oof"], th["tau"]), task3.hard_hurdle(h["p_test"], h["mu_test"], th["tau"]))
new["Hurdle E[y] + smearing"] = (task3.expected_value(h["p_oof"], h["mu_oof"], float(h["smear"])),
                                 task3.expected_value(h["p_test"], h["mu_test"], float(h["smear"])))
gate_new_oof = (np.clip(h["p_oof"] + tm["shift"], 0, 1) > 0.5).astype(float)
gate_new_test = (np.clip(h["p_test"] + tm["shift"], 0, 1) > 0.5).astype(float)

print("\nper-model OOF score (old features -> filled features):")
for k in new:
    print(f"  {k:24s} {nmae(y, m_old[k][0]):.5f} -> {nmae(y, new[k][0]):.5f}")

names = list(m_old)
P_old = np.column_stack([m_old[k][0] for k in names])
s_old, pred_old = t3_blend_score(y, folds, P_old, gate_oof)
m_new = {k: new.get(k, m_old[k]) for k in names}
P_new = np.column_stack([m_new[k][0] for k in names])
s_new, pred_new = t3_blend_score(y, folds, P_new, gate_new_oof)
print(f"\nblend: old {s_old:.5f} -> filled-feature members {s_new:.5f} ({s_new - s_old:+.5f})")
H, Pm = tr.historical_spending.isna().values, tr.prev_month_spending.isna().values
for g, msk in [("hist & prev both missing", H & Pm), ("hist missing only", H & ~Pm), ("rest", ~H)]:
    print(f"  {g:26s}: mean |err| {np.abs(y - pred_old)[msk].mean():7.0f} -> {np.abs(y - pred_new)[msk].mean():7.0f}")
np.savez(OUT / "t3_identity_pool.npz", P_oof=P_new, P_test=np.column_stack([m_new[k][1] for k in names]),
         gate_oof=gate_new_oof, gate_test=gate_new_test, names=np.array(names))
