"""Round 31, Task 5 item 2, follow-up: a self-null for the conditional residuals (label-free; Task 5 has only test.csv).

    python fe2_r31_t5_resid_selfnull.py

fe_r31's Gaussian-copula null cannot calibrate the residual tails: real residuals are heavy-tailed (non-linear, heteroscedastic
relations, zero-inflated columns), copula residuals are near-Gaussian, so the real data exceed the copula's 99.9% quantile 170x.
Self-null: the same models predict period 3 of each behaviour instead of period 4 (inputs: the other three periods of the
behaviour, all periods of the others, statics, flags). A group planted in period 4 makes the period-4 residual tails heavier than
the period-3 ones; without one, the two match (if anything period 4 is calmer: fg_r31 found fewer period-4 changes).
Reported: accounts beyond the period-3 score's 99 / 99.9% quantiles for the period-4 score (S2, Smax, economy+ / -), per-behaviour
counts of |r| > 6 MAD in period 4 vs period 3, and the upper-tail gap lambda of each.
"""
import sys

import numpy as np

sys.argv = ["fe2"]
_src = open(__file__.replace("fe2_r31_t5_resid_selfnull.py", "fe_r31_t5_conditional_resid.py"), encoding="utf-8").read()
exec(compile(_src.split("Rr = residuals(D0)")[0], "fe_head", "exec"))     # data, U, models, helpers of fe_r31 (no runs)


def residuals_t(D, t, seed=0):
    from sklearn.model_selection import KFold as _KF
    R = np.zeros((len(D), len(BEH)))
    for j, g in enumerate(BEH):
        y = nscore(D[f"{g}_{t}"].to_numpy(float))
        feats = [c for c in D.columns if c != f"{g}_{t}"]
        ok = ~np.isnan(y)
        pred = np.full(len(D), np.nan)
        for a, b in _KF(5, shuffle=True, random_state=seed).split(D):
            a = a[ok[a]]
            pred[b] = lgb.LGBMRegressor(**PRM, random_state=seed).fit(D.iloc[a][feats], y[a]).predict(D.iloc[b][feats])
        r = y - pred
        R[:, j] = np.nan_to_num(r / (np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826))
    return R


R4 = residuals_t(D0, 4)
log("period-4 residuals done")
R3 = residuals_t(D0, 3)
log("period-3 residuals done")
s4, s3 = summarise(R4), summarise(R3)
for k in s4:
    line = f"{k:10s}"
    for q in (0.99, 0.999):
        th = np.quantile(s3[k], q)
        line += f" | beyond the period-3 {q:.1%} quantile: period 4 {int(np.sum(s4[k] > th))} (period 3 {int(np.sum(s3[k] > th))})"
    line += f" | upper-tail gap lambda period 4 {lam_gap(s4[k]):.1f}, period 3 {lam_gap(s3[k]):.1f}"
    log(line)
rows = sorted(((g, int(np.sum(np.abs(R4[:, j]) > 6)), int(np.sum(np.abs(R3[:, j]) > 6)), lam_gap(np.abs(R4[:, j])), lam_gap(np.abs(R3[:, j])))
               for j, g in enumerate(BEH)), key=lambda t: -(t[1] - t[2]))
log("per behaviour, |r| > 6 MAD (period 4 / period 3) and gap lambda (period 4 / period 3), largest period-4 surplus first: "
    + "; ".join(f"{g} {a}/{b}, {la:.1f}/{lb:.1f}" for g, a, b, la, lb in rows[:8]))
log(f"total |r| > 6 MAD: period 4 {int(np.sum(np.abs(R4) > 6))}, period 3 {int(np.sum(np.abs(R3) > 6))}")
log("done")
