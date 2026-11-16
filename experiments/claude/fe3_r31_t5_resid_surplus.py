"""Round 31, Task 5 item 2, second follow-up: what is the period-4 surplus of large conditional residuals? (label-free method)

    python fe3_r31_t5_resid_surplus.py [first-release test.csv]

fe2_r31: on the competition data the period-4 residuals have heavier tails than the period-3 ones (S2 beyond the period-3 99.9%
quantile: 106 accounts vs 24). Two readings: (a) a hidden period-4 group; (b) a shape effect: period 4 is calmer (fg_r31: steps into
period 4 are ~16% smaller in every account), so the residual core is narrower and the same surprises are larger in MAD units.
Checks: the surplus accounts' behaviours, flag rates and direction coherence (a planted group shares a direction) on the
competition data, and the same computation on the first release, where the id block tells whether the surplus accounts are
anomalies (reference only; nothing is tuned on it).
"""
import sys

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root, repo on sys.path)

OLD = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = ["fe3"]
_src = open(__file__.replace("fe3_r31_t5_resid_surplus.py", "fe_r31_t5_conditional_resid.py"), encoding="utf-8").read()
_head = _src.split("Rr = residuals(D0)")[0]
_src2 = open(__file__.replace("fe3_r31_t5_resid_surplus.py", "fe2_r31_t5_resid_selfnull.py"), encoding="utf-8").read()
_rt = "def residuals_t" + _src2.split("def residuals_t")[1].split("R4 = residuals_t")[0]


def run(label, loader, truth_fn=None):
    from src import task5 as _t5
    _t5.load = loader
    g = {"__file__": __file__}
    exec(compile(_head, "fe_head", "exec"), g)
    exec(compile(_rt, "fe2_rt", "exec"), g)
    D0, BEH, u, df = g["D0"], g["BEH"], g["u"], g["df"]
    R4, R3 = g["residuals_t"](D0, 4), g["residuals_t"](D0, 3)
    s4, s3 = g["summarise"](R4), g["summarise"](R3)
    th = np.quantile(s3["S2"], 0.999)
    sel = s4["S2"] > th
    sel3 = s3["S2"] > th
    cos = lambda R: ((lambda W: ((W @ W.T).sum() - len(W)) / (len(W) * (len(W) - 1)))(R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)))
    flags = df.loc[u, ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]].to_numpy(float)
    g["log"](f"\n== {label}: S2 beyond the period-3 99.9% quantile: period 4 {sel.sum()}, period 3 {sel3.sum()}; |r| > 6 MAD: period 4 "
             f"{int(np.sum(np.abs(R4) > 6))}, period 3 {int(np.sum(np.abs(R3) > 6))}")
    g["log"]("  period-4 surplus accounts, behaviours with the largest mean |r|: "
             + ", ".join(f"{BEH[j]} {v:.1f}" for j, v in sorted(enumerate(np.abs(R4[sel]).mean(0)), key=lambda t: -t[1])[:6])
             + f"; mean pairwise cosine of their residual vectors {cos(R4[sel]):+.3f} (period-3 tail accounts {cos(R3[sel3]):+.3f}); "
             f"flag rates (pw, vpn, susp, mass) {np.nanmean(flags[sel], 0).round(3).tolist()} vs U {np.nanmean(flags, 0).round(3).tolist()}")
    dmax = np.abs(R4).max(1)
    g["log"](f"  share of accounts whose largest |r| is in one behaviour only (others < 3 MAD): surplus {np.mean((np.abs(R4[sel]) > 3).sum(1) == 1):.2f}, "
             f"period-3 tail {np.mean((np.abs(R3[sel3]) > 3).sum(1) == 1):.2f}")
    if truth_fn is not None:
        tr = truth_fn(df)[u]
        g["log"](f"  reference: true anomalies among U {int(tr.sum())}; among the period-4 tail accounts {int(tr[sel].sum())} of {sel.sum()}; among the "
                 f"period-3 tail accounts {int(tr[sel3].sum())} of {sel3.sum()}; with the true anomalies removed, period 4 beyond the period-3 "
                 f"99.9% quantile {int(np.sum(sel & ~tr))} vs period 3 {int(np.sum(sel3 & ~tr))}")
    return R4, R3


run("competition data", lambda: pd.read_csv("05_anomaly_detection/test.csv"))
if OLD:
    run("first release, reference only", lambda: pd.read_csv(OLD), lambda d: d.player_id.str[1:].astype(int).values >= 23819)
print("done", flush=True)
