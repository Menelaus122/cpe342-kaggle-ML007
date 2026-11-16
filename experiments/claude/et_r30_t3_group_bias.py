"""Round 30 (Jackpot loop), Task 3: median bias correction per segment x identity-repaired group (training rows only, CPU).

    python et_r30_t3_group_bias.py

The ratio stage 2 predicts log(y / base) and maps back with base * exp(r); a group whose base was rebuilt from the row identities
(30% of rows) or a segment could carry a small median bias. Groups = segment (0-3, missing) x {spending history repaired, complete}
= 10 groups. Two corrections, each fitted on the 4 training folds (L1) and applied to the 5th (pipeline Task 3 folds; input =
the round 28 nested OOF prediction, exact zeros kept at zero):
  multiplicative: yhat * s_g, s_g on a grid 0.80-1.20 (0.01)
  additive:       yhat + a_g for yhat > 0, a_g = the L1-optimal shift on a grid of quantiles of the group's residuals
Acceptance (user, 2026-11-16): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0005.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402

SIX = ["historical_spending", "total_transactions", "avg_transaction_value", "prev_month_spending", "purchases_on_discount", "discount_rate_used"]
tr = pd.read_csv(DATA_DIRS[3] / "train.csv", usecols=[task3.TARGET, "segment"] + SIX)
y = tr[task3.TARGET].values.astype(float)
grp = tr["segment"].fillna(-1).values * 2 + tr[SIX].isna().any(axis=1).values.astype(int)
folds = make_folds(y, kind="regression")
final = np.load("experiments/claude/outputs/r28/t3_oof_rebuilt.npz")["final"]
G = np.unique(grp)
GRID = np.round(np.arange(0.80, 1.2001, 0.01), 2)


def fit_mult(idx):
    return {g: GRID[np.argmin([np.abs(y[idx][grp[idx] == g] - s * final[idx][grp[idx] == g]).sum() for s in GRID])] for g in G}


def fit_add(idx):
    out = {}
    for g in G:
        m = (grp[idx] == g) & (final[idx] > 0)
        r = y[idx][m] - final[idx][m]
        cands = np.r_[0.0, np.quantile(r, np.linspace(0.3, 0.7, 41))]
        out[g] = cands[np.argmin([np.abs(r - a).sum() for a in cands])]
    return out


for label, fit, apply in [("multiplicative", fit_mult, lambda p, g, v: p * v),
                          ("additive", fit_add, lambda p, g, v: np.where(p > 0, np.maximum(p + v, 0), 0))]:
    cand = final.copy()
    chosen = []
    for a, b in folds:
        prm = fit(a)
        chosen.append(prm)
        for g in G:
            mb = grp[b] == g
            cand[b[mb]] = apply(final[b[mb]], g, prm[g])
    per = [nmae(y[b], cand[b]) - nmae(y[b], final[b]) for _, b in folds]
    d = nmae(y, cand) - nmae(y, final)
    ok = d > 0 and sum(p > 0 for p in per) >= 3 and min(per) >= -0.0005
    print(f"{label:15s} NMAE {nmae(y, cand):.5f} vs {nmae(y, final):.5f}: delta {d:+.5f}; folds +{sum(p > 0 for p in per)}/5 "
          f"({' '.join(f'{p:+.5f}' for p in per)}) -> {'PASS' if ok else 'fail'}")
    print("   fold-0 parameters per group (segment, repaired): " + "; ".join(f"({int(g // 2) if g >= 0 else 'NA'}, {int(g % 2)}): {v:.2f}"
                                                                         for g, v in chosen[0].items()))
