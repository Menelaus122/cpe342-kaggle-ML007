"""Round 30, Task 3: a multiplicative scale per segment on top of the pipeline prediction (training rows only, CPU).

    python ep_r30_t3_segment_scale.py

ej_r30_t3_forensics.py: Whales carry 79% of the absolute error. The pipeline learns one global scale for its blend; the L1-optimal
scale may differ by segment. Candidate: yhat_g = s_g * yhat for each segment g (0-3, missing = its own group), s_g from a grid
{0.80, 0.82, ..., 1.20} minimising sum |y - s_g * yhat| on the 4 training folds, applied to the 5th (pipeline Task 3 folds; input =
the round 28 nested OOF prediction). Variants: every group, Whales only. Also reported: the in-sample best scale per group.
Acceptance (user, 2026-11-16): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0005.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402

tr = pd.read_csv(DATA_DIRS[3] / "train.csv", usecols=[task3.TARGET, "segment"])
y = tr[task3.TARGET].values.astype(float)
seg = tr["segment"].fillna(-1).values
folds = make_folds(y, kind="regression")
final = np.load("experiments/claude/outputs/r28/t3_oof_rebuilt.npz")["final"]
GRID = np.round(np.arange(0.80, 1.2001, 0.02), 2)
GROUPS = [-1, 0, 1, 2, 3]
best_in = {g: GRID[np.argmin([np.abs(y[seg == g] - s * final[seg == g]).sum() for s in GRID])] for g in GROUPS}
print("in-sample L1-best scale per segment: " + ", ".join(f"{'missing' if g < 0 else g}: {best_in[g]:.2f}" for g in GROUPS))
for label, gs in [("every segment", GROUPS), ("Whales only", [3])]:
    cand = final.copy()
    chosen = []
    for a, b in folds:
        sc = {}
        for g in gs:
            m = seg[a] == g
            sc[g] = GRID[np.argmin([np.abs(y[a][m] - s * final[a][m]).sum() for s in GRID])]
            mb = seg[b] == g
            cand[b[mb]] = sc[g] * final[b[mb]]
        chosen.append(sc)
    per = [nmae(y[b], cand[b]) - nmae(y[b], final[b]) for _, b in folds]
    d = nmae(y, cand) - nmae(y, final)
    ok = d > 0 and sum(p > 0 for p in per) >= 3 and min(per) >= -0.0005
    print(f"{label:14s} NMAE {nmae(y, cand):.5f} vs {nmae(y, final):.5f}: delta {d:+.5f}; folds +{sum(p > 0 for p in per)}/5 "
          f"({' '.join(f'{p:+.5f}' for p in per)}) -> {'PASS' if ok else 'fail'}; scales chosen per fold "
          + "; ".join(",".join(f"{v:.2f}" for v in c.values()) for c in chosen))
