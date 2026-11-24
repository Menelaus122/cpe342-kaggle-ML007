"""Round 35, Task 3: nested shrinkage of the top 1% of the predictions (training rows only, CPU).

    python fw_r35_t3_tail_shrink.py

Hypothesis (user brief of 2026-11-16 ~22:25): the largest predictions belong to whales who sometimes spend far less than usual, so
pulling the top of the prediction range back lowers the absolute error:
    y_new = tau + g * (y_hat - tau)    for y_hat > tau,    tau = the 99th percentile of the predictions,    g in [0.85, 1.00].
Reference = the pipeline read-out fully nested (the `reference` vector saved by fn_r33_t3_meta.py, NMAE 0.79457). For each fold,
tau and g come from the other four folds' nested predictions (g on a 0.01 grid by the total absolute error; tau = their 99th
percentile) and are applied to the held-out fold. Controls with the same nesting: tau at the 98th and 99.5th percentile, and g free
in [0.70, 1.15] (does the data ask for shrinkage or for expansion?). Also printed: the absolute error by prediction quantile.
Acceptance (user, 2026-11-16, Task 3): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0002.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401
from src import task3
from src.config import DATA_DIRS
from src.metrics import nmae
from src.modeling import make_folds

y = pd.read_csv(DATA_DIRS[3] / "train.csv")[task3.TARGET].values.astype(float)
ref = np.load("experiments/claude/outputs/r33/t3_meta_variants.npz")["reference"]
folds = make_folds(y, kind="regression")
print(f"{len(y)} training rows; nested reference NMAE {nmae(y, ref):.5f}; share of the total absolute error in the top 1% / 5% of the "
      f"predictions: {np.abs(y - ref)[ref > np.quantile(ref, .99)].sum() / np.abs(y - ref).sum():.3f} / "
      f"{np.abs(y - ref)[ref > np.quantile(ref, .95)].sum() / np.abs(y - ref).sum():.3f}")
top = ref > np.quantile(ref, 0.99)
print(f"top 1% of the predictions ({top.sum()} rows): mean prediction {ref[top].mean():,.0f}, mean target {y[top].mean():,.0f}, median "
      f"target / prediction {np.median(y[top] / ref[top]):.3f}, share below half the prediction {np.mean(y[top] < 0.5 * ref[top]):.3f}, "
      f"share above twice {np.mean(y[top] > 2 * ref[top]):.3f}; in-sample best g at the 99th percentile: "
      + ", ".join(f"g {g:.2f}: {nmae(y, np.where(top, np.quantile(ref, .99) + g * (ref - np.quantile(ref, .99)), ref)) - nmae(y, ref):+.5f}"
                  for g in (0.85, 0.90, 0.95, 1.05, 1.10)))


def run(q, grid):
    out, picks = ref.copy(), []
    for a, b in folds:
        tau = np.quantile(ref[a], q)
        m = ref[a] > tau
        err = {g: np.abs(y[a][m] - (tau + g * (ref[a][m] - tau))).sum() for g in grid}
        g = min(err, key=err.get)
        picks.append(g)
        out[b] = np.where(ref[b] > tau, tau + g * (ref[b] - tau), ref[b])
    per = [nmae(y[b], out[b]) - nmae(y[b], ref[b]) for _, b in folds]
    d = nmae(y, out) - nmae(y, ref)
    ok = d > 0 and sum(p > 0 for p in per) >= 3 and min(per) >= -0.0002
    return d, per, picks, ok


G_USER = np.round(np.arange(0.85, 1.0001, 0.01), 2)
G_FREE = np.round(np.arange(0.70, 1.1501, 0.01), 2)
for name, q, grid in (("top 1%, g in [0.85, 1.00] (the brief)", 0.99, G_USER), ("top 2%, g in [0.85, 1.00]", 0.98, G_USER),
                      ("top 0.5%, g in [0.85, 1.00]", 0.995, G_USER), ("top 1%, g free in [0.70, 1.15]", 0.99, G_FREE),
                      ("top 5%, g free in [0.70, 1.15]", 0.95, G_FREE)):
    d, per, picks, ok = run(q, grid)
    print(f"  {name:38s} delta {d:+.5f}  folds +{sum(p > 0 for p in per)}/5 ({' '.join(f'{p:+.5f}' for p in per)})  g picked {picks} "
          f"-> {'PASS' if ok else 'fail'}")
print("done")
