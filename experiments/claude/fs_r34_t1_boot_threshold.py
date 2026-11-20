"""Round 34, Task 1 item 2b: a bootstrap-smoothed F2 cut-off (training rows only, CPU).

    python fs_r34_t1_boot_threshold.py

The pipeline's cut-off for a fold is the single F2 optimum of the other four folds' out-of-fold level-2 probabilities: the argmax
of a step function, so it carries sampling noise (the per-fold optima of the v22 level 2 spread over about 0.14-0.19).
Candidates, each from 200 bootstrap resamples of the training folds' rows (same resamples for all three):
  A  mean of the 200 resample optima;
  B  median of the 200 resample optima;
  C  the cut-off maximising the mean F2 over the 200 resamples (a smoothed F2 curve; grid of 0.002 over 0.05-0.40).
Reference = the pipeline rule. Level-2 probabilities = the v22 level 2 (cache of r33_t1_common.base_probs), on the pipeline folds
and stacker split seeds 1, 2, 3. Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import numpy as np

import r33_t1_common as c
from r33_t1_common import SPLITS, f2, fold_of, labels_from, log, y

B = 200
GRID = np.arange(0.05, 0.4001, 0.002)
base_p = c.base_probs()
log("base level 2 ready")


def boot_cuts(yy, p, rng):
    """Resample optima (exact F2 optimum of each resample) and the mean F2 curve over the resamples on GRID."""
    o = np.argsort(-p)
    ys, ps = yy[o].astype(float), p[o]
    n = len(ps)
    pos_grid = np.searchsorted(-ps, -GRID, side="right")                  # rows with p >= cut, for each grid cut
    last = np.r_[ps[1:] != ps[:-1], True]
    opt, curve = np.zeros(B), np.zeros(len(GRID))
    for b in range(B):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)         # resampling counts of the sorted rows
        tp, k, npos = np.cumsum(w * ys), np.cumsum(w), float((w * ys).sum())
        f = 5 * tp / (4 * npos + k)
        opt[b] = ps[int(np.argmax(np.where(last & (w > 0), f, -1)))]
        idx = np.maximum(pos_grid - 1, 0)
        curve += np.where(pos_grid > 0, f[idx], 0.0)
    return opt, curve / B


names = ["A mean of 200 bootstrap optima", "B median of 200 bootstrap optima", "C argmax of the bootstrap-mean F2 curve"]
cand = {n: [] for n in names}
base_labels = []
for k, (label, fl) in enumerate(SPLITS):
    fo, p = fold_of(fl), base_p[k]
    base_labels.append(labels_from(p, fo))
    labs = {n: np.zeros(len(y), int) for n in names}
    info = []
    for i in range(5):
        tr_, va = fo != i, fo == i
        opt, curve = boot_cuts(y[tr_], p[tr_], np.random.default_rng(100 * k + i))
        cuts = (opt.mean(), np.median(opt), GRID[int(np.argmax(curve))])
        for n, t in zip(names, cuts):
            labs[n][va] = p[va] >= t
        ref_t = c.optimize_threshold(y[tr_], p[tr_], 2.0)[0]
        own_t = c.optimize_threshold(y[va], p[va], 2.0)[0]
        info.append(f"fold {i}: pipeline {ref_t:.3f}, mean {cuts[0]:.3f}, median {cuts[1]:.3f}, curve {cuts[2]:.3f}; bootstrap SD {opt.std():.3f}; "
                    f"the fold's own optimum {own_t:.3f}")
    for n in names:
        cand[n].append(labs[n])
    log(f"{label}: base {f2(y, base_labels[-1]):.5f}; " + "; ".join(f"{n.split(' ')[0]} {f2(y, labs[n]):.5f}" for n in names))
    print("   " + "\n   ".join(info), flush=True)
print()
for n in names:
    c.report(n, cand[n], base_labels)
log("done")
