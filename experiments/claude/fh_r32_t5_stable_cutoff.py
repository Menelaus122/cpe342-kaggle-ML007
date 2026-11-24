"""Round 32, Task 5 item 3: Monte Carlo variance of the pocket's cut-off (label-free; Task 5 has only test.csv).

    python fh_r32_t5_stable_cutoff.py [first-release test.csv for the reference run]

v24's pocket (cross-fitted 3-column density ratio, 20 random halvings, 300 shuffle-null draws) gave k* 263-288 over seeds (about
25 borderline accounts; Jaccard 0.913-1.000). Two sources of Monte Carlo noise: the cross-fitting splits (the score itself) and the
null draws (the est_TP curve, whose slope at the cut-off is low, so a small error moves k* far). task5.density_ratio_curve now
streams the splits (one pair of fitted maps at a time), so many more splits and null draws fit in memory.
Settings compared, 5 master seeds each (splits and null draws both change with the seed):
  reps 20 / null 300 (v24), reps 200 / null 300, reps 200 / null 1000.
Reported per setting: k* per seed, Jaccard of the flagged sets (min / mean), accounts flagged under every seed and under any seed,
expected F3 (common group size within each run), and on the first release (reference only) F3 against the id block.
Target set by the user: Jaccard > 0.995 between runs. The cheapest setting that reaches it becomes the pipeline default.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
FEATS = ["accuracy_std_level", "kda_std_level"]
SETTINGS = [(20, 300), (200, 300), (200, 1000)]
SEEDS = [42, 1, 2, 3, 4]


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    pocket0, pinfo0 = task5.pocket_detector(X, gap, FEATS)
    log(f"\n===== {label}: gap families {gap.sum()}, min-rank pocket {pocket0.sum()}")
    for reps, n_perm in SETTINGS:
        sets, ks, ef, ref = [], [], [], []
        for s in SEEDS:
            fl, info = task5.pocket_refine(X, gap, pocket0, pinfo0, FEATS, seed=s, reps=reps, n_perm=n_perm)
            sets.append(fl)
            ks.append(info["k"])
            ef.append(info["expected_f"])
            if truth is not None:
                f = gap | fl
                tp = np.sum(f & truth)
                ref.append(10 * tp / (10 * tp + 9 * np.sum(~f & truth) + np.sum(f & ~truth)))
        jac = [np.sum(a & b) / np.sum(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
        line = (f"reps {reps:3d}, null {n_perm:4d}: k* {ks}; Jaccard min {min(jac):.4f}, mean {np.mean(jac):.4f}; in every seed "
                f"{int(np.all(sets, 0).sum())}, in any {int(np.any(sets, 0).sum())}; expected F3 {min(ef):.4f}-{max(ef):.4f}")
        if truth is not None:
            line += f" | reference F3 {min(ref):.4f}-{max(ref):.4f} (mean {np.mean(ref):.4f})"
        log(line)


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1:
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
