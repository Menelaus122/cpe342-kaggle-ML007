"""Round 30, Task 5: are the 2,148 flags stable under subsampling? (label-free; Task 5 has only test.csv, by design)

    python el_r30_t5_stability.py

The pipeline: gap families (task5.gap_clusters on the raw columns + wrap-aware geo features; 1,809 accounts) + the dependence-pocket
detector outside them (task5.pocket_detector on the pocket features found by the blind pair scan; 339 accounts) = 2,148 flags.
Here, 20 random subsamples of 80% of the accounts (without replacement: a bootstrap with replacement would create zero-width gaps)
re-run the whole procedure: gap scan, then the pocket detector (pocket features fixed to the full-data scan's, n_perm 100) with the
subsample's own gap flags as `known`. For every account: the share of the subsamples containing it in which it is flagged.
A noise-driven flag would come and go; a planted group's member would be flagged in nearly every subsample.
"""
import time

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
df = task5.load()
X, _ = task5.build_features(df)
_, gm = task5.gap_clusters(df, X)
gap = gm.any(axis=1).values
scan, noise = task5.dependence_pockets(X, ~gap)
sig = scan[scan.z > noise]
feats = sorted(set(sig.a) | set(sig.b))
pocket, info = task5.pocket_detector(X, gap, feats)
flag = gap | pocket
print(f"full data: gap families {gap.sum()}, pocket {pocket.sum()} on {feats} (k {info['k']}), total {flag.sum()} [{time.time() - t0:.0f}s]")

R, frac = 20, 0.8
rng = np.random.default_rng(0)
seen, hit_gap, hit_any = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
sizes = []
for r in range(R):
    idx = np.sort(rng.choice(len(df), int(frac * len(df)), replace=False))
    d, x = df.iloc[idx].reset_index(drop=True), X.iloc[idx].reset_index(drop=True)
    _, m = task5.gap_clusters(d, x)
    g = m.any(axis=1).values if m.shape[1] else np.zeros(len(idx), bool)
    p, inf = task5.pocket_detector(x, g, feats, n_perm=100, seed=r)
    seen[idx] += 1
    hit_gap[idx] += g
    hit_any[idx] += g | p
    sizes.append((g.sum() / frac, p.sum() / frac, inf["k"]))
    print(f"  subsample {r}: gap {g.sum()} (x1/0.8 = {g.sum() / frac:.0f}), pocket {p.sum()} [{time.time() - t0:.0f}s]", flush=True)
rate_g, rate_a = hit_gap / np.maximum(seen, 1), hit_any / np.maximum(seen, 1)
ok = seen >= 5
print(f"\nscaled sizes over the subsamples: gap {np.mean([s[0] for s in sizes]):.0f} +- {np.std([s[0] for s in sizes]):.0f}, "
      f"pocket {np.mean([s[1] for s in sizes]):.0f} +- {np.std([s[1] for s in sizes]):.0f}")


def dist(name, v, m):
    q = [np.mean(v[m] >= t) for t in (0.95, 0.9, 0.75, 0.5)]
    print(f"  {name:38s} n {m.sum():5d}: flagged in >= 95% of its subsamples {q[0]:.3f}, >= 90% {q[1]:.3f}, >= 75% {q[2]:.3f}, >= 50% {q[3]:.3f}")


print("share of subsamples in which an account is flagged (accounts seen in >= 5 subsamples):")
dist("gap-family accounts (gap scan only)", rate_g, ok & gap)
dist("gap-family accounts (any flag)", rate_a, ok & gap)
dist("pocket accounts (any flag)", rate_a, ok & pocket)
dist("unflagged accounts (any flag)", rate_a, ok & ~flag)
unst = ok & flag & (rate_a < 0.5)
new = ok & ~flag & (rate_a >= 0.5)
print(f"flagged on the full data but in < 50% of subsamples: {unst.sum()} (gap {np.sum(unst & gap)}, pocket {np.sum(unst & pocket)}); "
      f"unflagged on the full data but flagged in >= 50%: {new.sum()}")
np.savez("experiments/claude/outputs/r30/t5_stability.npz", rate_gap=rate_g, rate_any=rate_a, seen=seen, gap=gap, pocket=pocket)
print(f"[{time.time() - t0:.0f}s]")
