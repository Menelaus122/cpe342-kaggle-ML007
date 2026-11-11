"""Round 14: copy the Task 3 10-seed bag (by_r14_bag.py cv t3 / full t3) into the pipeline cache under the names scripts/nb_part2.py reads.

    python cb_r14_cache_t3_bag.py

Seed 42 is the original single-seed run and already cached (the bag script's seed-42 fits equal the cache exactly, checked here again).
Other seeds are stored as <name>_s<seed>. The notebook's cached(...) lambdas recompute exactly these fits if a file is missing.
"""
import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from by_r14_bag import BAG3, DIR, SEEDS  # noqa: E402
from src.utils import load_preds, save_preds  # noqa: E402

pairs = [("t3", "hurdle", "t3r_hurdle"), ("t3full", "hurdle", "t3r_hurdle_full")]
for nm, (_, _, _, key) in BAG3.items():
    pairs += [("t3", nm, key), ("t3full", nm, key + "_full")]

for stage, nm, key in pairs:
    z = np.load(DIR / f"{stage}_{nm}_42.npz")
    c = load_preds(key)
    diff = max(float(np.abs(z[k] - c[k]).max()) for k in z.files if k in c and z[k].ndim)
    assert diff == 0, f"seed-42 {stage} {nm} differs from cache {key}: {diff}"
    for s in SEEDS[1:]:
        z = np.load(DIR / f"{stage}_{nm}_{s}.npz")
        save_preds(f"{key}_s{s}", **{k: z[k] for k in z.files})
    print(f"{key}: seed 42 equals the cache; seeds {SEEDS[1:]} written as {key}_s<seed>")
