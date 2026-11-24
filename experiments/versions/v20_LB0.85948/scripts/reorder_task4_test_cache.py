"""Put the cached Task 4 test predictions into the row order of the 2026-11-12 test.csv.

    python scripts/reorder_task4_test_cache.py

The TAs re-released Task 4's test.csv on 2026-11-12 (columns id = ANS00001.., file_name, label). It lists the same 25,889 test
images as the first release (byte-identical, CRC-checked) but in a different order, and row i is ANS{i}. Every model's test
predictions were cached in the first release's order, so retraining is not needed: each test row is moved to the row that now
holds its image. Each rewritten cache gets a `test_file_name` array, which the notebook checks against test.csv.

Running it twice is safe: caches that already carry the new order are skipped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

OLD = task4.DATA / "_release_2026-11-03" / "test.csv"

old = pd.read_csv(OLD)
_, test = task4.load_frames()
old_names = (old.id.astype(str) + ".jpg").values
new_names = np.array(test.file_name.tolist(), dtype="U")   # plain unicode, so np.load needs no pickle
assert len(old_names) == len(new_names) == 25_889 and set(old_names) == set(new_names)
pos = pd.Series(np.arange(len(old_names)), index=old_names)
perm = pos.loc[new_names].values           # new row i  <-  old row perm[i]
assert (old_names[perm] == new_names).all()
print(f"new row order = old rows {perm[:5].tolist()} ...; rows that stay in place: {int((perm == np.arange(len(perm))).sum())}")

for f in sorted(PRED_DIR.glob("t4_*.npz")):
    z = dict(np.load(f, allow_pickle=True))   # only our own cache files
    if "test_file_name" in z:
        assert (z["test_file_name"].astype(str) == new_names).all(), f"{f.name}: stored order differs from test.csv"
        if z["test_file_name"].dtype == object:   # first version of this script stored an object array
            z["test_file_name"] = new_names
            np.savez(f, **z)
            print(f"{f.name}: already in the new order; test_file_name re-stored as unicode")
        else:
            print(f"{f.name}: already in the new order, skipped")
        continue
    keys = [k for k, v in z.items() if (k == "test" or k.startswith("test_")) and v.ndim >= 1 and len(v) == len(perm)]
    for k in keys:
        z[k] = z[k][perm]
    z["test_file_name"] = new_names
    np.savez(f, **z)
    print(f"{f.name}: reordered {keys}")
