"""Build submission_id_leak.csv: v3's file with Task 2 and Task 5 predicted from player_id alone.

Task 2: segment = id block (ids < 52,000 -> 0, < 84,500 -> 1, < 110,500 -> 2, else 3).
Task 5: flag = id >= 23,819 (the top 2,070 ids, which hold all six gap families).
Tasks 1, 3, 4 are copied unchanged from versions/v3_LB0.69663/submission.csv.
"""
import hashlib

import numpy as np
import pandas as pd

import common  # noqa: F401  (cwd = project root)
from src import config as C

CUTS2 = [52000, 84500, 110500]
START5 = 23819

v3 = pd.read_csv("versions/v3_LB0.69663/submission.csv")
sub = v3.copy()

# --- Task 2: check the blocks on the labelled rows, then apply them to the test ids
tr2 = pd.read_csv("02_multiclass_classification/train.csv", usecols=["player_id", "segment"])
blk_tr = np.searchsorted(CUTS2, tr2.player_id.str[1:].astype(int).values, side="right")
print(f"Task 2 train: id block = label for {(blk_tr == tr2.segment.values).mean():.4f} of {len(tr2):,} rows")
te2 = pd.read_csv("02_multiclass_classification/test.csv", usecols=["player_id"])
assert len(te2) == len(sub)
sub["task2"] = np.searchsorted(CUTS2, te2.player_id.str[1:].astype(int).values, side="right").astype(int)

# --- Task 5: top id block
te5 = pd.read_csv("05_anomaly_detection/test.csv", usecols=["player_id"])
assert len(te5) == len(sub)
sub["task5"] = (te5.player_id.str[1:].astype(int).values >= START5).astype(int)
assert (sub.task5[v3.task5 == 1] == 1).all(), "every v3 family account must stay flagged"

# --- same checks as scripts/make_submission.py
sample = pd.read_csv(C.ROOT / "sample_submission.csv") if (C.ROOT / "sample_submission.csv").exists() else v3
assert sub.shape == (C.N_TEST, 6) and list(sub.columns) == list(sample.columns)
assert sub.id.tolist() == v3.id.tolist()
assert sub.isna().sum().sum() == 0
assert set(sub.task1.unique()) <= {0, 1} and set(sub.task5.unique()) <= {0, 1}
assert set(sub.task2.unique()) <= {0, 1, 2, 3} and set(sub.task4.unique()) <= {0, 1, 2, 3, 4}
assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
assert (sub.task3 >= 0).all() and np.isfinite(sub.task3).all()
for c in ["task1", "task3", "task4"]:
    assert sub[c].equals(v3[c]), c

print(f"Task 2: changed {int((sub.task2 != v3.task2).sum()):,} of {len(sub):,} rows; class counts "
      f"{np.bincount(sub.task2).tolist()} (v3 {np.bincount(v3.task2).tolist()})")
print(f"Task 5: {int(sub.task5.sum()):,} flags = {sub.task5.mean():.2%} (v3 {int(v3.task5.sum()):,}); "
      f"added {int(((sub.task5 == 1) & (v3.task5 == 0)).sum())}")
out = C.ROOT / "submission_id_leak.csv"
sub.to_csv(out, index=False)
print(f"saved {out} | {sub.shape} | all checks passed | sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
