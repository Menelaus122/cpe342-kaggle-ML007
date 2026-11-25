"""Task 4 in sorted-filename order (user request, 2026-11-08).

Our Task 4 column maps ANS{i} to the i-th row of Task 4's test.csv (ids like scyd5lz, in unsorted order). Hypothesis for
the chance-level leaderboard score: the answer key lists the images in sorted filename order (s001rpz.jpg, s001six.jpg,
...). This file puts the SAME predictions in that order: ANS{j} = our prediction for the j-th image in sorted order.
Base = submission_v9_try.csv (last upload, 0.76242); only the task4 column is reordered, nothing is retrained.
Writes submission_t4_sorted_try.csv.
"""
import hashlib
import os

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)

D = "04_image_classification/_download/task4_image_classification"
test = pd.read_csv(f"{D}/test.csv")
ids = test.id.astype(str).tolist()
files = sorted(os.listdir(f"{D}/test"))
sorted_ids = [os.path.splitext(f)[0] for f in files]
assert len(ids) == len(set(ids)) == len(sorted_ids) == 25889 and set(ids) == set(sorted_ids)
assert sorted_ids == sorted(ids)                       # sorting ids == sorting filenames (all 7-char, lowercase)

base = pd.read_csv("submission_v9_try.csv")
pred_by_id = dict(zip(ids, base.task4.values))         # row i of test.csv <-> ANS{i} in our files
sub = base.copy()
sub["task4"] = np.array([pred_by_id[i] for i in sorted_ids], dtype=int)

assert sub.shape == (25889, 6) and sub.isna().sum().sum() == 0
assert set(sub.task4.unique()) <= {0, 1, 2, 3, 4}
assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
for c in ["id", "task1", "task2", "task3", "task5"]:
    assert sub[c].equals(base[c]), c
assert np.array_equal(np.bincount(sub.task4, minlength=5), np.bincount(base.task4, minlength=5))   # same predictions, new order
moved = float(np.mean(sub.task4.values != base.task4.values))
out = "submission_t4_sorted_try.csv"
sub.to_csv(out, index=False)
print(f"saved {out}: Task 4 reordered to sorted filenames; {moved:.1%} of rows now hold a different class than in v9; "
      f"other columns identical to v9 | sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
print("first rows: ANS00001 <- image", sorted_ids[0], "| ANS00002 <-", sorted_ids[1], "| ANS00003 <-", sorted_ids[2])
