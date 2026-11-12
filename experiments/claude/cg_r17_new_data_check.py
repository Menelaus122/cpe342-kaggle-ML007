"""Round 17 (data re-release of 2026-11-12): check the pipeline's new submission.csv against the last file of the old release.

    python cg_r17_new_data_check.py

Expected:
  * Task 1 and Task 3: identical to submission_clean_v13.csv (their data did not change, the caches were kept).
  * Task 4: submission_clean_v13.csv's Task 4 labels moved to the new test.csv row order (matched through the image file name).
  * Tasks 2 and 5: new test rows, so only summary statistics are compared.
Prints sha256 of the new file.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)

ROOT = Path.cwd()
T4 = ROOT / "04_image_classification" / "_download" / "task4_image_classification"
new, old = pd.read_csv(ROOT / "submission.csv"), pd.read_csv(ROOT / "submission_clean_v13.csv")
assert new.shape == old.shape == (25_889, 6) and new.id.equals(old.id)

for c in ("task1", "task3"):
    print(f"{c}: rows differing from submission_clean_v13.csv = {int((new[c] != old[c]).sum())}")

t_old, t_new = pd.read_csv(T4 / "_release_2026-11-03" / "test.csv"), pd.read_csv(T4 / "test.csv")
old_by_file = pd.Series(old.task4.values, index=t_old.id.astype(str) + ".jpg")
expected4 = old_by_file.loc[t_new.file_name].values
print(f"task4: rows equal to v13's labels moved to the new order = {np.mean(new.task4.values == expected4):.5f}; "
      f"rows equal to v13 in the OLD position = {np.mean(new.task4.values == old.task4.values):.4f}")
print("task4 class mix new:", new.task4.value_counts(normalize=True).sort_index().round(4).to_dict())

print("task2 class mix new:", new.task2.value_counts(normalize=True).sort_index().round(4).to_dict(),
      "| old:", old.task2.value_counts(normalize=True).sort_index().round(4).to_dict())
print(f"task5 flagged: new {int(new.task5.sum()):,} ({new.task5.mean():.2%}) | old {int(old.task5.sum()):,} ({old.task5.mean():.2%})")
print(f"sha256 submission.csv {hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest()}")
