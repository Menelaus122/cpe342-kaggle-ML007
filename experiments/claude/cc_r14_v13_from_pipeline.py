"""Round 14: final v13 files taken from the pipeline output, so the notebooks reproduce the upload exactly.

    python cc_r14_v13_from_pipeline.py

After the Task 3 bag was merged into scripts/nb_part2.py and the notebooks re-executed, submission.csv (clean pipeline) matches
ca_r14_build_v13.py up to the last cent (69 rows differ by 0.01 THB: the notebook averages seeds in a different floating-point order).
This script rewrites
  submission_v13.csv       = submission_v12_try.csv with the task3 field of every line replaced by the pipeline's task3 text,
  submission_clean_v13.csv = copy of the pipeline's submission.csv,
and prints the row-by-row diff against v12 for every task plus sha256.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)

ROOT = Path.cwd()
V12, PIPE = ROOT / "submission_v12_try.csv", ROOT / "submission.csv"
OUT_V13, OUT_CLEAN, PREV_CLEAN = ROOT / "submission_v13.csv", ROOT / "submission_clean_v13.csv", ROOT / "submission_clean_try.csv"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(V12) == "de380b025cdfe095154ffdfe5ef8d57fd7ed0e48160b90d3ceb94241ca29c384", "v12 file changed"

# the pipeline must differ from the step-by-step build by rounding only
pipe, built = pd.read_csv(PIPE), pd.read_csv(OUT_CLEAN)
d = (pipe.task3 - built.task3).abs()
print(f"pipeline vs ca_r14_build_v13.py: task3 rows differing {int((d > 0).sum())}, max |diff| {d.max():.4f} THB; other tasks differing "
      f"{ {c: int((pipe[c] != built[c]).sum()) for c in ['task1', 'task2', 'task4', 'task5']} }")
assert d.max() <= 0.0100001 and all((pipe[c] == built[c]).all() for c in ["task1", "task2", "task4", "task5"])


def lines_of(p):
    raw = p.read_bytes()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    return raw.split(eol), eol


lv, eol = lines_of(V12)
lp, _ = lines_of(PIPE)
hv, hp = lv[0].decode().split(","), lp[0].decode().split(",")
assert hv == hp
c3 = hv.index("task3")
n = len(pipe)
for i in range(1, n + 1):
    pv, pp = lv[i].decode().split(","), lp[i].decode().split(",")
    assert pv[0] == pp[0], f"id mismatch on line {i}"
    pv[c3] = pp[c3]
    lv[i] = ",".join(pv).encode()
OUT_V13.write_bytes(eol.join(lv))
OUT_CLEAN.write_bytes(PIPE.read_bytes())

a, b = pd.read_csv(V12), pd.read_csv(OUT_V13)
assert a.id.equals(b.id) and list(a.columns) == list(b.columns) and b.shape == (25_889, 6)
assert (b.task3 == pipe.task3).all()
print("\n=== submission_v13.csv vs v12, rows that differ per task ===")
for c in ["task1", "task2", "task3", "task4", "task5"]:
    print(f"  {c}: {int((a[c] != b[c]).sum()):,}")
dd = b.task3 - a.task3
nz = dd != 0
pos = (a.task3 > 0) & (b.task3 > 0)
print(f"Task 3 detail: rows changed {int(nz.sum()):,}; |change| mean {dd.abs().mean():.2f} THB, median over changed rows {dd[nz].abs().median():.2f}, "
      f"p99 {dd.abs().quantile(0.99):.2f}, max {dd.abs().max():.2f}")
print(f"  0 -> positive {int(((a.task3 == 0) & (b.task3 > 0)).sum())}, positive -> 0 {int(((a.task3 > 0) & (b.task3 == 0)).sum())}; "
      f"zero share {np.mean(a.task3 == 0):.4f} -> {np.mean(b.task3 == 0):.4f}")
print(f"  sum {a.task3.sum():,.2f} -> {b.task3.sum():,.2f} (ratio {b.task3.sum() / a.task3.sum():.5f}); max {a.task3.max():,.2f} -> {b.task3.max():,.2f}; "
      f"corr log1p {np.corrcoef(np.log1p(a.task3), np.log1p(b.task3))[0, 1]:.6f}")
print(f"  relative change on rows positive in both: median {np.median(np.abs(dd[pos]) / a.task3[pos]):.4f}, p95 {np.quantile(np.abs(dd[pos]) / a.task3[pos], 0.95):.4f}")
c0 = pd.read_csv(PREV_CLEAN)
print("submission_clean_v13.csv (= pipeline submission.csv) vs submission_clean_try.csv:", {c: int((c0[c] != pipe[c]).sum()) for c in c0.columns[1:]})
print("submission_clean_v13.csv vs submission_v13.csv:", {c: int((pipe[c] != b[c]).sum()) for c in c0.columns[1:]})
print(f"sha256 submission_v13.csv                          {sha(OUT_V13)}")
print(f"sha256 submission_clean_v13.csv = submission.csv   {sha(OUT_CLEAN)}  {sha(PIPE)}")
print(f"sha256 v12 (unchanged)                             {sha(V12)}")
