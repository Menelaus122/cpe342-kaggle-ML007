"""Join the five per-task parts (outputs/submission_parts/task{k}.csv, written by task{k}/task{k}.ipynb)
into submission.csv, with the same checks as the master notebook.

    python scripts/make_submission.py [--out submission.csv]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = ap.parse_args()

    sample = pd.read_csv(C.SAMPLE_SUB)
    sub = sample[["id"]].copy()
    for k in range(1, 6):
        path = C.OUT / "submission_parts" / f"task{k}.csv"
        if not path.exists():
            sys.exit(f"missing {path.relative_to(ROOT)}: run task{k}/task{k}.ipynb first")
        part = pd.read_csv(path)
        assert part.id.tolist() == sub.id.tolist(), f"task{k}: id order differs from sample_submission.csv"
        sub[f"task{k}"] = part[f"task{k}"].values

    expected_ids = [f"ANS{i:05d}" for i in range(1, C.N_TEST + 1)]
    assert sub.shape == (C.N_TEST, 6) and list(sub.columns) == list(sample.columns)
    assert sub.id.tolist() == expected_ids, "ID order mismatch"
    assert sub.isna().sum().sum() == 0, "NaNs in submission"
    assert set(sub.task1.unique()) <= {0, 1} and set(sub.task5.unique()) <= {0, 1}
    assert set(sub.task2.unique()) <= {0, 1, 2, 3} and set(sub.task4.unique()) <= {0, 1, 2, 3, 4}
    assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
    assert (sub.task3 >= 0).all() and np.isfinite(sub.task3).all()
    sub.to_csv(args.out, index=False)
    print(f"saved {args.out} | {sub.shape} | all checks passed")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
