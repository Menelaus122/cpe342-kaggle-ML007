"""Freeze the current state into versions/<name>/ so a submission can always be rolled back.

Run it right after uploading a submission to Kaggle (one snapshot per upload):

    python scripts/snapshot_version.py v4 --lb 0.70123 --note "Task 5 union, Task 1 one-SE threshold"

Copies submission.csv, the notebooks, src/, scripts/ and the small outputs (cached predictions, tuned params,
format variants, figures, Optuna DB) and writes VERSION.md. Then commit and tag it:

    git add versions/<name> && git commit -m "Snapshot <name> (LB ...)" && git tag <name> && git push --tags
"""
import argparse
import datetime as dt
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["submission.csv", "CPE342_Master_Pipeline.ipynb", "README.md", "requirements.txt"]
DIRS = ["src", "scripts", "task1", "task2", "task3", "task4", "task5"]
OUT_DIRS = ["params", "preds", "format_variants", "figures", "submission_parts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="folder name, e.g. v4")
    ap.add_argument("--lb", default="not submitted", help="public leaderboard score of this submission")
    ap.add_argument("--note", default="", help="one line: what changed")
    args = ap.parse_args()

    dest = ROOT / "versions" / (args.name if args.lb == "not submitted" else f"{args.name}_LB{args.lb}")
    if dest.exists():
        raise SystemExit(f"{dest.relative_to(ROOT)} already exists")
    (dest / "outputs").mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints")
    for f in FILES:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, dest / f)
    for d in DIRS:
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, dest / d, ignore=ignore)
    for d in OUT_DIRS:
        if (ROOT / "outputs" / d).exists():
            shutil.copytree(ROOT / "outputs" / d, dest / "outputs" / d, ignore=ignore)
    if (ROOT / "outputs" / "optuna.db").exists():
        shutil.copy2(ROOT / "outputs" / "optuna.db", dest / "outputs" / "optuna.db")

    sha = hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest()
    (dest / "VERSION.md").write_text(
        f"# {args.name}: public leaderboard {args.lb}\n\n"
        f"Snapshot taken {dt.datetime.now():%Y-%m-%d %H:%M}.\n\n"
        f"{args.note}\n\n"
        f"submission.csv sha256: `{sha}`\n\n"
        "Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.\n"
        "Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.\n",
        encoding="utf-8")
    print(f"snapshot written to {dest.relative_to(ROOT)} (submission sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
