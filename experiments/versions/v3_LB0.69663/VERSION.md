# v3 — public leaderboard 0.69663 (uploaded 2026-11-04)

Snapshot of the code, notebook, cached predictions and the exact submission file that scored 0.69663.

What this version contains vs. v2 (0.59610):
- Task 1: threshold tuned for the positive-class F2 defined on the instruction slides (tau = 0.163).
- Task 5: 1-D density-gap detector -> 6 gap-separated anomaly families (1,811 accounts flagged).
- Class columns written as plain integer codes.
- Tasks 2, 3, 4 predictions unchanged.

Roll back:
    copy versions/v3_LB0.69663/submission.csv          -> submit as-is
    copy versions/v3_LB0.69663/{src,scripts,outputs/*}  -> back into the project root to restore the code
Task 4 checkpoints (outputs/models, 1.2 GB) are not copied: the Task 4 test probabilities they produced are in outputs/preds/t4_*.npz.
