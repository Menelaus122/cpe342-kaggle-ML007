# v41: public leaderboard 0.86928

Snapshot taken 2026-11-22 14:45.

v39 base + DINOv3-7B 144x272 multi-layer Mixup probe P_7b @ 0.20 (5-fold + full refit) replacing P7 @ 0.30; 210 Task 4 labels vs v39 (52 gated); +0.00051 over v39, +0.00037 over v38 (best Public LB)

submission.csv sha256: `623ed6dc8a0165e5481ee57d53fe4331d946cfe7b02ba7fd190c286763163a2c`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
