# v31: public leaderboard 0.86443

Snapshot taken 2026-11-18 20:22.

v30 + Task 4 DINOv3-L/16 member (144x272, lr 8e-5, layer decay 0.90, 18 epochs, HUD dropout 0.70, Colab A100) next to DINOv2-L with the same brightness/contrast gate (x8 normal, x2 dark frames; round 40; 302 Task 4 labels change; Task 4 class multipliers pinned off); Tasks 1, 2, 3, 5 identical to v30

submission.csv sha256: `19d62e487836f3320e23a4a6359118e1512c715d03a98c894f052df2e2c4ea11`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
