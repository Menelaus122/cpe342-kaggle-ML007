# v39: public leaderboard 0.86877

Snapshot taken 2026-11-22 02:33.

v38 + DINOv3 ViT-H+/16 192x384 member's full refit (0.5 x 5-fold + 0.5 x full, as every member; weights unchanged x40); 96 Task 4 labels vs v38 (29 gated); -0.00014 vs v38; best public remains v38 (0.86891). Note: preds/ pruned to 276 caches (excluding t4_feat_*); task4/task4.ipynb and outputs/submission_parts/task4.csv are still the v38 execution while P_v2 runs on the laptop GPU.

submission.csv sha256: `65bed49831716fbc4c4226548ed0eba16a9f59096c2102eed3760ee15192df9c`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
