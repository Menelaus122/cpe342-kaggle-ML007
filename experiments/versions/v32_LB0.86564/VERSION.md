# v32: public leaderboard 0.86564

Snapshot taken 2026-11-18 20:38.

v31 with Task 4 member weights re-balanced around DINOv3-L (x20 on every frame without the gate, older members x0.5: ConvNeXt-T / DeiT-III 0.5, HUD-dropout members 1.5; DINOv2-B 8, DINOv2-L 8 / 2 gated; probe 0.35; round 41, hg_r41_t4_reweight.py nested over 48 settings; 483 Task 4 labels change); Tasks 1, 2, 3, 5 identical to v31

submission.csv sha256: `52884708bd8df99056f584669a609bbc14fb89c3f93b043f79d5e2faad9e9edf`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
