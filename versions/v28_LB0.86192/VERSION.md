# v28: public leaderboard 0.86192

Snapshot taken 2026-11-17 14:18.

v27 + Task 4 MLP probe on frozen features of seven backbones (adds SigLIP2-SO400M, EVA-02-L, ConvNeXt-L CLIP) at 20% (round 38 S3; 259 Task 4 labels change); Tasks 1, 2, 3, 5 identical to v27

submission.csv sha256: `d725001f81ebb4661ebe5136ab83f0365f06db0ea442f7090312f59d83f7ef03`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
