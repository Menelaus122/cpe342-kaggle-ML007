# v20: public leaderboard 0.85948

Snapshot taken 2026-11-15 00:10.

v19 + DINOv2 ViT-B/14 fine-tuned at 126x238 (lr 5e-5, layer decay 0.8, 2 warm-up + 13 cosine epochs, HUD dropout) as an eighth Task 4 member, weighted three times in the probability average (round 28). Public score +0.00325 over v19.

submission.csv sha256: `5c5fe26efe805dcc79603133c4e775720c73497aa3fe8f6b2e6cbba9f130e360`

Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
