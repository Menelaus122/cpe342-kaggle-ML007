# v12: public leaderboard 0.76248

Snapshot taken 2026-11-11. v11 (0.76242) with one Task 3 value changed: predictions are capped at the training maximum of 300,000, so ANS14698 goes from 431,946.09 to 300,000 (`scripts/nb_part2.py` section 7.4; test `bv2_t3_followup.py` C1). Task 3 OOF 0.79423 -> 0.79434 (+0.00011, 95% CI [+0.00003, +0.00021], 4/5 folds, 7 OOF rows). Public score +0.00006, the size predicted if the row is in the public split. Tasks 1, 2, 4, 5 identical to v11 (Tasks 2 and 5 use player_id; README question #6).

submission.csv (= submission_v12_try.csv, the uploaded file) sha256: `de380b025cdfe095154ffdfe5ef8d57fd7ed0e48160b90d3ceb94241ca29c384`

submission_clean_pipeline.csv (the pipeline's own output, no player_id) sha256: `061cb9866cab7de5d203799a890834c6a4186790e38ec107bf90bd9f33620b28`

Code, notebooks, cached predictions, params and figures are the pipeline state that produced the Task 3 column.
The Task 4 frozen-feature files from round 11 (t4_feat_*.npz, t4_round11_probes.npz; experiments only) are not copied.
Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
