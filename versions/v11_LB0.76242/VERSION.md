# v11: public leaderboard 0.76242 (same as v9)

Snapshot taken 2026-11-11. v9 (0.76242) with only Task 1 replaced by the round-11 pipeline prediction: four extra stack
members built the way last year's 1st-place team built theirs (github.com/Saltywan/CPE342-KAGGLE): RandomForest, LightGBM,
XGBoost and CatBoost on 24 of the 31 raw columns with training-median imputation, each averaged over seeds 42 and 7
(`src/task1.py` C24 / build_c24, `scripts/nb_part2.py` §7.1 and §7.6). Task 1 nested OOF F2 0.8367 → 0.8383 (+0.0016,
95% CI [+0.0008, +0.0024], 5/5 folds better, test-like +0.0023; `bn_t1_combined.py`, `bo_t1_seedavg.py`). 387 Task 1 labels
differ from v9 (test flags 15,537 → 15,690). Same public score as v9: the expected ~+0.0003 is below the public split's noise.
Tasks 2, 3, 4, 5 identical to v9 (Tasks 2 and 5 use player_id; README question #6).

submission.csv (= submission_v11_try.csv, the uploaded file) sha256: `5727b53e4455858979fa522593468a9beec437e7e893f87473746a740c9d9999`

submission_clean_pipeline.csv (the pipeline's own output, no player_id) sha256: `418599d056ef95e613e5114ce8c0940ffbdee482af28f1f8b2411af3992e7343`

Code, notebooks, cached predictions, params and figures are the pipeline state that produced the Task 1 column.
The Task 4 frozen-feature files from round 11 (t4_feat_*.npz, t4_round11_probes.npz; experiments only) are not copied.
Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
