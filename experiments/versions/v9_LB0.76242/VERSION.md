# v9: public leaderboard 0.76242 (same as v8)

Snapshot taken 2026-11-08. v8 (0.76242) with only Task 3 replaced by the prediction of the clean pipeline, after the v8 Task 3
recipe was merged into `src/task3.py` and the notebooks (identity fill, imputed segment, ratio stage 2 with LightGBM / XGBoost /
CatBoost, Optuna-tuned stage 1 and ratio stage 2, exact LP L1 blend). Task 3 OOF 0.79423 (v8 script 0.79403). Same public score
as v8, as expected (predicted change about +0.00004). Tasks 1, 2, 4, 5 identical to v8 (Tasks 2 and 5 use player_id; README
question #6).

submission.csv (= submission_v9_try.csv, the uploaded file) sha256: `6ae78310f3d2d32041553e49bbe78a6579be9afe8bf84b8361a82ded95312500`

submission_clean_pipeline.csv (the pipeline's own output, no player_id) sha256: `8a04601cd12799fe419807c8580aa4d9a13257bc21d28098a3ae7778156c5620`

Code, notebooks, cached predictions, params and figures are the pipeline state that produced the Task 3 column.
Roll back: submit this folder's submission.csv, or copy its src/, scripts/ and outputs/ over the project root.
Task 4 checkpoints are not copied; their test probabilities are in outputs/preds/t4_*.npz.
