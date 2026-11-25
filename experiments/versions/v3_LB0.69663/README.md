# cpe342-kaggle

Karena Thailand Player Intelligence: CPE342 hackathon solution (5 tasks, one submission).

| # | Task | Model | Metric |
|---|------|-------|--------|
| 1 | Anti-cheat pre-filter | LightGBM / XGBoost / CatBoost / ExtraTrees / LogReg → LogReg stacker → F2 threshold moving | F2 |
| 2 | Player segmentation | GBDTs + LogReg on latent engagement/spending indices → stacker → class-wise multipliers | Macro F1 |
| 3 | 30-day spending | L1 (median) GBDTs, two-stage hurdle mixture-median, Tweedie → out-of-fold L1 blend | NMAE |
| 4 | Game title from screenshots | ConvNeXt-Tiny (CNN) + DeiT-III-S (ViT), 5-fold + full-data models, GPU corruption-matched augmentation, Mixup/CutMix, EMA, TTA | Macro F1 |
| 5 | Account security (unsupervised) | 1-D density-gap detector finds 6 gap-separated anomaly families (no hand-set thresholds); 6-detector score ensemble as cross-check | F3 |

## Layout
```
CPE342_Master_Pipeline.ipynb   10-section master notebook (EDA -> features -> CV -> tuning -> stacking -> SHAP -> submission)
src/                           config, metrics, CV engine, plotting theme, one module per task (task1.py ... task5.py)
scripts/train_task4.py         Task 4 K-fold GPU fine-tuning (caches OOF/test probabilities + checkpoints)
scripts/tune.py                Optuna study for one task/model (persists to outputs/optuna.db)
scripts/run_tuning.sh          the tuning schedule used
scripts/build_notebook.py      regenerates the notebook from scripts/nb_part*.py
outputs/                       cached predictions, tuned params, figures, logs, model checkpoints
submission.csv                 final submission (25,889 rows, ids ANS00001..ANS25889)
```
Task 4 images are downloaded from the Drive link in `04_image_classification/README.md` and extracted to
`04_image_classification/_download/task4_image_classification/`.

## Reproduce
```bash
python -m venv .venv && .venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --sys-prefix --name cpe342

.venv/Scripts/python -u scripts/train_task4.py --tag cnxt --epochs 10          # 5-fold validation, ~2 h on a laptop RTX 4070
.venv/Scripts/python -u scripts/train_task4.py --tag cnxt --epochs 10 --full   # final model on all images (Train+Val)
.venv/Scripts/python -u scripts/train_task4.py --model deit3_small_patch16_384.fb_in22k_ft_in1k --tag deit3 \
    --h 192 --w 384 --lr 1e-4 --epochs 10 [--full]                               # second backbone (ViT)
bash scripts/run_tuning.sh                                             # optional: notebook falls back to defaults
python scripts/build_notebook.py
JUPYTER_PATH=.venv/share/jupyter .venv/Scripts/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=cpe342 CPE342_Master_Pipeline.ipynb
```
## Competition rules
- Validation uses 5-fold CV on the training data, and final models are retrained on all labelled data ("combine Train+Val").
- Test data is only ever predicted. Task 5 has only an unlabeled test.csv, so it is modelled unsupervised as the brief intends.
- Pre-trained weights are used in Task 4 only. No AutoML (Optuna is used for tuning).
- Metrics follow the instruction slides: F2/F3 are the positive-class formulas (`FBETA_AVERAGE = "binary"` in `src/config.py`; only Task 4 is marked macro). Class columns are written as plain integer codes, as defined in the datacards.
- No leaderboard probing. Two conventions are ambiguous in the course material: the Task 4 label format, and macro vs positive-class F-beta. Both are questions for the TAs. `outputs/format_variants/` holds the same predictions in the other conventions, so the one matching their answer can be submitted.
