# CPE342 Kaggle: Karena Thailand Player Intelligence (team ML007)

Source code of team **ML007** for the CPE342 Machine Learning project (KMUTT, 1/2026): five machine-learning tasks combined in
one Kaggle submission.

- **Final submission:** `submission.csv` (version v41, public leaderboard **0.86928**).
- **Notebooks:** each task has one notebook that rebuilds its column of that file.
- **Technical report:** in [`report/`](report/).

## Repository layout
The final code follows the structure on slide 6 of the project instructions. Experiments and history are kept apart in
`experiments/`.

```
task1/task1.ipynb       Task 1  Anti-cheat pre-filter          binary classification       F2 (cheater)
task2/task2.ipynb       Task 2  Player segmentation            4-class classification      macro F1
task3/task3.ipynb       Task 3  Monthly spending               zero-inflated regression    NMAE score
task4/task4.ipynb       Task 4  Game title detection           5-class image classification macro F1
task4/README.md                 how each Task 4 network was trained (scripts and Colab notebooks)
task4/colab/                    Colab notebooks that trained the large Task 4 members
task5/task5.ipynb       Task 5  Account security               unsupervised anomaly detection F3 (anomaly)
src/                    shared code imported by the notebooks (config, metrics, CV engine, one module per task)
scripts/                make_submission.py (joins the five columns), Task 4 trainers, Optuna tuning, notebook builders
outputs/                cached out-of-fold / test predictions (preds/), tuned parameters (params/, optuna.db),
                        the five submission columns (submission_parts/), figures
01_… 05_…               competition data and datacards (Task 4 images: see "Data" below)
submission.csv          final submission (v41)
requirements.txt        Python packages (pinned)
report/                 technical report
experiments/            NOT part of the final code: about 250 experiment scripts and their findings, every
                        submission file, version snapshots, Colab trials (see experiments/README.md)
```

## Final models
| Task | Final model | OOF score |
|---|---|---|
| 1 Anti-cheat | LightGBM / XGBoost / CatBoost / ExtraTrees / LogReg on 49 engineered features, plus RF / LightGBM / XGBoost / CatBoost on 24 raw columns; 10-seed bags. Level 2: mean log-odds of a LogReg stacker and two small tree meta-models. Threshold τ tuned for $F_2$ | $F_2$ 0.8387 |
| 2 Segmentation | GBDTs + LogReg on latent engagement / spending indices, then a multinomial stacker. Averaged with a KDE naive Bayes; class multipliers chosen by a soft macro-$F_1$ search | macro $F_1$ 0.7577 |
| 3 Spending | Repaired features (exact row identities, imputed segment). L1 GBDTs, hurdle models with a ratio-to-history second stage, and Tweedie (10-seed bags). Exact LP blend for L1, zero gate, cap at the training maximum | NMAE score 0.7946 |
| 4 Game title | 14 fine-tuned image backbones (ConvNeXt, DeiT-III, EfficientNetV2, DINOv2, DINOv3 ViT-L / ViT-H+, C-RADIOv3) at input sizes close to the frames' 230×120 shape, HUD dropout, 5-fold + Train+Val refit. Plus a 20% probe on frozen DINOv3-7B features. Decision rule argmax p·π^−0.75 | macro $F_1$ 0.9956 |
| 5 Account security | Six gap-separated anomaly families + a dependence-pocket detector (cross-fitted density ratio against a shuffle null): 2,053 flagged accounts | no labels |

How each choice was made is explained in the notebooks and in the technical report.

## How to run

### 1. Environment
Tested with Python 3.14.6 on Windows 11. A GPU is not needed to run the notebooks (it is only needed to retrain the Task 4
networks).
```bash
git clone https://github.com/Menelaus122/cpe342-kaggle-ML007.git
cd cpe342-kaggle-ML007
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130   # or the CPU wheels
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name cpe342      # the notebooks use the "cpe342" kernel
```

### 2. Data
- **Tasks 1, 2, 3, 5 and the Task 4 CSV files** are in the repository (`01_…` to `05_…`). They are identical to the Kaggle
  download of the re-released data (12 Nov 2026).
- **Task 4 images** (1.3 GB) are not in git. Download them from the Drive link in
  [`04_image_classification/README.md`](04_image_classification/README.md). Extract them to
  `04_image_classification/_download/task4_image_classification/{train,test}/`.

### 3. Run the notebooks
Open `task1/task1.ipynb` … `task5/task5.ipynb` in Jupyter or VS Code and run all cells. Each notebook writes its column to
`outputs/submission_parts/task<k>.csv`. Then join the five columns into `submission.csv`:
```bash
python scripts/make_submission.py
```
The same from the command line:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=cpe342 task1/task1.ipynb          # ... task2 to task5
python scripts/make_submission.py
```

**Run time.**
- On the laptop used for the project, all five notebooks take about 42 minutes: Task 1 2.5, Task 2 2.5, Task 3 8,
  Task 4 2.5 and Task 5 27 minutes.
- The expensive models are not retrained: their out-of-fold and test predictions are cached in `outputs/preds/*.npz`, and the
  tuned hyper-parameters in `outputs/params/` and `outputs/optuna.db`.

**Reproducibility check (25 Nov 2026).** A fresh run of the five notebooks reproduces `submission.csv` exactly in Tasks 1, 3,
4 and 5. In Task 2, 25,888 of 25,889 rows match. The exception is ANS19723: it sits on the boundary between Casual and Social,
and floating-point differences in the Task 2 refit can move it. `submission.csv` in this repository is the file that was
submitted.

**Retraining a model.**
- Delete its cache file (for example `outputs/preds/t2_lgb.npz`) and the notebook trains it again.
- Task 3's 10-seed bags take about 1¾ hours from scratch.
- The Task 4 networks are trained outside the notebook, on a GPU: see [`task4/README.md`](task4/README.md).

**Windows notes.**
- If the console uses a Thai code page (cp874), set `PYTHONIOENCODING=utf-8`.
- If the `cpe342` kernel is not found, set `JUPYTER_PATH=.venv/share/jupyter`.

### How the notebooks are built
The five notebooks are generated from one set of cell definitions, so shared steps are identical across tasks:
- `scripts/nb_part1.py` and `scripts/nb_part2.py` hold the cells.
- `scripts/build_task_notebooks.py` writes `task1/` … `task5/`.

To change a notebook, edit those files, rebuild with `python scripts/build_task_notebooks.py`, and re-execute.

## Competition rules followed
- **Data:** only the provided training data. Every final model is refitted on all labelled rows (Train+Val).
- **Test set:** only ever predicted. No pseudo-labelling, no fine-tuning on test, no test statistics in validation or model
  choices (TA ruling of 15 Nov 2026).
  - Exception: Task 5 ships only an unlabelled `test.csv`, so its unsupervised detectors are fitted on it, as the brief
    intends.
- **Pre-trained models:** used for Task 4 only. Tasks 1, 2, 3 and 5 are trained from scratch.
- **No AutoML.** Optuna is used for hyper-parameter tuning.
- **No leaderboard probing.** Every candidate was judged on out-of-fold predictions with a rule fixed before the run; the
  leaderboard was never used to choose between variants.
- **Metrics:** as in the instruction slides. $F_2$ / $F_3$ are the positive-class formulas, Task 4 is macro $F_1$, and
  Task 3 is the NMAE score $1/(1+\sum|y-\hat y|/\sum y)$.

## Acknowledgements
The 24-raw-column Task 1 members follow last year's 1st-place public repository
([Saltywan/CPE342-KAGGLE](https://github.com/Saltywan/CPE342-KAGGLE)).
