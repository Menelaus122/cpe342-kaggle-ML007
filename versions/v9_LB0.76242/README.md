# CPE342 Kaggle: Karena Thailand Player Intelligence (team ML007)

Our solution to the CPE342 (KMUTT, 1/2026) hackathon: **five machine-learning tasks, one Kaggle submission**.

| Deadline | What |
|---|---|
| **24 Nov 2026** | Kaggle competition closes (70% of the grade) |
| **30 Nov 2026** | Technical report (20%) + source code on GitHub, one folder per task (10%) |

## Current status

| # | Task | Type | Metric | Final recipe | OOF score |
|---|------|------|--------|--------------|-----------|
| 1 | Anti-cheat pre-filter | binary | $F_2$ (cheater) | LightGBM / XGBoost / CatBoost / ExtraTrees / LogReg → LogReg stacker → threshold τ = 0.163 (OOF optimum) | 0.8369 |
| 2 | Player segmentation | 4-class | $F_1$ (macro) | GBDTs + LogReg on latent engagement/spending indices → stacker → class-wise multipliers | 0.7549 |
| 3 | 30-day spending | regression | NMAE score | repaired features (exact row identities, imputed segment) → L1 GBDTs, hurdles with a ratio-to-history stage 2 (LightGBM / XGBoost / CatBoost), Tweedie → out-of-fold L1 blend (exact LP) → median zero gate | 0.7942 |
| 4 | Game title from screenshots | 5-class image | macro $F_1$ | ConvNeXt-Tiny + DeiT-III-S, 5-fold + full-data models, GPU augmentation, Mixup/CutMix, EMA, TTA | 0.9887 |
| 5 | Account security | unsupervised | $F_3$ (anomaly) | 6 gap-separated anomaly families (1,811 accounts); score ensemble and rules as cross-checks | no labels* |

\*Task 5 has no labels. A second gap scan of the derived change features finds no account outside the six families.

### Leaderboard log
Add a row for **every** upload (we have a daily limit), then snapshot it (see [Versions](#versions-and-roll-back)).

| Date | Version | Public LB | What changed |
|---|---|---|---|
| 2026-11-03 | v1 | 0.59593 | first full pipeline (float labels) |
| 2026-11-03 | v2 | 0.59610 | final models refit on Train+Val |
| 2026-11-04 | v3 | **0.69663** | Task 1 threshold for positive-class $F_2$; Task 5 gap families; integer class codes |
| 2026-11-04 | v4 | 0.69429 | Task 5 union (2,285 accounts); Task 1 τ 0.163 → 0.190. Lower than v3: both hedges rolled back |
| 2026-11-05 | v5 | 0.76 | **player_id leak test** (not in the pipeline): Task 2 = id block, Task 5 = top 8% of ids; Tasks 1/3/4 = v3. Allowed? TA question #6. Keep v3 as the final selection until they answer |
| 2026-11-06 | v6 | 0.76202 | v5 + Task 3 round 2 (missing spending values rebuilt from exact row identities, whale-only model). Uses player_id like v5 |
| 2026-11-06 | v7 | 0.76224 | v6 + Task 3 round 3 (ratio-to-history target, imputed segment; whale model dropped). Task 3 OOF 0.79331. Uses player_id like v5 |
| 2026-11-06 | v8 | **0.76242** | v7 + Task 3 tuning (Optuna ratio stage 2, XGB/Cat ratio members). Task 3 OOF 0.79403. Uses player_id like v5 |
| — | current `submission.csv` | not uploaded | clean pipeline (no player_id): Tasks 1, 2, 4 = v3; **Task 3 = the v8 recipe, merged into the pipeline on 2026-11-08** (`task3.build_repaired`, ratio stage 2, exact L1 blend); Task 5 = families + dependence pocket (2,191 flags = 1,811 families + 380; blind 2-D scan + permutation-null cut-off, `src/task5.py`) |

**Open questions for the TAs** (papit.vieng@kmutt.ac.th, phanita.siri@kmutt.ac.th):
1. Task 4 label format (integer codes as in the datacard, or text such as `fiveN` as on the slide) and whether row *i* of Task 4's `test.csv` maps to `ANS{i}`.
2. $F_2$ / $F_3$: positive class (slide formula) or macro?
3. Task weights in the leaderboard average.
4. Task 1's `train.csv` has 13,184 rows without a label. May they be used (semi-supervised), or should they be ignored as now?
5. Task 5: which behaviours count as anomalous (e.g. do persistently near-inactive or silent accounts count)?
6. `player_id` is not random. In Task 2 the segments sit in four id blocks (97% pure; using the id would raise OOF macro-F1 from 0.755 to 0.968). In Task 5 all anomalies sit in the top 8% of ids. May we use `player_id` as a feature? (Details: `experiments/claude/FINDINGS.md`.)

`outputs/format_variants/` already holds the current predictions in the other conventions, ready for whatever they answer.

---

## Quick start

### 1. Clone and create the environment (Python 3.14, NVIDIA GPU for Task 4)
```bash
git clone https://github.com/Menelaus122/cpe342-kaggle-ML007.git
cd cpe342-kaggle-ML007
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name cpe342        # the notebooks use the "cpe342" kernel
```

### 2. Data
* **Tasks 1, 2, 3, 5:** the CSVs are in the repo (`01_…`, `02_…`, `03_…`, `05_…`).
* **Task 4 images (1.3 GB, not in git):** download from the Drive link in `04_image_classification/README.md` and extract so that the layout is
  `04_image_classification/_download/task4_image_classification/{train,test,train.csv,test.csv}`.
* **Task 4 checkpoints (not in git):** only needed for the Grad-CAM cell. Download them from the GitHub release:
  ```bash
  gh release download t4-checkpoints -D outputs/models
  ```

### 3. Run
Everything expensive is cached in `outputs/preds/*.npz` (out-of-fold + test predictions of every model) and `outputs/optuna.db`, so a full run takes about 20 minutes and does not retrain Task 4.

```bash
# the five task notebooks (the course deliverable), then join their columns
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=cpe342 task1/task1.ipynb      # ... task2 to task5
python scripts/make_submission.py                                   # -> submission.csv

# or the master notebook (all five tasks in one file); it writes the identical submission.csv
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=cpe342 CPE342_Master_Pipeline.ipynb
```
Opening the notebooks in Jupyter / VS Code and running all cells works too.

Windows notes: set `PYTHONIOENCODING=utf-8` if the console uses a Thai code page (cp874), and, if the `cpe342` kernel is not found, set `JUPYTER_PATH=.venv/share/jupyter`.

---

## Repository layout
```
task1/task1.ipynb … task5/task5.ipynb   one notebook per task (course deliverable), 10 sections each:
                                        imports → loading → EDA → features → CV → tuning → final model
                                        → feature importance → submission column → conclusion
CPE342_Master_Pipeline.ipynb            the same cells for all five tasks in one notebook
src/                                    shared code: config, metrics, CV engine, plotting theme, one module per task
  config.py                             paths, seed, folds, FBETA_AVERAGE switch, Task 4 class order
  metrics.py                            exact competition metrics + threshold / class-weight tuning
  modeling.py                           K-fold runner for LightGBM / XGBoost / CatBoost / sklearn, Train+Val refit
  task1.py … task5.py                   loading + feature engineering (+ detectors for Task 5, training for Task 4)
scripts/
  nb_part1.py, nb_part2.py              ★ SOURCE OF TRUTH for all notebook cells
  build_notebook.py                     nb_part*.py → CPE342_Master_Pipeline.ipynb
  build_task_notebooks.py               nb_part*.py → task1/ … task5/ notebooks
  make_submission.py                    outputs/submission_parts/task*.csv → submission.csv (+ checks)
  snapshot_version.py                   freeze the current state into versions/<name>/
  train_task4.py                        Task 4 K-fold GPU fine-tuning (writes outputs/preds/t4_*.npz)
  tune.py, run_tuning.sh                Optuna studies (persist in outputs/optuna.db)
outputs/
  preds/                                cached OOF + test predictions (delete a file to retrain that model)
  params/, optuna.db                    tuned hyper-parameters
  submission_parts/                     one column per task, written by the task notebooks
  format_variants/                      the same predictions in other label/metric conventions
  figures/, logs/                       figures used in the notebooks/report; Task 4 training histories
versions/                               frozen earlier submissions (code + predictions + submission.csv)
experiments/<your-name>/                personal sandbox (see experiments/README.md)
01_… 05_…                               competition data and datacards
CPE342_Project_Instruction.pdf          official instructions (metrics on slides 15/23/25, format on slide 24)
```

---

## Working together

### Git workflow
`main` always holds a working pipeline. Work on a branch and merge through a pull request:
```bash
git switch main && git pull
git switch -c <your-name>/<topic>        # e.g. <your-name>/task2-features
# ... work, commit ...
git push -u origin <your-name>/<topic>
gh pr create --fill                       # the other person reviews and merges
```
* Put the **out-of-fold score before → after** in every PR that changes a model. We only merge changes that improve OOF (or that fix a bug).
* Keep PRs to one task where possible; then two people rarely touch the same file.
* Try ideas first in `experiments/<your-name>/`, then move what works into `src/` + `scripts/nb_part*.py`.

### Notebooks are generated: edit the sources, not the `.ipynb`
All notebooks are built from `scripts/nb_part1.py` and `scripts/nb_part2.py` (one `md(...)` / `code(...)` call per cell). This keeps the master and per-task notebooks identical and avoids JSON merge conflicts.
1. Change the logic in `src/` and/or the cell text in `scripts/nb_part*.py`.
2. Rebuild: `python scripts/build_notebook.py && python scripts/build_task_notebooks.py`.
3. Re-execute the affected notebooks (commands above) and commit the executed notebooks together with the source change.

If git reports a conflict in an `.ipynb`, don't merge it by hand: accept either side, then rebuild and re-execute.

### Cached predictions
`outputs/preds/<name>.npz` is reused whenever it exists. If you change a model or its features, **delete its cache file** (e.g. `outputs/preds/t2_lgb.npz`) so the notebook retrains it, and commit the new file. Task 4 caches come from `scripts/train_task4.py` (2–3 GPU hours per backbone).

### Submitting to Kaggle
1. Upload `submission.csv`. Only one person uploads per day: agree in chat first, because the team shares the daily limit.
2. Add the score to the **Leaderboard log** above.
3. Freeze it: `python scripts/snapshot_version.py v5 --lb 0.7xxx --note "what changed"`, then commit and `git tag v5 && git push --tags`.

### Versions and roll-back
| Folder | Public LB | Contents |
|---|---|---|
| `versions/v2_LB0.59610/` | 0.59610 | submission file only |
| `versions/v3_LB0.69663/` | 0.69663 | code, notebook, cached predictions, figures, exact submission file |
| `versions/v4_LB0.69429/` | 0.69429 | the same, for the Task 5 union / Task 1 τ = 0.190 variant |
| `versions/v5_LB0.76/` | 0.76 | submission file + the script that makes it (player_id leak test) |
| `versions/v6_LB0.76202/` | 0.76202 | submission file + scripts (Task 3 round 2 on the v5 file) |
| `versions/v7_LB0.76224/` | 0.76224 | submission file + scripts (Task 3 round 3 on the v5 file) |
| `versions/v8_LB0.76242/` | 0.76242 | submission file + scripts (Task 3 tuning on the v7 file) |
| `versions/pre_t3_merge_backup_2026-11-08/` | not uploaded | notebooks, submission parts and `submission.csv` just before Task 3 v8 was merged into the pipeline |

`submission_v9_try.csv` (not uploaded) = v8 with only Task 3 replaced by the merged pipeline's prediction (Task 3 OOF 0.79423 vs 0.79403). Expected change about +0.00004, below what the leaderboard can resolve; it makes the uploaded Task 3 column come from the code we hand in.

To roll back, upload that folder's `submission.csv`, or copy its `src/`, `scripts/` and `outputs/` over the project root.

---

## Competition rules (must hold for every change)
- **Data:** only the provided Train/Validation data. Combining Train+Val for the final models is required (§7.6 retrains every model on all labelled rows).
- **Test set:** only ever predicted. No pseudo-labelling or fine-tuning on test. Task 5 ships only an unlabeled `test.csv`, so its unsupervised detectors are fitted on it as the brief intends.
- **Pre-trained models:** allowed for Task 4 only. Tasks 1, 2, 3 and 5 are trained from scratch.
- **No AutoML** frameworks. Optuna for hyper-parameter tuning is fine.
- **No leaderboard probing:** never upload variants just to see which scores better, and respect the daily submission limit. Ambiguities go to the TAs ("When in doubt, ask first!").
- **Metrics** follow the instruction slides: $F_2$ / $F_3$ are the positive-class formulas (`FBETA_AVERAGE = "binary"` in `src/config.py`; only Task 4 is marked macro). Class columns are plain integer codes, as defined in the datacards.
