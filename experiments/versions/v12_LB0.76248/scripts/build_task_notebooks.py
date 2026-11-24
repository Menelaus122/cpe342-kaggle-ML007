"""Build one self-contained notebook per task (task1/task1.ipynb ... task5/task5.ipynb).

The cells come from the same definitions as the master notebook (scripts/nb_part1.py, nb_part2.py), so the
per-task notebooks and CPE342_Master_Pipeline.ipynb always run identical code and produce identical predictions.
Cells that cover several tasks in the master notebook (loading, folds, tuning, Train+Val refit, submission)
are rebuilt here for one task. Each notebook follows the same 10-section layout and ends by writing its own
column to outputs/submission_parts/task{k}.csv; scripts/make_submission.py joins the five parts.

    python scripts/build_task_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cpe342 task1/task1.ipynb
"""
import re
from pathlib import Path

import nbformat

from build_notebook import load_cells

ROOT = Path(__file__).resolve().parents[1]
CELLS = load_cells("nb_part1") + load_cells("nb_part2")

TASKS = {
    1: dict(name="Anti-Cheat Pre-Filter", kind="Binary classification", target="is_cheater", metric="$F_2$ (positive class: cheater)"),
    2: dict(name="Player Segmentation", kind="4-class classification", target="segment", metric="$F_1$ (macro over the 4 segments)"),
    3: dict(name="Monthly Spending", kind="Zero-inflated regression", target="spending_30d", metric="NMAE score $1/(1+\\sum|y-\\hat y|/\\sum y)$"),
    4: dict(name="Game Title Detection", kind="5-class image classification", target="class", metric="Macro $F_1$"),
    5: dict(name="Account Security", kind="Unsupervised anomaly detection", target="is_anomaly", metric="$F_3$ (positive class: anomaly)"),
}
TOC = ["Imports & Configuration", "Data Loading & Overview", "Exploratory Data Analysis", "Data Preparation & Feature Engineering",
       "Model Training & Evaluation", "Hyperparameter Tuning", "Stacking Ensemble (K-Fold) & Final Decision",
       "Feature Importance Analysis", "Submission", "Conclusion"]


# ------------------------------------------------------------------ helpers
def pick(substr, kind=None):
    hits = [src for k, src in CELLS if substr in src and (kind is None or k == kind)]
    assert len(hits) == 1, f"{len(hits)} cells match {substr!r}"
    return hits[0]


def _tasks_in(text):
    m = re.search(r"Tasks?\s+([\d][\d ,and]*)", text)
    return {int(d) for d in re.findall(r"\d", m.group(1))} if m else None


def md_for_task(src, k, lead=None):
    """Keep the parts of a master markdown cell that belong to task k.
    * H1 lines and anchors are dropped (each notebook writes its own section headers);
    * a heading that names other tasks starts a block that is dropped, up to the next heading;
    * text before the first heading belongs to `lead` (None = generic, always kept);
    * bullets that start with "Task j" / "Tasks i and j" are kept only if they name k."""
    out, owner = [], lead
    for line in src.split("\n"):
        if line.startswith("<a id=") or re.match(r"^# ", line):
            continue
        if line.startswith("#"):
            t = _tasks_in(line)
            owner = None if t is None else (k if k in t else -1)
        if owner is not None and owner != k:
            continue
        b = re.match(r"^\s*[*-] (\*\*)?Tasks? [\d][^:]*:", line)
        if b and k not in (_tasks_in(line) or {k}):
            continue
        out.append(line)
    text = "\n".join(out).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def section(n, extra=""):
    return f'<a id="{n}"></a>\n# {n}. {TOC[n - 1]}' + (f"\n\n{extra}" if extra else "")


def split_refit(k):
    """Header + the `# ---- Task k` block of the master Train+Val refit cell."""
    src = pick("from src.modeling import refit_full, refit_full_sklearn")
    head, rest = src.split("# ---- Task 1", 1)
    blocks = dict(re.findall(r"# ---- Task (\d)\n(.*?)(?=\n# ---- Task \d|\ndisplay\(pd\.DataFrame\(changes)", "# ---- Task 1" + rest, re.S))
    body = blocks[str(k)].rstrip()
    return (f"{head}# ---- Task {k}\n{body}\n\n"
            f'display(pd.DataFrame(changes, columns=["task", "check", "value", "note"]).set_index("task").round(4))\n'
            f"pred{k} = pred{k}_final   # used by section 9")


SETUP = pick("import os, sys, json, time, warnings, subprocess").replace(
    'ROOT = Path.cwd()\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))',
    'ROOT = Path.cwd()\nif not (ROOT / "src").exists():          # notebook lives in taskN/; the shared code is one level up\n'
    '    ROOT = ROOT.parent\nos.chdir(ROOT)\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))')
assert "os.chdir(ROOT)" in SETUP
METRIC_CHECK = pick("# Sanity-check the metric implementations on toy inputs")
DATACARD = pick("**Data card summary**")

LOAD = {
    1: 't1_train, t1_test, t1_unlabeled = task1.load()\nprint(f"Task 1: dropped {t1_unlabeled:,} training rows whose label is missing")\n'
       'reduce_mem_usage(t1_train, verbose=True)\n'
       'display(pd.DataFrame({"train rows": [len(t1_train)], "test rows": [len(t1_test)], "features": [t1_test.shape[1] - 1],\n'
       '                      "cheater share": [t1_train.is_cheater.mean()],\n'
       '                      "missing (train / test)": [f"{t1_train.drop(columns=\'is_cheater\').isna().mean().mean():.3f} / {t1_test.isna().mean().mean():.3f}"]}))',
    2: 't2_train, t2_test = task2.load()\nreduce_mem_usage(t2_train, verbose=True)\nseg_names = ["Casual", "Grinder", "Social", "Whale"]\n'
       'display(pd.DataFrame({"train rows": [len(t2_train)], "test rows": [len(t2_test)], "features": [t2_test.shape[1] - 1],\n'
       '                      "missing rate": [t2_train.drop(columns="segment").isna().mean().mean()]}))',
    3: 't3_train, t3_test = task3.load()\nreduce_mem_usage(t3_train, verbose=True)\nseg_names = ["Casual", "Grinder", "Social", "Whale"]\n'
       'display(pd.DataFrame({"train rows": [len(t3_train)], "test rows": [len(t3_test)], "features": [t3_test.shape[1] - 1],\n'
       '                      "zero share": [(t3_train.spending_30d == 0).mean()], "missing rate": [t3_train.drop(columns="spending_30d").isna().mean().mean()]}))',
    4: 't4_train, t4_test = task4.load_frames()\n'
       'display(pd.DataFrame({"train images": [len(t4_train)], "test images": [len(t4_test)], "size": ["230x120 RGB"], "classes": [len(C.T4_CLASSES)]}))',
    5: 't5_test = task5.load()\nreduce_mem_usage(t5_test, verbose=True)\n'
       'display(pd.DataFrame({"accounts (test only, no labels)": [len(t5_test)], "columns": [t5_test.shape[1] - 1],\n'
       '                      "missing rate": [t5_test.isna().mean().mean()]}))',
}
TEST_FRAME = {1: "t1_test", 2: "t2_test", 3: "t3_test", 4: "t4_test", 5: "t5_test"}

_cv = pick("def cached(name, fn, force=False):")
_CV_TAIL = _cv[_cv.index("def cached(name, fn, force=False):"):]
CV_SETUP = {
    k: "from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score\n"
       "from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier\nfrom sklearn.linear_model import LogisticRegression\n"
       "from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
       "from sklearn.preprocessing import QuantileTransformer\n\n"
       + {1: "folds1 = make_folds(y1)\nprint(\"fold sizes:\", [len(v) for _, v in folds1])\n",
          2: "folds2 = make_folds(y2)\nprint(\"fold sizes:\", [len(v) for _, v in folds2])\n",
          3: "folds3 = make_folds(y3, kind=\"regression\")\nprint(\"zero share per fold:\", [round((y3[v] == 0).mean(), 3) for _, v in folds3])\n",
          4: "", 5: ""}[k]
       + "\n" + _CV_TAIL
    for k in TASKS
}

ABLATION = {
    1: 'raw1 = [c for c in t1_train.columns if c not in ("player_id", "is_cheater")]\nb = {}\n'
       'b["T1 raw"] = cached("t1_base_raw", lambda: run_cv("lgb", "binary", BASE, X1[raw1], y1, X1_test[raw1], folds1, name="T1 raw"))\n'
       'b["T1 FE"] = cached("t1_base_fe", lambda: run_cv("lgb", "binary", BASE, X1, y1, X1_test, folds1, name="T1 FE"))\nrows = []\n'
       'for k in ("T1 raw", "T1 FE"):\n    thr, s = optimize_threshold(y1, b[k]["oof"], 2.0)\n'
       '    rows.append([k, f"{AVG_NAME} F2 @ OOF-tuned threshold", s, f"AUC {roc_auc_score(y1, b[k][\'oof\']):.4f}, thr {thr:.3f}"])\n'
       'display(pd.DataFrame(rows, columns=["experiment", "metric", "OOF score", "note"]))',
    2: 'raw2 = [c for c in X2.columns if c in t2_train.columns]                     # original columns, noise removed\nb = {}\n'
       'b["T2 raw"] = cached("t2_base_raw", lambda: run_cv("lgb", "multiclass", BASE, X2[raw2], y2, X2_test[raw2], folds2, n_classes=4, name="T2 raw"))\n'
       'b["T2 FE"] = cached("t2_base_fe", lambda: run_cv("lgb", "multiclass", BASE, X2, y2, X2_test, folds2, n_classes=4, name="T2 FE"))\n'
       'b["T2 FE+TE"] = cached("t2_base_fe_te", lambda: run_cv("lgb", "multiclass", BASE, X2, y2, X2_test, folds2, n_classes=4,\n'
       '                                                       fold_fn=task2.te_fold_fn, name="T2 FE+TE"))\n'
       'display(pd.DataFrame([[k, "macro-F1 (argmax)", macro_f1(y2, b[k]["oof"].argmax(1))] for k in b], columns=["experiment", "metric", "OOF score"]))',
    3: 'raw3 = [c for c in t3_train.columns if c not in ("player_id", "spending_30d")]\nb = {}\n'
       'b["T3 raw (L1 log1p)"] = cached("t3r_base_raw_log", lambda: run_cv("lgb", "regression", BASE, X3[raw3], np.log1p(y3), X3_test[raw3], folds3, name="T3 raw log"))\n'
       'b["T3 FE (L1 log1p)"] = cached("t3r_base_fe_log", lambda: run_cv("lgb", "regression", BASE, X3, np.log1p(y3), X3_test, folds3, name="T3 FE log"))\n'
       'b["T3 FE (L1 raw THB)"] = cached("t3r_base_fe_raw", lambda: run_cv("lgb", "regression", BASE, X3, y3, X3_test, folds3, name="T3 FE raw"))\n'
       'display(pd.DataFrame([[k, "NMAE", nmae(y3, np.clip(b[k]["oof"] if "raw THB" in k else np.expm1(b[k]["oof"]), 0, None))] for k in b],\n'
       '                     columns=["experiment", "metric", "OOF score"]))',
}

AUDIT = {
    1: 'from src import audit\nwith pd.option_context("display.max_colwidth", 120):\n'
       '    display(pd.Series(audit.task_audit(t1_train, t1_test, "is_cheater"), name="Task 1").to_frame())',
    2: 'from src import audit\nwith pd.option_context("display.max_colwidth", 120):\n'
       '    display(pd.Series(audit.task_audit(t2_train, t2_test, "segment"), name="Task 2").to_frame())',
    3: 'from src import audit\nwith pd.option_context("display.max_colwidth", 120):\n'
       '    display(pd.Series(audit.task_audit(t3_train, t3_test, "spending_30d", regression=True), name="Task 3").to_frame())',
    4: 'from src import audit\n'
       'display(pd.DataFrame(audit.image_audit(task4.DATA, {"train": t4_train, "test": t4_test})).T.rename_axis("Task 4 images"))',
    5: 'from src import audit\nwith pd.option_context("display.max_colwidth", 200):\n'
       '    display(pd.Series(audit.task_audit(None, t5_test), name="Task 5").to_frame())',
}

TUNE_PARAMS = {
    1: 'spw = (y1 == 0).sum() / (y1 == 1).sum()\nP1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})\n'
       'P1_cat = tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})\n'
       'P1_xgb = {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw}\n'
       'shown = [("P1_lgb", P1_lgb), ("P1_cat", P1_cat), ("P1_xgb", P1_xgb)]',
    2: 'P2_lgb = tuned("task2_lgb", {"learning_rate": 0.05})\nP2_cat = tuned("task2_cat", {"learning_rate": 0.08, "depth": 6})\n'
       'P2_xgb = {"learning_rate": 0.04, "max_depth": 5, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.5}\n'
       'shown = [("P2_lgb", P2_lgb), ("P2_cat", P2_cat), ("P2_xgb", P2_xgb)]',
    3: 'P3_lgb = tuned("task3_lgb", {"learning_rate": 0.05})\nP3_xgb = tuned("task3_xgb", {"learning_rate": 0.05, "max_depth": 7})\n'
       'shown = [("P3_lgb", P3_lgb), ("P3_xgb", P3_xgb)]',
}
STUDY_N = {1: 40, 2: 15, 3: 30}


def tuning_cells(k):
    tuned_def = pick("def tuned(name, fallback):").split("\nspw = ", 1)[0]
    study = (pick('STORAGE = f"sqlite:///')
             .replace("STUDIES = [(1, \"lgb\", 40), (3, \"lgb\", 30), (2, \"lgb\", 15)]", f"STUDIES = [({k}, \"lgb\", {STUDY_N[k]})]"))
    assert f"STUDIES = [({k}," in study
    plot = pick("lgb_studies = [s for s in").replace('("task1_lgb", "task2_lgb", "task3_lgb")', f'("task{k}_lgb",)')
    params = (tuned_def + "\n\n" + TUNE_PARAMS[k] + "\nfor name, v in shown:\n"
              "    print(f\"{name}: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})}\")")
    return [("code", study), ("code", plot), ("code", params)]


SAVE_PART = r"""
sample_sub = pd.read_csv(C.SAMPLE_SUB)
part = pd.DataFrame({{"id": sample_sub.id, "task{k}": {expr}}})
assert len(part) == C.N_TEST == len({test}) and part.id.tolist() == [f"ANS{{i:05d}}" for i in range(1, C.N_TEST + 1)]
assert part["task{k}"].notna().all(){check}
part_dir = C.OUT / "submission_parts"
part_dir.mkdir(exist_ok=True)
part.to_csv(part_dir / "task{k}.csv", index=False)
print(f"saved {{part_dir / 'task{k}.csv'}} | {{part.shape}}")
display(part.head())
{describe}
"""
SAVE = {
    1: dict(expr="pred1.astype(int)", check=" and set(part.task1.unique()) <= {0, 1}",
            describe='print(f"flagged as cheaters: {part.task1.mean():.2%} (train prevalence {y1.mean():.2%}) at τ={THR1:.3f}")'),
    2: dict(expr="pred2.astype(int)", check=" and set(part.task2.unique()) <= {0, 1, 2, 3}",
            describe='print("predicted segment shares:", dict(zip(seg_names, np.round(np.bincount(pred2, minlength=4) / len(pred2), 3))))'),
    3: dict(expr="np.round(pred3, 2)", check=" and (part.task3 >= 0).all()",
            describe='print(f"zero share {(part.task3 == 0).mean():.3f} | median {part.task3.median():,.0f} | mean {part.task3.mean():,.0f} THB")'),
    4: dict(expr="pred4.astype(int)", check=" and set(part.task4.unique()) <= {0, 1, 2, 3, 4}",
            describe='print("predicted class shares:", dict(zip(C.T4_CLASSES, np.round(np.bincount(pred4, minlength=5) / len(pred4), 3))))'),
    5: dict(expr="pred5.astype(int)", check=" and set(part.task5.unique()) <= {0, 1}",
            describe='print(f"flagged accounts: {part.task5.sum():,} ({part.task5.mean():.2%})")'),
}

RECAP = {k: next(line for line in pick("### Methodology recap").split("\n") if line.startswith(f"| {k} ")) for k in TASKS}


# ------------------------------------------------------------------ notebook bodies
def task_cells(k):
    t = TASKS[k]
    cells = []
    md = lambda s: cells.append(("markdown", s))
    code = lambda s: cells.append(("code", s))

    # title
    toc = "\n".join(f"{i}. [{name}](#{i})" for i, name in enumerate(TOC, 1))
    md(f"# Task {k}: {t['name']}\nCPE342 Karena Thailand Player Intelligence. {t['kind']}, target `{t['target']}`, metric {t['metric']}.\n\n"
       f"This notebook is the Task {k} slice of `CPE342_Master_Pipeline.ipynb`: the same cells, the same shared code in `src/` and the same "
       f"cached artefacts in `outputs/`, so both produce identical predictions. It ends by writing `outputs/submission_parts/task{k}.csv`; "
       f"`python scripts/make_submission.py` joins the five parts into `submission.csv`.\n\n"
       "**Rules followed.** Only the provided data are used; test rows are only ever predicted"
       + (" (Task 5 ships only an unlabeled `test.csv`, so its unsupervised detectors are fitted on it as the brief intends)" if k == 5 else "")
       + "; every score and decision comes from 5-fold out-of-fold validation" + (", and the final models are retrained on Train+Val combined" if k < 5 else "")
       + "; " + ("ImageNet-pretrained weights are used, which the rules allow for Task 4 only" if k == 4 else "models are trained from scratch")
       + "; no AutoML" + (" (Optuna for tuning)" if k <= 3 else "") + "; no leaderboard probing.\n\n## Table of Contents\n" + toc)

    # 1. imports
    md(section(1, "Global seeds, warning suppression, GPU discovery, and the exact competition metric, checked on toy inputs."))
    code(SETUP)
    code(METRIC_CHECK)

    # 2. loading
    bullet = next(line for line in DATACARD.split("\n") if line.startswith(f"* **Task {k}**"))
    md(section(2, "Row *i* of `test.csv` maps positionally to submission id `ANS{i:05d}` (25,889 rows).\n\n**Data card.**\n" + bullet))
    code(f"from src import task{k}\n" + LOAD[k])

    # 3. EDA
    md(section(3))
    if k == 1:
        code(pick('counts = pd.Series({"Legitimate"')); code(pick("top6 = d.abs()"))
        md(md_for_task(pick("Every top feature is *lower* for cheaters"), 1, lead=1))
    elif k == 2:
        code(pick('vc = t2_train.segment.value_counts()')); code(pick('cats = ["loyalty_tier"'))
        md(md_for_task(pick("sit on a clean **2×2 latent design**"), 2, lead=2))
    elif k == 3:
        code(pick("y3 = t3_train.spending_30d"))
        md(md_for_task(pick("The target is **zero-inflated**"), 3, lead=3))
    elif k == 4:
        code(pick("fig, axes = plt.subplots(5, 6"))
        md(md_for_task(pick("Every image is exactly **230×120"), 4, lead=4))
    else:
        code(pick("groups5 = [g for g in task5.temporal_groups")); code(pick('lat = t5_test[[f"login_lat_{i}"'))
        md(md_for_task(pick("For every behaviour, the deviation of **period 4"), 5, lead=5))

    # 3.6 data-quality audit (same checks as the master notebook, this task only)
    md("### Data-quality audit\nAdversarial validation (train vs test), missing values vs target, impossible values, duplicates and "
       "collinearity, computed by `src/audit.py` on features only (see the master notebook §3.6).")
    code(AUDIT[k])
    md(md_for_task(pick("**Audit findings.**"), k))

    # 4. feature engineering
    fe_md = {1: "## 4.1 Task 1", 2: "## 4.2 Task 2", 3: "## 4.3 Task 3", 4: "## 4.4 Task 4", 5: "## 4.5 Task 5"}[k]
    fe_code = {1: "X1, y1, X1_test = task1.build", 2: "X2, y2, X2_test = task2.build", 3: "X3, y3, X3_test, repair3 = task3.build_repaired",
               4: "aug = task4.GPUAugment()", 5: "X5, dev5 = task5.build_features"}[k]
    md(section(4, md_for_task(pick(fe_md, "markdown"), k)))
    code(pick(fe_code))

    # 5. training & evaluation
    proto = {1: "`StratifiedKFold` on the label", 2: "`StratifiedKFold` on the segment",
             3: "`StratifiedKFold` on a zero / positive-quantile bin of `spending_30d`, so every fold has the same share of non-spenders and whales",
             4: "`StratifiedKFold` on the game label (`scripts/train_task4.py`)", 5: None}[k]
    if k <= 4:
        md(section(5, f"**Validation protocol.** 5 folds (seed 42), {proto}. Every model produces out-of-fold (OOF) predictions for every "
                      "training row: they give honest scores, feed threshold/weight tuning and serve as stacking inputs. Test predictions "
                      "average the 5 fold models. Results are cached in `outputs/preds/*.npz`; delete a file to retrain that model."))
    else:
        md(section(5, "Task 5 has no labels, so there is no train/validation split. The detectors are fitted on the unlabeled accounts, "
                      "and their agreement with the gap-separated families is the diagnostic."))
    code(CV_SETUP[k])
    if k <= 3:
        md("### 5.1 Baseline and feature-engineering ablation (default LightGBM)")
        code(ABLATION[k])
        md(md_for_task(pick("**Reading the ablation**"), k))
    elif k == 4:
        md("### 5.1 Per-fold learning curves of the fine-tuned backbones")
        code(pick("T4_BACKBONES = {"))
    else:
        md(md_for_task(pick("### 5.3 Task 5: anomaly families", "markdown"), 5))
        code(pick("cols5 = task5.detector_columns(X5)"))
        md(md_for_task(pick("**Reading the diagnostics.**"), 5, lead=5))

    # 6. tuning
    if k <= 3:
        md(section(6, md_for_task(pick("# 6. Hyperparameter Tuning (Optuna, TPE)"), k, lead=None)))
        cells.extend(tuning_cells(k))
    elif k == 4:
        md(section(6, "No Optuna search for Task 4: one fine-tuning run costs 2-3 GPU hours on this laptop. The recipe uses standard "
                      "fine-tuning settings (AdamW, weight decay 0.05, 1-epoch warm-up + cosine schedule, 10 epochs, batch 64, label "
                      "smoothing 0.1, Mixup/CutMix p = 0.8, EMA 0.999). The learning rate is set per backbone family: 2e-4 for the CNN "
                      "and 1e-4 for the ViT, the usual ratio for fine-tuning the two families. The only post-hoc parameters, the class-wise decision multipliers, are tuned on OOF probabilities in §7."))
    else:
        md(section(6, "Task 5 has no labels, so there is no objective to tune against. Instead, §5 shows that the gap detector's output "
                      "does not depend on its settings: the identical 1,811 accounts are flagged for λ ≥ 50, 100 or 300, neighbourhoods "
                      "of 100–400 accounts, tails of 5–15% and minimum group sizes of 10–50."))

    # 7. final model
    if k == 1:
        md(section(7, md_for_task(pick("# 7. Stacking Ensemble (K-Fold) & Metric-Aware Post-Processing"), k)))
    elif k == 2:
        md(section(7, "**Level 1:** three GBDTs with different tree-growth biases (LightGBM leaf-wise, XGBoost depth-wise, CatBoost "
                      "oblivious trees with native categoricals) and a multinomial logistic regression on Gaussian-rank features, all on "
                      "identical folds. **Level 2:** a regularised multinomial logistic-regression stacker on the level-1 OOF "
                      "log-probabilities, itself cross-validated on the same folds. **Post-processing:** class-wise multipliers "
                      r"$\hat y=\arg\max_c w_c P_{ic}$ tuned on the stacked OOF probabilities for macro-$F_1$."))
    elif k == 3:
        md(section(7, "Level-1 regressors are blended with non-negative weights that minimise total absolute error (fitted out-of-fold), "
                      "followed by a median-consistent zero gate and one scale factor."))
    elif k == 4:
        md(section(7, "Fold models of architecturally different backbones are averaged in probability space, and class-wise decision "
                      r"multipliers $\hat y=\arg\max_c w_c P_{ic}$ are tuned on the OOF blend for macro-$F_1$."))
    else:
        md(section(7, "The unsupervised evidence (gap families, score ensemble, hard-evidence rules) is weighed for the $F_3$ metric: "
                      "the gap families are flagged, and the other two serve as cross-checks."))
    if k == 1:
        code(pick("def logit(p):")); code(pick("THR1_opt, SCORE1_opt = optimize_threshold"))
    elif k == 2:
        md(pick("## 7.2 Task 2: Player segmentation (macro-F1)", "markdown"))
        code(pick("num2 = [c for c in X2.columns if c not in cat2]"))
        md(md_for_task(pick("The confusion matrix mirrors the 2×2 design"), 2, lead=2))
    elif k == 3:
        md(md_for_task(pick("The confusion matrix mirrors the 2×2 design"), 3, lead=2))
        code(pick("m3 = {}\nr = cached")); code(pick("names3 = list(m3)"))
    elif k == 4:
        md(md_for_task(pick("## 7.4 Task 4: Game title detection", "markdown"), 4))
        code(pick("y4 = t4_train.label.values")); code(pick("# Distribution-shift diagnostics"))
    else:
        md(md_for_task(pick("## 7.5 Task 5:", "markdown"), 5))
        code(pick("members = [k for k in det_scores"))
    if k <= 4:
        refit_md = md_for_task(pick("## 7.6 Final models: combine Train+Val", "markdown"), k).split("\n", 1)[1]
        if k == 4:   # keep the rule quote and the closing paragraph, drop the tabular-model bullets
            refit_md = "\n".join(line for line in refit_md.split("\n") if not line.startswith("* **") or "Task 4" in line)
        md("## 7.6 Final model: combine Train+Val (competition rule)\n" + refit_md)
        code(split_refit(k))

    # 8. feature importance
    if k <= 3:
        md(section(8, "Exact **TreeSHAP** (Lundberg & Lee) values come from a LightGBM model refit on fold 0 with the tuned parameters, "
                      "explaining 4,000 held-out rows"
                      + (", complemented by model-agnostic **permutation importance** (the metric drop when a column is shuffled on the held-out fold)." if k != 2 else ".")))
        code(pick("def fold0_lgb(params, X, y, objective"))
        key = {1: ("mdl1, va1 = fold0_lgb", "**Business reading (anti-cheat).**"), 2: ("mdl2, va2 = fold0_lgb", "**Business reading (segmentation).**"),
               3: ("mdl3, va3 = fold0_lgb", "**Business reading (spending).**")}[k]
        code(pick(key[0])); md(pick(key[1]))
    elif k == 4:
        md(section(8, "**Grad-CAM** (Selvaraju et al., 2017) on the last ConvNeXt stage shows which pixels drive each decision."))
        code(pick("cam_model = task4.create_model")); md(pick("**Business reading (game detection).**"))
    else:
        md(section(8, "Every alert is explained by the behaviour that isolates it (the family's gap column)."))
        code(pick("flagged = np.where(pred5 == 1)[0]")); md(pick("**Business reading (security).**"))

    # 9. submission part
    fmt = ("a THB amount rounded to 2 decimals" if k == 3 else "a plain integer code, as defined in the datacard")
    md(section(9, f"The Task {k} column of the submission: `task{k}` is {fmt}. The part is written to `outputs/submission_parts/task{k}.csv` "
                  "with the `id` column of `sample_submission.csv`; `python scripts/make_submission.py` checks and joins the five parts."))
    code(SAVE_PART.strip("\n").format(k=k, test=TEST_FRAME[k], **SAVE[k]))

    # 10. conclusion
    md(section(10))
    code('display(pd.DataFrame(summary, columns=["task", "metric", "OOF score", "final recipe"]).set_index("task").style.format({"OOF score": "{:.4f}"}))')
    md("| Task | Key insight | What moved the score |\n|---|---|---|\n" + RECAP[k])
    return cells


def main():
    for k in TASKS:
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "cpe342", "display_name": "Python (cpe342 venv)", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        for kind, src in task_cells(k):
            nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src))
        out = ROOT / f"task{k}" / f"task{k}.ipynb"
        out.parent.mkdir(exist_ok=True)
        nbformat.write(nb, out)
        print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
