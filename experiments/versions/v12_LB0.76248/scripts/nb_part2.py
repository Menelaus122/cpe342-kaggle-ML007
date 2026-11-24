"""Notebook cells, part 2: sections 5-10."""

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ========================================================== 5. training & evaluation
md(r"""
<a id="5"></a>
# 5. Model Training & Evaluation

**Validation protocol.** Every tabular model uses the *same* 5-fold split per task (seed 42):

* Tasks 1 and 2 use `StratifiedKFold` on the label.
* Task 3 stratifies on a zero / positive-quantile bin of `spending_30d`, so every fold has the same share of non-spenders and whales.
* Task 4 uses `StratifiedKFold` on the game label (`scripts/train_task4.py`).

Each model produces **out-of-fold (OOF)** predictions for every training row. These give honest metric estimates, feed threshold and weight tuning, and serve as level-1 inputs for stacking. Test predictions average the 5 fold models. Early stopping monitors each fold's validation loss, which is standard practice. Decision thresholds and weights are only ever tuned on OOF predictions.

Results are cached in `outputs/preds/*.npz`, so re-running the notebook is fast. Delete a file to retrain that model.
""")
code(r"""
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer

folds1 = make_folds(y1)
folds2 = make_folds(y2)
folds3 = make_folds(y3, kind="regression")
print("fold sizes T1:", [len(v) for _, v in folds1], "| T3 zero-share per fold:", [round((y3[v] == 0).mean(), 3) for _, v in folds3])

def cached(name, fn, force=False):
    # Return {'oof','test',...} arrays, training via fn() only when no cache exists.
    r = load_preds(name)
    if r is None or force:
        t0 = time.time()
        out = fn()
        r = {k: np.asarray(v) for k, v in out.items() if k in ("oof", "test", "p_oof", "mu_oof", "p_test", "mu_test", "sigma", "smear", "n_iter")}
        save_preds(name, **r)
        print(f"  trained {name} in {time.time() - t0:.0f}s")
    return r

BASE = {"learning_rate": 0.05}
summary = []   # running scoreboard
""")
md(r"""
### 5.1 Baselines and feature-engineering ablation (default LightGBM)
""")
code(r"""
raw1 = [c for c in t1_train.columns if c not in ("player_id", "is_cheater")]
raw2 = [c for c in X2.columns if c in t2_train.columns]                     # original columns, noise removed
raw3 = [c for c in t3_train.columns if c not in ("player_id", "spending_30d")]

b = {}
b["T1 raw"] = cached("t1_base_raw", lambda: run_cv("lgb", "binary", BASE, X1[raw1], y1, X1_test[raw1], folds1, name="T1 raw"))
b["T1 FE"] = cached("t1_base_fe", lambda: run_cv("lgb", "binary", BASE, X1, y1, X1_test, folds1, name="T1 FE"))
b["T2 raw"] = cached("t2_base_raw", lambda: run_cv("lgb", "multiclass", BASE, X2[raw2], y2, X2_test[raw2], folds2, n_classes=4, name="T2 raw"))
b["T2 FE"] = cached("t2_base_fe", lambda: run_cv("lgb", "multiclass", BASE, X2, y2, X2_test, folds2, n_classes=4, name="T2 FE"))
b["T2 FE+TE"] = cached("t2_base_fe_te", lambda: run_cv("lgb", "multiclass", BASE, X2, y2, X2_test, folds2, n_classes=4,
                                                       fold_fn=task2.te_fold_fn, name="T2 FE+TE"))
b["T3 raw (L1 log1p)"] = cached("t3r_base_raw_log", lambda: run_cv("lgb", "regression", BASE, X3[raw3], np.log1p(y3), X3_test[raw3], folds3, name="T3 raw log"))
b["T3 FE (L1 log1p)"] = cached("t3r_base_fe_log", lambda: run_cv("lgb", "regression", BASE, X3, np.log1p(y3), X3_test, folds3, name="T3 FE log"))
b["T3 FE (L1 raw THB)"] = cached("t3r_base_fe_raw", lambda: run_cv("lgb", "regression", BASE, X3, y3, X3_test, folds3, name="T3 FE raw"))

rows = []
for k in ("T1 raw", "T1 FE"):
    thr, s = optimize_threshold(y1, b[k]["oof"], 2.0)
    rows.append([k, f"{AVG_NAME} F2 @ OOF-tuned threshold", s, f"AUC {roc_auc_score(y1, b[k]['oof']):.4f}, thr {thr:.3f}"])
for k in ("T2 raw", "T2 FE", "T2 FE+TE"):
    rows.append([k, "macro-F1 (argmax)", macro_f1(y2, b[k]["oof"].argmax(1)), ""])
for k in ("T3 raw (L1 log1p)", "T3 FE (L1 log1p)", "T3 FE (L1 raw THB)"):
    p = b[k]["oof"] if "raw THB" in k else np.expm1(b[k]["oof"])
    rows.append([k, "NMAE", nmae(y3, np.clip(p, 0, None)), ""])
ablation = pd.DataFrame(rows, columns=["experiment", "metric", "OOF score", "note"])
display(ablation)
""")
md(r"""
**Reading the ablation**

* **Task 1:** the engineered ratios leave AUC and F2 essentially unchanged. The signal is additive and noisy, and trees already find these interactions. We keep the features because the composite indices are useful for explanation.
* **Task 2:** the latent-factor indices give a clear macro-F1 gain. Leakage-free in-fold target encoding of the nominal columns is neutral, which confirms the EDA verdict that they carry no information. The final GBDTs therefore rely on native categorical handling (CatBoost's ordered target statistics are themselves a leakage-free target encoding) instead of explicit TE columns.
* **Task 3:** once the identity fill has repaired the raw spending columns (section 4.3), the engineered ratios add nothing to the `log1p` model (0.7830 raw vs 0.7823 with features). More interesting, **the L1 model fitted on raw THB beats the L1 model fitted on `log1p`** (0.7867), although both estimate the conditional median. In log space every account's error weighs the same. On the raw scale, model capacity flows to the whales who dominate $\sum|y-\hat y|$, and that sum is exactly what NMAE measures.

### 5.2 Task 4: per-fold learning curves of the fine-tuned backbones
""")
code(r"""
T4_BACKBONES = {   # tag -> (timm model, input HxW) as trained by scripts/train_task4.py
    "cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)),
    "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
}
fold_done = lambda tag: all((C.PRED_DIR / f"t4_{tag}_fold{f}.npz").exists() for f in range(N_FOLDS))
if not fold_done("cnxt") and RUN_T4_TRAINING:
    subprocess.run([sys.executable, "-u", "scripts/train_task4.py", "--model", T4_BACKBONES["cnxt"][0],
                    "--tag", "cnxt", "--epochs", "10"], check=True)
T4_TAGS = [tag for tag in T4_BACKBONES if fold_done(tag)]
assert "cnxt" in T4_TAGS, "Task 4 fold caches missing (run scripts/train_task4.py or set RUN_T4_TRAINING=True)"
print("Task 4 backbones with complete 5-fold caches:", {t: T4_BACKBONES[t] for t in T4_TAGS})

log_dir = C.OUT / "logs"
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.3))
for j, tag in enumerate(T4_TAGS):
    for f in range(N_FOLDS):
        h = pd.DataFrame(json.loads((log_dir / f"t4_{tag}_fold{f}_history.json").read_text()))
        lab = tag if f == 0 else None
        axes[0].plot(h.epoch + 1, h.loss, color=CAT[j], alpha=0.55, lw=1.4, label=lab)
        axes[1].plot(h.epoch + 1, h.val_f1, color=CAT[j], alpha=0.55, lw=1.4, label=lab)
axes[0].set_title("Training loss (soft-target CE, Mixup/CutMix)"); axes[0].set_xlabel("epoch")
axes[1].set_title("Validation macro-F1 of the EMA weights (5 folds each)"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0.7, 1.0)
axes[1].legend(loc="lower right")
plt.tight_layout(); savefig(fig, "t4_curves"); plt.show()
""")
md(r"""
### 5.3 Task 5: anomaly families and detector diagnostics (no labels)
Before scoring detectors, we ask how the anomalies are *distributed*. EDA showed that `behavior_stability` is uniform on 0.70–0.95 for 99% of accounts, while 274 accounts sit on 0.20–0.40 with **nothing in between**. A long tail does not leave an empty gap. A separate population of planted or anomalous behaviour does.

`task5.gap_clusters` turns this into a label-free detector that runs on every column. In each 10% tail it tests the widest empty gaps. With $d$ the density of the 200 accounts next to the gap and $w$ its width, $\lambda = d\,w$ is the number of accounts we would *expect* inside the gap if the bulk simply continued, and $P(\text{gap empty}) \approx e^{-\lambda}$. Accounts beyond a gap with $\lambda \ge 50$ form an **isolated tail cluster**. Groups under 20 accounts are individual outliers, not populations; this excludes, for example, longitudes that wrap past ±180° for players near the date line. Groups over 3% are zero-inflated counts. Clusters that share accounts are merged into **families** (union-find, overlap ≥ 50%). One family is one behaviour seen through several columns: a bot shows low IP entropy *and* clockwork login times *and* constant session lengths.

The result does not hinge on these settings. The flagged set is the identical 1,811 accounts for $\lambda \ge$ 50, 100 or 300, neighbourhoods of 100–400 accounts, tails of 5–15%, and minimum group sizes of 10–50. Only a much looser $\lambda \ge 20$ adds 40 borderline accounts.

The families then serve as a yardstick for the six statistical detectors: which behaviours does each one see in its top 7%?
""")
code(r"""
cols5 = task5.detector_columns(X5)
Z5 = task5.detector_matrix(X5, cols5, "robust")
gc_table, gc_masks = task5.gap_clusters(t5_test, X5)
gc_fams = task5.merge_families(gc_masks)
FAMILY_NAMES = {"friend_removes_4": "skill boost + friend purge", "behavior_stability": "erratic (low stability)",
                "location_changes_4": "takeover / VPN abuse", "activity_level": "near-inactive",
                "communication_frequency": "silent (no chat)", "ip_hash_entropy": "single-IP bots"}
def fam_name(groups):
    for g in groups:
        for key, nm in FAMILY_NAMES.items():
            if g.startswith(key):
                return nm
    return groups[0]
fam_mask = {fam_name(f): gc_masks[f].any(axis=1).values for f in sorted(gc_fams, key=lambda f: -gc_masks[f].any(axis=1).sum())}
gap5 = gc_masks.any(axis=1).values
display(gc_table[["group", "size", "share", "gap_from", "gap_to", "expected_in_gap"]].style.format(
    {"share": "{:.2%}", "gap_from": "{:.3f}", "gap_to": "{:.3f}", "expected_in_gap": "{:,.0f}"}).hide(axis="index"))
display(pd.DataFrame([[nm, int(m.sum()), m.mean(), len(f)] for (nm, m), f in zip(fam_mask.items(), sorted(gc_fams, key=lambda f: -gc_masks[f].any(axis=1).sum()))],
                     columns=["family", "accounts", "share", "columns that isolate it"]).style.format({"share": "{:.2%}"}).hide(axis="index"))
print(f"{len(gc_table)} isolated tail clusters -> {len(gc_fams)} families -> {gap5.sum():,} accounts ({gap5.mean():.2%})")

det_scores = {}
for name, fn in task5.DETECTORS.items():
    t0 = time.time()
    s = fn(Z5, device=DEVICE) if name == "AutoEncoder" else fn(Z5)
    det_scores[name] = task5.robust_z(s)
    print(f"  {name:17s} {time.time() - t0:5.1f}s")

BUDGET = 0.07
cov = pd.DataFrame({name: {**{k: (s >= np.quantile(s, 1 - BUDGET))[m].mean() for k, m in fam_mask.items()},
                           "ALL families": (s >= np.quantile(s, 1 - BUDGET))[gap5].mean()}
                    for name, s in det_scores.items()}).T
fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
sns.heatmap(cov, annot=True, fmt=".2f", cmap=SEQ_CMAP, vmin=0, vmax=1, ax=axes[0], cbar=False, annot_kws={"size": 8})
axes[0].set_title(f"Share of each family inside a detector's top {BUDGET:.0%}"); axes[0].tick_params(axis="x", rotation=30, labelsize=7.5)
plt.setp(axes[0].get_xticklabels(), ha="right", rotation_mode="anchor")
corr5 = pd.DataFrame(det_scores).corr("spearman")
sns.heatmap(corr5, annot=True, fmt=".2f", cmap=SEQ_CMAP, vmin=0, vmax=1, ax=axes[1], cbar=False, annot_kws={"size": 7.5})
axes[1].set_title("Spearman agreement between detectors"); axes[1].tick_params(labelsize=7.5)
plt.tight_layout(); savefig(fig, "t5_detectors"); plt.show()
""")
md(r"""
**Reading the diagnostics.** Six families account for 7.0% of accounts. Each is cut off from normal behaviour by a gap that should hold hundreds to tens of thousands of accounts if it were just a tail.

* **Every detector except LOF sees the spike families.** Teleports, transfers and purchase spikes (takeover / VPN abuse), and skill jumps with friend purges, land in the top 7% for 96–100% of their accounts.
* **Coverage of the "too low" families depends on the gap's distance from the bulk.** Very low `behavior_stability` and single-IP bots sit 5–6 robust SDs (1.4826·MAD) below the median. The extreme-value detector, autoencoder and MCD catch the low-stability family (0.97 / 1.00 / 0.80). Bots are caught mostly by the extreme-value detector (0.81).
* **Near-inactive and silent accounts are invisible to all of them (≤ 0.22).** Their activity and chat levels sit only about 2 robust SDs below the median. In 70 dimensions they are swamped by accounts that are merely noisy, the classic weakness of distance and density scores on bounded features. Yet their gaps are unmistakable: zero accounts where hundreds are expected.
* **LOF covers little of anything (≤ 0.29).** Local density ratios in 70 dimensions mostly react to noise.

The gap-cluster detector therefore finds what the statistical detectors miss. §7.5 checks whether the score ensemble and the hard-evidence rules find anything *beyond* the families, since under $F_3$ a missed anomaly costs far more than a false alarm.
""")

# ============================================================ 6. hyperparameter tuning
md(r"""
<a id="6"></a>
# 6. Hyperparameter Tuning (Optuna, TPE)

Optuna's multivariate **Tree-structured Parzen Estimator** (Bayesian optimisation) searches each model's space, and each trial is scored with the task's **exact** metric on 3-fold OOF predictions:

* **Task 1:** $F_2$ *after* OOF threshold optimisation, so the search rewards ranking quality where it matters for the threshold. The stored study used the macro reading; both readings reward the same ranking quality, only the final threshold differs. `scale_pos_weight` (cost-sensitive learning) is part of the space.
* **Task 2:** macro-$F_1$ of the argmax.
* **Task 3:** NMAE of the L1 (median) model on raw THB. Two hurdle components were tuned the same way (`experiments/claude/ag_t3_tune.py`, same folds): stage 1 for log-loss and the ratio stage 2 for L1 on the log ratio. Their parameters are stored in `outputs/params/task3_stage1.json` and `outputs/params/task3_ratio_stage2.json`.

Studies run from `scripts/tune.py` (via `scripts/run_tuning.sh`) and persist in `outputs/optuna.db`. This section loads them, and runs them first if `RUN_OPTUNA=True`. Final models use the tuned parameters with **half the tuned learning rate** (early stopping then finds proportionally more trees), which usually adds a little accuracy at no tuning cost.
""")
code(r"""
STORAGE = f"sqlite:///{(C.OUT / 'optuna.db').as_posix()}"
optuna.logging.set_verbosity(optuna.logging.WARNING)
STUDIES = [(1, "lgb", 40), (3, "lgb", 30), (2, "lgb", 15)]   # LightGBM is the primary learner; XGB/CatBoost keep robust defaults
studies, rows = {}, []
for t, m, n in STUDIES:
    name = f"task{t}_{m}"
    if RUN_OPTUNA:
        subprocess.run([sys.executable, "-u", "scripts/tune.py", "--task", str(t), "--model", m, "--trials", str(n)], check=True)
    try:
        st = optuna.load_study(study_name=name, storage=STORAGE)
    except KeyError:
        st = None
    studies[name] = st
    if st is not None and len(st.trials):
        done = [tr for tr in st.trials if tr.state.name == "COMPLETE"]
        rows.append([name, len(done), st.best_value, done[0].value, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.best_params.items()})])
display(pd.DataFrame(rows, columns=["study", "trials", "best OOF score", "first-trial score", "best params"]))
""")
code(r"""
lgb_studies = [s for s in ("task1_lgb", "task2_lgb", "task3_lgb") if studies.get(s) is not None]
fig, axes = plt.subplots(2, len(lgb_studies), figsize=(4.4 * len(lgb_studies), 6.2), squeeze=False)
labels = {"task1_lgb": "Task 1: F2 (study objective: macro)", "task2_lgb": "Task 2: macro-F1", "task3_lgb": "Task 3: NMAE"}
for j, s in enumerate(lgb_studies):
    df = studies[s].trials_dataframe().query("state == 'COMPLETE'")
    ax = axes[0, j]
    ax.scatter(df.number, df.value, s=14, color=MUTED, label="trial")
    ax.plot(df.number, df.value.cummax(), color=CAT[0], lw=2, label="best so far")
    ax.set_title(labels[s]); ax.set_xlabel("trial")
    lo = df.value.quantile(0.1); ax.set_ylim(lo - (df.value.max() - lo) * 0.1, df.value.max() + (df.value.max() - lo) * 0.15)
    imp = optuna.importance.get_param_importances(studies[s])
    imp = pd.Series(imp).sort_values()
    axes[1, j].barh(imp.index, imp.values, color=CAT[0], height=0.6)
    axes[1, j].set_title("fANOVA hyper-parameter importance", fontsize=10); axes[1, j].tick_params(labelsize=7.5)
axes[0, 0].legend(loc="lower right")
plt.tight_layout(); savefig(fig, "optuna"); plt.show()
""")
code(r"""
def tuned(name, fallback):
    p = load_params(name)                                   # written when a study completes
    if p is None and studies.get(name) is not None and len(studies[name].trials):
        p = studies[name].best_params                       # partially-run study: best trial so far
    p = dict(fallback) if p is None else dict(p)
    if "learning_rate" in p:
        p["learning_rate"] = p["learning_rate"] / 2      # finer steps for the final fit
    return p

spw = (y1 == 0).sum() / (y1 == 1).sum()
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
P1_cat = tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})
P1_xgb = {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw}
P2_lgb = tuned("task2_lgb", {"learning_rate": 0.05})
P2_cat = tuned("task2_cat", {"learning_rate": 0.08, "depth": 6})
P2_xgb = {"learning_rate": 0.04, "max_depth": 5, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.5}
P3_lgb = tuned("task3_lgb", {"learning_rate": 0.05})
P3_xgb = tuned("task3_xgb", {"learning_rate": 0.05, "max_depth": 7})
for k, v in [("P1_lgb", P1_lgb), ("P1_cat", P1_cat), ("P2_lgb", P2_lgb), ("P2_cat", P2_cat), ("P3_lgb", P3_lgb), ("P3_xgb", P3_xgb)]:
    print(f"{k}: {json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})}")
""")

# ================================================================== 7. stacking
md(r"""
<a id="7"></a>
# 7. Stacking Ensemble (K-Fold) & Metric-Aware Post-Processing

**Level 1:** diverse model families trained on identical folds: three GBDTs with different tree-growth biases (LightGBM leaf-wise, XGBoost depth-wise, CatBoost oblivious trees), plus a bagged randomised-tree ensemble (ExtraTrees) and a linear model (logistic regression on Gaussian-rank features). Task 1 adds four members built the way last year's 1st-place team built theirs (CPE342 1/2025, public repo [Saltywan/CPE342-KAGGLE](https://github.com/Saltywan/CPE342-KAGGLE)): Random Forest, LightGBM, XGBoost and CatBoost on 24 of the 31 raw columns with training-median imputation, each averaged over two seeds. They see the data differently from our members, and adding them raises the nested OOF F2 by +0.0016 (95% CI [+0.0008, +0.0024], better in 5/5 folds, also better under test-like missingness).

**Level 2:** a regularised logistic-regression stacker on the level-1 OOF log-odds. The stacker is itself cross-validated **on the same folds**, so its reported score is also out-of-fold. Its test input is the fold-averaged level-1 test predictions.

**Post-processing:** the decision rule is tuned on stacked OOF predictions to maximise the *exact* metric:
* Task 1: threshold moving (Provost, 2000).
* Tasks 2 and 4: class-wise multipliers $\hat y=\arg\max_c w_c P_{ic}$.
* Task 3: an L1 blend and a scale factor.
* Task 5: fusion and contamination.

## 7.1 Task 1: Anti-cheat (F2)
""")
code(r"""
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

make_et1 = lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.4, class_weight="balanced",
                                        n_jobs=-1, random_state=SEED)
make_lr1 = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 QuantileTransformer(output_distribution="normal", random_state=SEED),
                                 LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
m1 = {}
m1["LightGBM"] = cached("t1_lgb", lambda: run_cv("lgb", "binary", P1_lgb, X1, y1, X1_test, folds1, name="T1 lgb"))
m1["XGBoost"] = cached("t1_xgb", lambda: run_cv("xgb", "binary", P1_xgb, X1, y1, X1_test, folds1, name="T1 xgb"))
m1["CatBoost"] = cached("t1_cat", lambda: run_cv("cat", "binary", P1_cat, X1, y1, X1_test, folds1, name="T1 cat"))
m1["ExtraTrees"] = cached("t1_et", lambda: run_cv_sklearn(make_et1, "binary", X1.values, y1, X1_test.values, folds1, name="T1 et"))
m1["LogReg"] = cached("t1_lr", lambda: run_cv_sklearn(make_lr1, "binary", X1.values, y1, X1_test.values, folds1, name="T1 logreg"))

# Four more members built the way last year's 1st-place team built theirs (CPE342 1/2025, github.com/Saltywan/CPE342-KAGGLE):
# 24 of the 31 raw columns, training-median imputation, near-default settings, each averaged over two seeds. Their different
# inputs make them diverse: OOF F2 +0.0016 (95% CI [+0.0008, +0.0024], 5/5 folds), test-like +0.0023 (round 11, bn_/bo_ scripts).
R1, R1_test = task1.build_c24(t1_train, t1_test)
P24 = {"lgb": {"learning_rate": 0.05, "num_leaves": 31}, "xgb": {"learning_rate": 0.05, "max_depth": 6}, "cat": {"learning_rate": 0.08, "depth": 6}}
SEEDS24 = (42, 7)
seed_kw = lambda kind, s: {"random_seed": s} if kind == "cat" else {"random_state": s}
make_rf24 = lambda s: (lambda: RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.7, n_jobs=-1, random_state=s))
seed_avg = lambda rs: {k: np.mean([r[k] for r in rs], axis=0) for k in ("oof", "test")}
m1["RF-24"] = cached("t1_rf24", lambda: seed_avg([run_cv_sklearn(make_rf24(s), "binary", R1.values, y1, R1_test.values, folds1,
                                                                 name=f"T1 rf24 seed {s}") for s in SEEDS24]))
for kind, nm in (("lgb", "LightGBM-24"), ("xgb", "XGBoost-24"), ("cat", "CatBoost-24")):
    m1[nm] = cached(f"t1_{kind}24", lambda kind=kind: seed_avg([run_cv(kind, "binary", {**P24[kind], **seed_kw(kind, s)}, R1, y1, R1_test,
                                                                      folds1, name=f"T1 {kind}24 seed {s}") for s in SEEDS24]))

def stack_binary(models, y, folds, C_=1.0):
    S = np.column_stack([logit(m["oof"]) for m in models.values()])
    St = np.column_stack([logit(m["test"]) for m in models.values()])
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = LogisticRegression(C=C_, max_iter=2000).fit(S[tr], y[tr]).predict_proba(S[va])[:, 1]
    final = LogisticRegression(C=C_, max_iter=2000).fit(S, y)
    return oof, final.predict_proba(St)[:, 1], final.coef_.ravel(), final

stack1_oof, stack1_test, coef1, STACKER1 = stack_binary(m1, y1, folds1)
avg1_oof = np.mean([m["oof"] for m in m1.values()], axis=0)
rows = []
for name, p in [*[(k, v["oof"]) for k, v in m1.items()], ("Mean of probabilities", avg1_oof), ("LogReg stacker (OOF)", stack1_oof)]:
    thr, s = optimize_threshold(y1, p, 2.0)
    thr_b, s_b = optimize_threshold(y1, p, 2.0, average="binary")
    rows.append([name, roc_auc_score(y1, p), s, thr, s_b, thr_b])
t1_table = pd.DataFrame(rows, columns=["model", "AUC", "macro-F2", "thr (macro)", "binary-F2", "thr (binary)"]).set_index("model")
display(t1_table.style.format("{:.4f}").highlight_max(subset=["AUC", "macro-F2", "binary-F2"], props="font-weight:bold"))
print("stacker log-odds weights:", dict(zip(m1, np.round(coef1, 3))))
""")
code(r"""
THR1_opt, SCORE1_opt = optimize_threshold(y1, stack1_oof, 2.0)
OTHER_AVG = "macro" if FBETA_AVERAGE == "binary" else "binary"
THR1_alt, _ = optimize_threshold(y1, stack1_oof, 2.0, average=OTHER_AVG)
grid, curve_main = threshold_curve(y1, stack1_oof, 2.0, FBETA_AVERAGE)
_, curve_alt = threshold_curve(y1, stack1_oof, 2.0, OTHER_AVG)
# One-standard-error band (Breiman et al., 1984): thresholds whose OOF F2 lies within one bootstrap SE of the
# optimum are statistically tied, so the exact choice inside the band matters little. The slides define the
# positive-class F2, so we keep its OOF optimum. (v4 tried the band's macro-friendly end, 0.190, as a hedge
# for the macro reading; that submission scored lower, see the README leaderboard log.)
rng1 = np.random.default_rng(SEED)
pred_opt1 = (stack1_oof >= THR1_opt).astype(int)
SE1 = np.std([f2(y1[i], pred_opt1[i]) for i in (rng1.integers(0, len(y1), len(y1)) for _ in range(200))])
tied1 = curve_main >= SCORE1_opt - SE1
THR1, SCORE1 = THR1_opt, SCORE1_opt
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.axvspan(grid[tied1].min(), grid[tied1].max(), color=CAT[0], alpha=0.08, lw=0)
ax.plot(grid, curve_main, color=CAT[0], label=f"{AVG_NAME} F2 (optimised)")
ax.plot(grid, curve_alt, color=CAT[1], label=f"{'macro' if OTHER_AVG == 'macro' else 'positive-class'} F2 (reference)")
ax.axvline(THR1, color=CAT[0], lw=1, alpha=0.6); ax.axvline(0.5, color=MUTED, lw=1)
ax.annotate(f"chosen τ = {THR1:.3f}\n{AVG_NAME} F2 = {SCORE1:.4f}", (THR1, SCORE1), xytext=(THR1 + 0.45, SCORE1 - 0.07), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.text(0.505, ax.get_ylim()[0] + 0.01, "default 0.5", color="#52514e", fontsize=8)
ax.set_xlabel("probability threshold τ"); ax.set_ylabel("OOF score"); ax.set_title("Task 1: threshold moving on stacked OOF probabilities")
ax.legend(loc="lower left"); plt.tight_layout(); savefig(fig, "t1_threshold"); plt.show()
pred1 = (stack1_test >= THR1).astype(int)
print(f"OOF {AVG_NAME} F2 at the default 0.5: {f2(y1, (stack1_oof >= 0.5).astype(int)):.4f} -> best {SCORE1_opt:.4f} at τ={THR1_opt:.3f}")
print(f"one-SE plateau (SE = {SE1:.4f}): τ in [{grid[tied1].min():.3f}, {grid[tied1].max():.3f}] -> chosen τ={THR1:.3f} (the OOF optimum), {AVG_NAME} F2 {SCORE1:.4f}")
print(f"regret check: at τ={THR1:.3f} the other reading ({OTHER_AVG}) scores {fbeta(y1, (stack1_oof >= THR1).astype(int), 2, OTHER_AVG):.4f} "
      f"vs {fbeta(y1, (stack1_oof >= THR1_alt).astype(int), 2, OTHER_AVG):.4f} at its own optimum τ={THR1_alt:.3f}")
print(f"test flagged as cheaters: {pred1.mean():.2%} (train prevalence {y1.mean():.2%})")
summary.append(["Task 1", f"{AVG_NAME} F2", SCORE1, "5 L1 models -> LogReg stacker -> tuned threshold"])
""")

md(r"""
## 7.2 Task 2: Player segmentation (macro-F1)
""")
code(r"""
num2 = [c for c in X2.columns if c not in cat2]           # sklearn models: numeric columns only
m2 = {}
m2["LightGBM"] = cached("t2_lgb", lambda: run_cv("lgb", "multiclass", P2_lgb, X2, y2, X2_test, folds2, n_classes=4, name="T2 lgb"))
m2["XGBoost"] = cached("t2_xgb", lambda: run_cv("xgb", "multiclass", P2_xgb, X2, y2, X2_test, folds2, n_classes=4, name="T2 xgb"))
m2["CatBoost"] = cached("t2_cat", lambda: run_cv("cat", "multiclass", P2_cat, X2, y2, X2_test, folds2, cat_cols=cat2, n_classes=4, name="T2 cat"))
make_lr2 = lambda: make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(output_distribution="normal", random_state=SEED),
                                 LogisticRegression(C=0.5, max_iter=3000))
m2["LogReg"] = cached("t2_lr", lambda: run_cv_sklearn(make_lr2, "multiclass", X2[num2].values, y2, X2_test[num2].values,
                                                      folds2, n_classes=4, name="T2 logreg"))

def stack_multi(models, y, folds, C_=0.5):
    S = np.column_stack([np.log(np.clip(m["oof"], 1e-6, 1)) for m in models.values()])
    St = np.column_stack([np.log(np.clip(m["test"], 1e-6, 1)) for m in models.values()])
    oof = np.zeros((len(y), 4))
    for tr, va in folds:
        oof[va] = LogisticRegression(C=C_, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    final = LogisticRegression(C=C_, max_iter=3000).fit(S, y)
    return oof, final.predict_proba(St), final

stack2_oof, stack2_test, STACKER2 = stack_multi(m2, y2, folds2)
W2, SCORE2 = optimize_class_weights(y2, stack2_oof)
rows = [[k, macro_f1(y2, v["oof"].argmax(1))] for k, v in m2.items()]
rows += [["Mean of probabilities", macro_f1(y2, np.mean([v["oof"] for v in m2.values()], 0).argmax(1))],
         ["LogReg stacker (OOF)", macro_f1(y2, stack2_oof.argmax(1))],
         ["Stacker + class weights", SCORE2]]
display(pd.DataFrame(rows, columns=["model", "macro-F1"]).set_index("model").style.format("{:.4f}").highlight_max(props="font-weight:bold"))
print("class-wise decision multipliers w_c:", dict(zip(seg_names, np.round(W2, 3))))
pred2 = apply_class_weights(stack2_test, W2)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
cm = confusion_matrix(y2, apply_class_weights(stack2_oof, W2), normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f", cmap=SEQ_CMAP, vmin=0, vmax=1, xticklabels=seg_names, yticklabels=seg_names, cbar=False, ax=axes[0])
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true"); axes[0].set_title("Task 2 OOF confusion (row-normalised)")
per_cls = f1_score(y2, apply_class_weights(stack2_oof, W2), average=None)
axes[1].bar(seg_names, per_cls, color=CAT[0], width=0.6, **viz.bar_gap())
for i, v in enumerate(per_cls):
    axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
axes[1].set_ylim(0, 1); axes[1].set_title("Per-segment F1")
plt.tight_layout(); savefig(fig, "t2_confusion"); plt.show()
summary.append(["Task 2", "macro-F1", SCORE2, "4 L1 models -> multinomial stacker -> class multipliers"])
""")
md(r"""
The confusion matrix mirrors the 2×2 design. Almost all errors fall between segments that differ in **only one** latent factor:
* Grinder→Casual and Whale→Social sit on the engagement axis.
* Social→Casual and Whale→Grinder sit on the spending axis.

Two-factor mistakes (Casual↔Whale, Grinder↔Social) are rare. The class multipliers slightly up-weight Social and Whale, trading a little Casual precision for minority-class recall, which is exactly what macro-F1 rewards.

## 7.3 Task 3: Monthly spending (NMAE)
The level-1 pool mixes *median* estimators (the right target for absolute error) with structured alternatives. Every member uses the repaired features of section 4.3:

* **L1 GBDTs on raw THB** (LightGBM, XGBoost, CatBoost-MAE) and **L1 LightGBM on `log1p`**.
* **Two-stage hurdle (Cragg, 1971).** A classifier gives $p=P(y>0)$ (LightGBM, Optuna-tuned for log-loss). An L1 regressor fitted on spenders only gives the centre $\mu$ of $\log(1+y)$, either directly or through the **ratio target** $\log(1+y)-\log(1+b)$ for a base column $b$ (`historical_spending` or `prev_month_spending`).
  * Spending is multiplicative: $y \approx b \times \text{segment factor} \times \text{noise}$. The within-segment spread (MAD) of $\log(y/b)$ is 0.205, versus 0.44–0.53 for $\log y$ itself, so the ratio target is much easier to learn.
  * Ratio stage 2 is fitted with LightGBM (Optuna-tuned for L1, and default parameters), XGBoost and CatBoost.
  * Rows without a usable base keep the plain hurdle's $\mu$.

  Each stage 2 is combined with $p$ three ways:
  * **Mixture median** (MAE-optimal under a log-normal positive part): $\hat y = 0$ if $p\le 0.5$, else $\exp\!\big(\mu+\sigma\,\Phi^{-1}(1-\tfrac{0.5}{p})\big)-1$. The scale $\sigma$ and a gate shift are tuned on OOF NMAE.
  * **Hard gate:** $\hat y = 1[p>\tau]\,(e^{\mu}-1)$, with $\tau$ tuned on OOF.
  * **Expected value with Duan smearing,** $p\,(e^{\mu}\bar{e^{\varepsilon}}-1)$. This is the MSE-optimal recipe, shown for contrast.
* **Tweedie GBDT** ($p=1.5$, log link): a mean estimator.

These are blended with **non-negative weights that minimise total absolute error**, with the weights fitted out-of-fold. Non-negative L1 blending is a linear program ($\min \sum(u+v)$ s.t. $Pw+u-v=y$, $w,u,v\ge 0$), so the weights are solved exactly with HiGHS instead of a local search (deterministic, and +0.00007 NMAE out-of-fold over the earlier Powell search). Three light post-processing steps follow:
* a **median-consistent zero gate:** wherever the hurdle classifier gives $P(y>0)\le 0.5$, the conditional median is exactly 0, so the prediction is set to 0;
* a single scale factor;
* a **cap at the largest training value** (300,000 THB, reached by 15 training rows, so the target looks truncated there). A prediction above any value the target can take only adds error. Out-of-fold, each fold's cap comes from its own training rows (+0.00011 NMAE, 95% CI [+0.00003, +0.00021], 4/5 folds; it touches very few rows).
""")
code(r"""
S1_3 = tuned("task3_stage1", {"learning_rate": 0.05})          # hurdle stage 1, Optuna-tuned for log-loss
R2_3 = tuned("task3_ratio_stage2", {"learning_rate": 0.05})    # ratio stage 2, Optuna-tuned for L1 on the log ratio
TW3 = {"learning_rate": 0.05, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "l1"}
m3 = {}
r = cached("t3r_lgb_raw", lambda: run_cv("lgb", "regression", P3_lgb, X3, y3, X3_test, folds3, name="T3 lgb L1 raw"))
m3["LGB-L1 raw THB"] = (np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None))
r = cached("t3r_xgb_raw", lambda: run_cv("xgb", "regression", P3_xgb, X3, y3, X3_test, folds3, name="T3 xgb L1 raw"))
m3["XGB-L1 raw THB"] = (np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None))
r = cached("t3r_cat_raw", lambda: run_cv("cat", "regression", {"learning_rate": 0.08, "depth": 8}, X3, y3, X3_test, folds3, name="T3 cat MAE"))
m3["Cat-MAE raw THB"] = (np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None))
r = cached("t3r_lgb_log", lambda: run_cv("lgb", "regression", P3_lgb, X3, np.log1p(y3), X3_test, folds3, name="T3 lgb L1 log"))
m3["LGB-L1 log1p"] = (np.clip(np.expm1(r["oof"]), 0, None), np.clip(np.expm1(r["test"]), 0, None))
r = cached("t3r_tweedie", lambda: run_cv("lgb", "regression", TW3, X3, y3, X3_test, folds3, name="T3 tweedie"))
m3["LGB-Tweedie (mean)"] = (np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None))

# hurdle: one stage 1 shared by every stage 2
h = cached("t3r_hurdle", lambda: task3.hurdle_cv("lgb", X3, y3, X3_test, folds3, clf_params=S1_3,
                                                 reg_params={"learning_rate": 0.05}))
RATIO3 = {"ratio[hist] LGB tuned": ("lgb", R2_3, "historical_spending"),
          "ratio[prev] LGB tuned": ("lgb", R2_3, "prev_month_spending"),
          "ratio[hist] LGB": ("lgb", {"learning_rate": 0.05}, "historical_spending"),
          "ratio[prev] LGB": ("lgb", {"learning_rate": 0.05}, "prev_month_spending"),
          "ratio[hist] XGB": ("xgb", {"learning_rate": 0.05}, "historical_spending"),
          "ratio[hist] Cat": ("cat", {"learning_rate": 0.08, "depth": 8}, "historical_spending")}
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
mu3 = {"Hurdle": (h["mu_oof"], h["mu_test"])}
ratio3 = {}
for tag, (kind, prm, bcol) in RATIO3.items():
    ratio3[tag] = cached(slug3(tag), lambda: task3.ratio_stage2_cv(kind, prm, X3, y3, X3_test, folds3, bcol, h["mu_oof"], h["mu_test"]))
    mu3[tag] = (ratio3[tag]["mu_oof"], ratio3[tag]["mu_test"])
RO3 = {}
for tag, (mu, mu_t) in mu3.items():
    RO3[tag] = task3.readouts(y3, h["p_oof"], mu)
    for ro_kind in ("mixture-median", "hard gate", "E[y] + smearing"):
        m3[f"{tag} {ro_kind}"] = (task3.apply_readout(ro_kind, h["p_oof"], mu, RO3[tag]),
                                  task3.apply_readout(ro_kind, h["p_test"], mu_t, RO3[tag]))
ro = RO3["Hurdle"]
print(f"hurdle: tuned sigma {ro['sigma']:.3f}, gate shift {ro['shift']:+.3f}; hard-gate tau={ro['tau']:.3f}; smearing factor={ro['smear']:.3f}")
print(f"stage-1 AUC P(y>0): {roc_auc_score(y3 > 0, h['p_oof']):.4f}")
pos3 = y3 > 0
display(pd.DataFrame([[t, np.mean(np.abs(np.log1p(y3[pos3]) - mu3[t][0][pos3]))] for t in mu3],
                     columns=["stage 2", "MAE of log1p(y) on spenders"]).set_index("stage 2").style.format("{:.4f}"))
""")
code(r"""
names3 = list(m3)
P3o = np.column_stack([m3[k][0] for k in names3])
P3t = np.column_stack([m3[k][1] for k in names3])

# out-of-fold blending: weights fitted on 4 folds' OOF predictions, applied to the 5th
blend_oof, cap_oof = np.zeros(len(y3)), np.zeros(len(y3))
for tr, va in folds3:
    w = task3.l1_blend_weights_exact(y3[tr], [P3o[tr, j] for j in range(P3o.shape[1])])
    blend_oof[va] = P3o[va] @ w
    cap_oof[va] = y3[tr].max()
W3 =task3.l1_blend_weights_exact(y3, [P3o[:, j] for j in range(P3o.shape[1])])
# median-consistent zero gate: if P(y>0) <= 0.5 the conditional median is exactly 0
gate_oof = (np.clip(h["p_oof"] + RO3["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
gate_test = (np.clip(h["p_test"] + RO3["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
scales = np.linspace(0.85, 1.15, 61)
sc_scores = [nmae(y3, blend_oof * s * gate_oof) for s in scales]
SCALE3 = float(scales[int(np.argmax(sc_scores))])
CAP3 = float(y3.max())                                           # the target never exceeds its training maximum
SCORE3 = nmae(y3, np.minimum(blend_oof * SCALE3 * gate_oof, cap_oof))
pred3 = np.minimum(np.clip(P3t @ W3 * SCALE3, 0, None) * gate_test, CAP3)

rows = [[k, nmae(y3, m3[k][0])] for k in names3]
rows += [["L1 blend (out-of-fold weights)", nmae(y3, blend_oof)],
         ["+ zero gate (P(y>0) <= 0.5 -> 0)", nmae(y3, blend_oof * gate_oof)],
         [f"+ scale {SCALE3:.3f}", nmae(y3, blend_oof * SCALE3 * gate_oof)],
         [f"+ cap at the training maximum, {CAP3:,.0f} THB (final)", SCORE3]]
display(pd.DataFrame(rows, columns=["model", "NMAE"]).set_index("model").style.format("{:.5f}").highlight_max(props="font-weight:bold"))
print("full-data blend weights:", {k: round(float(w), 3) for k, w in zip(names3, W3) if w > 1e-3})
summary.append(["Task 3", "NMAE", SCORE3, "repaired features; L1/hurdle/ratio-hurdle/Tweedie pool -> OOF L1 blend -> median zero gate -> scale -> cap"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
hb = axes[0].hexbin(np.log1p(y3), np.log1p(blend_oof * SCALE3), gridsize=60, cmap=SEQ_CMAP, bins="log", mincnt=1)
axes[0].plot([0, 12.6], [0, 12.6], color=MUTED, lw=1); axes[0].grid(False)
axes[0].set_xlabel("log1p(actual)"); axes[0].set_ylabel("log1p(OOF prediction)"); axes[0].set_title("Task 3: OOF predictions vs actual")
dec = pd.qcut(blend_oof * SCALE3, 10, labels=False, duplicates="drop")
calib = pd.DataFrame({"pred": blend_oof * SCALE3, "y": y3, "d": dec}).groupby("d").agg(pred=("pred", "median"), y=("y", "median"))
axes[1].plot(calib.index + 1, calib.y, marker="o", color=CAT[0], label="median actual")
axes[1].plot(calib.index + 1, calib.pred, marker="o", color=CAT[1], label="median prediction")
axes[1].set_yscale("symlog", linthresh=100); axes[1].set_xlabel("prediction decile"); axes[1].set_ylabel("THB (symlog)")
axes[1].set_title("Median calibration by decile"); axes[1].legend()
plt.tight_layout(); savefig(fig, "t3_calibration"); plt.show()
""")

md(r"""
## 7.4 Task 4: Game title detection (macro-F1)
For each backbone, every fold model (final EMA weights) predicts its held-out fold with h-flip TTA, which gives OOF probabilities. Test probabilities are the mean over the 5 folds × 2 views. The two backbones come from different architecture families, and their probabilities are averaged:
* **ConvNeXt-Tiny** (CNN, IN-12k) at 160×320;
* **DeiT-III Small/16** (Vision Transformer, IN-22k) at 192×384, with position embeddings resampled to the 12×24 token grid.

Class-wise multipliers are then tuned on the OOF blend for macro-F1, and kept only if they help. Two more backbones were tried and dropped on OOF evidence:
* EfficientNetV2-S diverged at lr 3e-4 (epoch-0 loss above uniform).
* DINOv2 ViT-S/14 (self-supervised, at 154×308, lr 5e-5, wider zoom augmentation 0.7–1.3) reached only 0.9705 macro-F1 on fold 0, versus 0.986 / 0.985 for the two kept backbones. Adding it lowered the fold-0 ensemble at every weight tried (0.9878 → 0.9863 at 1/3 weight, 0.9874 at 0.2), and it was *less* confident on test images, not more. Its run was stopped after fold 0 (`outputs/preds/t4_dino_fold0.npz`).

**Why a second architecture? A test-set shift.** The next cell compares the models' confidence on test images against held-out training images. Test images are visibly harder: far more low-confidence predictions and lower agreement between fold models. They also show a different class mix, with fewer 7-Eleven Knights screenshots. So the OOF macro-F1 is an *optimistic* estimate of leaderboard performance. The shift is not a low-level one: binned by brightness, saturation, contrast or the black rotation border, test confidence is lower in *every* bin. The test images mostly show content (scenes, maps, menus) that the training set does not cover. Ensembling architecturally different models is the standard remedy that stays within the rules, and ViTs are known to be more robust to distribution shift than CNNs. We deliberately do **not** apply transductive label-shift correction (EM prior re-estimation on test predictions, Saerens et al. 2002), because it would adapt the model using test data.
""")
code(r"""
y4 = t4_train.label.values
oof_tag, test_tag, per_fold = {}, {}, []
for tag in T4_TAGS:
    oof_tag[tag], test_tag[tag] = np.zeros((len(y4), 5)), np.zeros((len(t4_test), 5))
    for f in range(N_FOLDS):
        z = np.load(C.PRED_DIR / f"t4_{tag}_fold{f}.npz")
        oof_tag[tag][z["val_idx"]] = z["val_probs"]
        test_tag[tag] += z["test_probs"] / N_FOLDS
        per_fold.append([tag, f, f1_score(y4[z["val_idx"]], z["val_probs"].argmax(1), average="macro")])
pf = pd.DataFrame(per_fold, columns=["backbone", "fold", "macro-F1"]).pivot(index="backbone", columns="fold", values="macro-F1")
pf["OOF (all folds)"] = [macro_f1(y4, oof_tag[t].argmax(1)) for t in pf.index]
oof4 = np.mean([oof_tag[t] for t in T4_TAGS], axis=0)
test4 = np.mean([test_tag[t] for t in T4_TAGS], axis=0)
if len(T4_TAGS) > 1:
    pf.loc["probability average"] = [np.nan] * N_FOLDS + [macro_f1(y4, oof4.argmax(1))]
display(pf.round(4))
W4, SCORE4 = optimize_class_weights(y4, oof4)
base4 = macro_f1(y4, oof4.argmax(1))
print(f"OOF macro-F1 argmax {base4:.4f} -> with class multipliers {SCORE4:.4f} | accuracy {(oof4.argmax(1) == y4).mean():.4f}")
if SCORE4 - base4 < 0.0005:           # not worth the extra degrees of freedom
    W4, SCORE4 = np.ones(5), base4
pred4 = apply_class_weights(test4, W4)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
cm4 = confusion_matrix(y4, apply_class_weights(oof4, W4), normalize="true")
sns.heatmap(cm4, annot=True, fmt=".3f", cmap=SEQ_CMAP, vmin=0, vmax=1, xticklabels=C.T4_CLASS_NAMES, yticklabels=C.T4_CLASS_NAMES,
            cbar=False, ax=axes[0], annot_kws={"size": 8})
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true"); axes[0].set_title("Task 4 OOF confusion (row-normalised)")
axes[0].tick_params(labelsize=7.5)
pc4 = f1_score(y4, apply_class_weights(oof4, W4), average=None)
axes[1].barh(C.T4_CLASS_NAMES[::-1], pc4[::-1], color=CAT[0], height=0.6, **viz.bar_gap())
for i, v in enumerate(pc4[::-1]):
    axes[1].text(v, i, f" {v:.4f}", va="center", fontsize=8.5)
axes[1].set_xlim(0.9, 1.0); axes[1].set_title("Per-class F1 (OOF)")
plt.tight_layout(); savefig(fig, "t4_confusion"); plt.show()
summary.append(["Task 4", "macro-F1", SCORE4, f"{' + '.join(T4_TAGS)} 5-fold, GPU aug + Mixup/CutMix, EMA, h-flip TTA"])
""")
code(r"""
# Distribution-shift diagnostics: held-out train images (OOF) vs test images
fold_preds = {tag: [np.load(C.PRED_DIR / f"t4_{tag}_fold{f}.npz")["test_probs"].argmax(1) for f in range(N_FOLDS)] for tag in T4_TAGS}
rows = []
for tag in T4_TAGS:
    fp = fold_preds[tag]
    agree = np.mean([np.mean(fp[i] == fp[j]) for i in range(N_FOLDS) for j in range(i + 1, N_FOLDS)])
    rows.append([tag, (oof_tag[tag].max(1) < 0.7).mean(), (test_tag[tag].max(1) < 0.7).mean(), agree])
for i, a in enumerate(T4_TAGS):
    for b in T4_TAGS[i + 1:]:
        rows.append([f"{a} vs {b} (test)", np.nan, np.nan, np.mean(test_tag[a].argmax(1) == test_tag[b].argmax(1))])
display(pd.DataFrame(rows, columns=["model", "OOF share max-p<0.7", "TEST share max-p<0.7", "pairwise fold agreement on TEST"]).set_index("model").round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
bins = np.linspace(0.2, 1.0, 41)
axes[0].hist(oof4.max(1), bins=bins, color=CAT[0], alpha=0.75, density=True, label="held-out train (OOF)")
axes[0].hist(test4.max(1), bins=bins, color=CAT[1], alpha=0.6, density=True, label="test")
axes[0].set_xlabel("max class probability"); axes[0].set_title("Confidence drops on test: covariate shift"); axes[0].legend(loc="upper left")
xs = np.arange(5); wdt = 0.38
axes[1].bar(xs - wdt / 2, np.bincount(y4, minlength=5) / len(y4), wdt, color=CAT[0], label="train labels", **viz.bar_gap())
axes[1].bar(xs + wdt / 2, np.bincount(pred4, minlength=5) / len(pred4), wdt, color=CAT[1], label="test predictions", **viz.bar_gap())
axes[1].set_xticks(xs); axes[1].set_xticklabels(C.T4_CLASS_NAMES, fontsize=8); axes[1].set_title("Class mix: prior shift"); axes[1].legend()
plt.tight_layout(); savefig(fig, "t4_shift"); plt.show()
""")

md(r"""
## 7.5 Task 5: Final decision (six families + one dependence pocket; score ensemble and rules as cross-checks)
1. **Score ensemble.** Each retained detector's raw score is robust-z standardised (median/MAD) and the results are averaged (LOF excluded, §5.3). The knee (Kneedle) of the sorted log-excess marks where scores stop falling steeply.
2. **Isolated tail clusters (§5.3).** Every account in a gap-separated family is flagged. Their membership needs no threshold choice: the gap *is* the threshold, and it is set by the data.
3. **F3 asymmetry.** For a perfect ranker with true rate $r$ that flags a fraction $q$, $F_3(q) = \frac{10\,\min(q,r)/q}{9\,\min(q,r)/q + \min(q,r)/r}$. Missing anomalies is about 9× costlier than a false alarm, so a whole family the score ensemble cannot see (near-inactive, silent accounts) has to be caught by the family detector.
4. **Ensemble alerts outside the families.** They are inspected below. Their largest robust z is modest, and they are enriched for the four yes/no flags only because the ensemble reads those flags as inputs. Outside the families the flags are statistically independent: the number of accounts with 0, 1, 2 or 3 flags matches the independence prediction almost exactly. So *statistically* they look like the noisy edge of normal behaviour.
5. **Are the families complete?** The gap detector of §5.3 scans the raw columns. A planted behaviour could hide from it if it only shows up as a *change* against the account's own history. So we rerun the same detector on the derived per-account features: period-4 shift, jumps, burstiness, all-period levels and geo hops. Every isolated group it finds there lies inside the six families. Two independent views of the data agree that there is no further *gap-separated* population. Point 7 looks for a group that has no gap at all.
6. **Union or families?** Task 5 has no labels, so we cannot see the organisers' anomaly definition. $F_3$ weighs recall 9× more than precision: the Bayes-optimal $F_\beta$ decision flags every account whose anomaly probability exceeds $F^*/(1+\beta^2) \approx F^*/10$ (Lipton et al., 2014). That argues for adding the ensemble and rule alerts *if* they have a real chance of being anomalies. The table below scores each candidate rule as if each of the others were the truth. The evidence says the extra alerts do not clear that bar: they look like the noisy edge of normal behaviour (point 4), and nothing outside the families forms a population (point 5). We therefore do not add them. A submission using the union (v4: public LB 0.69429 against 0.69663 for the families) agrees with this.
7. **A group without a gap: dependence pockets.** A behaviour can be unusual in two features *at once* without being extreme in either, so it leaves no 1-D gap. Normal accounts draw their features almost independently: on a 20×20 quantile grid of a feature pair, every cell should then hold $n/400$ accounts. We scan all ~11,300 feature pairs outside the families (pairs built from the same behaviour, or from a built-in ratio such as km per login, are skipped) and record each pair's most over-full cell, $z = (\text{observed} - \text{expected})/\sqrt{\text{expected}}$. The noise level is the largest $z$ of the same scan on column-shuffled data, which keeps every marginal and destroys every group. Exactly **one** pocket clears it by far: very high match-to-match spread of KDA *and* of accuracy (erratic performance, e.g. a shared or boosted account). After flagging it, no pocket is left above the noise level.
8. **A label-free cut-off for the pocket.** Score = the lower of the two rank percentiles (high only if both are high). Shuffling the columns again gives, for every cut-off $k$, the expected number of normal accounts among the top $k$; so $k$ minus that number estimates the true members, without labels. We choose $k$ to maximise the expected $F_3$, counting the families as true. Each flagged account then has a much higher chance of being anomalous than the $F^*/10$ bar of point 6. The final flags are the **six families plus the pocket**.
""")
code(r"""
members = [k for k in det_scores if k != "LOF"]
M5 = np.column_stack([det_scores[k] for k in members])
w_sel, agree = task5.consensus_weights(np.column_stack(list(det_scores.values())))
display(pd.DataFrame({"LOO-consensus Spearman": agree, "in fusion": [k in members for k in det_scores]}, index=list(det_scores)).T)
consensus = M5.mean(axis=1)
s_sorted = np.sort(consensus)[::-1]
log_excess = np.log1p(np.clip(s_sorted - np.median(s_sorted), 0, None))
knee = task5.knee_point(log_excess)
Q5 = knee / len(consensus)
flag_ml = consensus >= s_sorted[knee]
print(f"score-ensemble knee: top {Q5:.2%} | families: {gap5.mean():.2%} | ensemble ROC-AUC for family membership: {roc_auc_score(gap5, consensus):.4f}")
rec = pd.DataFrame({"accounts": {k: int(m.sum()) for k, m in fam_mask.items()},
                    "caught by ensemble knee": {k: flag_ml[m].mean() for k, m in fam_mask.items()}})
display(rec.style.format({"caught by ensemble knee": "{:.1%}"}))

# ensemble-only alerts: do they look like a population, or like the noisy edge of normal behaviour?
extra = flag_ml & ~gap5
normal = ~flag_ml & ~gap5
zdf = pd.DataFrame(Z5, columns=cols5)
top_abs = zdf.abs().max(axis=1)
print(f"ensemble-only alerts: {extra.sum():,} | median of their largest |robust z|: {np.median(top_abs[extra]):.2f} "
      f"(families {np.median(top_abs[gap5]):.2f}, unflagged accounts {np.median(top_abs[normal]):.2f})")
print("binary-flag rates (password / vpn / odd-hour / mass-sale):",
      {nm: np.round(t5_test.loc[m, task5.FLAG_COLS].mean().values, 3).tolist() for nm, m in [("ensemble-only", extra), ("unflagged", normal)]})
# The ensemble reads the four flags as inputs, so ensemble-only alerts are enriched for flags by construction.
# Is "several flags at once" a population of its own? Outside the families, compare the number of flags
# per account with what independent flags would produce (Poisson-binomial).
Fl = t5_test.loc[~gap5, task5.FLAG_COLS].dropna()
pr = Fl.mean().values
expd = np.array([1.0])
for r in pr:
    expd = np.convolve(expd, [1 - r, r])
nflag = Fl.sum(axis=1).astype(int).value_counts().reindex(range(5), fill_value=0)
display(pd.DataFrame({"accounts observed": nflag.values, "expected if independent": np.round(expd * len(Fl), 1)},
                     index=pd.Index(range(5), name="flags set")).T)

# Completeness check: the same gap detector on the derived per-account features (period-4 change, jumps, levels, geo)
Xd5 = X5.drop(columns=[c for c in X5.columns if c in task5.STATIC_COLS or c in ("n_flags", "n_missing")])
gd_table, gd_masks = task5.gap_clusters(Xd5)
gd_any = gd_masks.any(axis=1).values
print(f"derived-feature gap scan: {len(gd_table)} isolated groups covering {gd_any.sum():,} accounts; "
      f"outside the six families: {int((gd_any & ~gap5).sum())}")

# Candidate rules, each scored as if each candidate were the ground truth
rules5 = task5.evidence_rules(t5_test)
rule5 = rules5.any(axis=1).values
cands5 = {"gap families": gap5, "ensemble knee + evidence rules": flag_ml | rule5, "union of all three": gap5 | flag_ml | rule5}
truths5 = {"gap families": gap5, "ensemble knee + evidence rules": flag_ml | rule5, "families ∩ (knee + rules)": gap5 & (flag_ml | rule5)}
rob5 = pd.DataFrame({f"truth = {tn}": {dn: f3(t.astype(int), d.astype(int)) for dn, d in cands5.items()} for tn, t in truths5.items()})
rob5["worst case"] = rob5.min(axis=1)
rob5.insert(0, "accounts flagged", [int(d.sum()) for d in cands5.values()])
display(rob5.style.format({c: "{:.3f}" for c in rob5.columns if c != "accounts flagged"}))
# point 7: blind dependence-pocket scan outside the families; point 8: label-free detector and cut-off
scan5, noise5 = task5.dependence_pockets(X5, ~gap5)
sig5 = scan5[scan5.z > noise5]
print(f"dependence-pocket scan outside the families: {len(scan5):,} feature pairs; noise level (max z on shuffled data) "
      f"{noise5:.1f}; pockets above it: {len(sig5)}")
display(scan5.head(5).round(1))
pocket_feats5 = sorted(set(sig5.a) | set(sig5.b))
pocket5, pinfo5 = task5.pocket_detector(X5, gap5, pocket_feats5)
print(f"pocket detector on {pocket_feats5}: estimated group size {pinfo5['group_size']:.0f}; cut-off k = {pinfo5['k']} "
      f"(estimated members {pinfo5['est_tp']:.0f}, precision {pinfo5['precision']:.2f}); expected F3 families only "
      f"{pinfo5['expected_f_known_only']:.3f} -> with the pocket {pinfo5['expected_f']:.3f}")
flag5 = gap5 | pocket5
pred5 = flag5.astype(int)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6))
ranks = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
axes[0].plot(ranks, log_excess, color=CAT[0], label="all accounts")
axes[0].axvline(Q5, color=CAT[1], lw=1.2)
axes[0].annotate(f"ensemble knee: top {Q5:.2%}", (Q5, log_excess[knee]), xytext=(Q5 + 0.05, log_excess[knee] + 1.0), fontsize=8.5,
                 arrowprops=dict(arrowstyle="-", color=MUTED))
axes[0].set_xlim(0, 0.3); axes[0].set_xlabel("fraction of accounts (sorted by ensemble score)")
axes[0].set_ylabel("log(1 + score excess over median)"); axes[0].set_title("Score-ensemble curve")
r_true = gap5.mean()
qs = np.linspace(0.2, 3, 200) * r_true
axes[1].plot(qs / r_true, task5.expected_fbeta(qs, r_true, 3), color=CAT[0], label="F3 (positive class)")
axes[1].plot(qs / r_true, task5.expected_fbeta(qs, r_true, 1), color=CAT[1], label="F1, for contrast")
axes[1].set_xlabel("flagged fraction / true anomaly rate"); axes[1].set_ylabel("score of a perfect ranker")
axes[1].set_title("Why F3 punishes missed anomalies"); axes[1].legend(loc="lower right")
plt.tight_layout(); savefig(fig, "t5_threshold"); plt.show()
print(f"FINAL Task 5 flags: {pred5.sum():,} = {pred5.mean():.2%} of accounts (six gap-separated families {gap5.sum():,} + dependence pocket {pocket5.sum():,})")
AGREE5 = roc_auc_score(gap5, consensus)
summary.append(["Task 5", "ensemble ROC-AUC vs families (agreement)", AGREE5,
                "6 gap-separated families (1-D density-gap detector) + 1 dependence pocket (2-D scan, permutation-null cut-off) + 6-detector ensemble cross-check"])
""")

md(r"""
## 7.6 Final models: combine Train+Val (competition rule)
The rules say: *"Combine Train+Val for training"* and *"✅ Allowed: combining Train and Validation sets for final model training after all validation is complete."* Everything above was validation: the 5-fold OOF predictions are what we scored, tuned and stacked on. Each fold model saw only 80% of the labelled data. Now, with every decision fixed, each level-1 model is **retrained once on 100% of the labelled rows**:

* **GBDTs:** the validated hyper-parameters are reused. The number of boosting rounds comes from early stopping on fold 0, ×1.15 for the 25% larger training set, so no hold-out is needed.
* **ExtraTrees / logistic regression / hurdle stages:** refitted on all rows.
* **Task 4:** one extra model per backbone is trained on all 31,546 images with the identical recipe (`scripts/train_task4.py --full`).

The final level-1 test input is the **average of the 5-fold models and the full-data model**. Both are legitimate final models, and averaging them lowers variance. **Nothing is re-tuned:** the stacker, the threshold τ, the class multipliers, the L1 blend weights, the zero gate and the scale all keep their out-of-fold values. Task 5 has no labelled data, so its detectors are already fitted on all available data.
""")
code(r"""
from src.modeling import refit_full, refit_full_sklearn
FULL_W = 0.5                                     # weight of the full-data model in the final test input
blend_full = lambda fold_test, full_test: (1 - FULL_W) * np.asarray(fold_test) + FULL_W * np.asarray(full_test)
changes = []

# ---- Task 1
f1 = {}
f1["LightGBM"] = cached("t1_lgb_full", lambda: refit_full("lgb", "binary", P1_lgb, X1, y1, X1_test, folds1, name="T1 lgb"))
f1["XGBoost"] = cached("t1_xgb_full", lambda: refit_full("xgb", "binary", P1_xgb, X1, y1, X1_test, folds1, name="T1 xgb"))
f1["CatBoost"] = cached("t1_cat_full", lambda: refit_full("cat", "binary", P1_cat, X1, y1, X1_test, folds1, name="T1 cat"))
f1["ExtraTrees"] = cached("t1_et_full", lambda: refit_full_sklearn(make_et1, "binary", X1.values, y1, X1_test.values))
f1["LogReg"] = cached("t1_lr_full", lambda: refit_full_sklearn(make_lr1, "binary", X1.values, y1, X1_test.values))
f1["RF-24"] = cached("t1_rf24_full", lambda: {"test": np.mean([refit_full_sklearn(make_rf24(s), "binary", R1.values, y1, R1_test.values)["test"]
                                                               for s in SEEDS24], axis=0)})
for kind, nm in (("lgb", "LightGBM-24"), ("xgb", "XGBoost-24"), ("cat", "CatBoost-24")):
    f1[nm] = cached(f"t1_{kind}24_full", lambda kind=kind: {"test": np.mean([refit_full(kind, "binary", {**P24[kind], **seed_kw(kind, s)}, R1, y1,
                                                                                         R1_test, folds1, name=f"T1 {kind}24 seed {s}")["test"]
                                                                              for s in SEEDS24], axis=0)})
St1 = np.column_stack([logit(blend_full(m1[k]["test"], f1[k]["test"])) for k in m1])
stack1_test_final = STACKER1.predict_proba(St1)[:, 1]
pred1_final = (stack1_test_final >= THR1).astype(int)
changes.append(["Task 1", "labels changed", np.mean(pred1_final != pred1), f"flag rate {pred1.mean():.2%} -> {pred1_final.mean():.2%}"])

# ---- Task 2
f2m = {}
f2m["LightGBM"] = cached("t2_lgb_full", lambda: refit_full("lgb", "multiclass", P2_lgb, X2, y2, X2_test, folds2, name="T2 lgb"))
f2m["XGBoost"] = cached("t2_xgb_full", lambda: refit_full("xgb", "multiclass", P2_xgb, X2, y2, X2_test, folds2, name="T2 xgb"))
f2m["CatBoost"] = cached("t2_cat_full", lambda: refit_full("cat", "multiclass", P2_cat, X2, y2, X2_test, folds2, cat_cols=cat2, name="T2 cat"))
f2m["LogReg"] = cached("t2_lr_full", lambda: refit_full_sklearn(make_lr2, "multiclass", X2[num2].values, y2, X2_test[num2].values))
St2 = np.column_stack([np.log(np.clip(blend_full(m2[k]["test"], f2m[k]["test"]), 1e-6, 1)) for k in m2])
pred2_final = apply_class_weights(STACKER2.predict_proba(St2), W2)
changes.append(["Task 2", "labels changed", np.mean(pred2_final != pred2), ""])

# ---- Task 3
r = {}
r["LGB-L1 raw THB"] = cached("t3r_lgb_raw_full", lambda: refit_full("lgb", "regression", P3_lgb, X3, y3, X3_test, folds3, name="T3 lgb raw"))
r["XGB-L1 raw THB"] = cached("t3r_xgb_raw_full", lambda: refit_full("xgb", "regression", P3_xgb, X3, y3, X3_test, folds3, name="T3 xgb raw"))
r["Cat-MAE raw THB"] = cached("t3r_cat_raw_full", lambda: refit_full("cat", "regression", {"learning_rate": 0.08, "depth": 8}, X3, y3, X3_test, folds3, name="T3 cat"))
r["LGB-L1 log1p"] = cached("t3r_lgb_log_full", lambda: refit_full("lgb", "regression", P3_lgb, X3, np.log1p(y3), X3_test, folds3, name="T3 lgb log"))
r["LGB-Tweedie (mean)"] = cached("t3r_tweedie_full", lambda: refit_full("lgb", "regression", TW3, X3, y3, X3_test, folds3, name="T3 tweedie"))
hf = cached("t3r_hurdle_full", lambda: task3.hurdle_full("lgb", X3, y3, X3_test, folds3, S1_3, {"learning_rate": 0.05}))
full3 = {k: np.clip(v["test"], 0, None) for k, v in r.items()}
full3["LGB-L1 log1p"] = np.clip(np.expm1(r["LGB-L1 log1p"]["test"]), 0, None)
p_fin = blend_full(h["p_test"], hf["p_test"])
mu_fin = {"Hurdle": blend_full(h["mu_test"], hf["mu_test"])}
for tag, (kind, prm, bcol) in RATIO3.items():
    rf = cached(slug3(tag) + "_full", lambda: task3.ratio_stage2_full(kind, prm, X3, y3, X3_test, folds3, bcol))
    bt3 = np.log1p(X3_test[bcol].values)
    ok3 = np.isfinite(bt3)
    mu_fin[tag] = mu_fin["Hurdle"].copy()
    mu_fin[tag][ok3] = bt3[ok3] + blend_full(ratio3[tag]["test"], rf["test"])[ok3]
P3t_final = []
for k in names3:
    tag = next((t for t in mu_fin if k.startswith(t + " ")), None)
    if tag is not None:                                   # hurdle read-outs: recompute from the blended stage inputs
        P3t_final.append(task3.apply_readout(k[len(tag) + 1:], p_fin, mu_fin[tag], RO3[tag]))
    else:
        P3t_final.append(blend_full(m3[k][1], full3[k]))
gate_test_final = (np.clip(p_fin + RO3["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
pred3_final = np.minimum(np.clip(np.column_stack(P3t_final) @ W3 * SCALE3, 0, None) * gate_test_final, CAP3)
changes.append(["Task 3", "corr(log1p before, after)", np.corrcoef(np.log1p(pred3), np.log1p(pred3_final))[0, 1],
                f"zero share {np.mean(pred3 == 0):.3f} -> {np.mean(pred3_final == 0):.3f}"])

# ---- Task 4
test4_final = []
for tag in T4_TAGS:
    pth = C.PRED_DIR / f"t4_{tag}_full.npz"
    test4_final.append(blend_full(test_tag[tag], np.load(pth)["test_probs"]) if pth.exists() else test_tag[tag])
    print(f"Task 4 {tag}: full-data model {'included' if pth.exists() else 'NOT found -> 5-fold ensemble only'}")
pred4_final = apply_class_weights(np.mean(test4_final, axis=0), W4)
changes.append(["Task 4", "labels changed", np.mean(pred4_final != pred4), ""])

display(pd.DataFrame(changes, columns=["task", "check", "value", "note"]).set_index("task").round(4))
pred1, pred2, pred3, pred4 = pred1_final, pred2_final, pred3_final, pred4_final   # used by section 9
""")

# ======================================================== 8. feature importance
md(r"""
<a id="8"></a>
# 8. Feature Importance Analysis
Exact **TreeSHAP** (Lundberg & Lee) values come from a LightGBM model refit on fold 0 with the tuned parameters, explaining 4,000 held-out rows. They are complemented by model-agnostic **permutation importance**, the metric drop when a column is shuffled on the held-out fold. Task 4 uses **Grad-CAM** on the last ConvNeXt stage, and Task 5 uses per-account *anomaly drivers*.
""")
code(r"""
import lightgbm as lgb
from sklearn.inspection import permutation_importance

def fold0_lgb(params, X, y, objective, n_classes=None):
    tr, va = (folds1 if objective == "binary" else folds2 if objective == "multiclass" else folds3)[0]
    from src.modeling import LGB_BASE
    p = {**LGB_BASE, **params, "objective": {"binary": "binary", "multiclass": "multiclass", "regression": "regression_l1"}[objective]}
    Model = lgb.LGBMRegressor if objective == "regression" else lgb.LGBMClassifier
    m = Model(**p).fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m, va

def tree_shap(model, Xs):
    # LightGBM's built-in TreeSHAP (pred_contrib) - exact, handles categoricals; drops the bias column
    c = model.booster_.predict(Xs, pred_contrib=True, num_iteration=model.best_iteration_)
    k = Xs.shape[1] + 1
    return c.reshape(len(Xs), -1, k)[:, :, :-1] if c.shape[1] > k else c[:, :-1]

def shap_panel(ax, sv, Xs, title, top=14):
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values().tail(top)
    ax.barh(imp.index, imp.values, color=CAT[0], height=0.65)
    ax.set_title(title); ax.set_xlabel("mean |SHAP|"); ax.tick_params(axis="y", labelsize=8)
    return imp

def dependence(ax, sv, Xs, col, color=CAT[0]):
    j = list(Xs.columns).index(col)
    x = Xs[col].values
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    m = ~np.isnan(x) & (x >= lo) & (x <= hi)
    ax.scatter(x[m], sv[m, j], s=3, alpha=0.35, color=color, rasterized=True)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_xlabel(col); ax.set_ylabel("SHAP value")
""")
md("## 8.1 Task 1: what separates cheaters?")
code(r"""
mdl1, va1 = fold0_lgb(P1_lgb, X1, y1, "binary")
Xs1 = X1.iloc[va1].sample(4000, random_state=SEED)
sv1 = tree_shap(mdl1, Xs1)
pi1 = permutation_importance(mdl1, X1.iloc[va1], y1[va1], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
imp1 = shap_panel(axes[0], sv1, Xs1, "TreeSHAP importance (log-odds)")
pim = pd.Series(pi1.importances_mean, index=X1.columns).sort_values().tail(14)
axes[1].barh(pim.index, pim.values, color=CAT[1], height=0.65); axes[1].set_title("Permutation importance (AUC drop)")
axes[1].tick_params(axis="y", labelsize=8)
dependence(axes[2], sv1, Xs1, imp1.index[-1]); axes[2].set_title(f"SHAP dependence: {imp1.index[-1]}")
plt.tight_layout(); savefig(fig, "t1_shap"); plt.show()
""")
md(r"""
**Business reading (anti-cheat).** Five account-context signals carry the model: `reports_received`, `crosshair_placement`, `account_age_days`, `friend_network_size` and `level`. Both TreeSHAP and permutation importance agree on them. Engineered mechanical ratios add little on top. The SHAP dependence plot shows a **step function**:
* exactly 4 reports gives about +5 log-odds, and only cheaters have it;
* 5–15 reports gives a mild push towards cheating;
* **20 or more reports gives about −5 log-odds, and such accounts are always legitimate.**

In this data the most-reported players are **innocent targets of frustrated opponents**. That is exactly why Karena's current report-driven system has a 70% false-positive rate. Practical consequences:
* **Stop ranking the review queue by report count.** Rank it by model probability. At the tuned threshold the pre-filter keeps recall high, as $F_2$ demands, so few real cheaters slip past the 500-cases-per-day bottleneck.
* **Auto-clear mass-reported veterans.** Accounts with 20+ reports, long tenure and large friend networks can be deprioritised, which protects legitimate high-skill players from wrongful bans.
* **Show each case's top SHAP drivers** to reviewers, so a 10-minute manual review becomes a quick confirmation.
""")
md("## 8.2 Task 2: what defines each segment?")
code(r"""
mdl2, va2 = fold0_lgb(P2_lgb, X2, y2, "multiclass")
Xs2 = X2.iloc[va2].sample(4000, random_state=SEED)
sv2 = tree_shap(mdl2, Xs2)                     # (n, k, f)
if sv2.ndim == 3 and sv2.shape[1] == 4:
    per_class = pd.DataFrame({seg_names[k]: np.abs(sv2[:, k, :]).mean(0) for k in range(4)}, index=Xs2.columns)
else:
    per_class = pd.DataFrame({seg_names[k]: np.abs(sv2[..., k]).mean(0) for k in range(4)}, index=Xs2.columns)
top2 = per_class.sum(1).sort_values().tail(12).index
fig, ax = plt.subplots(figsize=(9, 4.6))
yy = np.arange(len(top2)); hgt = 0.2
for k, sname in enumerate(seg_names):
    ax.barh(yy + (k - 1.5) * hgt, per_class.loc[top2, sname], height=hgt, color=CAT[k], label=sname, edgecolor=viz.SURFACE, linewidth=0.8)
ax.set_yticks(yy); ax.set_yticklabels(top2, fontsize=8); ax.set_xlabel("mean |SHAP| (per-class log-odds)")
ax.set_title("Task 2: top drivers per segment"); ax.legend(loc="lower right")
plt.tight_layout(); savefig(fig, "t2_shap"); plt.show()
""")
md(r"""
**Business reading (segmentation).** Two engineered latent indices plus `loyalty_tier` and `engagement_level` drive almost every decision. A practical consequence: **two scores per player (engagement and spending) are enough to target campaigns**.
* **Whales** (high/high) should get VIP care and exclusive bundles, *without* blanket discounts.
* **Socials** (high spending, low engagement) respond to cosmetics and social events.
* **Grinders** (high engagement, low spending) are the audience for ranked and tournament messaging.
* **Casuals** should get light-touch retention offers rather than esports promotion.
""")
md("## 8.3 Task 3: what drives next-month spending?")
code(r"""
mdl3, va3 = fold0_lgb(P3_lgb, X3, y3, "regression")
Xs3 = X3.iloc[va3].sample(4000, random_state=SEED)
sv3 = tree_shap(mdl3, Xs3)
pi3 = permutation_importance(mdl3, X3.iloc[va3], y3[va3], scoring="neg_mean_absolute_error", n_repeats=3, random_state=SEED, n_jobs=1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
imp3 = shap_panel(axes[0], sv3, Xs3, "TreeSHAP importance (THB)")
pim3 = pd.Series(pi3.importances_mean, index=X3.columns).sort_values().tail(14)
axes[1].barh(pim3.index, pim3.values, color=CAT[1], height=0.65); axes[1].set_title("Permutation importance (MAE increase, THB)")
axes[1].tick_params(axis="y", labelsize=8)
dependence(axes[2], sv3, Xs3, imp3.index[-1]); axes[2].set_title(f"SHAP dependence: {imp3.index[-1]}")
plt.tight_layout(); savefig(fig, "t3_shap"); plt.show()
""")
md(r"""
**Business reading (spending).** Monetary history dominates. The engineered `monetary_mean_log` (mean of log historical and log previous-month spend) is by far the strongest driver, and its SHAP curve is flat below ~7 (≈1,100 THB) and then rises steeply: only established spenders move the forecast. After it come raw previous-month and historical spend, `segment` (Task 2's output feeds Task 3), premium membership and VIP tier. Days since the last purchase act as a smaller brake. For Karena:
* **VIP desk:** route support by *predicted* 30-day spend rather than playtime. This avoids last quarter's mistake of white-glove service for 150 heavy players who spent little while a 255k-THB whale waited three days.
* **Discount targeting:** accounts whose predicted spend is high *without* any discount do not need a 50% coupon. Concentrate discounts on the predicted-zero/low band with recent activity.
* **Revenue forecasting:** the sum of predicted medians is a conservative, outlier-robust forecast. Pair it with the Tweedie mean model when finance needs an expected-value total.
""")
md("## 8.4 Task 4: where does the CNN look? (Grad-CAM)")
code(r"""
cam_model = task4.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False).to(DEVICE)
cam_model.load_state_dict(torch.load(C.MODEL_DIR / "t4_cnxt_fold0.pt", map_location=DEVICE))
gc = task4.GradCAM(cam_model, cam_model.stages[-1])
z0 = np.load(C.PRED_DIR / "t4_cnxt_fold0.npz")
va_idx0, vp0 = z0["val_idx"], z0["val_probs"]
pick = []
for k in range(5):
    ok = va_idx0[(y4[va_idx0] == k) & (vp0.argmax(1) == k)]
    pick += list(np.random.default_rng(k).choice(ok, 2, replace=False))
arr = task4.load_uint8([t4_train.path.iloc[i] for i in pick])
x = task4.normalize(task4.to_gpu_batch(arr, (160, 320), DEVICE))
cams, probs = gc(x)
fig, axes = plt.subplots(2, 5, figsize=(14, 3.6))
for i, ax in enumerate(axes.T.ravel()):
    img = task4.to_gpu_batch(arr[i:i + 1], (160, 320), DEVICE)[0].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img); ax.imshow(cams[i], cmap="magma", alpha=0.45); ax.axis("off")
    k = y4[pick[i]]
    ax.set_title(f"{C.T4_CLASS_NAMES[k]}  p={probs[i, k]:.2f}", fontsize=8.5)
fig.suptitle("Grad-CAM on the last ConvNeXt stage (fold-0 model, held-out images)", y=1.02, fontsize=11)
plt.tight_layout(); savefig(fig, "t4_gradcam"); plt.show()
del cam_model; torch.cuda.empty_cache()
""")
md(r"""
**Business reading (game detection).** The attention maps land on **game-specific HUD and art assets**: menus, minimaps, character art and vehicle or world textures, rather than on the noise or the black rotation borders. This shows the corruption-matched augmentation taught the network to ignore artefacts. On in-distribution screenshots (OOF macro-F1 ≈ 0.99) the classifier can auto-route the 2,000 daily support tickets to the right game team. It can also verify "Play Multiple Games" campaign screenshots at scale. The §7.4 shift analysis argues for a **confidence gate** in production: predictions with max probability < 0.7 go to a human. That is under 2% of in-distribution traffic, but about a fifth of the shifted test-style traffic, and those hard cases are the ones that should be collected and labelled to retrain the model.
""")
md("## 8.5 Task 5: why was each account flagged?")
code(r"""
flagged = np.where(pred5 == 1)[0]
# reason = the column that isolates the account most clearly (largest expected-in-gap among its clusters)
order = gc_table.group.tolist()
G = gc_masks[order].values[flagged]
R5 = rules5.values[flagged]
reason = pd.Series([gc_table.column[int(np.argmax(g))] if g.any() else
                    ("pocket: " + " x ".join(pocket_feats5) if pocket5[i] else
                     f"rule: {rules5.columns[int(np.argmax(r))]}" if r.any() else "score ensemble (multivariate)")
                    for i, g, r in zip(flagged, G, R5)]).str.replace(r"_[1-4]$", "", regex=True)
drv = reason.value_counts().head(14).sort_values()
fam_counts = pd.Series({k: int((m & (pred5 == 1)).sum()) for k, m in fam_mask.items()})
if pocket5.any():
    fam_counts["erratic performance (pocket)"] = int(pocket5.sum())
if (flag5 & ~gap5 & ~pocket5).any():
    fam_counts["no family (ensemble / rules)"] = int((flag5 & ~gap5 & ~pocket5).sum())
fam_counts = fam_counts.sort_values()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].barh(drv.index, drv.values, color=CAT[0], height=0.65); axes[0].set_title("Clearest reason for each alert (isolating column)")
axes[0].tick_params(axis="y", labelsize=8)
axes[1].barh(fam_counts.index, fam_counts.values, color=CAT[0], height=0.6)
for i, v in enumerate(fam_counts.values):
    axes[1].text(v, i, f" {v}", va="center", fontsize=8.5)
axes[1].set_title("Alerts by anomaly family"); axes[1].tick_params(axis="y", labelsize=8.5)
plt.tight_layout(); savefig(fig, "t5_drivers"); plt.show()
""")
md(r"""
**Business reading (security).** Every alert comes with a family and the behaviour that isolates it, so the security team can route each one:
* **Takeover / VPN abuse** (teleporting logins with a currency drain, or with a purchase spike behind a VPN): freeze outgoing transfers, force a password reset, add step-up authentication, and re-price VPN purchases.
* **Skill boost + friend purge** (sudden skill jump in the last period, friends removed): account sharing or selling. Review, and notify the owner.
* **Bots** (one IP, clockwork login times, constant session length, flat KDA): rate-limit, then batch-ban by IP.
* **Erratic performance** (the dependence pocket: KDA and accuracy swing far more from match to match than for any normal player): shared, boosted or resold accounts. Review; these alerts are the least certain (estimated precision about 0.5), which F3 accepts.
* **Silent and near-inactive accounts** (no chat at all, or barely active in every period): farm and mule accounts. Watch for transfers into them.
* **Erratic behaviour** (stability far below any normal player): resellers and shared accounts. Review.
* **Ensemble or rule alerts outside the families** (not flagged in the submission): a low-priority watch list, since they are statistically closest to normal players.

The current rule system has a 60% false-positive rate and treats every alert the same way. Typed, explained alerts let a small team triage efficiently. The family detector also needs no hand-set threshold: each gap in the data defines its own cut-off, so it adapts when behaviour drifts.
""")

# =================================================================== 9. submission
md(r"""
<a id="9"></a>
# 9. Submission
Row *i* of each task's `test.csv` maps to `ANS{i:05d}`. The file has the same ids, column order and comma delimiter as `sample_submission.csv`.

Class columns (Tasks 1, 2, 4, 5) are written as **plain integer codes** (`0`, `1`, ...). That is the coding the datacards define, e.g. Task 4 `0: 7-11 Knights … 4: Roblock`. The `1.0` style in `sample_submission.csv` is only an artefact of that file's empty cells, which force pandas to use floats. An integer code reads identically whether the scorer compares numbers or text. Task 3 is a THB amount rounded to 2 decimals. Tasks 1 and 5 are thresholded for the positive-class $F_\beta$ that the slides define.

**Open questions go to the TAs, not the leaderboard.** The rules forbid test-set probing and cap daily submissions, so we do *not* upload variants to see which scores better. Two conventions are ambiguous in the course material:
* the Task 4 label format: integer codes in the sample/datacard versus text names on the PDF slide;
* $F_\beta$ averaging: the positive-class formula on the slides versus "macro" in our original brief.

We ask the TAs (rule 8: *"When in doubt, ask first!"*). For the answer they give, the matching file is already written to `outputs/format_variants/`: the same predictions, only re-formatted or re-thresholded.
""")
code(r"""
sub = sample_sub[["id"]].copy()
sub["task1"] = pred1.astype(int)
sub["task2"] = pred2.astype(int)
sub["task3"] = np.round(pred3, 2)
sub["task4"] = pred4.astype(int)
sub["task5"] = pred5.astype(int)

# ---- sanity checks
expected_ids = [f"ANS{i:05d}" for i in range(1, C.N_TEST + 1)]
assert sub.shape == (25_889, 6), sub.shape
assert list(sub.columns) == list(sample_sub.columns)
assert sub.id.tolist() == expected_ids == sample_sub.id.tolist(), "ID order mismatch"
assert sub.isna().sum().sum() == 0, "NaNs in submission"
assert set(sub.task1.unique()) <= {0, 1} and set(sub.task5.unique()) <= {0, 1}
assert set(sub.task2.unique()) <= {0, 1, 2, 3} and set(sub.task4.unique()) <= {0, 1, 2, 3, 4}
assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
assert (sub.task3 >= 0).all() and np.isfinite(sub.task3).all()

out_path = C.ROOT / "submission.csv"
sub.to_csv(out_path, index=False)
check = pd.read_csv(out_path)
assert check.shape == (25_889, 6) and open(out_path).readline().strip() == "id,task1,task2,task3,task4,task5"
print(f"saved {out_path} | {check.shape} | delimiter ',' | all checks passed")
display(sub.head(8))

# Format / convention variants of the SAME predictions, for when the TAs answer the open questions.
# They are not leaderboard probes: upload only the single variant that matches the TAs' answer.
var_dir = C.OUT / "format_variants"
var_dir.mkdir(exist_ok=True)
for old in var_dir.glob("submission_*.csv"):
    old.unlink()
variants = {
    "float_labels": sub.astype({c: float for c in ["task1", "task2", "task4", "task5"]}),
    "task4_text_labels": sub.assign(task4=pd.Series(pred4).map(dict(enumerate(C.T4_CLASSES))).values),
    f"task1_{OTHER_AVG}_F2": sub.assign(task1=(stack1_test_final >= THR1_alt).astype(int)),
}
for name, df in variants.items():
    df.to_csv(var_dir / f"submission_{name}.csv", index=False)
print(f"format variants in {var_dir.name}/: {sorted(variants)}")
print(f"  task1_{OTHER_AVG}_F2: τ={THR1_alt:.3f}, flags {variants[f'task1_{OTHER_AVG}_F2'].task1.mean():.1%} (main file: τ={THR1:.3f}, flags {sub.task1.mean():.1%})")
""")
code(r"""
dist = pd.DataFrame({
    "task1 cheater rate": [sub.task1.mean(), y1.mean()],
    "task2 Casual/Grinder/Social/Whale": [np.round(np.bincount(pred2, minlength=4) / len(pred2), 3).tolist(), np.round(np.bincount(y2) / len(y2), 3).tolist()],
    "task3 zero share | median | mean": [f"{(sub.task3 == 0).mean():.3f} | {sub.task3.median():,.0f} | {sub.task3.mean():,.0f}",
                                         f"{(y3 == 0).mean():.3f} | {np.median(y3):,.0f} | {y3.mean():,.0f}"],
    "task4 class shares": [np.round(np.bincount(pred4, minlength=5) / len(pred4), 3).tolist(), np.round(np.bincount(y4) / len(y4), 3).tolist()],
    "task5 flagged rate": [sub.task5.mean(), np.nan],
}, index=["test predictions", "train reference"]).T
display(dist)
""")
md(r"""
The prediction distributions are consistent with the training priors, with a few explainable departures:
* **Task 1** flags more players than the 35% base rate, because the $F_2$-optimal threshold trades precision for recall.
* **Task 3** predicts *exact zeros* for roughly the non-spender share, thanks to the median-consistent gate, while keeping whales' scale, so the mean stays close to the training mean.
* **Task 4**'s class mix differs from training. This is the prior shift diagnosed in §7.4, not a pipeline error: the same code reproduces the training mix on OOF predictions.
""")

# =================================================================== 10. conclusion
md(r"""
<a id="10"></a>
# 10. Conclusion
""")
code(r"""
board = pd.DataFrame(summary, columns=["task", "metric", "OOF score", "final recipe"]).set_index("task")
display(board.style.format({"OOF score": "{:.4f}"}))
print("Task 5 has no labels: its row is the agreement (ROC-AUC) between the score ensemble and the gap families, not an accuracy.")
""")
md(r"""
### Methodology recap
| Task | Key insight | What moved the score |
|---|---|---|
| 1 Anti-cheat | Noisy additive signal; recall-weighted metric; test rows have fewer missing values | Cost-sensitive GBDTs (`scale_pos_weight` tuned) → 5-model stack → **threshold moving** on OOF (optimum inside a flat one-SE band) |
| 2 Segmentation | 2×2 latent design (engagement × spending) | **Noise-averaging latent indices**, ordinal tiers; stack + **class-wise multipliers** |
| 3 Spending | NMAE ⇔ total absolute error ⇒ predict the **median**; spending ≈ history × segment factor × noise | **Data repair** (exact row identities, rule-based segment), **ratio-target hurdles** (LightGBM / XGBoost / CatBoost) with mixture-median read-out, L1 GBDTs on raw THB; OOF L1 blend, median-consistent zero gate, scale |
| 4 Game title | Corrupted screenshots, 2:1 aspect; shifted test set | Aspect-preserving inputs, **GPU corruption-matched augmentation**, Mixup/CutMix, EMA, TTA; **CNN + ViT ensemble** for shift robustness |
| 5 Security | Anomalies are **separate populations**: each family sits beyond an empty gap in some behaviour | **1-D density-gap detector** (expected-in-gap statistic) finds 6 families with no hand-set threshold; a second scan of the derived change features finds nothing outside them |

For Tasks 1–4, the final test predictions average the 5-fold models with models **retrained on all labelled data** (Train+Val, per the rules, §7.6). Every decision rule keeps its out-of-fold value.

### Business impact for Karena Thailand
* **Anti-cheat:** a high-recall pre-filter turns 10,000 daily reports into a ranked queue. The 500 daily manual reviews go to the most probable cheaters first, and SHAP reasons speed up each decision.
* **Segmentation:** two interpretable scores route the ฿60M monthly campaign budget to the right message per segment, the precondition for the targeted 3–4× lift in campaign effectiveness.
* **Spending:** account-level median forecasts drive VIP routing and discount suppression for whales (the ฿6M+ monthly leakage), and support conservative revenue forecasting.
* **Game detection:** near-perfect automatic routing of screenshot tickets and campaign verification removes about 10 minutes of triage per ticket.
* **Security:** typed, explained alerts (takeover / VPN abuse / reseller / bot farm) replace a rule system with a 60% false-positive rate, and each alert type comes with a specific playbook.

### Limitations and future improvements
* **Metric averaging.** The macro-vs-positive-class choice for $F_2$ moves the optimal Task 1 threshold. We follow the slides (positive class). If the TAs confirm macro, flip `FBETA_AVERAGE` and re-run.
* **Task 1** uses 13k unlabeled training rows only implicitly. Semi-supervised learning on *training* rows (never test) is a natural next step, as are a focal-loss LightGBM objective and monotone constraints for interpretability.
* **Task 3** could add distributional GBDTs (quantile regression at the mixture-median level, or NGBoost-style log-normal heads) to replace the constant-σ log-normal assumption.
* **Task 4** test images show a covariate shift (unseen scenes, lower confidence) and a class-prior shift relative to training, so the leaderboard will likely sit below the OOF score. Next steps:
  * larger or CLIP-pretrained backbones, which are known to be more shift-robust (a quick DINOv2-S run underfit at a conservative learning rate; ViT-B/16 costs ~4.5 GPU-hours per 5-fold run on this laptop);
  * multi-scale/multi-crop TTA;
  * if the rules permit transductive post-processing, EM label-shift correction of the class priors.
* **Task 5:** with a small analyst-labelled sample, the fusion weights and contamination level could be calibrated directly, and a temporal autoencoder could model the 4-period sequences.
""")
