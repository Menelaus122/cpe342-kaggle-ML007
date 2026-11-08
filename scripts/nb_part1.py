"""Notebook cells, part 1: title + sections 1-4."""

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# =============================================================================== title
md(r"""
# Karena Thailand Player Intelligence: CPE342 Master Pipeline

End-to-end, reproducible solution for the five Karena Player Intelligence tasks:

| # | Task | Type | Target | Metric |
|---|------|------|--------|--------|
| 1 | Anti-Cheat Pre-Filter | Binary classification | `is_cheater` | $F_2$ |
| 2 | Player Segmentation | 4-class classification | `segment` | Macro $F_1$ |
| 3 | Monthly Spending | Zero-inflated regression | `spending_30d` | NMAE |
| 4 | Game Title Detection | 5-class image classification | `class` | Macro $F_1$ |
| 5 | Account Security | Unsupervised anomaly detection | `is_anomaly` | $F_3$ |

**Competition rules, and how this notebook follows them**

| Rule | How it is met |
|---|---|
| Provided datasets only; no external data | Only the five task folders are read. Pre-trained ImageNet weights are used in Task 4 alone, which the rules allow |
| No test data for training (no pseudo-labelling / fine-tuning) | Test features are only ever *predicted*. Encoders, scalers and statistics are fitted on training rows (target encoding inside each CV fold). Task 5 ships only an unlabeled `test.csv`, so its unsupervised detectors are fitted on it as the brief intends; no labels exist |
| Validate on Train/Val, then *combine Train+Val* for the final model | Every score and decision comes from 5-fold out-of-fold validation. **§7.6 then retrains every model on 100% of the labelled data** |
| Tasks 1, 2, 3, 5 trained from scratch | Gradient-boosted trees, ExtraTrees, logistic regression and a small autoencoder, all initialised from scratch |
| No AutoML; tuning tools allowed | Hand-built pipelines; Optuna for hyper-parameters |
| No test-set probing; daily submission limits | No leaderboard-driven choices. Format and metric ambiguities are put to the TAs (see §9) |

**Code layout.** Reusable logic lives in `src/` (one module per task plus shared metrics/CV engine). The GPU-heavy Task 4 fine-tuning runs from `scripts/train_task4.py`, and Optuna studies from `scripts/tune.py`. Both persist artifacts in `outputs/`, which this notebook loads, or re-creates when the caches are absent.

## Table of Contents
1. [Imports & Configuration](#1)
2. [Data Loading & Overview](#2)
3. [Exploratory Data Analysis](#3)
4. [Data Preparation & Feature Engineering](#4)
5. [Model Training & Evaluation](#5)
6. [Hyperparameter Tuning](#6)
7. [Stacking Ensemble (K-Fold)](#7)
8. [Feature Importance Analysis](#8)
9. [Submission](#9)
10. [Conclusion](#10)
""")

# ======================================================================= 1. imports
md(r"""
<a id="1"></a>
# 1. Imports & Configuration

Global seeds, warning suppression, GPU discovery and the **exact competition metrics**:

* $F_\beta = (1+\beta^2)\dfrac{P\cdot R}{\beta^2 P + R}$ with $\beta=2$ (Task 1) and $\beta=3$ (Task 5). Both weight recall $\beta^2$ times more than precision.
* Macro $F_1$: the unweighted mean of per-class $F_1$, so minority classes count as much as majority ones (Tasks 2 and 4).
* $\text{NMAE} = \dfrac{1}{1 + \sum|y-\hat y| / \sum y}$ (Task 3). It is a monotone transform of total absolute error, so the optimal forecast is the conditional **median**.

> **Averaging switch.** The official instruction slides define $F_2 = 5PR/(4P+R)$ and $F_3 = 10PR/(9P+R)$, the textbook positive-class formulas (cheater = 1, anomaly = 1). Only Task 4 is marked "F1 (Macro)". We therefore optimise the **positive-class** score (`FBETA_AVERAGE = "binary"` in `src/config.py`). The optimal threshold differs materially between the two readings, so the macro score is reported alongside, and one switch flips the whole pipeline.
""")
code(r"""
import os, sys, json, time, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C
from src.config import SEED, N_FOLDS, FBETA_AVERAGE
from src.utils import seed_everything, quiet, reduce_mem_usage, timer, save_preds, load_preds, load_params
from src.metrics import (fbeta, f2, f3, macro_f1, nmae, optimize_threshold, threshold_curve,
                         optimize_class_weights, apply_class_weights)
from src.modeling import make_folds, run_cv, run_cv_sklearn
from src import viz
from src.viz import CAT, SEQ_CMAP, DIV_CMAP, MUTED, savefig

quiet()
seed_everything(SEED)
viz.set_theme()
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

import torch, lightgbm, xgboost, catboost, sklearn, optuna
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | lightgbm {lightgbm.__version__} | "
      f"xgboost {xgboost.__version__} | catboost {catboost.__version__} | sklearn {sklearn.__version__} | optuna {optuna.__version__}")
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
AVG_NAME = "positive-class" if FBETA_AVERAGE == "binary" else "macro"
print(f"SEED={SEED}  N_FOLDS={N_FOLDS}  FBETA_AVERAGE={FBETA_AVERAGE!r}")

# Execution switches: heavy artifacts are cached in outputs/ and re-used unless forced.
RUN_OPTUNA = False          # True -> (re)run Optuna studies in-notebook (slow); False -> load stored studies
RUN_T4_TRAINING = False     # True -> launch the Task 4 GPU K-fold fine-tuning if fold caches are missing
""")
code(r"""
# Sanity-check the metric implementations on toy inputs
y_t = np.array([1, 1, 1, 0, 0, 0, 0, 0])
y_p = np.array([1, 1, 0, 1, 0, 0, 0, 0])
P, R = 2 / 3, 2 / 3
assert abs(fbeta(y_t, y_p, 2, "binary") - 5 * P * R / (4 * P + R)) < 1e-12
assert abs(fbeta(y_t, y_p, 3, "binary") - 10 * P * R / (9 * P + R)) < 1e-12
assert abs(nmae([0, 10, 30], [5, 10, 20]) - 1 / (1 + 15 / 40)) < 1e-12
print("metric checks passed | macro-F2 toy:", round(f2(y_t, y_p, 'macro'), 4), "| binary-F2 toy:", round(f2(y_t, y_p, 'binary'), 4))
""")

# ===================================================================== 2. loading
md(r"""
<a id="2"></a>
# 2. Data Loading & Overview

Each task ships its own `train.csv`/`test.csv`. Row *i* of every task's `test.csv` maps **positionally** to submission id `ANS{i:05d}` (all five test sets have exactly 25,889 rows). Task 4 images (230×120 JPEG) were downloaded from the course Drive link in `04_image_classification/README.md`.
""")
code(r"""
from src import task1, task2, task3, task4, task5

t1_train, t1_test, t1_unlabeled = task1.load()
t2_train, t2_test = task2.load()
t3_train, t3_test = task3.load()
t4_train, t4_test = task4.load_frames()
t5_test = task5.load()
sample_sub = pd.read_csv(C.SAMPLE_SUB)

print(f"Task 1: dropped {t1_unlabeled:,} training rows whose label is missing (cannot be used for supervised learning)")
overview = pd.DataFrame([
    ["1 Anti-cheat", len(t1_train), len(t1_test), t1_test.shape[1] - 1, "is_cheater", "F2 (2 classes)", t1_train.drop(columns="is_cheater").isna().mean().mean()],
    ["2 Segmentation", len(t2_train), len(t2_test), t2_test.shape[1] - 1, "segment", "Macro F1 (4 cls)", t2_train.drop(columns="segment").isna().mean().mean()],
    ["3 Spending", len(t3_train), len(t3_test), t3_test.shape[1] - 1, "spending_30d", "NMAE", t3_train.drop(columns="spending_30d").isna().mean().mean()],
    ["4 Game title", len(t4_train), len(t4_test), "230x120 RGB", "class", "Macro F1 (5 cls)", 0.0],
    ["5 Security", 0, len(t5_test), t5_test.shape[1] - 1, "is_anomaly", "F3 (no labels)", t5_test.isna().mean().mean()],
], columns=["task", "train_rows", "test_rows", "features", "target", "metric", "missing_rate"])
display(overview)
assert all(len(t) == C.N_TEST for t in (t1_test, t2_test, t3_test, t4_test, t5_test)), "test sizes must equal 25,889"
print("sample_submission:", sample_sub.shape, list(sample_sub.columns), "| ids", sample_sub.id.iloc[0], "...", sample_sub.id.iloc[-1])
""")
code(r"""
# Memory optimisation: downcast numerics (lossless for these value ranges)
for name, df in [("t1_train", t1_train), ("t2_train", t2_train), ("t3_train", t3_train), ("t5_test", t5_test)]:
    print(f"{name:9s}", end=" ")
    reduce_mem_usage(df, verbose=True)

# Missing-value profile: every tabular task has ~5-12% MCAR-looking gaps spread over all columns
miss = pd.DataFrame({
    "task1": t1_train.drop(columns=["player_id", "is_cheater"]).isna().mean().describe()[["mean", "min", "max"]],
    "task2": t2_train.drop(columns=["player_id", "segment"]).isna().mean().describe()[["mean", "min", "max"]],
    "task3": t3_train.drop(columns=["player_id", "spending_30d"]).isna().mean().describe()[["mean", "min", "max"]],
    "task5": t5_test.drop(columns=["player_id"]).isna().mean().describe()[["mean", "min", "max"]],
}).T.rename(columns=lambda c: f"col_missing_{c}")
display(miss)
""")
md(r"""
**Data card summary**

* **Task 1** has 31 behavioural / mechanical-skill features. 13,184 training rows carry no label and are dropped. Classes are 65% legitimate and 35% cheater.
* **Task 2** has 34 numeric plus 10 categorical features. The datacard itself flags `random_metric_1..3` as pure noise. Several tier columns are ordinal.
* **Task 3** has 32 numeric features, including `segment` from Task 2. The target is in THB, with 48% exact zeros and a long right tail (max 300k).
* **Task 4** labels are strings. The datacard's integer coding is their alphabetical order: `0 = 7-11k, 1 = fiveN, 2 = freefried, 3 = pube, 4 = roblock`. `sample_submission.csv` holds integers, so we submit integers.
* **Task 5** has 29 behaviours × 4 consecutive periods, plus static account attributes and 4 security flags. There are no labels.
""")

# ========================================================================= 3. EDA
md(r"""
<a id="3"></a>
# 3. Exploratory Data Analysis
## 3.1 Task 1: Anti-cheat
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 2.2]})
counts = pd.Series({"Legitimate": (t1_train.is_cheater == 0).sum(), "Cheater": (t1_train.is_cheater == 1).sum(), "Unlabeled\n(dropped)": t1_unlabeled})
axes[0].bar(counts.index, counts.values, color=[CAT[0], CAT[1], MUTED], width=0.6, **viz.bar_gap())
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
axes[0].set_title("Label balance"); axes[0].set_ylabel("rows")

# Univariate separability (|Cohen's d|) of every feature
num = t1_train.drop(columns=["player_id", "is_cheater"])
g = t1_train.groupby("is_cheater")
d = ((g[num.columns].mean().loc[1] - g[num.columns].mean().loc[0]) / num.std()).sort_values(key=np.abs)
axes[1].barh(d.index[-14:], d.values[-14:], color=[CAT[1] if v > 0 else CAT[0] for v in d.values[-14:]], height=0.65)
axes[1].axvline(0, color=MUTED, lw=0.8)
axes[1].set_title("Top-14 features by standardized mean difference (cheater − legit)")
axes[1].set_xlabel("Cohen's d  (orange: higher for cheaters, blue: lower)")
plt.tight_layout(); savefig(fig, "t1_eda_balance"); plt.show()
""")
code(r"""
top6 = d.abs().sort_values(ascending=False).index[:6]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
for ax, c in zip(axes.ravel(), top6):
    for k, lab in [(0, "Legitimate"), (1, "Cheater")]:
        sns.kdeplot(t1_train.loc[t1_train.is_cheater == k, c].dropna(), ax=ax, color=CAT[k], lw=2, label=lab, cut=0)
    ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
axes[0, 0].legend()
fig.suptitle("Task 1: class-conditional densities of the most separating features", y=1.0, fontsize=11)
plt.tight_layout(); savefig(fig, "t1_eda_kde"); plt.show()

fig, ax = plt.subplots(figsize=(9.5, 7.5))
corr = t1_train.drop(columns="player_id").corr()
sns.heatmap(corr, cmap=DIV_CMAP, vmin=-1, vmax=1, center=0, square=True, cbar_kws={"shrink": 0.6}, ax=ax,
            xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=6.5); ax.set_title("Task 1: Pearson correlation (incl. target)")
plt.tight_layout(); savefig(fig, "t1_eda_corr"); plt.show()
print("Strongest linear correlations with is_cheater:")
print(corr["is_cheater"].drop("is_cheater").sort_values(key=np.abs, ascending=False).head(8).round(3).to_string())
""")
md(r"""
**Findings.**
* **Every top feature is *lower* for cheaters** in this data: `reports_received` (d ≈ −0.7), `crosshair_placement`, K/D, headshot %, game sense, account age and level. This contradicts the textbook aimbot story of inflated K/D and headshot %, so the model must learn from the data rather than from domain priors. The counter-intuitive report sign matters too: heavily reported accounts are mostly *legitimate* players reported by frustrated opponents, which matches the brief's 70% false-positive complaint.
* **Class-specific supports.** Several features look near-uniform with *shifted* ranges per class. `reports_received` > 19, `account_age_days` > 1,000 and `crosshair_placement` > 0.94 occur only for legitimate players, and crosshair < 0.82 occurs only for cheaters. Trees capture these hard boundaries for free, and the residual difficulty lives in the overlap region.
* Pairwise correlations are weak and the separating signal is spread over many features. That is GBDT territory, with a realistic AUC ceiling near 0.89.

## 3.2 Task 2: Player segmentation
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 2.3]})
seg_names = ["Casual", "Grinder", "Social", "Whale"]
vc = t2_train.segment.value_counts().sort_index()
axes[0].bar(seg_names, vc.values, color=CAT[0], width=0.6, **viz.bar_gap())
for i, v in enumerate(vc.values):
    axes[0].text(i, v, f"{v / vc.sum():.0%}", ha="center", va="bottom", fontsize=8.5)
axes[0].set_title("Segment distribution (imbalanced)")

num2 = t2_train.select_dtypes("number").drop(columns=["segment"])
zmed = ((t2_train.groupby("segment")[num2.columns].mean() - num2.mean()) / num2.std()).T
zmed.columns = seg_names
zmed = zmed.loc[zmed.abs().max(axis=1).sort_values(ascending=False).index[:18]]
sns.heatmap(zmed.T, cmap=DIV_CMAP, center=0, vmin=-0.8, vmax=0.8, ax=axes[1], cbar_kws={"shrink": 0.8, "label": "mean z-score"})
axes[1].set_title("Per-segment mean z-score: a 2x2 design (engagement x spending)")
axes[1].tick_params(axis="x", labelsize=7, rotation=60)
plt.tight_layout(); savefig(fig, "t2_eda_profile"); plt.show()
""")
code(r"""
cats = ["loyalty_tier", "engagement_level", "region", "player_type_tag"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
for ax, c in zip(axes, cats):
    ct = pd.crosstab(t2_train[c].fillna("NA"), t2_train.segment, normalize="index")
    ct.columns = seg_names
    sns.heatmap(ct, cmap=SEQ_CMAP, vmin=0, vmax=0.65, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False, ax=ax)
    ax.set_title(c); ax.set_ylabel(""); ax.tick_params(labelsize=7.5)
fig.suptitle("P(segment | category): tiers are informative, region/tag are noise", y=1.03, fontsize=11)
plt.tight_layout(); savefig(fig, "t2_eda_cats"); plt.show()
""")
md(r"""
**Findings.** The four segments sit on a clean **2×2 latent design**:

| | low spending | high spending |
|---|---|---|
| **low engagement** | Casual (0) | Social (2) |
| **high engagement** | Grinder (1) | Whale (3) |

Engagement is carried by many weak indicators (play frequency, session length, streaks, friends, chat, tournaments, peak hours), and spending by another set (total and monthly spend, purchase frequency, rare items, VIP, loyalty tier). `loyalty_tier` and `engagement_level` are strong ordinal proxies. `region`, `platform`, `device_type`, `payment_method`, `language`, `account_status` and `player_type_tag` show flat conditional distributions, i.e. no signal. The `random_metric_*` columns are documented noise.

*Data-quality note:* `region` shows 26% missing values versus about 8% elsewhere. pandas' default `na_values` parse the region code **"NA" (North America)** as missing, so that bucket merges North America with genuinely missing rows. It is harmless here because region carries no signal. In production, read that column with `keep_default_na=False`.

## 3.3 Task 3: Monthly spending
""")
code(r"""
y3 = t3_train.spending_30d
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
axes[0].hist(np.log1p(y3), bins=80, color=CAT[0])
axes[0].set_title(f"log1p(spending_30d): {(y3 == 0).mean():.1%} exact zeros")
axes[0].set_xlabel("log1p(THB)")
hb = axes[1].hexbin(np.log1p(t3_train.prev_month_spending), np.log1p(y3), gridsize=55, cmap=SEQ_CMAP, bins="log", mincnt=1)
axes[1].set_xlabel("log1p(prev_month_spending)"); axes[1].set_ylabel("log1p(spending_30d)")
axes[1].set_title("Strongest predictor, with a zero 'hurdle' band"); axes[1].grid(False)
seg = t3_train.groupby("segment").spending_30d.agg(zero_rate=lambda s: (s == 0).mean(), median="median")
axes[2].bar(seg_names, seg["zero_rate"], color=CAT[0], width=0.6, **viz.bar_gap())
for i, (zr, m) in enumerate(zip(seg["zero_rate"], seg["median"])):
    axes[2].text(i, zr, f"median\n{m:,.0f}", ha="center", va="bottom", fontsize=7.5)
axes[2].set_ylim(0, 0.95); axes[2].set_title("Share of non-spenders by segment"); axes[2].set_ylabel("P(spending_30d = 0)")
plt.tight_layout(); savefig(fig, "t3_eda_target"); plt.show()
print("skewness of y:", round(y3.skew(), 2), "| quantiles of positive y:", y3[y3 > 0].quantile([.1, .5, .9, .99]).round(0).to_dict())
print("Top correlations with target:", t3_train.drop(columns="player_id").corr()["spending_30d"].drop("spending_30d").sort_values(ascending=False).head(6).round(3).to_dict())
""")
md(r"""
**Findings.** The target is **zero-inflated** (48% zeros) and **heavily right-skewed** (skew ≈ 4.2). Past spending (`historical_spending`, `prev_month_spending`, ρ ≈ 0.87) dominates. Among spenders, log-spend follows log prev-month spend in a tight band. So *how much* a spender spends is very predictable, and the hard part is the **zero hurdle**: whether they spend at all. The segment from Task 2 is the next strongest signal: 72% of Casuals spend nothing, while Whales have a median of ~55k THB. Because NMAE is an absolute-error metric, the zero mass matters. Any account whose probability of spending is below 50% should be predicted as exactly 0.

## 3.4 Task 4: Game screenshots
""")
code(r"""
from PIL import Image
fig, axes = plt.subplots(5, 6, figsize=(13, 5.4))
for r, cls in enumerate(C.T4_CLASSES):
    ids = t4_train[t4_train["class"] == cls].sample(6, random_state=7)
    for c, p in enumerate(ids.path):
        axes[r, c].imshow(Image.open(p)); axes[r, c].axis("off")
    axes[r, 0].text(-12, 60, f"{r}: {C.T4_CLASS_NAMES[r]}", ha="right", va="center", fontsize=9)
fig.suptitle("Training screenshots per class (230x120): rotated, colour-inverted, hue-shifted, noisy", y=1.0, fontsize=11)
plt.tight_layout(); savefig(fig, "t4_eda_samples"); plt.show()

fig, ax = plt.subplots(figsize=(6.5, 2.6))
vc4 = t4_train["class"].value_counts().reindex(C.T4_CLASSES)
ax.barh(C.T4_CLASS_NAMES[::-1], vc4.values[::-1], color=CAT[0], height=0.6, **viz.bar_gap())
for i, v in enumerate(vc4.values[::-1]):
    ax.text(v, i, f" {v:,} ({v / vc4.sum():.0%})", va="center", fontsize=8.5)
ax.set_xlim(0, vc4.max() * 1.25); ax.set_title("Task 4 class balance (train)")
plt.tight_layout(); savefig(fig, "t4_eda_balance"); plt.show()
sizes = pd.Series([Image.open(p).size for p in t4_train.path.sample(300, random_state=0)]).value_counts()
print("image sizes (W,H) in a 300-image sample:", sizes.to_dict())
""")
md(r"""
**Findings.** Every image is exactly **230×120 (aspect ≈ 1.92:1)**. The screenshots are deliberately corrupted: random rotations and shears with black borders, colour inversion, extreme hue shifts, heavy Gaussian noise and blur. The classes are imbalanced (7-Eleven Knights 39% vs Roblock 10%). Implications: (1) keep the wide aspect ratio rather than squash to a 224×224 square; (2) use training augmentations that *reproduce* these corruptions; (3) after training, tune the macro-F1 decision rule on OOF probabilities.

## 3.5 Task 5: Account-security time windows
""")
code(r"""
groups5 = [g for g in task5.temporal_groups(t5_test) if g not in ("login_lat", "login_lon")]
rows = []
for gname in groups5:
    v = t5_test[[f"{gname}_{i}" for i in range(1, 5)]].values.astype(float)
    sd = np.nanstd(v[:, :3])
    dev4 = np.abs(v[:, 3] - np.nanmean(v[:, :3], axis=1)) / sd      # period 4 vs own periods 1-3
    dev1 = np.abs(v[:, 0] - np.nanmean(v[:, 1:3], axis=1)) / sd      # control: period 1 vs periods 2-3
    rows.append([gname, np.nanpercentile(dev1, 99.5), np.nanpercentile(dev4, 99.5)])
tail = pd.DataFrame(rows, columns=["group", "control (period 1)", "period 4"]).sort_values("period 4")
fig, ax = plt.subplots(figsize=(8.5, 6.4))
ax.hlines(tail.group, tail["control (period 1)"], tail["period 4"], color=MUTED, lw=1.2)
ax.scatter(tail["control (period 1)"], tail.group, color=CAT[0], s=28, label="control: period 1 vs periods 2-3", zorder=3)
ax.scatter(tail["period 4"], tail.group, color=CAT[1], s=28, label="period 4 vs own periods 1-3", zorder=3)
ax.set_xscale("log"); ax.set_xlabel("99.5th percentile of |deviation| (population SDs, log scale)")
ax.set_title("Task 5: injected behaviour changes concentrate in period 4"); ax.legend(loc="lower right")
ax.tick_params(axis="y", labelsize=7.5)
plt.tight_layout(); savefig(fig, "t5_eda_tails"); plt.show()
""")
code(r"""
lat = t5_test[[f"login_lat_{i}" for i in range(1, 5)]]
lon = t5_test[[f"login_lon_{i}" for i in range(1, 5)]]
invalid = (lat.abs() > 90).any(axis=1) | (lon.abs() > 180).any(axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
axes[0].scatter(t5_test.login_lon_4, t5_test.login_lat_4, s=2, color=MUTED, alpha=0.5, label="period-4 logins")
axes[0].scatter(t5_test.login_lon_4[invalid], t5_test.login_lat_4[invalid], s=10, color=CAT[1], label=f"coords past ±90° / ±180° (n={invalid.sum()})")
axes[0].axhspan(90, 140, color=CAT[7], alpha=0.06); axes[0].axhspan(-110, -90, color=CAT[7], alpha=0.06)
axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude"); axes[0].legend(loc="lower left", markerscale=2)
axes[0].set_title("Login geography: out-of-range coordinates")
flags = task5.FLAG_COLS
fr = t5_test[flags].mean().sort_values()
axes[1].barh(fr.index, fr.values, color=CAT[0], height=0.55, **viz.bar_gap())
for i, v in enumerate(fr.values):
    axes[1].text(v, i, f" {v:.1%}", va="center", fontsize=8.5)
axes[1].set_xlim(0, fr.max() * 1.3); axes[1].set_title("Security flag base rates")
plt.tight_layout(); savefig(fig, "t5_eda_geo"); plt.show()
""")
md(r"""
**Findings.** For every behaviour, the deviation of **period 4 from the account's own periods 1–3** behaves like the control (period 1 vs 2–3) up to the ~98th percentile. Then it explodes for specific families:
* **Takeover / boosting:** `skill_rating`, `skill_trajectory`, `kda_mean`, `accuracy_mean` jump, `friend_removes` spikes and reaction time drops. These co-occur.
* **Credential theft / VPN abuse:** `location_changes`, `currency_transfers` and `purchase_*` spikes with long login jumps, plus a drop in `play_pattern_consistency`.
* **Out-of-range coordinates are not a rule on their own.** About two thirds belong to those teleporting accounts. The rest are players near the date line whose longitude wraps past ±180° with no actual travel. The geo signal is therefore the wrap-aware great-circle jump (haversine), not the coordinate range.

So one anomaly signature is a **change relative to the account's own baseline**, which drives the feature engineering below. §5.3 shows a second kind that never changes: accounts that are abnormal in *every* period (single-IP clockwork bots, near-zero activity or chat, very low `behavior_stability`).
""")

# ================================================================ 4. preparation
md(r"""
<a id="4"></a>
# 4. Data Preparation & Feature Engineering

**General policy.** GBDTs handle missing values natively, using a learned default direction per split, so tabular NaNs are left as-is for trees. All statistics used by the features (z-score means/SDs, frequencies) are computed on **training rows only** and then applied to test. Supervised encoders (target encoding) are fitted **inside each CV fold**.

## 4.1 Task 1: mechanical-anomaly ratios
*Domain hypothesis, tested in §5.1:* aimbots inflate *mechanical* skill without the matching *game sense*. We build ratios (`hs_per_accuracy`, `kd_per_reaction`, `accuracy_per_reaction`, `dmg_per_survival`), skill-vs-experience rates (`level_per_day`, `kd_per_level`, `skill_per_playtime`), report intensities (`reports_per_day`, `reports_per_friend`), and two composite z-indices: `mech_index`, `sense_index` and their gap `mech_minus_sense`.
""")
code(r"""
X1, y1, X1_test = task1.build(t1_train, t1_test)
new1 = [c for c in X1.columns if c not in t1_train.columns]
print(f"Task 1 matrix: {X1.shape}, test {X1_test.shape}; engineered: {new1}")
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, c in zip(axes, ["mech_minus_sense", "reports_per_day", "hs_per_accuracy"]):
    lo, hi = X1[c].quantile([0.005, 0.995])
    for k, lab in [(0, "Legitimate"), (1, "Cheater")]:
        sns.kdeplot(X1.loc[y1 == k, c].clip(lo, hi).dropna(), ax=ax, color=CAT[k], lw=2, label=lab, cut=0)
    ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
axes[0].legend(); plt.tight_layout(); savefig(fig, "t1_fe"); plt.show()
""")
md(r"""
## 4.2 Task 2: latent-factor indices, ratios and encodings
* **Noise removal:** drop `random_metric_1..3`.
* **Ordinal encoding:** `engagement_level` (Low→Veteran), `loyalty_tier` and `skill_tier` (Bronze→Diamond).
* **Composite indices:** NaN-aware means of z-scored indicators give `engagement_index` and `spending_index` (plus their product and difference). Averaging ~15 noisy, partly-missing measurements of the same latent factor removes most of the noise, which trees cannot do on their own.
* **Behavioural ratios:** `spend_per_session = total_spending_thb / (play_frequency · avg_session_duration)`, `social_engagement = friend_count / play_frequency`, `monetization_velocity = avg_monthly_spending · spending_frequency`, and others.
* **Nominal categoricals:** frequency encoding, plus **leakage-free K-fold target encoding**. Per-class rates are smoothed toward the prior, $\hat p_{c,k} = \frac{\sum_{i \in c} 1[y_i=k] + m\,\pi_k}{n_c + m}$, and fitted inside every CV fold through `task2.te_fold_fn`.
""")
code(r"""
X2, y2, X2_test = task2.build(t2_train, t2_test)
cat2 = [c for c in X2.columns if str(X2[c].dtype) == "category"]
print(f"Task 2 matrix: {X2.shape} | categorical (native): {cat2}")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), sharex=True, sharey=True)
xs, ys = X2.engagement_index.clip(-1.2, 1.4), X2.spending_index.clip(-1.2, 1.6)
for k, ax in enumerate(axes):
    ax.scatter(xs, ys, s=1, color="#e4e3df", rasterized=True)
    m = y2 == k
    ax.hexbin(xs[m], ys[m], gridsize=35, cmap=SEQ_CMAP, mincnt=3)
    ax.set_title(seg_names[k]); ax.set_xlabel("engagement_index"); ax.grid(False)
axes[0].set_ylabel("spending_index")
fig.suptitle("Engineered latent indices recover the 2x2 segment design", y=1.03, fontsize=11)
plt.tight_layout(); savefig(fig, "t2_fe_latent"); plt.show()
""")
md(r"""
## 4.3 Task 3: RFM proxies and monetary interactions
Recency (`days_since_last_purchase`, recency ratios), frequency (`purchase_frequency`, transactions, sessions) and monetary terms (`log_hist`, `log_prev`, `prev_over_hist`, `hist_per_streak = historical_spending / daily_login_streak`, `txn_value_x_freq`), engagement × event terms (`event_x_session = event_participation_rate · avg_session_length`), and polynomial interactions of past spending (`prev_sq`, `prev_x_hist`, `prev_x_segment`).

**Data repair before feature engineering.** Two facts about the raw columns are used. Neither uses the target or any test row for fitting:

* **Exact identities within a row.** `historical_spending = total_transactions × avg_transaction_value` (relative error 2e-16), `purchases_on_discount = total_transactions × discount_rate_used`, and `prev_month_spending / historical_spending` always lies in 0.30–0.40. Each column is about 5.8% missing at random, so most gaps can be filled exactly from the same row (`task3.fill_identities`). The median ratio is used only when nothing exact is available.
* **`segment` is a rule of the features.** It is set by `event_participation_rate` cut-offs plus `historical_spending`, and a classifier recovers it with hold-out accuracy 0.999. The 5.8% missing segments are filled by a classifier trained on training rows with a known segment (`task3.impute_segment`), and a `segment_imputed` flag is kept.
""")
code(r"""
X3, y3, X3_test, repair3 = task3.build_repaired(t3_train, t3_test)
print("values filled from identities (train):", repair3["fill_train"])
print("values filled from identities (test): ", repair3["fill_test"])
print(f"segment imputer hold-out accuracy: {repair3['segment_accuracy']:.4f} "
      f"({int(X3.segment_imputed.sum()):,} train / {int(X3_test.segment_imputed.sum()):,} test rows filled)")
new3 = [c for c in X3.columns if c not in t3_train.columns]
spear = X3[new3].corrwith(pd.Series(y3), method="spearman").sort_values(key=np.abs, ascending=False)
print(f"Task 3 matrix: {X3.shape}")
display(spear.to_frame("Spearman rho with spending_30d").T.round(3))
""")
md(r"""
## 4.4 Task 4: aspect-preserving resize and corruption-matched augmentation
Inputs are resized to **160×320** (2:1). That is a 4% horizontal stretch, versus the 48% width compression of a square 224×224 resize. Training augmentation: horizontal flip (never vertical, since HUDs are vertically anchored), aspect-correct affine (±25° rotation, ±8° shear, 0.85–1.15 scale, ±8% shift, black fill like the data), **RandAugment** (2 ops, magnitude 9, re-drawn for every chunk of 8 images), per-sample brightness/contrast/saturation and **full-circle hue rotation** in YIQ space, random inversion, grayscale, Gaussian noise / blur and random erasing. Each batch also gets **Mixup** (α = 0.4) or **CutMix** (α = 1) with probability 0.8, plus **label smoothing** 0.1.

*Engineering note:* the first version applied these per image in CPU DataLoader workers. On this Windows laptop that starved the GPU (utilisation swung between 0% and 100%) and deadlocked once on worker re-spawn. All 31.5k images are now decoded once into a 2.6 GB uint8 array and **augmented batched on the GPU**, roughly 2.3× faster.
""")
code(r"""
# The whole augmentation pipeline runs batched on the GPU (see src/task4.py: GPUAugment)
aug = task4.GPUAugment()
one = task4.load_uint8([t4_train.path.iloc[3]])
x = task4.to_gpu_batch(np.repeat(one, 8, axis=0), (160, 320), DEVICE)
torch.manual_seed(0)
xa = aug(x)
mean = torch.tensor(task4.MEAN, device=DEVICE).view(1, 3, 1, 1); std = torch.tensor(task4.STD, device=DEVICE).view(1, 3, 1, 1)
vis = (xa * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
fig, axes = plt.subplots(2, 4, figsize=(12, 3.2))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x[0].permute(1, 2, 0).cpu().numpy() if i == 0 else vis[i])
    ax.set_title("original (resized 160x320)" if i == 0 else f"augmented #{i}", fontsize=9); ax.axis("off")
plt.tight_layout(); savefig(fig, "t4_fe_aug"); plt.show()
""")
md(r"""
## 4.5 Task 5: impossible travel, self-referential deviations and burstiness
* **Geodesic features:** Haversine distance between consecutive login coordinates (1→2→3→4), max hop, total path, period-4 distance from the account's historical centroid, km per login in period 4, and an **invalid-coordinate flag** ($|lat|>90°$ or $|lon|>180°$).
* **Self-referential deviation (for each of the 27 behaviours):** `dev4` = (x₄ − mean(x₁..x₃)) / population SD, `selfz4`, the same scaled by the account's own variability, `maxjump`, the largest first difference, `burst`, max / mean, and `cv`, the dispersion over the 4 periods.
* Static attributes and the 4 security flags (plus their count).
""")
code(r"""
X5, dev5 = task5.build_features(t5_test)
geo5 = [c for c in X5.columns if c.startswith("geo_")]
selfz5 = [c for c in X5.columns if c.endswith("_selfz4")]
dyn5 = [c for c in X5.columns if c.endswith(("_maxjump", "_burst", "_cv"))]
static5 = task5.STATIC_COLS + ["n_flags"]
print(f"Task 5 engineered matrix: {X5.shape} | dev4: {len(dev5)} | selfz4: {len(selfz5)} | dynamics: {len(dyn5)} | geo: {len(geo5)}")
display(X5[["geo_hop_34", "geo_p4_from_home_km", "geo_invalid_coord", "skill_rating_dev4", "friend_removes_dev4",
            "currency_transfers_dev4", "location_changes_dev4"]].describe(percentiles=[.5, .99, .999]).T.round(3))
""")
