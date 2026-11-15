# experiments/

Personal sandbox. Each teammate works in their own sub-folder (`experiments/<github-username>/`), so two people
never edit the same scratch notebook.

* Anything goes here: quick notebooks, scripts, ablations, plots.
* Code and notebooks are committed; generated data (`*.csv`, `*.npy`, `*.npz`, `outputs/`) is git-ignored.
* Nothing in the main pipeline imports from `experiments/`.
* When an idea works, move it into `src/` (and the notebook cells in `scripts/nb_part*.py`) through a pull request,
  with its out-of-fold score in the PR description.

To make the shared code importable from a notebook in `experiments/<you>/`:

```python
import os, sys
from pathlib import Path
ROOT = Path.cwd().parents[1]          # experiments/<you>/ -> project root
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src import config as C, task1    # etc.
```

## Test data in old experiment scripts (retired 2026-11-15)

The TAs ruled on 2026-11-15 that test data, even unlabeled, may not be used to design the validation or to choose augmentation
(test-time augmentation at prediction is allowed). The main pipeline (`src/`, `scripts/`, `task1/`-`task5/`) now uses test data
only to produce the submission. The scripts below are kept as a record of what was run; they used test data for diagnostics,
and some of those diagnostics fed decisions (see `claude/FINDINGS.md`, "Audit after the TA ruling"). They must not be used for
new decisions.

* Task 1 "test-like" check (validation rows re-weighted / masked to the test's missing-cell rates): `bd_phase2_t1t2`, `bj_t1_tune`,
  `bk_lastyear_methods`, `bm_t1_rf_confirm`, `bn_t1_combined`, `bo_t1_seedavg`, `bq_t1_paradigms`, `bq2_t1_followup`, `by_r14_bag`,
  `bz_r14_eval`, `cd_r15_blindspots`, `ce_r16_t1_members`, `cf_r16_eval`, `dv_r27_t1_bag_l2`, `ee_r28_t1_zscore_l2`.
* Train-vs-test classifiers (adversarial validation, test-likeness weights): `bc_audit_phase1`, `bl_t4_testlike`.
* Class-prior estimates on test predictions (EM / BBSE, predicted test shares): `df_r25_t4_prior_stack`, `dk_r26_prior_shift`,
  `dy_r28_t4_member_alpha`, `cy_r23_t4_prior_temp`.
* Confidence / agreement of models on test images: `bb_t4_alignment_check`, `bi_t4_probe_eval`, `co_r19_t4_views_allfolds`,
  `cq_r20_t4_blend`, `cv_r22_t4_eval`, `dc_r24_t4_test_agreement`, `dp_r27_t2_kde_blend`, `dq_r27_t4_probe_views`, `dr_r27_t4_probe_fusion`.
