# experiments/ (not part of the final submission)

Everything here is research, trial and error, or history. The final code is in `task1/` … `task5/`, `src/` and `scripts/`
(see the main [README](../README.md)), and none of it imports from this folder.

| Folder / file | Contents |
|---|---|
| `claude/` | About 250 experiment scripts, one per check, named `<id>_r<round>_<task>_<topic>.py`, from 44 rounds of experiments. `claude/FINDINGS.md` records every check, its result and the decision, including everything that failed. |
| `HISTORY.md` | Every Kaggle upload with its public score and what changed (v1–v41), and the list of version snapshots. |
| `submissions/` | Every uploaded or prepared submission file (`submission_v<k>*.csv`). The final one, v41, is also `submission.csv` at the repository root. |
| `versions/` | Frozen snapshots of the code, cached predictions and submission file of each upload, for roll-back. |
| `colab/` | Colab notebooks of Task 4 members that were tried and not kept (DINOv2-g/14, a first DINOv3-L recipe, DINOv3-7B at 192×384 and with LoRA, multi-resolution TTA) and their builders. |
| `scripts/` | One-off tools: `snapshot_version.py` (writes `versions/`), `build_v41_p7b_submission.py` (standalone check that v41 = v39 + the P_7b probe), `reorder_task4_test_cache.py` (moved old Task 4 caches to the re-released test order), `train_task4_probe.py` (the 7-backbone probe used in v28–v40, replaced by P_7b in v41). |
| `master_pipeline/` | `CPE342_Master_Pipeline.ipynb`: all five tasks in one notebook, built from the same cells as the task notebooks. Its saved outputs are from the v32 execution. It is kept for reference and is not the deliverable. |

Scripts in `claude/` were written for the layout before the reorganisation of 25 Nov 2026. A few of them read files that have
moved since then (see the table at the top of `HISTORY.md`). Data they generate (`claude/outputs/`, `*.npz`, `*.npy`, `*.csv`)
is git-ignored. To import the shared code from a notebook in this folder:

```python
import os, sys
from pathlib import Path
ROOT = Path.cwd().parents[1]          # experiments/claude/ -> project root
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src import config as C, task1    # etc.
```

## Test data in old experiment scripts (retired 2026-11-15)

The TAs ruled on 2026-11-15 that test data, even unlabeled, may not be used to design the validation or to choose augmentation
(test-time augmentation at prediction is allowed). The final pipeline (`src/`, `scripts/`, `task1/`-`task5/`) uses test data
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
