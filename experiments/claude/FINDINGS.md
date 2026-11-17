# Task 2 / Task 3 improvement search (2026-11-04)

All numbers are out-of-fold (OOF) on the notebook's own 5-fold splits. `common.py` rebuilds the
notebook's features, folds and baseline scores from `outputs/preds/` (reproduces 0.7549 and 0.78917).
Run any script from this folder with the project venv, e.g. `python c_t2_nb.py`.

## Task 2 (macro-F1, baseline 0.7549)

**What the data looks like** (`b_t2_explore.py`, `e_t2_latent_check.py`)
- Within a segment the features are almost uncorrelated: average |Spearman| 0.005, max 0.05. The data was
  generated as "pick a segment, then draw each feature (nearly) independently".
- There is a weak hidden score per factor: within a segment, engagement-engagement pairs correlate +0.001 to
  +0.008 and spending-spending pairs +0.004 to +0.017 (largest for Whales). The noise columns show 0.
- Only engagement features, spending features, `loyalty_tier`, `engagement_level`, `vip_tier`, achievement and
  collection columns carry signal. region, platform, device, payment, language, account_status,
  player_type_tag and skill_tier carry none.
- Missingness is random (missing rates differ across segments by at most 0.009).
- Ceiling: the most confident 31% of predictions (p ≥ 0.9) are still wrong 5.5% of the time. That looks
  like label noise, so no model will get far above about 0.76.

**Experiments**

| Model | argmax | + class weights | log-loss |
|---|---|---|---|
| current LightGBM (`t2_lgb`) | 0.7531 | 0.7543 | 0.6077 |
| histogram Naive Bayes, 16 bins (`c_t2_nb.py`) | 0.7531 | 0.7545 | 0.6271 |
| LightGBM stumps = additive model, FE features (`d_t2_stumps.py`) | 0.7525 | 0.7545 | 0.6161 |
| LightGBM + in-fold logistic factor scores (`f_t2_factor.py`) | 0.7540 | 0.7549 | 0.6058 |

| Stack (`g_t2_stack.py`) | macro-F1 |
|---|---|
| current 4-model stack | 0.7549 |
| + factor-LightGBM, NB-16, stumps-FE | **0.7559** |
| the same comparison on 3 other stacker splits | +0.0012 / +0.0007 / +0.0014 |

## Task 3 (NMAE score, baseline 0.78917)

**Where the error is** (`a_baseline.py`, `h_t3_explore.py`)
- Whales (`segment == 3`) are 14% of rows but 87% of all spending and **79.5% of the total absolute error**:
  - **21.4%:** whales who spend 0 this month (1,042 rows). Not predictable: AUC of P(y>0) within whales is 0.57.
  - **58.1%:** whales who spend. Typical log error (MAD) 0.118.
  - **6.2%:** rows with a missing segment.
- `historical_spending` and `prev_month_spending` are nearly the same signal (0.994 correlation within whales).
  Log-linear formulas reach MAD 0.129, which is worse than the current blend's 0.118 (`i_t3_whale.py`). The
  blend's residual does not correlate with any feature (max |rho| 0.04), so what is left looks like noise.

**Experiments** (`j_t3_whale_model.py`)

| Change | Task 3 score |
|---|---|
| current blend | 0.78917 |
| whale rows predicted by a whale-only LightGBM (L1 on log1p) | 0.78951 |
| whale rows = 50/50 average of the whale-only model and the current blend | **0.78999** |
| whale-only L1 on raw THB | 0.78762 (worse) |
| whale-only linear-tree LightGBM | 0.636 (unstable, dropped) |

## Conclusion
Both tasks are close to their noise ceiling. The best changes found are +0.0010 (Task 2) and +0.0008
(Task 3). With equal task weights that is about **+0.0004 on the leaderboard**. The gap to the top
(0.697 → 0.748) does not come from Tasks 2 and 3.

## Task 1 (positive-class F2, baseline 0.8356 at τ = 0.190; OOF optimum 0.8369 at τ = 0.163)

**What the data looks like** (`k_t1_explore.py`, `l_t1_middle.py`)
- Three groups:
  - **Obvious cheaters:** 8,745 rows with p ≥ 0.95, 99.9% cheaters. These are new accounts: median age
    178 vs 510 days, 80 vs 143 friends, faster level progression, lower crosshair placement.
  - **Obvious clean players:** 24,716 rows with p < 0.02, 0.02% cheaters.
  - **Hidden cheaters:** about 19,600 cheaters inside an ambiguous middle of about 52,000 rows (38% cheaters).
    They differ from clean players only by small shifts in the same direction on about 8 skill features
    (lower K/D, headshot %, accuracy, aim smoothness, spray control, game sense; slower reactions).
    The stack separates them with AUC 0.69 within the middle.
- 16 of 31 features carry no signal on their own (AUC ≈ 0.50).
- Missingness is random (class missing rates differ by ≤ 0.006).

**Experiments**

| Change | AUC | best F2 |
|---|---|---|
| current LightGBM | 0.8902 | 0.8362 |
| LightGBM + in-fold logistic scores (all features / skill features) (`m_t1_factor.py`) | 0.8896 | 0.8354 |
| current stack | 0.8910 | 0.8369 |
| stack + that model | 0.8911 | 0.8368 |

**Threshold vs missing values** (`n_t1_missing.py`)

| NaNs per row | best F2 | best τ |
|---|---|---|
| 0 | 0.865 | 0.170 |
| 1–2 | 0.850 | 0.16–0.17 |
| 3 | 0.841 | 0.183 |
| 4–5 | 0.831 | 0.149 |
| 6+ | 0.826 | 0.139 |

- τ = 0.190 costs at most 0.003 in any group.
- Test rows have fewer NaNs (2.5 vs 3.7 per row), so the test F2 is likely a bit above the OOF estimate.
  Nothing to change.
- If the TAs confirm positive-class F2, moving τ from 0.190 to the OOF optimum 0.163 is worth +0.0013.

**13,184 unlabelled training rows** are dropped by `task1.load()`. They look like the labelled rows
(3.7 NaNs per row). Semi-supervised use would draw only on training data, but it is a grey area: ask the TAs
first. The expected gain is small.

## Overall
Tasks 1, 2 and 3 are all close to their noise ceilings. Everything found adds up to about +0.002 of task
score, which is under +0.001 on the leaderboard. The gap to the top is in how Tasks 4 and 5 are scored.

## Task 5 (no labels; v4 = union of 6 gap families + ensemble knee + evidence rules = 2,285 accounts)

**Completeness check of the families** (`o_t5_gapscan.py`)
- The notebook runs the 1-D gap detector on the raw columns only. I ran the same detector on the derived
  per-account change features: period-4 shift vs own history (`*_dev4`, `*_selfz4`), jumps, burstiness,
  all-period levels and geo hops.
- It finds 37 separated groups covering 1,537 accounts. **Every one of them is already in the six
  families** (0 new accounts against the families and 0 against the v4 union).
- So a second, independent view of the data agrees: there is no undiscovered separated population.

**Should v4 flag more?** Under positive-class F3, F3 = 10·TP / (9·K + N_flagged). Flagging one more account
pays off only if its chance of being an anomaly exceeds about F3/10 ≈ 9%.
- **Ensemble-only alerts:** they look like the noisy edge of normal behaviour (§7.5 of the notebook).
- **Unflagged accounts:** they show no sign of being anomalous at all.
- **Cost of guessing:** if the truth were the families, flagging another 1,000 normal accounts would cut F3
  from 0.974 to 0.925.

So the v4 union is already the right decision with the evidence we have.

**What would still help:** knowing the organisers' anomaly definition, e.g. whether persistent
"unusual but legitimate" behaviour (near-inactive or silent accounts) counts. That is a question for the
TAs, not something the data can answer.

# player_id leak check (2026-11-05)

Scripts: `p_t2_id.py`, `q_t2_id_deep.py`, `r_id_other_tasks.py`, `s_t5_id.py` (logs in `outputs/*.log`).
Run the Task 2 scripts on the **original** `train.csv` (as committed): the cached OOF predictions are matched
to rows by position.

## Task 2: the id gives the segment away
- Train and test ids are one shuffled sequence P000000..P129999 (test ≈ 20% of every id range).
- Sorted by id, the segments come in **four contiguous blocks**: ids 0–51,999 → 0, 52,000–84,499 → 1,
  84,500–110,499 → 2, 110,500–129,999 → 3. That is exactly 40 / 25 / 20 / 15% of 130,000.
- 97.0% of rows carry their block's label. The other 3.0% (≈1% to each other class) are spread at random,
  and their **features match the block, not the label**: the features-only model agrees with their label 7%
  of the time and with the block 78% (the same as for normal rows). So they are **random label flips**, and
  this is the label noise behind the ~0.76 ceiling we found earlier.
- 5-fold OOF macro-F1: features-only stack 0.7549; id only 0.965–0.968; block + features 0.9678. The
  features add nothing once the block is known, and nothing can spot the flipped labels (AUC 0.51).
- No periodic pattern (id mod m, last digit): only the blocks.

## Tasks 1 and 3: no id signal
Spearman(id, target) ≈ 0, flat per id decile, id-only OOF AUC 0.505 (Task 1) / Spearman 0.005 (Task 3).

## Task 5: the anomalies are the top 8% of ids
- Test ids P000000..P025888; every one of our 1,811 flagged accounts has id ≥ 23,819. The block
  23,819–25,888 holds 2,070 accounts = 7.996%.
- Inside it, by id: takeover 310 | skill boost + friend purge 310 | single-IP bots 207 | VPN teleport 207 |
  then 1,036 = 4 × 259 interleaved: erratic stability, near-inactive, silent, and **259 accounts we do not
  flag**.
- Those 259 are a **seventh anomaly type: performance volatility**. `kda_std` and `accuracy_std` are about
  2 normal-SD higher, and `performance_stability` is higher too. There is no density gap, which is why the gap
  detector misses them. A z-sum of `kda_std_level` + `accuracy_std_level` + `performance_stability_level`
  separates them from normal accounts with AUC 0.966. Its top 400 accounts catch 241 of the 259.
- Of v4's 474 extra flags, only 16 are inside the block. If the block is the truth, F3 is v3 0.886 and v4
  0.874. That fits v4 scoring below v3.

## Rules
Using `player_id` (Task 2 blocks, Task 5 id range) is exploiting a data-generation leak. The rules do not
mention ids, but "do not attempt to reverse-engineer or obtain test set labels" + "When in doubt, ask
first!" make it a TA question. The Task 5 volatility detector uses only features, but it was found through
the id, so mention it in the same question. Nothing here is merged into the pipeline.

## Task 2 without the id (`u_t2_clean_labels.py`)
- **No feature encodes the row order.** Inside each block, the largest |Spearman(id, feature)| is 0.020
  (noise level 0.016, 136 tests). Categorical columns vs id decile: smallest p 0.033 over 40 tests.
- **Training on the clean segments does not help.** Using the train ids to undo the 3% flipped labels gives
  macro-F1 0.7546 vs 0.7544 on the given labels. The features identify the true segment only 79% of the time
  either way. With symmetric label noise the best class is the same, so this is expected.
- So **about 0.755–0.76 is the features-only ceiling**. GBDTs, Naive Bayes, additive stumps, factor models and
  the stack all land there. The 0.968 comes only from the id.

# Second improvement round, Tasks 1 / 3 / 5 (2026-11-05)

## Data checks
- **No cross-task link:** the same `player_id` in Tasks 1, 2 and 3 is a different player (shared columns
  correlate 0.00, segments agree at chance, 28.6%). No join to exploit.
- **Tasks 1 and 3 ids:** no relation to any feature (max |Spearman| 0.009).
- **No artefacts:** no clipping, integer or decimal-precision differences between classes beyond what the
  trees already see.

## Task 1: still at its ceiling
| Method | AUC alone | Stack F2 (baseline 0.8369) |
|---|---|---|
| histogram Naive Bayes (missing values marginalised) (`w_t1_nb.py`) | 0.838 | 0.8370 |
| latent-class Naive Bayes, 2+2 / 3+3 / 2+4 components | 0.856–0.867 | 0.8367–0.8368 |
| tabular MLP, quantile inputs + missing flags (`z_t1_mlp.py`) | 0.842 | 0.8369 |

- **Same drop with missing values:** AUC falls with the number of NaNs at the same rate for Naive Bayes and
  LightGBM (0.915 → 0.862). That is lost information, not poor NaN handling.
- **Trees win:** the trees beat smooth models by about 0.05 AUC, so the signal is threshold-like.
- **Class weight:** already tuned (about 1.0).

## Task 3: exact identities between columns → +0.0011 (`v_identity_scan.py`, `x_t3_identity.py`)
- **The identities:** `historical_spending = total_transactions × avg_transaction_value` (relative error
  2e-16), `purchases_on_discount = total_transactions × discount_rate_used`, and
  `prev_month_spending / historical_spending` lies in 0.30–0.40.
- **Filling:** each column is ~5.8% missing (completely at random), so most gaps can be filled exactly from
  the same row (`t3_fill.py`; no target information).
- **Result:** LightGBM members gain +0.0017 to +0.0034 each, blend 0.78917 → 0.79026. On rows missing both
  history columns, mean |error| falls from 7,618 to 4,577.
- **All members refitted** (`x2_t3_identity_all.py`): XGB 0.78523 → 0.78615, Cat 0.77947 → 0.77796,
  Tweedie 0.77644 → 0.77795. Blend **0.79035**.
- **+ whale-only model** (50/50 on whale rows, as in round one): **0.79097**. Total 0.78917 → 0.79097
  (+0.0018), about +0.0004 on the leaderboard with equal weights.
- **Target structure:** for y > 0, log(y / prev_month) ≈ segment offset (+0.18 / +0.66 / +0.84 / +1.56)
  + noise (MAD about 0.2). P(y = 0) by segment: 0.72 / 0.48 / 0.31 / 0.07.

## Task 5: a feature-only detector for the 7th group (`y2_t5_pocketscan.py`, `y3_t5_volatility.py`)
- **A 2-component Gaussian mixture per feature pair fails** (`y_t5_pairscan.py`): on non-Gaussian features
  it splits the bulk instead of isolating a small group.
- **Blind "dependence pocket" scan:** normal accounts draw features almost independently, so on a 20×20
  quantile grid every cell should hold n/400 accounts. Across 11,367 feature pairs the strongest real pocket
  is `accuracy_std_level` top 5% × `kda_std_level` top 5%: 222 accounts vs 60 expected, z = 20.9. The only
  stronger pocket, km-per-login × login count, is mechanical (a ratio feature). Graded afterwards with the
  id, 183 of the 222 are among the 259.
- **Label-free threshold:**
  - Score = Fisher combination of the three volatility features.
  - Null = each column shuffled independently, which keeps the marginals and destroys any group.
  - Expected normals above each cut-off come from the null, and k maximises the expected F3.
  - Result: k = 324 (estimated precision 0.51). Id check: 204 of 324 are among the 259 (precision 0.63).
- **Task 5 F3 if the id block is the truth:** 0.886 → 0.970.
- The scan and the detector never use the id. But we first learned the group exists through the id, so this
  is in TA question #4 before it goes into the pipeline.

## Round 3 (2026-11-06): Task 3 ratio target, segment imputation

v6 (= the 0.76 file + round-2 Task 3) scored **0.76202** on the public leaderboard (snapshot `versions/v6_LB0.76202`).

* **Multiplicative target** (`aa_t3_target_structure.py`): for y > 0, log(y / historical_spending) has within-segment
  MAD 0.205 (prev_month_spending 0.207), versus 0.46-0.58 for any other base and 0.44-0.53 for log(y) itself.
  So y ~ hist x segment factor x noise.
* **Ratio-target hurdle** (`ab_t3_resid.py`): stage 2 learns log1p(y) - log1p(base) instead of log1p(y).
  Stage-2 MAE on positives 0.1420 -> 0.1372; its mixture-median read-out alone scores 0.7930, above the whole
  v6 blend. With the ratio members in the pool the whale-only model no longer helps (0.79307 -> 0.79246), so it is dropped.
* **Segment imputation** (`ac_t3_segment.py`, `t3_segment.py`): segment is a rule of the features
  (event_participation_rate cut-offs at 35 / 55 plus historical_spending; LightGBM hold-out accuracy 0.9993).
  The 5.8% missing segments are filled by a classifier trained on train rows only. Small gain (ratio member 0.79302 -> 0.79322).
* **Zeros** (`af_t3_zeros.py`): within a segment, y = 0 is close to random (AUC 0.57-0.63). Nothing more to extract.
* No row-order leak in Task 1 or Task 3 train files (row-index AUC 0.502, Spearman 0.002).
* **Full rebuild** (`ae_t3_v7.py`, notebook recipe incl. Train+Val refit): blend OOF **0.79331**
  (v6 0.79097, v3 0.78917). `submission_v7_try.csv` = v6 with only Task 3 replaced. Expected about +0.0005 LB with equal weights.

### Round 3b: tuning (2026-11-06)
v7 scored **0.76224** (snapshot `versions/v7_LB0.76224`). Then `ag_t3_tune.py` (Optuna, 40 + 30 trials, same folds) and `ah_t3_v8.py`:
* Ratio stage 2 (LightGBM): stage-2 log MAE 0.13649 -> 0.13541 (small leaves, lower learning rate); member 0.79327 -> 0.79373.
* Stage 1 P(y > 0): log-loss 0.55343 -> 0.55235; no blend gain (0.79331 -> 0.79329).
* XGBoost / CatBoost ratio members: 0.79341 / 0.79376 alone; they take half the blend weight.
* Blend OOF **0.79403** (v7 0.79331, +0.0007), i.e. about +0.00014 LB with equal weights. `submission_v8_try.csv` = v7 with Task 3 replaced.
The ratio formulation was the step change; tuning and diversity add little on top. Task 3 is close to its ceiling again.

## Round 4 (2026-11-06): Tasks 1 and 5, no leakage (`ai_t1_round4.py`, `aj_t5_round4.py`)

### Task 1: error analysis
- **Validation:** nested threshold (tau chosen on the other 4 folds); significance from a paired Poisson bootstrap of the F2 difference (1,000 resamples).
- **Baseline B0 (notebook stack):** nested F2 **0.8364** (the non-nested OOF optimum quoted earlier is 0.8369).
- **Confusion matrix at tau:** TN 31,981 · FP 27,110 · FN 788 · TP 30,937. Recall 0.975, precision 0.533: F2 buys recall with false alarms.
- **Failure mode 1, missingness:** AUC 0.915 with 0 NaNs, 0.863 with 6+; the FP rate rises from 0.37 to 0.52. Recall stays about 0.975 in every group.
- **Failure mode 2, skilled cheaters:** the 788 missed cheaters have *above-average* headshot %, accuracy, spray control and K/D. Caught cheaters are below average, and clean players sit in between. Only 42 of the 788 have p < 0.05.
- **Calibration:** the stacker is well calibrated by decile.

| Experiment | Change | OOF AUC | OOF F2 (nested τ) | Δ vs B0 | 95% CI | Decision |
|---|---|---|---|---|---|---|
| B0 | notebook stack (5 members) | 0.8910 | 0.8364 | – | – | baseline |
| E1 | in-fold kNN target features (k = 25 / 100, nested) in LightGBM | 0.8907 | 0.8369 | +0.0005 | [−0.0002, +0.0012] | discard |
| E1b | LightGBM + kNN as an extra stack member | 0.8910 | 0.8362 | −0.0002 | [−0.0007, +0.0003] | discard |
| E2a | + 4-seed bagged LightGBM | 0.8914 | 0.8363 | −0.0001 | [−0.0006, +0.0005] | discard |
| E2b | + bagged + recall-weighted (spw × 3) LightGBM | 0.8914 | 0.8364 | +0.0001 | [−0.0006, +0.0007] | discard |
| E3a | stacker + missing-count buckets (× mean logit) | 0.8909 | 0.8364 | 0.0000 | [−0.0004, +0.0005] | discard |
| E3b | one τ per missing-count bucket (nested) | 0.8910 | 0.8367 | +0.0003 | [−0.0002, +0.0008] | discard |

Nothing clears the bar. Task 1 stays as it is; with the earlier rounds (NB, latent NB, MLP, factor scores) this is strong evidence it is at its ceiling.

### Task 5: label-free error analysis and experiments
There are no labels, so there is no OOF score. Decisions use the **expected F3 under a permutation null**: shuffle each column independently, count the shuffled accounts above each cut-off, and treat the six gap families as true. The id is printed only as a reference column and is never used to decide.

- **Error analysis:** a blind 20×20 quantile scan of 11,319 feature pairs outside the families, against a noise level of max z = 9.1 on shuffled data (2 runs). Exactly **one** pocket lies above noise: `accuracy_std_level` × `kda_std_level`, z = 20.9 (222 accounts vs 60 expected). After flagging it, no pair is above noise (top z = 8.2).
- **Feature choice:** the detector's features are taken programmatically from the pockets above noise. A first run that also used `performance_stability_level` (a choice traced back to the id analysis) was replaced. The blind 2-feature version is also better on every measure.

| Experiment | Change | Flags | Est. precision of new flags | Expected F3 (label-free) | Δ vs B0 | Decision | Reference only: F3 if id block = truth |
|---|---|---|---|---|---|---|---|
| T5-B0 | six gap families (pipeline) | 1,811 | – | 0.9091 | – | baseline | 0.8860 |
| E1 | Fisher score on the pocket features, k* = 283 | 2,094 | 0.45 | 0.9595 | +0.050 | discard | 0.9660 |
| E2a | min-rank ("both high"), k* = 323 | 2,134 | 0.61 | **0.9919** | **+0.083** | **KEEP** | 0.9830 |
| E2b | cross-fitted density-ratio classifier (real vs shuffled) | 1,822 | – | 0.9018 | −0.007 | discard (fails on 2 features) | 0.8855 |
| E3a | min-rank, cut-off = expected-F3 optimum | 2,134 | 0.61 | 0.9919 | +0.083 | KEEP | 0.9830 |
| E3b | min-rank, cut-off = estimated FDR 50% (k = 402) | 2,213 | 0.50 | 0.9901 | +0.081 | discard | 0.9831 |
| E3c | min-rank, cut-off = estimated group size (k = 201) | 2,012 | 0.75 | 0.9746 | +0.066 | discard | 0.9602 |

- **Pipeline:** `src/task5.py` gains `dependence_pockets` and `pocket_detector`. Master notebook §7.5 points 7–8 and the final Task 5 cell: `pred5 = families | pocket`.
- **Cut-off stability:** with its own shuffles the pipeline picks k = 380 (expected F3 0.9913, reference 0.9827). The expected-F3 curve is flat between k ≈ 320 and 400.

## Round 5 (2026-11-06): last attempts at the 0.00055 gap

v8 scored **0.76242** (snapshot `versions/v8_LB0.76242`); the top of the leaderboard is 0.76297.

- **Task 3: spending scales with history and last month together** (Huber fit): y ∝ hist^0.59 × prev^0.40, residual MAD 0.2045 vs 0.2055 for hist alone. Negligible.
- **Task 3: quantile read-out** (`ak_t3_quantile.py`): the L1-optimal prediction is the (1 − 0.5/p) quantile. LightGBM quantile models at 12 levels are well calibrated (coverage within 0.01 of each level). The member scores 0.79365 alone, and the blend goes 0.79403 → 0.79410 (+0.00007). The reason is in the residual:
  - its spread is not predictable (R² of |residual| = 0.010);
  - it is extremely heavy-tailed (skew 5.2, excess kurtosis 73): rare spending spikes that nothing predicts.
- **Task 3: level-2 LightGBM-L1 stacker** (`al_t3_l2stack.py`): 0.7914–0.7921, below the linear L1 blend (0.7941). Discard.
- **Task 1: pseudo-labelling the 13,184 unlabelled training rows** (`am_t1_pseudo.py`, teacher trained per fold, no test rows): student AUC within ±0.001 of the teacher. Stack F2 0.8364 → 0.8369 at best, which is inside the noise found in round 4 (CI half-width about 0.0007). Discard. (Also TA question #5.)

**Conclusion.** Every task except Task 4 is at a measured ceiling:
- Task 1: about 15 experiments, all inside the bootstrap noise.
- Task 2: 3% random label flips.
- Task 3: unpredictable zeros and spending spikes; constant spread.
- Task 5: the id block is complete.

The remaining 0.00055 is below what clean, OOF-verified changes can deliver. The only large lever is Task 4.

## Round 6 (2026-11-06): Task 1 again (`an_t1_deep.py`, `ao_t1_round6.py`, `ap_t1_combined.py`)
- **Deep-dive:**
  - No duplicate rows, within train or between train and test.
  - 18 raw columns carry no signal even in distribution shape (KS < 0.012, overall and in the ambiguous middle).
  - `level_progression_speed` has a different spread but no new information: the difference is a heavy upper tail among clean new accounts, already captured by `level_per_day`.
  - Cheaters form 3 sub-types (Gaussian mixture): "aimbot-like" (smooth aim, few reports, ~3k), "skilled" (fast reactions, high game sense, spray control and win rate, ~3k), and the bulk.

| Experiment | Change | Stack AUC | OOF F2 (nested τ) | Δ vs B0 | 95% CI | Decision |
|---|---|---|---|---|---|---|
| B0 | notebook stack | 0.8910 | 0.8364 | – | – | baseline |
| F1 | LightGBM without the 18 no-signal columns (replaces LightGBM) | 0.8915 | 0.8366 | +0.0002 | [−0.0005, +0.0010] | discard |
| F2 | 4-class LightGBM (clean + 3 in-fold cheater sub-types) | 0.8903 | 0.8358 | −0.0006 | [−0.0013, +0.0001] | discard |
| C1 | reduced features for LGB (3-seed bag), XGB, Cat + ET + LR | 0.8914 | 0.8368 | +0.0004 | [−0.0004, +0.0013] | discard |
| C2 | both views: 5 original + 3 reduced members | 0.8917 | 0.8368 | +0.0005 | [−0.0003, +0.0011] | discard |
| C2b | C2 + one τ per missing-count bucket | 0.8917 | 0.8368 | +0.0004 | [−0.0003, +0.0011] | discard |

Dropping the no-signal columns raises member AUC in 14 of 15 fold × model cases (+0.0002 to +0.0026). So it is a real but tiny effect: about +0.0005 F2, roughly +0.0001 on the leaderboard, and not significant for F2.

## Round 7 (2026-11-06): looking for anything new (`aq_row_checks.py`, `ar_lb_noise.py`, `as_ta_baseline.py`)

**No new structure in the data.**
- **Row position:** train labels and our test predictions are flat across the csv file order for Tasks 1, 2, 3 and 5 (all |Spearman| ≤ 0.003, decile means flat). Only `player_id` was checked before; the file order carries nothing either.
- **Same row across tasks:** row *i* of one task is not the same player as row *i* of another. Shared columns correlate at |Spearman| ≤ 0.009 (n = 22k–90k), the Task 2 label matches the Task 3 `segment` column on 28.0% of rows (chance), and the Task 1 cheater rate is flat across the Task 2 segment of the same row.
- **Task 4 image files:** their timestamps follow neither the `test.csv` order nor sorted-id order (|Spearman| ≤ 0.01).

**The grade is not rank-based** (instruction PDF, slide 6). For teams above the baseline, Grade% = β + (100% − β)·Φ((score − μ)/σ), where μ and σ are the mean and SD of all scores above the baseline. With β = 60% (the slide's example) and plausible μ/σ (0.72–0.75 / 0.01–0.025), the 0.00055 gap to the top is worth +0.1 to +0.4 percentage points of the Kaggle part.

**Leaderboard noise** (`ar_lb_noise.py`, our OOF predictions as the population; the "competitor" is a close variant of our own models):
- The difference between two near-equal submissions has an SD of 0.00044 if 30% of the rows are public and 0.00033 at 50%. Tasks 1 + 3 contribute about 0.0002; Task 4 at chance contributes about 0.0003.
- So a 0.00055 lead flips on the private board about 12–15% of the time with no change at all. Real competitors differ more from us than our own variants do, so this is a lower bound.
- Task 4 against a shuffled key: macro-F1 0.200 ± 0.003.

**The TAs' own baselines point to a Task 4 mismatch on our side** (`as_ta_baseline.py`). Slide 27: the ground baseline (LR / Random Forest / CNN / Isolation Forest, models from class) scores 0.68 on the public board, and the challenge baseline scores 0.75. Default versions of the tabular parts (80/20 split, median imputation):

| Task | Model | Score |
|---|---|---|
| 1 | LR / RF, threshold 0.5 or best | 0.49 / 0.60 / 0.76 / 0.82 |
| 2 | RF / LR | 0.72 / 0.75 |
| 3 | RF | 0.76 |
| 5 | Isolation Forest (contamination auto, 0.05, 0.08, 0.1) | 0.35–0.46 |

- **Ground baseline:** solving 0.68 = 0.2·(S1 + … + S5) gives a Task 4 score of **0.61–1.08** for every combination.
- **Challenge baseline:** reaching 0.75 needs Task 4 ≥ **0.36**, even if its tabular parts matched our best leak-free models and its Task 5 were perfect.
- **Our side:** our uploads imply Task 4 ≈ 0.21 (chance), while our 5-fold OOF macro-F1 is 0.988 and the test predictions look plausible on a fresh contact sheet.
- **Reading:** with equal weights, which our upload history supports, the answer key works with the TAs' convention, and our Task 4 column (row *i* of `test.csv` → `ANS{i}`, datacard codes) does not match it. A random row order, or a code permutation that leaves one class in place, both give about 0.2. A text-versus-integer mismatch would give 0.
- **Our own upload history agrees.** Task 5 of each version was graded against the id block:
  - v2 (Task 5 F3 0.658, Task 1 F2 0.753, Task 4 written as floats `1.0`) implies Task 4 ≈ **0.03**.
  - v3 has the same Task 4 predictions written as integers `1` and implies Task 4 ≈ **0.22**.
  - So the integer format is read correctly, but the matches stay at chance level: a row-order or code mapping that differs from ours. This also explains why the text-label variant scored the same as v1, since both gave Task 4 = 0.
- **A possible mechanism (speculative):** the Task 4 files inside the Drive zip (`test.csv`, `train.csv`, images) are dated 2025-11-01, while the zip's folders are dated September–October 2026. If this year's answer key was built from a fresh shuffle of the test set, the Drive `test.csv` order would not match it for anyone, while the TAs' own baseline would.
- **Value:** about +0.15 on the leaderboard (0.2 × (0.95 − 0.21)), roughly 250 times the gap to the top.
- **Route:** only the TAs can answer it (question #1). Uploading order or code variants would be probing.

## Round 8 (2026-11-08): Task 3 zeros, Tasks 1 and 3 again, Task 3 v8 merged into the pipeline

### Do the many zero predictions cost score? No (`at_t3_zeros_check.py`)
The metric only sees the total absolute error. Raising a prediction from 0 to a small c adds c for every row that really
spent 0 and removes c for every row that spent more than c. So 0 is the best prediction for any group of rows in which
more than half spent nothing, i.e. wherever P(y = 0 | x) > 0.5 (the median rule).

| | training labels | v8 OOF predictions | v8 test predictions |
|---|---|---|---|
| share of zeros, all rows | 0.481 | 0.497 | 0.498 |
| segment 0 (72% of its labels are 0) | 0.724 | 0.982 | 0.982 |
| segment 1 | 0.480 | 0.413 | 0.408 |
| segments 2 and 3 (whales) | 0.306 / 0.071 | 0.000 / 0.000 | 0.000 / 0.000 |

- We predict about as many zeros as there really are, and the test predictions match the OOF predictions segment by segment (no shift).
- Of the 51,646 OOF rows predicted 0, 70.3% really spent 0. In every P(y>0) bin below the gate (0.482) zeros outnumber spenders; above it spenders outnumber zeros. The gate sits exactly at the cross-over.
- **Where the error is:** spenders we predict as 0 are 14.8% of rows but only **2.6%** of the total error (mean 471 THB: small spenders, mostly segment 0). Rows we predict as spending but who spent 0 cause **33.0%** (mostly whales and segment-2 players who skip a month; within a segment this is close to random, AUC 0.57–0.63). Size errors on spenders cause 64.5%.

| What if we predicted fewer zeros? | OOF score | vs v8 |
|---|---|---|
| v8 | 0.79403 | – |
| no zero gate (the blend is tiny there anyway) | 0.79402 | −0.00002 |
| every zero replaced by 10 / 50 / 200 THB | 0.79391 / 0.79340 / 0.79108 | −0.0001 / −0.0006 / −0.0030 |
| gate at P(y>0) > 0.30 (30% zeros) | 0.79402 | −0.00001 |
| gate at P(y>0) > 0.60 / 0.70 (more zeros) | 0.79276 / 0.78727 | −0.0013 / −0.0068 |
| same ratio model, mean read-out (no zeros) vs median read-out | 0.79052 vs 0.79376 | −0.0032 |
| oracle: zero exactly on the rows that spent 0 (not achievable) | 0.85189 | +0.058 |

So the zeros are the metric-optimal choice. The only large potential is knowing *who skips a month*, which the features do not reveal.

### Task 1: three new hypotheses (`av_t1_round8.py`)
**Error analysis.** The stack loses most of its accuracy when a key column is missing: AUC 0.897 → 0.853 without
`account_age_days` (14.5% of train rows), 0.901 → 0.852 without `friend_network_size` (22.3%), 0.901 → 0.850 without
`reports_received` (22.3%), 0.895 → 0.859 without `crosshair_placement` (12.8%). Several of these columns are predictable from
the others (`account_age_days` R² 0.71, `level_progression_speed` 0.62, `friend_network_size` 0.49). **But the test set is
missing these values far less often:** `account_age_days` 0.6% (train 14.5%), `friend_network_size` 4.7% (22.3%),
`crosshair_placement` 2.8% (12.8%); 78% of test rows have all four key columns vs 45% of train rows.

| Experiment | Change | OOF AUC | OOF F2 (nested τ) | Δ vs reference | 95% CI | Δ on rows with the key columns | Decision |
|---|---|---|---|---|---|---|---|
| B0 | notebook stack | 0.8910 | 0.8364 | – | – | – | baseline |
| control | same LightGBM refitted (reference for E1a/E3) | 0.8911 | 0.8361 | −0.0003 vs B0 | – | – | – |
| E1a | 8 predictable columns imputed (imputers on train features only), LightGBM swapped in | 0.8916 | 0.8372 | +0.0011 vs control | [+0.0003, +0.0018] | −0.0001 | discard (see below) |
| E1b | imputed LightGBM as a 6th member | 0.8919 | 0.8368 | +0.0004 vs B0 | [−0.0002, +0.0011] | −0.0002 | discard |
| E2 | specialist LightGBM on the ambiguous middle (63% of rows), re-stacked | 0.8911 | 0.8366 | +0.0003 vs B0 | [−0.0001, +0.0006] | +0.0004 | discard |
| E3 | E1 model + 5 stochastic imputations averaged at prediction | 0.8906 | 0.8362 | +0.0001 vs control | [−0.0007, +0.0008] | −0.0005 | discard |

**Why E1a is not kept.** Its whole gain comes from rows with a missing key column, which the test set rarely has. Reweighting the
OOF rows to the test's missing-value rates (missingness is random and label-free, so this is a fair estimate of test F2;
nothing is fitted) shrinks the gain to +0.0007 vs control, CI [−0.0002, +0.0018], and +0.0003 vs B0, CI [−0.0006, +0.0013].
So Task 1 stays unchanged (now ~25 experiments, all inside the noise for the test population).

### Task 3: three new hypotheses (`au_t3_round8.py`, `aw_t3_exact_blend.py`)
Same folds, same 26-member v8 pool, paired Poisson bootstrap of the score difference (1,000 resamples).

| Experiment | Change | OOF score | Reference | Δ | 95% CI | Decision |
|---|---|---|---|---|---|---|
| B0 | v8: 26 members, Powell L1 blend, zero gate, scale | 0.79403 | – | – | – | baseline |
| H1a | one gate threshold + scale per segment (chosen on the other folds) | 0.79399 | B0 with nested global threshold + scale (0.79395) | +0.00005 | [−0.00002, +0.00011] | discard |
| H1b | one scale per segment | 0.79399 | same | +0.00004 | [−0.00002, +0.00011] | discard |
| H1c | one threshold per segment | 0.79395 | same | 0.00000 | [−0.00001, +0.00002] | discard |
| H2a | zero gate learned from the metric (label sign(y − \|y − u\|), weight \|y − \|y − u\|\|, in-fold) | 0.79402 | B0 | −0.00001 | [−0.00002, −0.00001] | discard |
| H2b | H2a with its cut-off chosen on the other folds | 0.79313 | B0 | −0.00090 | [−0.00104, −0.00076] | discard |
| H2c | zero only where both gates say 0 | 0.79402 | B0 | −0.00002 | [−0.00002, −0.00001] | discard |
| H3a | + ratio stage 2 weighted by 1 + hist (member alone 0.79272 vs 0.79373 unweighted) | 0.79405 | B0 | +0.00002 | [−0.00005, +0.00009] | discard |
| H3a′ | + ratio stage 2 weighted by sqrt(1 + hist) (member 0.79363) | 0.79406 | B0 | +0.00003 | [−0.00003, +0.00009] | discard |
| H3b | + one-stage L1 on log1p(y) − log1p(hist), all rows (member 0.79179; its zeros come out by themselves) | 0.79404 | B0 | +0.00001 | [−0.00005, +0.00007] | discard |
| H3-all | + all 8 new members (Powell blend) | 0.79415 | B0 | +0.00011 | [+0.00004, +0.00018] | Powell artefact, see below |
| **P1** | **same 26 members, L1 blend solved exactly (linear program, HiGHS)** | **0.79411** | B0 | **+0.00007** | **[+0.00001, +0.00014]** | **KEEP** |
| H3-all exact | + all 8 new members, exact blend | 0.79414 | P1 | +0.00003 | [−0.00002, +0.00009] | discard |

- **Why the gate ideas fail:** the gate decision is already at the cross-over point, and on the gated rows the blend is tiny anyway (see the zero analysis above), so there is nothing to win. The metric-trained gate (H2) learns to keep the blend almost everywhere because the weights are tiny on exactly those rows.
- **Why weighting fails:** weighting the log-space model by spending scale makes the few largest whales dominate each leaf (high variance). The member gets worse alone and adds nothing to the blend.
- **Powell vs exact:** non-negative L1 blending is a linear program, and Powell's local search stops about 0.0001 short of the optimum. H3-all's apparent gain was mostly Powell finding a better point in a bigger pool; with the exact solver the extra members add +0.00003 (CI includes 0).
- **Leaderboard value:** P1 is worth about +0.00002 with equal weights, so it is merged for determinism and correctness, not for the score.

### Pipeline update (brief step 5)
- **Task 3 v8 recipe merged into the clean pipeline:** `src/task3.py` gains `fill_identities`, `impute_segment`, `build_repaired`,
  `ratio_stage2_cv` / `ratio_stage2_full`, `readouts` / `apply_readout` and `l1_blend_weights_exact`. The notebook's Task 3 sections
  (4.3, 5.1, 7.3, 7.6) use them, with new caches `outputs/preds/t3r_*` and the Optuna parameters in `outputs/params/task3_stage1.json` and
  `task3_ratio_stage2.json`. The per-task notebooks were rebuilt from the same cells.
- **Master notebook re-executed without errors:** Task 3 OOF **0.79423** (v3 pipeline 0.78917; v8 experiment script 0.79403; the same
  pool with the Powell blend 0.79414, so retraining noise is about ±0.0001).
- **`submission.csv` regenerated** (25,889 rows, checks passed): Tasks 1, 2, 4, 5 unchanged; Task 3 = the merged recipe
  (corr(log1p) with v8's Task 3 0.998, zero share 0.500 vs 0.498).
- **`submission_v9_try.csv`** = v8 with only Task 3 replaced by the pipeline's. Expected change vs v8 about +0.00004 on the leaderboard,
  below what the board can resolve. Its point is that the Task 3 column now comes from the code we hand in.
- **Uploaded 2026-11-08: v9 scored 0.76242, the same as v8** (snapshot `versions/v9_LB0.76242`). So the merged pipeline reproduces
  v8's Task 3 quality on the hidden test rows.
- Pre-merge notebooks and `submission.csv` are backed up in `versions/pre_t3_merge_backup_2026-11-08/`.

### How can the top team be at 0.76297? (2026-11-08)
The score fits two very different profiles (equal task weights):

| Profile | T1 | T2 | T3 | T4 | T5 | Leaderboard |
|---|---|---|---|---|---|---|
| ours (v8/v9): player_id structure in T2/T5, Task 4 misaligned | 0.84 | 0.968 | 0.794 | ~0.21 | ~1.0 | 0.7624 |
| clean tabular models + **working Task 4** + Isolation-Forest-level Task 5 | 0.84 | 0.755 | 0.79 | 0.95 | 0.48 | 0.763 |

- Our player_id gains (+0.21 on T2, +0.52 on T5) are about the same size as a working Task 4 (+0.74), so the two totals coincide.
- The TA baselines fit the second profile and cannot be reached with Task 4 at chance: default tabular models (T1 0.82, T2 0.75, T3 0.76) + a CNN at ~0.95 + Isolation Forest (0.46) give 0.748 (the challenge baseline is 0.75); simpler models + a weaker CNN (~0.7) give ~0.68 (the ground baseline).
- Under the first profile the 0.00055 gap is about one SD of the leaderboard's own noise (public-row sampling plus Task 4 chance variation between teams).
- Either way the leaders are not ahead on Tasks 1 or 3. The difference between teams sits in Task 4.
- **Task 1 metric reading checked with our own uploads:** v3 → v4 (τ 0.163 → 0.190 plus the Task 5 union) changed the score by −0.00234. Positive-class F2 predicts −0.00265 (off by 0.0003, inside the noise); macro-F2 predicts −0.00023 (off by 0.0021). So the scorer uses positive-class F2, as we assumed.
- **Sharper TA question #1:** Task 4's `test.csv` ids are random strings (`scyd5lz`, `slbh6z4`, ...) in unsorted order, while the image folder sorts as `s001rpz.jpg`, `s001six.jpg`, .... Nothing states whether ANS{i} is the i-th row of `test.csv` or the i-th image in sorted-filename order (or some other order).
- **Sorted-filename order tested (v10, user request): 0.76257 vs v9 0.76242.** +0.00015 on the leaderboard = +0.00075 on Task 4, i.e. still
  chance level (a correct order would give ~0.91). Two random alignments differ by about ±0.0009 on the leaderboard, so the +0.00015 is
  noise. The answer key is in neither `test.csv` order nor sorted-filename order. The Task 4 zip on Drive is unchanged since 2026-10-05
  (same size as our 2026-11-03 copy), so there is no newer data version. Only the TAs can resolve it.

## Round 9 (2026-11-08): methods from the literature (`ay_t3_research.py`, `az_t1_research.py`, `ba_t1_ranker_check.py`)

**Task 3 in context** (OOF): predict 0 for everyone 0.500 · segment median 0.620 · history × segment factor 0.722 · default
random forest 0.760 · v3 pipeline 0.789 · **current pipeline 0.794** · our model + knowing who spends 0 (oracle) 0.852.

### Task 3 (baseline B0 = merged pipeline pool, exact LP blend, 0.79423)
| Experiment | Method (source) | OOF | Δ vs B0 | 95% CI | Decision |
|---|---|---|---|---|---|
| R1 | BG/NBD + Gamma-Gamma customer-base features (Fader, Hardie & Lee 2005) in a new hurdle | 0.79424 | +0.00001 | [−0.00001, +0.00004] | discard |
| R2 | zero-inflated lognormal network, mixture-median read-out (Wang, Liu & Miao 2019) | 0.79423 | +0.00000 | [+0.00000, +0.00000] | discard |
| R1+R2 | both | 0.79424 | +0.00001 | [−0.00001, +0.00004] | discard |
| R3 | L1-optimal multiplier per prediction decile (nested) | 0.79410 | −0.00013 | [−0.00018, −0.00007] | discard |
| R4 | ensemble selection with replacement (Caruana et al. 2004) instead of LP weights | 0.79429 | +0.00006 | [+0.00004, +0.00008] | significant but negligible (~+0.00001 LB); not merged |

- **BG/NBD does not fit this data:** the maximum-likelihood fit degenerates (r, α → 0; a, b → very large), so P(alive) ≈ 1 for almost
  everyone. Stage-1 AUC 0.7848 vs 0.7849, stage-2 MAE 0.1359 vs 0.1360. The synthetic recency/frequency columns do not behave like
  a purchase process.
- **ZILN network:** AUC 0.780 (GBDT stage 1: 0.785), log MAE on spenders 0.153 (GBDT ratio models: 0.136). Alone 0.7835 with the
  median read-out, 0.7791 with the paper's mean read-out (the mean is the wrong target for an absolute-error metric). The LP blend
  gives it zero weight.

### Task 1 (baseline B0 = notebook stack, nested F2 0.8364)
| Experiment | Method (source) | OOF AUC | OOF F2 | Δ | 95% CI | Decision |
|---|---|---|---|---|---|---|
| Q1 | focal-loss LightGBM γ = 1 as 6th member (Lin et al. 2017) | 0.8912 | 0.8369 | +0.0006 | [+0.0000, +0.0011] | borderline, not kept |
| Q1′ | focal loss γ = 2 | 0.8915 | 0.8364 | +0.0001 | [−0.0005, +0.0006] | discard |
| Q2 | pairwise-ranking XGBoost (rank:pairwise) as 6th member | 0.8912 | 0.8373 | +0.0009 | [+0.0002, +0.0015] | discard: test-weighted +0.0002, CI [−0.0009, +0.0012] |
| Q3 | expected-F2 plug-in decision rule (Lipton et al. 2014; Nan et al. 2012) | 0.8910 | 0.8367 | +0.0003 | [−0.0002, +0.0008] | discard |

- Q3's cut-off per fold is 0.167–0.168. On the test predictions it would cut at 0.170 (59.5% flagged vs 60.0% at 0.163), so the
  decision rule we use is already where the theory puts it. Not applied (it would use test predictions to set the cut-off).
- **Q2 check (`ba_t1_ranker_check.py`, stopped by Claude Code for low system memory after its main result):** seed 42 reproduces
  +0.0009 (CI [+0.0003, +0.0015]), but reweighted to the test set's missing-value rates the gain is +0.0002, CI [−0.0009, +0.0012],
  and on rows with all four key columns +0.0001. Like the imputation idea in round 8, the gain sits in rows missing key columns,
  which the test set rarely has. Task 1 stays unchanged.
- **Full check, re-run 2026-11-09:** seed 7 gives OOF +0.0004 (CI [−0.0002, +0.0010]), test-weighted +0.0002 (CI [−0.0008, +0.0012]);
  seed 42 OOF +0.0009, test-weighted +0.0002 (CI [−0.0009, +0.0012]); both seeds averaged OOF +0.0009, test-weighted +0.0004
  (CI [−0.0007, +0.0014]). The OOF gain depends on the seed and none of the test-weighted intervals is above 0: discard.
- **Round 9 conclusion:** no pipeline change. Every literature method either adds nothing or helps only training rows that the
  test set does not resemble.

## Task 4 convention confirmed (2026-11-10, `bb_t4_alignment_check.py`)
The convention was confirmed: Task 4 uses the datacard codes and ANS{i} = row i of `test.csv` (ANS00001 = first row). That is exactly
what v1–v9 use (v10's sorted order is wrong). End-to-end check of our column:
1. v9's task4 = argmax of the cached ensemble probabilities on 97.9% of rows (the rest are the class-wise multipliers); nothing is reordered when writing.
2. A saved checkpoint re-run on `test.csv` rows by file name reproduces the cached probabilities (same class on 600/600 rows), so cache row i = `test.csv` row i.
3. The same checkpoint scores 98.7% on held-out training images with the text labels mapped to the datacard codes.
So our Task 4 file follows the stated convention exactly, yet the upload history (equal weights) still implies Task 4 ≈ 0.21. The cause
is outside our file: the Task 4 answer key / scorer, or unequal task weights. Only the TAs can check (e.g. score v9's Task 4 column on its own).

## Round 10 (2026-11-10): forensic audit + generalisation checks (`bc_audit_phase1.py`, `bd_phase2_t1t2.py`, `bf_phase2_t5.py`, `bg_phase2_t3.py`)
Acceptance rule for this round: better on >= 4 of 5 folds, fold std not larger, paired-bootstrap 95% CI above 0, and (Task 1) not worse
under the test's missing-value pattern.

### Phase 1 audit
| Check | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| train-vs-test AUC (all / missing indicators only / median-filled values) | 0.901 / 0.903 / 0.894 | 0.500 / 0.496 / 0.498 | 0.498 / 0.498 / 0.498 | – | test only |
| largest value shift (KS on observed values) | 0.010 | 0.012 | 0.010 | brightness 115 → 105 | – |
| informative missing columns (p < 0.001) | 0 of 31 | 0 of 44 | 0 of 32 | – | – |
| impossible values | none (bounded scores clipped at their limits) | none (`account_age_days = 999` is not a sentinel) | none | 0 unreadable, all 230×120, 41 near-blank train frames | 207 accounts off-globe (139 teleports, 68 date-line wraps, already handled); `refund_count_2/3 = −1` in 16% = floor, not anomaly-related |
| duplicates / constant / collinear | none / none / none | none / none / none | none / none / hist ~ prev (known identity) | no duplicate images | none / none / per-period lat, lon, skill series |

- **Task 1's shift is only in which cells are missing** (test misses key columns far less often, a few others more often). Values do not
  shift and missingness carries no label signal (MCAR). Under a test-like missing-value pattern the LightGBM member scores F2 0.847 vs 0.835
  plain, so the test F2 should be above the OOF estimate.

### Task 1 (B0 nested F2 0.8364, folds 0.8347 / 0.8369 / 0.8374 / 0.8343 / 0.8386, std 0.0016)
| Exp | Change | F2 | Δ | 95% CI | folds better | fold std | test-like member Δ | Decision |
|---|---|---|---|---|---|---|---|---|
| A1 | missing-value indicators + native NaN | 0.8363 | +0.0002 | [−0.0003, +0.0008] | 4/5 | 0.0017 | −0.0001 | discard |
| A2 | median imputation | 0.8362 | +0.0000 | [−0.0007, +0.0008] | 3/5 | 0.0020 | −0.0006 | discard |
| A3 | masking-robust training (10% random cells masked, rows duplicated) | 0.8365 | +0.0003 | [−0.0003, +0.0011] | 3/5 | 0.0021 | +0.0009 | discard |
| A4 | cut-off = median of per-fold optimal cut-offs | 0.8365 | +0.0001 | [−0.0002, +0.0005] | 3/5 | 0.0014 | – | discard |
| A5 | isotonic calibration of members before stacking | 0.8363 | −0.0001 | [−0.0006, +0.0005] | 2/5 | 0.0009 | – | discard |
| A6 | LightGBM averaged over 3 seeds | 0.8362 | +0.0001 | [−0.0006, +0.0007] | 3/5 | 0.0017 | −0.0005 | discard |

Stacker calibration: Brier 0.1245, ECE 0.010 (already calibrated; the logistic stacker is a Platt-type calibrator).

### Task 2, features only (B0 nested macro-F1 0.7536, folds 0.7535 / 0.7481 / 0.7544 / 0.7558 / 0.7562, std 0.0029)
| Exp | Change | macro-F1 | Δ | 95% CI | folds better | fold std | Decision |
|---|---|---|---|---|---|---|---|
| B1 | class multipliers = median of per-fold optima | 0.7540 | +0.0004 | [−0.0003, +0.0013] | 3/5 | 0.0027 | discard |
| B3 | LightGBM averaged over 3 seeds | 0.7534 | −0.0004 | [−0.0010, +0.0001] | 2/5 | 0.0023 | discard |

Stacker calibration: log-loss 0.6058, multiclass Brier 0.3272.

### Task 5: the brief's consensus ensemble vs the pipeline
| Detector | flags | gap families caught | seed stability (Jaccard) | F3 vs id block (reference only) |
|---|---|---|---|---|
| pipeline: 6 gap families + dependence pocket | 2,191 | 100% | deterministic | **0.983** |
| consensus (IF, ECOD, COPOD, MCD, AE), robust scaling, knee cut | 12,942 | 94% | 0.935 | 0.605 |
| same, cut at the pipeline's count | 2,191 | 61% | 0.869 | 0.552 |
| robust scaling + PCA 90% subspace, knee / same count | 12,939 / 2,191 | 72% / 48% | 0.905 / 0.833 | 0.457 / 0.420 |
| quantile scaling, knee / same count | 12,935 / 2,191 | 90% / 39% | 0.921 / 0.704 | 0.572 / 0.358 |
| quantile scaling + PCA, knee / same count | 12,944 / 2,191 | 88% / 37% | 0.934 / 0.810 | 0.566 / 0.345 |

The knee of a fused score is degenerate on this data (it cuts at half of all accounts), and at any matched count the consensus misses
most of the families. The pipeline stays.

### Task 3 (B0 = merged pipeline 0.79423, folds 0.7876 / 0.7946 / 0.7977 / 0.8011 / 0.7903, std 0.00487)
Already in place and not re-tested: L1 objectives (the median is the NMAE-optimal target), Tweedie as a mean member, folds stratified on
zero / positive-quantile bins of the target, the zero gate at the median cross-over (round 8). Huber is not used because it targets
something between the mean and the median.

| Exp | Change | OOF | Δ | 95% CI | folds better | fold std | Decision |
|---|---|---|---|---|---|---|---|
| C2 | the 4 ratio stage-2 models with ~99% of the blend weight averaged over seeds 42 / 2026 / 777 | 0.79434 | +0.00011 | [−0.00001, +0.00023] | 4/5 | 0.00498 | discard (CI touches 0; ~+0.00002 LB) |
| C3 | C2 + stage-1 P(y>0) averaged over the same seeds | 0.79435 | +0.00011 | [−0.00001, +0.00023] | 4/5 | 0.00497 | discard |

Seed averaging lowers every member's stage-2 error a little (0.1357 → 0.1350 for CatBoost), so the effect is real but too small to
clear the bar. It is the closest of all round-10 changes.

### Multi-seed summary (brief step 3)
3-seed averaging was tested for the main GBDT of every task: Task 1 +0.0001 (3/5 folds), Task 2 −0.0004 (2/5), Task 3 +0.00011 (4/5,
CI touching 0). None clears the acceptance rule, so the single-seed models stay.

### Deliverables
`src/audit.py` + section 3.6 of the master notebook and a "Data-quality audit" cell in each task notebook. No model or decision changed.

## Round 11 (2026-11-10): Task 4 on the test images, Task 1 tuning (`bh_t4_foundation_feats.py`, `bi_t4_probe_eval.py`, `bj_t1_tune.py`)

### Task 4: how good are we on the TEST images? (label-free)
The fine-tuned models are much less sure on test than out-of-fold: median max-probability 0.89 vs 0.91, and only 38–46% of test images
above 0.9 vs 73–82% OOF. The fold models agree with each other on 90–92% of test images, ConvNeXt and DeiT on 90%.
Three published estimators of accuracy without labels (they read our own predictions on the unlabeled test images, nothing is fitted):
ATC (Garg et al. 2022), DoC (Guillory et al. 2021), agreement-on-the-line (AGL, Baek et al. 2022; 15 model pairs, probit fit r = 0.94).

Frozen foundation features + logistic-regression probe (same 5 folds), to see whether web-scale features lose less under the shift:

| Model | OOF acc | OOF macro-F1 | ATC | DoC | AGL | agrees with our ensemble on test | predicted test shares (7-11k / fiveN / freefried / pube / roblock) |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T (fine-tuned) | 0.9886 | 0.9865 | 0.869 | 0.897 | 0.910 | 0.945 | 0.17 / 0.24 / 0.15 / 0.26 / 0.17 |
| DeiT-III-S (fine-tuned) | 0.9862 | 0.9839 | 0.897 | 0.908 | 0.901 | 0.954 | 0.17 / 0.25 / 0.15 / 0.25 / 0.18 |
| CLIP ViT-B/16, frozen | 0.8683 | 0.8526 | 0.737 | 0.782 | 0.696 | 0.675 | 0.25 / 0.20 / 0.16 / 0.27 / 0.14 |
| DINOv2-B (reg4), frozen | 0.9153 | 0.9036 | 0.798 | 0.819 | 0.757 | 0.741 | 0.21 / 0.21 / 0.14 / 0.28 / 0.16 |
| SigLIP-2 B/16, frozen | 0.8774 | 0.8631 | 0.764 | 0.802 | 0.707 | 0.712 | 0.23 / 0.19 / 0.15 / 0.26 / 0.17 |
| all three concatenated | 0.9382 | 0.9297 | 0.815 | 0.858 | 0.793 | 0.779 | 0.21 / 0.21 / 0.14 / 0.27 / 0.17 |
| (train label shares) | | | | | | | 0.39 / 0.14 / 0.18 / 0.19 / 0.10 |

- Our fine-tuned ensemble is estimated at **~0.87–0.91 accuracy on test** (vs 0.99 OOF): a real domain shift, worth ~0.08–0.1 of
  Task 4 *if* the answer key is fixed.
- Frozen features are worse on test by every estimator (0.70–0.86), so they are not more robust here → **discarded**.
- Every model, including the independent frozen ones, predicts only ~17–25% 7-11 Knights on test vs 39% in train: the test set very
  likely has a different class mix (label shift), not a model that fails on 7-11 Knights.

### Task 1: tuning every boosted-tree member (`bj_t1_tune.py`, studies in `outputs/optuna_round11.db`)
Before: LightGBM tuned (40 trials, 2026-11-03); XGBoost and CatBoost on fixed settings. Optuna TPE, 30/30/25 trials on a 3-fold split,
positive-class F2 at the best cut-off, trial 0 = current settings. Then each tuned member (learning rate halved, notebook rule) was swapped
into the stack on the 5 pipeline folds against the same member refitted with its current settings (control stack F2 0.8366, folds
0.8343 / 0.8375 / 0.8370 / 0.8350 / 0.8394, std 0.0018).

| Member | tuning folds: current → best | member F2 (5 folds): current → tuned | stack F2 | Δ | 95% CI | folds better | fold std | test-like member Δ | Decision |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.8354 → 0.8364 | 0.8340 → 0.8359 | 0.8371 | +0.0005 | [−0.0002, +0.0012] | 3/5 | 0.0017 | +0.0031 | discard (closest) |
| CatBoost | 0.8351 → 0.8359 | 0.8358 → 0.8358 | 0.8364 | −0.0003 | [−0.0009, +0.0004] | 2/5 | 0.0019 | +0.0005 | discard |
| LightGBM (re-tuned, positive-class F2) | 0.8360 → 0.8368 | 0.8347 → 0.8359 | 0.8367 | +0.0001 | [−0.0006, +0.0007] | 3/5 | 0.0017 | +0.0010 | discard |
| all three tuned | | | 0.8367 | +0.0001 | [−0.0007, +0.0008] | 3/5 | 0.0018 | +0.0015 | discard |

Tuning lifts single members by up to +0.002 F2, but the stack already averages their differences away: at most +0.0005 for the stack,
inside the noise. The pipeline keeps its settings; tuned values are in `outputs/bj_t1_tuned_params.json`.

### Last year's 1st-place repo (github.com/Saltywan/CPE342-KAGGLE, CPE342 1/2025; public; read on 2026-11-11)
Same competition data (same Task 4 test images and ids). What they did, and what it tells us:

| Task | Their method | Ours | Result on our folds |
|---|---|---|---|
| 1 | 24 of 31 raw columns, median imputation, RF / LGB / XGB / Cat (Optuna) + LightGBM meta-model, cut-off on macro F2 | 49 features, native NaN, 5 members + LR stacker, positive-class F2 | their recipe 0.8376 vs ours 0.8364; their members alone are strong (Cat 0.8370) → **adopted as extra members** (below) |
| 2 | stack + **`player_id` features** (number, normalised, quartile bucket) | `player_id` blocks in v9 | they used the id too and still won both boards: useful for the TA question on `player_id` |
| 3 | median imputation, LGB-MAE / XGB-squared / Cat-MAE + LightGBM-MAE meta | identity repair, ratio hurdle, L1 blend | their recipe **0.78276** vs ours **0.79423** → nothing to take |
| 4 | ConvNeXt-V2-T, 224×224, no augmentation, class-weighted CE, trained on **56,318** labelled images (train + validation) | ConvNeXt-T + DeiT-III-S, 31,546 images | `test.csv` has `id = ANS00001…` + `file_name = scyd5lz.jpg…`: **ANS{i} = row i of test.csv, exactly our order** |
| 5 | PyOD PCA, contamination 5% | 6 gap families + pocket | much weaker than ours (Isolation-Forest level) |

- Their first five Task 4 test predictions (printed in the notebook) agree with ours on 3/5; the two disagreements are dark,
  very noisy images where all 12 of our models agree with each other. A test-like re-weighting of our OOF (`bl_t4_testlike.py`:
  train-vs-test classifier on DINOv2 / CLIP embeddings, AUC 0.82–0.84) gives **0.984 accuracy on test-like training images**
  (0.981 in the most test-like decile; 0.982 in the darkest quintile), so the disagreements do not indicate a broken model.
- Last year's teams had ~24.8k more labelled Task 4 images than we do. Whether this year has a validation set is a TA question.

### Task 1: their ideas on our folds (`bk_lastyear_methods.py`, `bm_t1_rf_confirm.py`, `bn_t1_combined.py`, `bo_t1_seedavg.py`)
| Variant | OOF F2 | Δ | 95% CI | folds better | test-like Δ | Decision |
|---|---|---|---|---|---|---|
| LightGBM meta-model instead of LR stacker | 0.8368 | +0.0005 | [−0.0002, +0.0012] | 4/5 | – | discard |
| + RandomForest (median-imputed), vs cached members | 0.8374–0.8375 (3 seeds) | +0.0010…+0.0011 | all > 0 | 4–5/5 | +0.0012 | superseded: vs refitted members only +0.0001 (2-seed avg) |
| their whole recipe (LGB meta) | 0.8376 | +0.0012 | [+0.0002, +0.0021] | 4/5 | – | reference |
| **+ their 4 members (24 cols, median), 2-seed average** | **0.8383** | **+0.0016** | **[+0.0008, +0.0024]** | **5/5** | **+0.0023** [+0.0004, +0.0042] | **KEEP → merged** |
| + RF + their 4 (2-seed avg) | 0.8381 | +0.0014 | [+0.0006, +0.0022] | 4/5 | +0.0022 | keep-level, not chosen (fewer folds) |
| + their CatBoost only (2-seed avg) | 0.8379 | +0.0012 | [+0.0004, +0.0020] | 5/5 | +0.0020 | keep-level, smaller |

(Baseline for the last four rows = all our members refitted, F2 0.8367.) Single seeds vary: + their 4 gave +0.0015 with seed 42
and +0.0004 with seed 7, hence the 2-seed averaging in the pipeline. LightGBM ablation: our 49 features 0.8351, their 24 columns
0.8355 (median) / 0.8363 (native NaN), all 31 raw columns 0.8344 / 0.8364: the gain comes from diversity, not from one better input.

### v11 upload and a second look at last year's repo (2026-11-11)
- **v11 = v9 + round-11 Task 1: public 0.76242, the same as v9.** The expected ~+0.0003 (Task 1 weight × the 387 changed labels
  on the public share of the rows) is below what the public split can show. The +0.0016 nested OOF gain (5/5 folds) is the evidence;
  v11 is the better final pick between the two. Snapshot: `versions/v11_LB0.76242/`.
- **Same test set as last year.** Their Task 4 notebook predicts ~25.9k images (405 batches of 64) and their `test.csv` starts at
  `scyd5lz.jpg`, like ours (25,889 rows). Their labelled set was 56,318 images, class counts {0: 15,865, 1: 8,601, 2: 12,277,
  3: 8,838, 4: 10,737}; ours is 31,546 {12,429, 4,407, 5,596, 5,926, 3,188}. Every one of our counts is below theirs, consistent
  with this year's train being a subset of last year's train + validation. Their held-out accuracy was 0.989 (8% split, epoch 10),
  the same level as our OOF; nothing in their Task 4 recipe beats ours.
- **Task 2:** their `player_id` features (number, normalised, z, quartile bucket) in a GBDT stack cannot beat the id block:
  our id-block OOF 0.968 is the ceiling set by the 3% random label flips (block + features 0.9678). Nothing to take.
- **Last year's leaderboard score is not in the repo.** Kaggle renders the leaderboard with JavaScript, so it could not be read
  here. Because the test set is the same, last year's top public score would show how much of this year's gap to it is Task 4
  scoring. Do **not** late-submit our predictions to last year's competition: it scores against the same hidden test labels,
  which is test-set probing.

### Task 5: is there a second leaking column? (`bp_t5_leak_scan.py`, `bp2_t5_leak_scan.py`, 2026-11-11)
Prompted by a rumour of "one more leaking feature". Reference = the id block (2,070 accounts). Result: **no second leak.**
- No column encodes the id: largest |Spearman(id, column)| outside the block 0.017 (noise ~0.012). NaN count per row and file
  row position vs block: AUC 0.501 / 0.501. No duplicated rows, no non-integer value shared between rows, no constant
  period-4/period-3 ratios, decimal digits as written in the csv differ by < 0.7 digits (friend_removes only, from the injection).
- Best single-column AUC for the block is 0.63 (friend_adds_4).
- **Values that occur only inside the block:** device_count_4 = 4 (156), guild_changes_1 = 3 (112), guild_changes_4 = 4 (97);
  339 accounts, 0 outside. These are injected anomaly values (outside the normal range), i.e. real signal, and **all 339 are
  already flagged** by the clean pipeline. Adding them changes nothing (F3 vs block 0.9827 → 0.9827).
- Flag columns are weak: password_changed = 1 → 25% in block, mass_item_sale = 1 → 34%, vpn_usage = 1 → 12%.
- v11's Task 5 is the id block itself (F3 1.0 if the block is the truth), so no second leak could raise v11.

## Round 12 (2026-11-11): the "break the plateau" brief (`bq0_row_lag.py`, `bq_t1_paradigms.py`, `bq2_t1_followup.py`, `br_t2_maha.py`)
Only the parts not tested in earlier rounds were run; same folds, nested cut-offs, test-like check and acceptance rule.
**Nothing passes. No pipeline change, no new submission.**

**Rules applied.** The denoising autoencoder is fitted on training-fold rows only (fitting it on test rows would be using the test
set for a training-related purpose). TabPFN / TabICL are pre-trained models, which the rules allow for Task 4 only. AutoGluon and
similar AutoML frameworks are prohibited. No test labels, pseudo-labels or leaderboard probing.

**Past CPE342 repositories (Phase 1).** 12 repositories from the 1/2025 class besides Saltywan's (notebooks downloaded as data, never
run). Best printed CV per task: T1 F2 0.8368 (tangIwza: LGB/XGB/Cat equal blend; ours 0.8383); T2 0.9685 with `player_id`
(PasinS47: the same id blocks we found; ours 0.968), features only ≤ 0.754 (ours 0.7536); T3 normalised MAE 0.271 at best
(PPANUT; ours ≈ 0.259, i.e. score 0.794); Liliable2's "NMAE 0.170" is computed on spenders only (zeros left out), not comparable;
T4 held-out F1 0.935–0.992 (ours 0.988). No repo beats us on any task. Not used: `KwanchanokThungsuk/Player_Intelligence_System`
(created 2026-11-05 with this year's data, so a current competitor); `chotanansub/karena-kaggle` holds only a README title and banners.

**Methods research.** TabM (Gorishniy et al., ICLR 2025; parameter-efficient MLP ensemble) and RealMLP (Holzmüller et al., NeurIPS
2024) are the strongest recent tabular nets and appear in recent Kaggle wins, usually blended with GBDTs via hill climbing or ridge
stacking. Both papers report roughly GBDT-level accuracy, not a consistent margin.

**Row order (Phase 2.1).** Neighbour-to-neighbour (lag-1) dependence in file order and in id order, Tasks 1–3 train/test and Task 5:
no feature or target exceeds 4/√n in any file, except the known Task 2 id blocks (94% of id-neighbours share a segment vs 28% by
chance). Cross-task links were ruled out in round 7.

### Task 1 (baseline = current pipeline stack, 9 members, nested F2 0.8383)
| Exp | Change | OOF F2 | Δ | 95% CI | folds better | test-like Δ | Decision |
|---|---|---|---|---|---|---|---|
| N1 | spline (quantile knots) + missing flags + scaled logistic regression member (alone 0.8081) | 0.8379 | −0.0004 | [−0.0009, +0.0001] | 2/5 | +0.0001 | discard |
| N2 | LightGBM + in-fold class-conditional Mahalanobis distances (Ledoit-Wolf), seed 42 | 0.8385 | +0.0002 | [+0.0001, +0.0004] | 5/5 | +0.0002 | passed, but see seeds |
| N2 | same, seed 7 / seed 2026 | 0.8384 / 0.8383 | +0.0001 / +0.0001 | include 0 | 4/5 / 3/5 | +0.0001 / −0.0001 | discard |
| N2 | same, 3-seed average | 0.8383 | +0.0000 | [−0.0001, +0.0002] | 2/5 | +0.0002 | **discard** (seed 42 was luck) |
| N3 | LightGBM + 128-d swap-noise DAE bottleneck (DAE fitted on training-fold rows) | 0.8382 | −0.0001 | [−0.0003, +0.0002] | 2/5 | +0.0001 | discard |
| N4 | TabM-style MLP ensemble (k = 16, BCE; alone 0.8144) | 0.8383 | +0.0000 | [−0.0001, +0.0002] | 1/5 | −0.0000 | discard |
| N5 | TabM with BCE + (1 − soft-F2) loss (alone 0.8130) | 0.8382 | −0.0001 | [−0.0003, +0.0002] | 1/5 | −0.0002 | discard |
| S1 | rank averaging of the 9 members, no stacker | 0.8290 | −0.0093 | [−0.0105, −0.0080] | 0/5 | −0.0103 | discard |
| S2 | Caruana hill climbing on F2 (nested; weights mostly Cat / their Cat / LGB) | 0.8370 | −0.0013 | [−0.0020, −0.0006] | 0/5 | −0.0010 | discard |

The first S2 run fed logits to the 0.05–0.95 cut-off grid (0.7659); the fixed run maps back to probabilities. The logistic stacker
stays the best combiner: it uses all nine members and is fitted, not searched.

### Task 2, features only (no-`player_id` fallback; v11's Task 2 is the id block)
| Exp | Change | macro-F1 | Δ | 95% CI | folds better | Decision |
|---|---|---|---|---|---|---|
| B0 | notebook stack + nested class multipliers | 0.7536 | – | – | – | baseline |
| M1 | + LightGBM with 4 in-fold class Mahalanobis distances (alone 0.7542 vs plain 0.7536) | 0.7535 | −0.0001 | [−0.0007, +0.0007] | 3/5 | discard |
| M2 | class multipliers shrunk 30% toward 1 | 0.7538 | +0.0002 | [−0.0003, +0.0007] | 3/5 | discard |

### Brief items already tested in earlier rounds (not repeated)
- Task 1: tabular MLP (round 3), in-fold kNN target features (round 4, E1), focal loss and pairwise ranking (round 9, Q1/Q2),
  expected-F2 decision rule (Q3), isotonic calibration (round 10, A5), Optuna tuning (round 11). The metric is positive-class F2,
  not macro (v3 → v4 upload pair, round 7).
- Task 3: L1 / quantile (12 levels) / Tweedie members, median read-out, hurdle with exact zeros, L1-optimal decile multipliers
  (R3), Caruana selection (R4: +0.00006, negligible), zero-inflated lognormal network (R2).
- Task 4: class-wise multipliers are already in the pipeline; frozen CLIP / DINOv2 / SigLIP probes (round 11) were worse. More
  backbones are not run: Task 4 scores at chance on the leaderboard until the TAs fix the key (README question #1), so a better
  Task 4 model cannot move the score now.
- Task 5: trajectory features (per-period shifts, jumps, burstiness, geo hops; round 2), multi-view ensemble (IF / ECOD / COPOD /
  MCD / autoencoder, round 10: F3 0.35–0.61 vs pipeline 0.983), the F3 cut-off rule (flag when P(anomaly) > F3/10 ≈ 9%).
  v11's Task 5 is the id block, so nothing can raise it.

## Round 13 (2026-11-11): cross-task links, calibration, Task 4 labels (`bs_cross_task.py`, `bt_t5_cutoff.py`, `bu_t4_labels.py`, `bv_t3_scale_whale.py`, `bv2_t3_followup.py`)

**Rules kept:** auxiliary models use training rows only, test rows are only predicted, and no upload variants were made. The brief asked for two Task 5 variants "to isolate Task 5's impact on the leaderboard"; that is the probing the rules forbid, and the numbers below say both would score lower anyway. `sample_submission.csv`: only column names and types were read.

**1. Cross-task links: none.**
- Exact entity match on all matching columns (4-10 columns, T1/T2/T3 in every train/test combination): 0 rows, equal to the shuffled baseline. Pairs with Task 5 share one column only and match at exactly the chance count.
- 82 pairings of matching columns by ANS position or by `player_id`, train and test: largest |Spearman| 0.0368 (n = 4,682, about 2.5 standard errors, the size expected from 82 tries). Task 2 segment = Task 3 segment on 28.0% of rows.
- v11 by ANS id: mean predicted Task 3 spending by Task 2 class is 10,380 / 10,353 / 10,583 / 10,722 (a linked Whale would be many times higher). Every chi-square p > 0.19, Cramer's V <= 0.0153. The clean file agrees.
- Transfer (rank-mapped shared columns, fitted on the other task's training rows): Task 3 member +0.00014 (4/5) but weight 0.0 in the blend in all 5 folds, so +0.00000; Task 2 member +0.0000 (3/5); Task 1 member F2 -0.0003 (3/5). All discard.

**2. Task 5 cut-off: v11 is already at the top.**
- v11 flags 2,070 = 7.996%, identical to the id block. The score of each upload implies Task 5 = 0.99-1.01 given Task 4 = 0.20-0.22.
- Consensus score alone: best F3 against the block is 0.664 at the top 18%. Knee at 5.90%; the score falls from 124 to 3.3 between the top 3% and 4%.
- Adding consensus flags to v11 lowers F3 against the block: top 10% 0.943, top 20% 0.852, top 35% 0.740.
- If the true rate were 12%, v11 would score about 0.69 and the leaderboard total would be about 0.70, not 0.762. That scenario is excluded by the uploads we already have.

**3. Task 4: the label-order hypothesis cannot explain the score.**
- Datacard, slides 20-21 and our code all use the alphabetical order. Simulating the order quoted in the brief on OOF gives macro-F1 0.0028, not 0.21 (no class stays in place).
- Permutations by fixed points: 0 -> 0.003, 1 -> 0.200, 2 -> 0.397, 3 -> 0.594, 5 -> 0.988. A shuffled row order gives 0.200 +- 0.002. So 0.21 fits a shuffled key or a code permutation with exactly one class in place (45 of 120), and only the TAs can tell which.
- Slide 24 shows Task 4 as text; text uploads scored Task 4 = 0, integers about 0.21, floats about 0.03. Worth stating in TA question #1.
- Model: 2-backbone OOF macro-F1 0.9881, accuracy 0.9899, weakest class 0.9802, h-flip TTA already in. Above the brief's 0.95 bar, so no upgrade was trained.

**4. Task 3: the exact L1 blend already solves the scale.** Baseline B0 = pipeline, 0.79423, scale 1.000.

| Variant | NMAE | delta | 95% CI | folds | Decision |
|---|---|---|---|---|---|
| A1 nested scale alpha 0.90-1.15 | 0.79423 | 0 | 0 | 0/5 | discard (alpha = 1.000 in every fold) |
| A2 nested alpha + zero cut-off tau | 0.79423 | 0 | 0 | 0/5 | discard (tau = 0 in every fold) |
| A3 sum-matching alpha | 0.79423 | -0.00000 | [-0.00006, +0.00005] | 2/5 | discard |
| W1 raw-scale Huber member | 0.79423 | 0 | 0 | 0/5 | discard (weight 0.0 in all 5 folds; member alone 0.78206) |
| W2 top-spender raw L1 model | 0.79423 | 0 | 0 | 0/5 | discard (nested weight 0 in every fold) |
| E1 Task 2 segment member in the blend | 0.79423 | 0 | 0 | 0/5 | discard (weight 0.0) |
| **C1 cap predictions at the training maximum** | **0.79434** | **+0.00011** | **[+0.00003, +0.00021]** | **4/5** | **passes; 7 OOF rows** |

- sum(pred)/sum(y) = 1.0052: no undershoot. Whales are not under-predicted: on the top 1% by prediction, sum(y)/sum(pred) = 0.974 and median y/pred = 1.006; about 50% of y lies above the prediction in every decile. The "0.707" on the top 1% by actual y is selection on the outcome.
- Top 1% of absolute errors = 39.5% of total error, of which 47.7% is actual zeros (a whale who skips a month), the part already known to be unpredictable.
- C1 on the test file: the target is capped at 300,000 (15 training rows sit exactly there). One v11 test prediction is above it (431,946). If that row's true value is at most 300,000, its error falls by exactly 131,946 THB whatever the true value is: about +0.0003 NMAE, about +0.00006 on the leaderboard if the row is scored. This is below the public-board noise (SD about 0.0004), so an upload could not confirm it.
- **C1 merged into the pipeline** (`scripts/nb_part2.py`, section 7.4): out-of-fold cap = each fold's training maximum, test cap = `y3.max()`, applied after the scale and the zero gate. All six notebooks re-executed with no errors; Task 3 OOF reads 0.79434. Compared with the pre-cap outputs exactly one Task 3 row changed (ANS14698, 431,946.09 → 300,000) and Tasks 1, 2, 4, 5 are identical. `submission_v12_try.csv` = `submission_v11_try.csv` with only that value capped (one line differs from v11, sha256 `de380b025cdfe095154ffdfe5ef8d57fd7ed0e48160b90d3ceb94241ca29c384`). Clean pipeline `submission.csv` = `submission_clean_try.csv`, sha256 `061cb9866cab7de5d203799a890834c6a4186790e38ec107bf90bd9f33620b28`.

## Round 14 (2026-11-11): v12 audit, Task 5 cut-off, Task 3 post-processing, 10-seed bagging (`bw_r14_audit.py`, `bx_r14_t3_post.py`, `by_r14_bag.py`, `bz_r14_eval.py`, `ca_r14_build_v13.py`, `cb_r14_cache_t3_bag.py`)

v12 scored **0.76248** (+0.00006 over v11, exactly the cap's predicted effect, so the capped row is in the public split). **Rules kept:** every fit uses training rows only, test rows are only predicted, no upload variants.

**1. v12 audit (`outputs/bw_r14_audit.log`).**

| Task | OOF | Model / rule | v12 test predictions |
|---|---|---|---|
| 1 (F2) | 0.83829 at the OOF cut-off, 0.83794 nested | 9-member log-odds stack; LGB/XGB/Cat/ET 1 seed, the four 24-column members 2 seeds, LogReg deterministic; tau = 0.15035 | 15,690 ones (60.605%); train prevalence 34.93% |
| 2 (macro-F1) | 0.96777 (clean pipeline 0.75359 nested) | id block, cuts 52,000 / 84,500 / 110,500; features unused | classes 0-3: 10,352 / 6,462 / 5,211 / 3,864 |
| 3 (NMAE) | 0.79434, folds .7877/.7947/.7977/.8013/.7903 | blend, zero gate, scale 1.000, cap 300,000 | zeros 50.02% (OOF 49.89%, train 48.12%); sum 270,797,197 |
| 5 (F3) | no labels | `id >= 23,819` | 2,070 ones (7.996%) |

**2. Task 5 cut-off (`outputs/bw_r14_audit.log`): no change.** Equal weights put Task 5 at about 1.00 (Task 4 = 0.21). The 20 lowest family ids are consecutive from 23,819. The 500 ids below the cut: 0 family, 0 pocket, evidence-rule rate 0.2%, mean consensus rank 0.459 (control band 20,000-20,499: 0.478); the 500 above: 100% family. Change points: consensus 23,818, families 23,819, evidence rules 23,819. No impossible-travel or device-switch extreme lies outside the block; login-burst extremes do, but the earlier gap scan found no cluster there. Adding the 500 ids below costs 0.0236 F3 (LB -0.0047) if they are normal; break-even needs a 10% hit rate.

**3. Task 3 zero-snap and joint scale (`outputs/bx_r14_t3_post.log`): no change.** Nested fine-grid zero-snap picks tau 0-2 THB: delta -0.000000, 0/5. Nested Nelder-Mead (tau, alpha): tau 0, alpha 1.0001-1.0005, +0.0000002, 3/5. In every prediction bin above 1 THB fewer than half the rows are y = 0, so snapping loses; sum(pred)/sum(y) = 1.0051.

**4. 10-seed bagging (seeds 42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182; features, hyper-parameters and folds unchanged).** Seed-42 refits equal the pipeline cache exactly (max diff 0; ET 3e-16). Classification: log-odds / log-probability averaging; Task 3: probability for stage 1, log scale for stage 2. Stacker, cut-off, multipliers, readouts, L1 weights, gate and scale are re-learned on the bagged OOF, nested.

Task 1 (`outputs/bz_r14_eval_t1.log`; B0 nested 0.83794, fold std 0.00117, test-like 0.85046). Single-seed spread of the whole stack over the 10 seeds: mean 0.83788, std 0.00030 (0.83731-0.83831), so v12 is a typical draw.

| Variant | nested F2 | delta | 95% CI | folds | fold std | test-like delta | Decision |
|---|---|---|---|---|---|---|---|
| 3-seed bag | 0.83835 | +0.00041 | [-0.00001, +0.00086] | 3/5 | 0.00158 | +0.00010 | discard |
| 5-seed bag | 0.83829 | +0.00034 | [-0.00010, +0.00079] | 3/5 | 0.00129 | -0.00026 | discard |
| 10-seed bag | 0.83833 | +0.00039 | [-0.00010, +0.00089] | 3/5 | 0.00141 | -0.00038 | discard |

Task 3 (`outputs/bz_r14_eval_t3.log`; B0 = v12 members + cap, 0.79434, fold std 0.00491). Bagged: both hurdle stages and the four ratio stage-2 models with 99.9% of the blend weight; the two near-zero-weight ratio members stay single-seed.

| Variant | NMAE | delta | 95% CI | folds | fold std | Decision |
|---|---|---|---|---|---|---|
| 3-seed bag | 0.79450 | +0.00016 | [+0.00005, +0.00027] | 4/5 | 0.00496 | passes |
| 5-seed bag | 0.79452 | +0.00018 | [+0.00007, +0.00031] | 5/5 | 0.00495 | passes |
| **10-seed bag** | **0.79458** | **+0.00024** | **[+0.00011, +0.00037]** | **5/5** | 0.00498 | **passes (used)** |

Task 2 (clean fallback only; v12's Task 2 is the id block; `outputs/bz_r14_eval_t2.log`; B0 = clean pipeline, nested macro-F1 0.75359, fold std 0.00293). Bagged: LightGBM, XGBoost, CatBoost (geometric mean over seeds); LogReg stays; stacker and multipliers re-learned nested.

| Variant | nested macro-F1 | delta | 95% CI | folds | fold std | Decision |
|---|---|---|---|---|---|---|
| 3-seed bag | 0.75376 | +0.00017 | [-0.00060, +0.00083] | 3/5 | 0.00171 | discard |
| 5-seed bag | 0.75348 | -0.00012 | [-0.00082, +0.00059] | 3/5 | 0.00209 | discard |
| 10-seed bag | 0.75369 | +0.00009 | [-0.00058, +0.00081] | 3/5 | 0.00196 | discard |

Summary of step 4: only Task 3 passes. Tasks 1 and 2 stay as in v12 and in the clean pipeline.

**5. v13 (`ca_r14_build_v13.py`, `outputs/ca_r14_build_v13.log`).** `submission_v13.csv` = v12 with only the Task 3 column replaced by the 10-seed bag, following the pipeline path (0.5 x 5-fold + 0.5 x Train+Val refit per seed, then the seed average). Checks before writing: the same code fed the cached v12 members reproduces the uploaded v12 Task 3 text on 25,889 of 25,889 rows; the seed-42 Train+Val refits equal the cache (diff 0); the bagged OOF equals the evaluation (0.794578). Re-learned blend: scale 1.000, gate shift -0.0004 (v12 +0.0010), weights ratio[hist] Cat 0.673, ratio[hist] XGB 0.195, ratio[prev] LGB tuned 0.133.
- Rows differing from v12: task1 0, task2 0, **task3 12,945**, task4 0, task5 0. Task 3: median change on changed rows 16.53 THB, mean over all rows 67.83, p99 1,264, max 6,312; relative change on rows positive in both: median 0.53%, p95 1.87%; 8 rows 0 -> positive, 11 positive -> 0; zero share 0.5002 -> 0.5003; sum ratio 1.00015; max still 300,000; corr(log1p) 0.99936.
- `submission_v13.csv` sha256 `538d250be0078afed41090a0290148e78a83a62465cfeb9271468c729d40c76a`; `submission_clean_v13.csv` (clean fallback + the same Task 3) sha256 `7239937c8c925deba9b1a50f84facceef4f498e992662b1e09f493620b9728e6`.
- Expected leaderboard effect at equal weights: +0.00005, below the public-board noise. The decision rests on the out-of-fold evidence, not on the public score.
- **Merged into the pipeline** (`scripts/nb_part2.py`, sections 7.3 and 7.6; user-approved): `SEEDS3`, per-seed caches `t3r_*_s<seed>` copied from the bag run by `cb_r14_cache_t3_bag.py` (seed-42 files equal the existing cache; the notebook lambdas recompute any missing seed with the same parameters). All six notebooks re-executed, 0 error outputs; the Task 3 cell reads 0.79458 with the weights above. The pipeline `submission.csv` matched `ca_r14_build_v13.py` except 69 Task 3 values that differ by exactly 0.01 THB (seed averaging in a different floating-point order), so the final files take Task 3 from the pipeline (`cc_r14_v13_from_pipeline.py`): `submission_v13.csv` sha256 `c878a5677f4bab1a30134ae6d795bbc5cff265483b73fc34a008e47c63625eb0` (v12 + pipeline Task 3; same diff counts as above), `submission_clean_v13.csv` = `submission.csv` sha256 `bf4c4a364cf0c35b2518b8776b77823e1bbc288d9bfead45b33efd9f448c72f2`. The three format-variant files change in Task 3 only. The earlier sha256 values in this item are superseded.

## Round 15 (2026-11-12): five "structural blind spot" checks (`cd_r15_blindspots.py`, logs `outputs/cd_r15_t124.log`, `outputs/cd_r15_t3.log`)

Baseline = the current pipeline / v13. Nothing passes, so no new submission (v13 stays the candidate). Every fit uses training rows only; nested = chosen on four folds, scored on the fifth.

**1. Binary vs macro F2 (Task 1).** The code uses positive-class F2 (`src/config.py` `FBETA_AVERAGE = "binary"`, `optimize_threshold` default); the macro reading exists only as `outputs/format_variants/submission_task1_macro_F2.csv`. OOF: tau_binary 0.15035 (binary 0.83829, macro 0.71475; 15,690 test ones = v12); tau_macro 0.400 (binary 0.74093, macro 0.78015; 9,897 ones, **5,793 test labels flip**); max-min compromise 0.355 (binary 0.77770, macro 0.77742; 4,363 flips). Switching to tau_macro would cost 0.097 F2 = **0.0195 LB** if the scorer is binary; the compromise costs 0.061 F2 = 0.012 LB. Our own uploads already settle it: v3 -> v4 observed -0.00234, binary predicts -0.00265, macro -0.00023. Task 5 has no cut-off to choose: v12/v13 = the id block, which is optimal under both readings if the block is the truth.

**2. Task 3 tail (B0 0.794578, the 10-seed pipeline).**
- Error by group of the prediction: predicted 0 = 51,902 rows, 2.6% of the error; deciles 8-10 of the positive predictions (7,768-300,000 THB) = 83.7% of the error and 93.0% of total y. Every decile has 48-51% of y above the prediction (a centred median), and sum(pred)/sum(y) is 0.90-1.02 in deciles 8-10. No compression of the tail.
- Nested power calibration a * yhat^b: a 0.94-0.98, b 1.002-1.006, -0.00010, CI [-0.00015, -0.00004], 2/5 -> discard. With a zero cut-off: tau0 0-0.1 THB, same result.
- Raw-scale members: LGB-L1, XGB-L1, Cat-MAE (already in the pool) get weight 0.0 in every fold. New raw Huber members (LightGBM 0.78193, CatBoost 0.78027 alone) also get 0.0 in every fold (delta exactly 0). Forced 10/20/30% mixes of raw L1 or Huber: -0.00024 to -0.00207, 0/5 folds in every case.
- Why: an L1 model on log1p(y) (and the mixture-median read-out) estimates the conditional median of y itself, because the median commutes with a monotone transform. The median is already the absolute-error optimum, so it is not a "relative error" estimate that a raw-scale model would correct.

**3. player_id (Tasks 2 and 5).**
- Task 2: the id block decides 100% of v12/v13 test rows; features are never consulted (no tie-break). The clean feature pipeline disagrees on 5,429 test rows (21%). Train flips (label != block) = 3,117 = 2.997%, spread evenly over the other three classes of each block. Features cannot see them: AUC 0.507, mean P(block) 0.685 on flipped rows vs 0.690 on the rest, the feature argmax equals the true label on 7.3% of flips. Neighbouring ids carry nothing: P(flip | previous id flipped) 0.0298 vs base 0.0300; the majority label of the k nearest training-fold ids (k = 2, 6, 20) reproduces the block (agreement 0.978-1.000) and never recovers a flip. Nested soft prior (features override the block only above a learned log-odds margin): the margin chosen is 5 in every fold, i.e. never override, +0.00000. 0.968 is the random-flip ceiling.
- Task 5: the id rule decides 100% of rows; detectors are not combined. Disagreements (round 14): 145 pocket flags and 69 evidence-rule hits outside the block (the latter the known date-line false positives); 24 block accounts no detector flags. The 500 ids below the cut look like a far control band.

**4. Task 4.** `t4_round11_probes.npz` = round-11 frozen CLIP / DINOv2 / SigLIP linear probes (worse than the fine-tuned models, discarded). v12/v13 Task 4 = ConvNeXt-Tiny + DeiT-III-S (5-fold + full-data models, h-flip TTA, class multipliers), datacard codes, test.csv row order. If the key is unrelated to our column, class k scores 2 p_k q_k / (p_k + q_k) in expectation, which is maximised at p = q and gives exactly 0.200: re-mixing classes cannot beat chance (our mix 0.200 against our own predicted mix; 0.184 against the training mix). Random alignments on a 12,944-row public half differ by SD 0.0037 on Task 4 = 0.00075 LB; the 0.00067 gap to #1 is 0.9 SD. Building a class-mix variant would mean tuning to a broken key and could only be checked by uploading (probing), so nothing was built.

**5. Seeds and cut-off smoothing.** Seeds were settled in round 14: Task 1 10-seed bag discarded (v13 keeps 1 seed for LGB/XGB/Cat/ET and 2 for the 24-column members), Task 2 is the id block (clean 10-seed bag discarded), Task 3 already uses 10 seeds. Task 1 cut-off rules vs B0 (pipeline argmax, nested 0.83794):

| Rule | nested F2 | delta | 95% CI | folds | test-like | full-data tau | test flips | Decision |
|---|---|---|---|---|---|---|---|---|
| Gaussian-smoothed curve, sigma 0.005 | 0.83810 | +0.00016 | [-0.00004, +0.00035] | 4/5 | +0.00004 | 0.1525 | 42 | discard |
| sigma 0.01 | 0.83798 | +0.00004 | [-0.00018, +0.00025] | 3/5 | -0.00014 | 0.1535 | 57 | discard |
| sigma 0.02 | 0.83808 | +0.00013 | [-0.00016, +0.00043] | 3/5 | +0.00008 | 0.1565 | 122 | discard |
| sigma 0.04 | 0.83824 | +0.00030 | [-0.00002, +0.00059] | 3/5 | +0.00026 | 0.1565 | 122 | discard |
| median of 1,000 bootstrap argmaxes | 0.83802 | +0.00008 | [-0.00006, +0.00022] | 2/5 | -0.00004 | 0.1510 | 12 | discard |

The OOF curve is flat: F2 stays within 0.0005 of the best for tau 0.1415-0.1665, a band holding 426 test rows, so any of these cut-offs changes at most ~0.0001 LB.

## Round 16 (2026-11-12): micro-gain harvesting (`ce_r16_t1_members.py`, `cf_r16_eval.py`; logs `outputs/cf_r16_t1.log`, `cf_r16_t2t5.log`, `cf_r16_t5.log`, `cf_r16_t3.log`)

The brief asked to keep +0.0001 to +0.0004 variance-reduction gains even when the CI overlaps zero. Two rules are printed: strict (as before) and **relaxed = mean nested gain > 0 (and for Task 1 test-like gain >= 0)**. v14 would use the relaxed rule. TabM is within the rules: our own PyTorch BatchEnsemble MLP trained from scratch on training folds (not pre-trained, not AutoML). **Nothing passes even the relaxed rule, so no v14 was built (it would equal v13).**

**Task 1** (B0 = v13, nested 0.83794, test-like 0.85046). TabM and spline-LR retrained with test predictions (`outputs/r16_t1_members.npz`; OOF corr with round 12: 0.999 / 0.99999); 10-seed bag refitted on Train+Val (`by_r14_bag.py full t1`). Alone: TabM 0.81492, spline-LR 0.80825, 10-seed stack 0.83858; corr(stack, TabM) 0.907. Order fixed before the run: (1) brief recipe, (2) bag + smoothed cut-off, (3) bag + argmax.

| Variant | nested F2 | delta | 95% CI | folds | test-like delta | test flips vs v13 | strict | relaxed |
|---|---|---|---|---|---|---|---|---|
| (1) 0.88 bag + 0.08 TabM + 0.04 spline, smoothed cut | 0.83754 | -0.00041 | [-0.00095, +0.00019] | 2/5 | -0.00046 | 260 | discard | discard |
| (2) 10-seed bag, smoothed cut | 0.83831 | +0.00037 | [-0.00007, +0.00084] | 3/5 | -0.00041 | 138 | discard | discard |
| (3) 10-seed bag, argmax cut | 0.83833 | +0.00039 | [-0.00008, +0.00085] | 3/5 | -0.00038 | 151 | discard | discard |
| (4) brief blend, argmax cut | 0.83725 | -0.00069 | [-0.00128, -0.00013] | 2/5 | -0.00096 | 270 | discard | discard |
| (5) brief blend, bootstrap-median cut | 0.83720 | -0.00074 | [-0.00132, -0.00018] | 1/5 | -0.00105 | 260 | discard | discard |
| bag + 5% TabM, smoothed | 0.83826 | +0.00031 | [-0.00018, +0.00079] | 3/5 | -0.00034 | 166 | discard | discard |
| bag + 10% TabM, smoothed | 0.83790 | -0.00005 | [-0.00058, +0.00047] | 3/5 | -0.00040 | 234 | discard | discard |
| bag + 15% TabM, smoothed | 0.83737 | -0.00058 | [-0.00118, -0.00000] | 2/5 | -0.00131 | 326 | discard | discard |
| bag + 8% TabM + 8% spline, smoothed | 0.83679 | -0.00115 | [-0.00180, -0.00049] | 0/5 | -0.00173 | 350 | discard | discard |

The brief's blend lowers F2: mixing in members 0.02-0.03 weaker that are 0.85-0.91 correlated adds bias, not just variance reduction. The 10-seed bag's +0.0004 on ordinary OOF rows turns into -0.0004 on test-like rows (the test has far fewer missing cells), so its expected test effect is negative.

**Task 3** (B0 = v13, 0.794578). Forced 5/8/12% blends (gated, capped) of seven other families: LGB-L1 log1p, Tweedie, XGB-L1 raw, single-seed ratio[hist] LGB, plain hurdle, new CatBoost-MAE raw depth 6 (alone 0.78055), new ridge log1p stage 2 + mixture-median (non-GBDT, alone 0.77912): all -0.00001 to -0.00049, 0-2/5 folds. Cap: in-sample NMAE peaks at 300k (250k 0.794323 ... 300k 0.794578 ... 350k 0.794462); nested choice 290-300k, -0.000014. v13 test: 7 predictions >= 250k, 1 at the 300k cap.

**Task 5.** The 25 most extreme outside-block accounts by consensus: 19 carry the `impossible_coords` rule (the known date-line coordinate artefact), the rest are geo extremes; none is a pocket flag. Top 0.1% inside the block: geo_km_per_login_4 21/21, geo_hop_max 26/26, device_count_maxjump 100/100, device_count_dev4 31/31 (the generated anomaly types sit entirely in the block), login_count_maxjump 1/26 (a natural tail). F3: adding 20 pays +0.0039 if half are real, -0.0010 if none; break-even P = 0.20. No labelled way to verify, evidence points the other way -> no change.

**Task 2.** total_spending_thb and avg_monthly_spending by id-block class in test match train by true label (e.g. Whale median 45,396 vs 45,024); every class bottoms out near 500 THB, no true Whale has 0 spending, and flipped rows look like their block. Flip rate is flat across spending deciles (2-4%). No deterministic contradiction -> no change.

## Round 17 (2026-11-12): the TAs' data re-release (`scripts/reorder_task4_test_cache.py`, `cg_r17_new_data_check.py`)

The TAs announced a data fix: Task 1 and Task 3 unchanged, Task 2 and Task 5 new datasets, Task 4 `test.csv` corrected, schema unchanged. Sources: the Kaggle zip (`player-intelligence-system-cpe-342-ml`, downloaded by the user) for Tasks 1, 2, 3, 5 and the TAs' Drive zip (modified 2026-11-12 07:55 UTC, 625.5 MB) for Task 4.

**What changed (checked by sha256 against git HEAD).**
* Task 1, Task 3, all datacards and `sample_submission.csv`: byte-identical.
* Task 2 train/test and Task 5 test: fully regenerated. No row's numeric features match an old row; column means move by at most 0.02 SD; same shapes (104,000 / 25,889 / 25,889).
* **Task 2's player_id leak is gone.** Class shares are flat over all ten id deciles (Casual 0.391-0.399), and an id-only LightGBM gets OOF macro-F1 0.141 (old data: 0.968). Train and test ids do not overlap.
* **Task 4:** the 57,435 images are byte-identical (zip CRC), and the train labels are identical (now integer codes, columns `id, file_name, label`). `test.csv` lists the same 25,889 test images in a **new order**: ANS00001 = `sjqyfwc.jpg` (old row 22,721), and no row kept its position. Our old-order Task 4 column agrees with the correctly re-ordered one on **21.25%** of rows, which matches the ~0.21 the leaderboard implied for Task 4 on every upload from v1 to v12: the answer key was in the new order all along.

**What we ran.**
* Task 4: `scripts/reorder_task4_test_cache.py` moves every cached test prediction to the row that now holds its image and stores `test_file_name`. The notebook asserts it; `train_task4.py` now writes it. Fresh inference of the fold-0 ConvNeXt and DeiT checkpoints on 512 new-order rows: argmax agrees 100%, max |prob diff| 0.004 (bf16). The old CSVs are kept in `_release_2026-11-03/`. `src/task4.load_frames` reads both formats.
* Task 2: all `t2_*` caches deleted and retrained (tuned hyperparameters reused). OOF macro-F1: LightGBM 0.7559, XGBoost 0.7559, CatBoost 0.7545, LogReg 0.7435, stacker 0.7561, **+ class multipliers 0.7568** (old data 0.7549).
* Task 5: the unchanged label-free pipeline finds the same structure as before: 33 tail clusters → **6 families, 1,809 accounts (6.99%)**; the derived-feature gap scan adds 0; the blind pocket scan finds exactly one pocket again (accuracy_std × kda_std, z above the noise level of 10.9), with cut-off k = 339 (estimated 194 members) → **2,148 flags (8.30%)**, expected F3 0.990 if the families are true.
* Tasks 1 and 3 reuse their caches: their columns are identical to `submission_clean_v13.csv` (0 rows differ).

**Result.** `submission.csv` (sha256 53c386ca…) is the first file whose every column belongs to the current test sets. Nothing in it uses player_id. Every earlier file (v1-v13 and the clean files) has Task 4 in the wrong order and Task 2/5 predictions for test rows that no longer exist; do not upload them.

## Round 18 (2026-11-12): v14 scored 0.84873, the "two-front" brief (`ch_r18_t4_shift.py`, `cj_r18_t4_tta.py`, `ci_r18_t2_audit.py`; logs in `outputs/`)

**Where the gap is.** v14 = 0.84873 (top 0.85787). v12 (old data, T2/T5 by id block, T4 in the wrong order) = 0.76248 gives T1 + T3 ≈ 1.644 → T1 ≈ 0.85 on test (the test-like estimate was 0.8505). Then v14 leaves T4 + T5 ≈ 1.84. With T5 ≈ 0.98 as on the old data, **T4 ≈ 0.86 on test** (OOF 0.989), in line with round 11's label-free 0.87–0.91. All 10 fold models agree on only 74% of test images.

**Task 4, held-out-cluster validation (training images only).** K-means (20 clusters per class) on frozen DINOv2-B features, StratifiedGroupKFold(5) by cluster, fold 0 (6,354 held-out images in 20 clusters). A frozen DINOv2 probe drops from 0.911 (stratified) to 0.829 (held-out clusters), close to its estimated test accuracy, **but fine-tuned models hardly drop at all** (ConvNeXt 0.9836 vs 0.990 stratified). So this split does not reproduce what happens on the test images, and it can only rank recipes inside the training distribution:

| Recipe (fold 0, same rows, 10 epochs, h-flip TTA) | acc | macro-F1 | Δ acc vs ConvNeXt [95% CI] |
|---|---|---|---|
| ConvNeXt-T in12k (pipeline) | 0.9836 | 0.9805 | — |
| DeiT-III-S (pipeline) | 0.9811 | 0.9787 | −0.0025 [−0.0055, +0.0006] |
| ConvNeXt-T, class-balanced sampling | 0.9816 | 0.9783 | −0.0020 [−0.0046, +0.0005] |
| EVA02-S (MIM in22k) | 0.9762 | 0.9735 | −0.0074 [−0.0107, −0.0041] |
| EfficientNetV2-S in21k | 0.9806 | 0.9779 | −0.0030 [−0.0061, +0.0000] |
| ConvNeXt-T in22k | 0.9819 | 0.9792 | −0.0017 [−0.0044, +0.0008] |
| pipeline pair (ConvNeXt + DeiT) | 0.9852 | 0.9826 | |
| all six averaged | 0.9862 | 0.9837 | +0.0010 over the pair (~6 images) |

Decision rules on the same held-out set (ConvNeXt): logit adjustment logits − τ·log π_train: τ = 0 → 0.9805, 0.5 → 0.9787, 1 → 0.9757 (this split keeps the train class mix, so it cannot show a label-shift gain). Class multipliers fitted on the held-out set itself (an upper bound): 0.9843, but with unstable, extreme weights (Roblock 0.36 vs 0.21 for the class-balanced model) → not transferable. `task4/train.csv` carries no metadata (id, file_name, label only).

TTA / photometric restoration (pipeline fold-0 checkpoints on their own 6,310 validation images; dark or low-contrast subset = 833): every view set lies within ±0.0006 F1 of the pipeline's original + flip (ensemble 0.9878). Autocontrast lowers the hard subset (0.9799 → 0.9775), gamma for dark images changes nothing, ±10° rotations and zoom-out 0.85 move 0–5 images. The training augmentation already covers darkness, inversion, rotation and noise.

**Task 2 on the new data: 0.757 is the Bayes ceiling.**
* Stack 0.7561 → + multipliers 0.7568 in-sample, **0.7564 nested**. Members' P(true class) correlate 0.99 (LGB/XGB/Cat) and 0.94 (LogReg).
* Confusion: Casual F1 0.822, Grinder 0.746, Whale 0.748, Social 0.711 (recall 0.678). Most confused pairs: Casual↔Grinder (8,439 rows), Casual↔Social (5,737), Social↔Whale (4,042).
* The old feature design still fits: every engagement/spending indicator keeps its sign; the 3 `random_metric` columns are pure noise (|ρ| ≤ 0.003); class shares in the engagement × spending grid 0.394/0.202/0.250/0.154.
* Features are independent given the class (within-class |rank corr| mean 0.004–0.010, max 0.054). Under that independence a naive-Bayes rule on per-feature class histograms is the Bayes classifier: it scores **0.7549 (0.7559 with multipliers)**, the same as the GBDTs. Adding it to the stack: +0.0001 nested.
* Calibration is exact (accuracy 0.811 at mean p 0.811; 0.961 at p ≥ 0.95): the remaining errors are label noise. The brief's 0.775 is not reachable.

**Decision:** nothing passes; no v15. The Task 4 test shift is real but no train-only validation reproduces it, so no Task 4 change can be shown to help. Options left need a ruling: (1) prior correction toward the test class mix (1,095 labels move; TA question), (2) using unlabeled test images to characterise the shift for augmentation design (grey area; ask first).

## Round 19 (2026-11-12): forensic investigation of the re-released data (`ck_r19_forensics_tab.py`, `cl_r19_forensics_t4.py`; logs in `outputs/`)

Model training was paused for this (ConvNeXt-in22k fold 0 is cached; `train_task4.py` resumes from there). Diagnostics only; Task 4 opened training images and files only.

### 1. What the TAs changed
* **Task 2: a fresh sample from the same generator.** Within-class KS old vs new ≤ 0.016 (5% critical value ~0.014 at the smallest class; 120 tests), class-conditional means differ by ≤ 0.026 SD (sampling noise ~0.016), categorical value sets identical, total-variation ≤ 0.005. No new row lies near an old row (median nearest-neighbour distance new→old 2.97 vs new→new 2.88): the rows were re-drawn, not perturbed. `player_id`s were re-assigned at random (train + test still use 129,889 of P000000–P129999, but a different 111 are missing; rank corr(row, id) ≈ 0).
* **Task 5: same 25,889 ids, re-drawn rows, same generator** (per-column KS ≤ 0.017, NaN rate 0.0487 → 0.0488, invalid coordinates 207 → 218 rows, binary-flag rates within 0.003). Our flags are now flat over id deciles (7.7–8.7%), so the old "top 8% of ids" pattern is gone.
* **Task 4:** only the test row order (round 17).

### 2. Task 2: how the features are generated (new train)
* Continuous columns: 95–96k distinct values, printed with the shortest round-trip float (15–17 significant digits), so the number of decimal places is just the value's magnitude (rank corr with integer digits −0.37 to −0.42). Re-stacking the OOF probabilities with decimal-place and significant-digit features: log-loss 0.60490 → 0.60495, macro-F1 unchanged → **no information**. (Decimal places do differ by class, p = 1e-81, only because the classes differ in magnitude.)
* Count columns (login_streak, friend_count, tournament_entries, rare_items_count, …) are integers ≥ 0 with 0.4–1.8% at 0; bounded columns are clipped: team_play_percentage [5, 100], win_rate_ranked [25, 75], achievement/collection [10, 95], ranked_participation [0, 100] (4.7% / 4.8% at the ends), item_type_preference_* [0.05, 0.95]. Within each class most continuous columns are best fitted by a normal (or a gamma for the low-activity classes), i.e. clipped normal draws per class. random_metric_1/2 are uniform [0, 1000] / [0, 1], random_metric_3 an integer normal: pure noise, as before.
* NaN: ~7.7% per column, MCAR (NaN count vs label χ² p = 0.59; vs OOF error AUC 0.51). Row order carries nothing (p = 0.29).
* **Hidden sub-groups:** within-class top eigenvalues sit at the random-matrix bound for Casual (1.062 vs 1.058), Grinder (1.082 vs 1.074) and Social (1.088 vs 1.082), so those classes are single populations of independent draws. **Whale has one extra factor** (1.269 vs 1.094): vip_tier, avg_monthly_spending, spending_frequency, rare_items_count, total_spending, peak_concurrent_hours, friend_count load on it, i.e. a continuous within-Whale "intensity" level, not two archetypes. Naive Bayes (which ignores it) still equals the GBDTs, so it does not move the ceiling. (Gaussian-mixture BIC prefers 2–3 components in every class, but only because clipped and integer columns are not Gaussian; uninformative.)

### 3. Tasks 1, 3, 5: per-row precision artifacts
* Task 1: decimal places, zero count, NaN count, integer-valued count vs is_cheater: AUC 0.48–0.51. Nothing.
* Task 3: zero count vs spending > 0 has AUC 0.34 and Spearman −0.47 with log spending, but it comes from the binary columns (tournament_participation, owns_limited_edition, is_premium_member, vip_status, guild_membership, segment = 0) that the model already uses. The L1 member's residuals are flat in the zero count, and an L1 re-fit on [member prediction, zero count] is worse (NMAE 0.27341 → 0.27365). Nothing.
* Task 5: decimals vs our flags AUC ≤ 0.57, explained by magnitude (geo km); the share of short-precision cells differs by ≤ 0.03 between families, pocket and unflagged accounts. No generation artifact marks the anomalies.

### 4. Task 5 archetypes (new test)
Families: (1) 517 takeover / VPN teleport (password_changed + vpn_usage, location jump of hundreds of km at period 4, currency transfers and purchases up), (2) 310 skill boost + friend purge, (3) 265 near-inactive, (4) 263 silent (no communication), (5) 247 low behavior_stability, (6) 207 single-IP clockwork bots (low IP entropy, constant session lengths and login times); plus the pocket, 339 erratic-performance accounts. These cover the brief's bot farms (6), account theft (1, 2) and VPN regional exploiters (inside 1).
**No geo-only group is missed:** among the 23,672 unflagged accounts with valid coordinates, **none** has a maximum hop above the top-1% threshold of all accounts (4,385 km); the largest unflagged hop is 335 km and the tail has no gap (largest step 11 km vs a median step of 0.1 km). vpn_usage = 1 alone is common (3,919 accounts, 15%) and is not an anomaly by itself.

### 5. Task 4 (training images and files only)
* **No metadata signal.** File names: 7 characters, always `s` + 6 random base-36 characters, Cramér's V with the label ≤ 0.036 at every position. Row order random (23,578 label runs vs 23,606 expected). Every image is 230×120 RGB baseline JPEG with the same quantisation table (one quality setting), 4:2:0 chroma, only a JFIF header, and all were written within 68 s (2025-11-22 00:13:32–00:14:40) with no class pattern: one batch re-encode by a script.
* **Corruptions are independent random draws, already covered by our augmentation:** 42% of images rotated (black corners; 36–45% by class), noise σ continuous and roughly uniform up to ~0.1 (no discrete levels), brightness / contrast / saturation follow content (by class). Low contrast predicts our OOF errors best (AUC 0.65).
* **No twins:** with an inversion-invariant gradient descriptor only 4 images have a neighbour above 0.98 similarity and 122 above 0.95 (84% same label, i.e. mostly unrelated low-texture frames). Training images are not corrupted copies of each other, so the stratified OOF is not inflated by near-duplicates (OOF accuracy 0.980 on the 122 vs 0.990 elsewhere).
* **The model is mainly a HUD reader.** Fold-0 ConvNeXt on 1,500 of its validation images: full image 0.982; centre 3×3 of a 5×5 grid greyed (border/HUD only) **0.952**; border ring greyed (centre only) **0.919** (Free Fried 0.882, PUBE 0.848). Single-cell occlusion hurts most along the **bottom strip** (on-screen controls / skill bar; 7–21 ×10⁻³ drop of P(true class) per cell vs 0–8 in the centre), then the right edge. The test images it is unsure about (crowds, dark frames) look HUD-less, and centre-only accuracy (0.92) is about the size of the estimated test accuracy (0.86–0.91). This is a hypothesis, not a measurement on test.

### Training adjustments derived from this (ranked by expected leaderboard impact)
1. **Task 4: make the model use scene content, not only the HUD.** Train-time "HUD dropout" (grey out the bottom strip / border ring at random) and score it train-only on full images, centre-only and bottom-masked validation images. Up to the whole Task 4 test gap (~0.07 of Task 4, ~+0.014 LB) if the test failures are HUD-less frames; unknown otherwise. One fold-0 run (~25 min) tells whether full-image accuracy holds while centre-only rises.
2. **Task 4: 4-backbone ensemble** (ConvNeXt + DeiT + ConvNeXt-in22k + EffNetV2), +0.002 accuracy on held-out clusters (CI > 0); ideally trained with item 1.
3. **Task 4: plain argmax instead of the in-sample class multipliers** (nested −0.0002 for the multipliers; they move 548 test labels).
4. Tasks 1, 2, 3, 5: no artifact, formula or missed group found; nothing to change.

### Round 19b: HUD dropout, fold-0 test (`src/task4.GPUAugment._hud`, `train_task4.py --p-hud`, `cm_r19_t4_hud_eval.py`)
Recipe: with probability 0.5 per training image, one of four modes before the corruptions: bottom strip 15–25% hidden; top strip 12–18% + the four corners hidden; all borders hidden (central 60–70% kept); or zoom into the central 60–75% (no fill). Hidden areas get the image's mean colour or uniform noise. ConvNeXt-T, pipeline fold 0, otherwise identical to `t4_cnxt_fold0`. Scored on the 6,310 fold-0 validation images with h-flip TTA; grey = 0.5 (not the training fill).

| View | ConvNeXt | ConvNeXt-HUD | Δ acc [95% CI] | both averaged |
|---|---|---|---|---|
| full frame | 0.9878 (F1 0.9859) | 0.9872 (0.9852) | −0.0006 [−0.0030, +0.0017] | **0.9880** (0.9863) |
| bottom 20% greyed | 0.9811 | 0.9815 | +0.0003 [−0.0024, +0.0032] | **0.9838** |
| centre only (5×5 outer ring greyed) | 0.9336 (PUBE 0.890) | **0.9609** (PUBE 0.944) | **+0.0273 [+0.0219, +0.0326]** | 0.9578 |
| bottom 20% cut, resized | 0.9808 | 0.9789 | −0.0019 [−0.0048, +0.0011] | **0.9826** |
| central 70% zoom (no grey) | 0.9607 | **0.9683** | **+0.0076 [+0.0035, +0.0116]** | **0.9688** |

* Passes: full-frame accuracy is unchanged within noise, accuracy without the HUD rises clearly, and the natural zoom view (no grey area the HUD model might recognise) confirms it. Averaging a normal and a HUD-dropout model is best or near-best on every view → final Task 4 = existing ConvNeXt + DeiT **plus** HUD-dropout ConvNeXt, ConvNeXt-in22k, EffNetV2-S and DeiT-III-S.
* "Dual view" (0.75·P(view) + 0.25·P(70% zoom of the view)): +0.001–0.003 on centre-only but −0.0013 on full frames for the HUD model → not used.
* Pipeline (`scripts/nb_part2.py` §5.2/§7.4): the six tags above; class multipliers kept only if they also pass a nested check (round 18: −0.0002 nested).

### Round 19c: six-backbone Task 4 ensemble, all folds (`cn_r19_t4_hud_ensemble.py`, `co_r19_t4_views_allfolds.py`)
HUD-dropout versions (p = 0.5) of ConvNeXt-T, ConvNeXt-T in22k, EfficientNetV2-S in21k and DeiT-III-S, trained on the pipeline's stratified folds + a full-data fit each (`train_task4.py --p-hud 0.5`, 527 GPU minutes). Every fold model predicts only its own validation images (31,546 out of fold), h-flip TTA; occluded views as in round 19b (grey = 0.5).

| OOF accuracy / macro-F1 | full frame | bottom 20% greyed | centre only | central 70% zoom |
|---|---|---|---|---|
| ConvNeXt | 0.9886 / 0.9865 | 0.9809 | 0.9316 | 0.9578 |
| DeiT-III | 0.9862 / 0.9839 | 0.9789 | **0.7006** (PUBE 0.342) | 0.9622 |
| ConvNeXt-HUD | 0.9867 / 0.9845 | 0.9810 | 0.9602 | 0.9679 |
| ConvNeXt-in22k-HUD | 0.9856 / 0.9832 | 0.9810 | 0.9605 | 0.9695 |
| EffNetV2-S-HUD | 0.9841 / 0.9816 | 0.9775 | 0.9450 | 0.9651 |
| DeiT-III-HUD | 0.9840 / 0.9814 | 0.9780 | 0.9527 | 0.9671 |
| **pair (v14)** | 0.9899 / 0.9881 | 0.9847 | 0.9295 (PUBE 0.876) | 0.9669 / 0.9624 |
| **six** | **0.9900 / 0.9883** | **0.9857** | **0.9635** (PUBE 0.946) | **0.9744 / 0.9705** |

Six − pair, paired bootstrap over rows (95% CI) and per-fold accuracy:
* full frame: macro-F1 +0.0002 [−0.0006, +0.0010], accuracy +0.0001 [−0.0005, +0.0008]; folds +0.0008 +0.0010 −0.0006 +0.0000 −0.0005 → **no loss**.
* bottom greyed: accuracy +0.0010 [+0.0002, +0.0019], 5/5 folds.
* centre only: macro-F1 +0.0326 [+0.0303, +0.0350], 5/5 folds (+0.027 to +0.044).
* **70% zoom (no grey area): macro-F1 +0.0080 [+0.0068, +0.0094], 5/5 folds (+0.006 to +0.009).**

* Each HUD single model loses 0.002–0.005 OOF on full frames against its normal counterpart, but the ensemble does not: averaging normal and HUD-dropout models keeps the full-frame score and inherits the robustness.
* The normal DeiT-III collapses when the border is grey (0.70; PUBE 0.34): the ViT treats the grey ring as out-of-distribution. Its HUD-dropout version holds 0.95.
* Class multipliers on the six-model blend: in-sample 0.9888, nested 0.9880 (−0.0003 vs argmax) → dropped, plain argmax.
* Test labels vs v14: 548 change from dropping the v14 multipliers, 701 from the new backbones, 919 in total (3.5%). Predicted shares move most for Free Fried (0.136 → 0.150) and PUBE (0.269 → 0.253). Label-free diagnostic only (not used to choose): mean max-probability on test 0.806 (pair) → 0.790 (six), as expected from averaging more diverse models.
* Decision: the six-backbone argmax ensemble goes into the pipeline (it was pre-registered in round 19b; the all-fold views confirm it). Whether the test gap is HUD-less frames remains a hypothesis: the gain on test is somewhere between 0 and the zoom/centre gains above.

**v15 uploaded 2026-11-13: public LB 0.85032 (+0.00159 over v14).** Only the Task 4 column changed, so Task 4 rose by about +0.008 (if the LB is the mean of the five task scores). If the 548 rows moved by dropping the multipliers are net zero, the 701 rows moved by the new backbones gained about 207 net correct labels, +0.30 per changed row: between the out-of-fold rates on full frames (+0.03) and on the 70% zoom (+0.57), close to an even mix of the two. This supports the HUD hypothesis but is one noisy public-split measurement. Gap to the top (0.85787): 0.00755 ≈ +0.038 on one task.

## Round 20 (2026-11-13): Task 4 blend rules, multi-crop inference, bigger / stronger HUD backbones (`cp_r20_t4_z75_views.py`, `cq_r20_t4_blend.py`, `cr_r20_t4_candidates.py`)
Objective fixed in advance: mean of out-of-fold macro-F1 on full frames and on the 70% centre zoom (the HUD-free view without grey area; v15's LB gain fits an even mix of the two). All 31,546 training images, each predicted by the fold model that did not train on it; rule parameters chosen on four folds and scored on the fifth.

| Rule (nested) | full | 70% zoom | centre only | objective vs v15 [95% CI], folds better |
|---|---|---|---|---|
| v15: mean of six | 0.9883 | 0.9705 | 0.9587 | — |
| group weight on the four HUD models (picked 0.6–0.75) | −0.0003 | −0.0004 | −0.0002 | −0.0004 [−0.0006, −0.0001], 0/5 |
| entropy gate (more HUD weight where the legacy pair's max-prob < τ) | −0.0003 | −0.0004 | −0.0002 | −0.0004 [−0.0007, −0.0000], 3/5 |
| six free model weights (Powell) | −0.0005 | +0.0002 | +0.0012 | −0.0001 [−0.0005, +0.0003], 3/5 |
| multi-crop: (1−b)·P(view) + b·P(central 75% zoom), with group weight | −0.0012 | −0.0001 | +0.0010 | −0.0006 [−0.0011, −0.0002], 0/5 |
| multi-crop with the gate | −0.0007 | −0.0003 | +0.0003 | −0.0005 [−0.0009, −0.0001], 1/5 |
| multi-crop with free weights | −0.0013 | −0.0020 | +0.0007 | −0.0016 [−0.0021, −0.0011], 0/5 |

* In-sample, more weight on the HUD models trades full frames for HUD-free views almost one for one (w = 0.9: full −0.0010, zoom +0.0006); v15's equal mean (w = 2/3) is already on the plateau. A second, zoomed view costs full-frame accuracy from b = 0.1 on (0.9883 → 0.9879) and helps only the centre-only view: the HUD models already saw centre zooms in training. **Nothing beats the plain mean.**
* PUBE Mobile ↔ Free Fried is 15–19% of the errors in every view (zoom 136 of 809; centre-only 221 of 1,152); the lowest recalls without HUD are Roblock (0.953 zoom) and FiveN (0.960). A perfect PUBE/FF specialist would add at most ~0.4% on the zoom view, so none was trained.
* Fold-0 candidates as a seventh member (pass rule fixed in advance: objective CI > 0 and full ≥ −0.0005):

| fold 0, macro-F1 | full | 70% zoom | centre only | objective |
|---|---|---|---|---|
| ConvNeXt-T HUD p=0.5 (cnxt_hud) | 0.9852 | 0.9632 | 0.9538 | 0.9742 |
| ConvNeXt-S in12k HUD p=0.5 (cnxtS_hud) | 0.9872 | 0.9675 | 0.9593 | 0.9774 |
| ConvNeXt-T HUD p=0.8 (cnxt_hud8) | 0.9832 | 0.9646 | 0.9565 | 0.9739 |
| v15 six | 0.9887 | 0.9700 | 0.9590 | 0.9793 |
| six + cnxtS_hud | 0.9892 | 0.9709 | 0.9607 | 0.9801 |
| six + cnxt_hud8 | 0.9885 | 0.9706 | 0.9615 | 0.9795 |

  Seven − six: cnxtS_hud objective +0.0008 [−0.0002, +0.0018], full +0.0005 [−0.0001, +0.0013] → **fail** (CI includes 0, though positive on every view); cnxt_hud8 +0.0002 [−0.0008, +0.0012] → fail. Stronger HUD dropout trades full-frame accuracy for little. ConvNeXt-S is the better single model on every view, but at batch 64 it overflows the 8 GB GPU into shared memory (~990 s per epoch vs ~120 s for Tiny): 5 folds + full would take ~16 h as is.
* **No v16:** no rule or candidate passed, so the pipeline stays at v15. With Task 4 at ~0.87 on test against 0.97 on the HUD-free proxy, most of the remaining test gap is content these train-only proxies cannot simulate; the open TA questions (prior alignment, test-image-informed augmentation) are the remaining large levers.

## Round 21 (2026-11-13): Task 2 generative models, Task 5 tail (`cs_r21_t2_generative.py`, `ct_r21_t5_tail.py`)
**Task 2.** Generative classifiers fitted per fold on the round-19 data-generating process (pipeline stack: nested 0.7564):

| model | alone, nested multipliers | as a 5th stack member, nested Δ [95% CI], folds |
|---|---|---|
| histogram naive Bayes (round 18) | 0.7553 | −0.0000 [−0.0007, +0.0007], 3/5 |
| censored-normal (Tobit) naive Bayes | 0.7518 | −0.0000 [−0.0007, +0.0008], 3/5 |
| KDE marginals + clip-bound masses | 0.7572 | −0.0002 [−0.0009, +0.0005], 1/5 |
| KDE + Gaussian copula for Whale | 0.7564 | +0.0003 [−0.0005, +0.0010], 3/5 |
| KDE + copula for every class | 0.7557 | +0.0001 [−0.0005, +0.0008], 4/5 |

* The normal assumption costs (Tobit 0.7518; sklearn GaussianNB with median imputation 0.604): several marginals are skewed (gamma-like for the low-activity classes). The Whale copula does not move the ceiling.
* Not pre-registered: 0.5 stack + 0.5 KDE naive Bayes, nested 0.7575 (+0.0011 [+0.0002, +0.0020], 5/5); with the blend weight also chosen nested (0.5-0.75) +0.0011 [+0.0000, +0.0021], 5/5; histogram / Tobit blends do nothing. **But** the rule-strict version (density grid on training values only instead of the train + test range; 99.99% identical predictions) gives +0.0008 [−0.0001, +0.0017], 4/5. A change touching 0.01% of rows moves the gain by 0.0003, so it is noise-level (about +0.0002 LB at best). Task 2 stays as it is (user rule: lock unless clearly past 0.7570).

**Task 5.** v15 flags 2,148 (six families 1,809 + pocket 339).
* Detector scale: the pipeline's robust-z cross-check fusion is 92% Mahalanobis (MCD); the final flags do not use it. Rank-mean / robust-z mean / soft-OR fusions recover only 52-58% of the flags in their top 2,148 (Jaccard 0.51-0.87 between fusions).
* A "gap" of 469 unflagged accounts under the MCD-dominated fusion is an artefact: they are exactly the unflagged accounts with `mass_item_sale = 1` (rare binary column, near-zero variance in the MCD core), otherwise normal (median max |z| 1.12 vs 1.01). Rank fusion has no gap (lambda <= 8.9; families need >= 50).
* F3 arithmetic: if all 2,148 flags were true and the true rate were 10% / 12% / 14%, F3 would be 0.844 / 0.713 / 0.618; the public score leaves T4 + T5 of about 1.85, so a 12-14% true rate is incompatible with it.

## Round 22 (2026-11-13): plan and code (runs queued behind ConvNeXt-Small)
* `GPUAugment(p_grid=...)`: GridMask or one large block hiding 65-85% of the frame (unit-checked: 61-84% hidden; defaults bit-identical). A low-light augmentation was written and removed again before any run (section 7.4: test confidence is lower in every brightness / contrast bin; round 18 gamma views gave nothing). Mosaic not added (CutMix/Mixup p = 0.8 already).
* `task4.create_model`: backbones in `NATIVE_NORM` (SigLIP / CLIP / EVA02-CLIP) get a forward pre-hook mapping ImageNet-normalised input to their own mean/std (SigLIP: 0.5/0.5; checked: pixel 0.5 reaches the patch embedding as 0.0). Existing members are unchanged.
* `train_task4.py`: `--p-grid`, `--layer-decay`, `--grad-ckpt` (ConvNeXt-Small at batch 64: peak 6.9 -> 2.1 GB, 574 ms/step; the 2.6 GB image array on the GPU had pushed it into shared memory, ~990 s/epoch).
* Queue: seven-vs-six all-fold check (`cw_r21_t4_seven.py`), fold 0 of `cnxt_grid` (ConvNeXt-T, HUD 0.5 + GridMask 0.2) and `siglip2_hud` (SigLIP 2 ViT-B/16, lr 2e-5, layer decay 0.75, HUD 0.5), `cv_r22_t4_eval.py` (pass rule: objective CI > 0, full >= −0.0005, real dark / low-contrast frames not lower); SigLIP folds 1-4 + full only if it passes, otherwise the held-out-cluster runs (`ch_r18_t4_shift.py cnxt_hud cnxt_grid siglip2_ld`).
* **ConvNeXt-Small (cnxtS_hud) over all five folds** (`cw_r21_t4_seven.py`; trained with `--grad-ckpt`, ~45 min per fold, full refit 45 min). Out-of-fold macro-F1 on 31,546 images:

| | full | bottom grey | centre grey | zoom 70% |
|---|---|---|---|---|
| cnxtS_hud alone | 0.9861 | 0.9812 | 0.9587 | 0.9667 |
| cnxt_hud alone | 0.9845 | 0.9774 | 0.9542 | 0.9635 |
| six (v15) | 0.9883 | 0.9831 | 0.9587 | 0.9705 |
| seven (+ cnxtS_hud) | 0.9886 | 0.9842 | 0.9608 | 0.9715 |

  Seven − six: objective +0.0007 [+0.0002, +0.0011], better in 5/5 folds (+0.0008 / +0.0009 / +0.0007 / +0.0003 / +0.0008); full frames +0.0003 [−0.0001, +0.0008]. Pre-registered rule (mean objective gain > 0, full >= −0.0005) -> **include**: v16 = v15 with cnxtS_hud as the seventh Task 4 member (plain mean, unchanged decision rule). The fold-0 estimate of round 20 (+0.0008, CI through 0) was the same size; five folds narrow the interval.

## Round 23 (2026-11-13): audit after dropping to 3rd (`cx_r23_t2_audit.py`, `cy_r23_t4_prior_temp.py`, `da_r23_t5_extend.py`, `db_r23_t5_flag_pockets.py`)
**Task 2** (stack nested 0.7564; OOF confusion: Casual-Grinder 8,439 errors, Social recall 0.678 / F1 0.711; accuracy 0.7702, weighted-F1 0.7688).
* Non-linear within-class dependence: binned mutual information of all 325 continuous pairs per class vs column-shuffled data: Casual max 0.0064 (noise 0.0065), Grinder 0.0100 (0.0100), Social 0.0124 (0.0128) -> independent given the class, not just uncorrelated.
* Temperature-scaled KDE naive Bayes: log-loss 0.6250 -> 0.6202 (stack 0.6047); same macro-F1.
* LightGBM + the brief's ratios and row-wise group z-summaries: 0.7559 -> 0.7547 alone; in the stack −0.0002 / −0.0005. MLP (3 seeds) 0.7494 alone, +0.0000 in the stack; Nystroem-RBF 0.7331, −0.0005; KNN 0.62-0.66, −0.0006 to −0.0000. Multipliers with a (w − 1)^2 penalty (lambda 0.05-0.2): −0.0003 to −0.0005. **Nothing passes.**
* Nominal columns on the new data: Cramer's V <= 0.008, p > 0.1 (all seven). player_id: all five tasks draw from one universe P000000-P129999 (80% of T2 train ids also in T1/T3 train), but the same id is not the same player: same-named columns correlate 0.00 across tasks, other tasks' columns eta^2 <= 0.0007 with the segment. No duplicates. Predicted class mix on test = on OOF (no prior shift).
* A competitor's "+0.01 over naive Bayes": our accuracy / weighted-F1 already sit +0.013 above our macro-F1, and single folds range 0.7536-0.7593, so a metric or hold-out difference explains it; GaussianNB (0.604) would not.

**Task 4.** (a) Prior: macro-F1 on OOF as is vs re-weighted to a balanced 20% class mix (rows weighted 0.2 / pi):

| rule | training mix: full / zoom70 | balanced mix: full / zoom70 | test labels changed |
|---|---|---|---|
| argmax (v15) | **0.9883** / **0.9705** | 0.9869 / 0.9678 | - |
| p * pi^-0.5 | 0.9876 / 0.9694 | 0.9879 / 0.9697 | 631 |
| p * pi^-0.75 | 0.9870 / 0.9687 | **0.9882** / **0.9705** | 917 |
| p * pi^-1 | 0.9859 / 0.9661 | 0.9878 / 0.9701 | 1,172 |

A bet on the test class mix: it wins as much under a balanced test as it loses under the training mix. The exponent uses no test data, but the reason to switch (predicted test mix: 17% 7-Eleven vs 39% in training) is the grey area of the unsent TA question -> not applied without the user / TA.
(b) Calibration: all backbones are similarly under-confident (temperature 0.52-0.55; EfficientNetV2 0.43); averaging calibrated probabilities (0.9880 / 0.9703) or log-probabilities (0.9882 / 0.9704) does not beat the plain mean (0.9883 / 0.9705).

**Tasks 1, 3, 5 sanity.** T1: mean P(cheater) 0.349 OOF and test; flag rate 64.3% OOF vs 60.7% test because test rows miss fewer values (8.0% vs 12.0% NaN), which round 11 showed raises F2. T3: test mean 10,462 vs training target 10,497, zero share 0.50 vs 0.48 (median gate). T5: unique ids, 0/1. Row order: T1/T2/T3/T5 map by position; a scrambled column would show on the LB (e.g. T2 near 0.25).

**Task 5 extension to 10-12%** (a competitor flags more than 8.3%). Label-free; F3 needs more than ~10% anomalies among the added accounts.
* Pocket continuation: its permutation null estimates 7 members among the next 440 (1.5%) -> expected F3 0.990 -> 0.972 (10%), 0.948 (12%).
* Ensemble (rank mean) additions: high flag rates only because the ensemble reads the flags (circular; v4 on the first release: +474 such alerts lowered the LB by 0.0023, i.e. about 4% anomalies).
* Family edges, pooled: password_changed 0.066 vs 0.033 looked like 16-22% members; per family it vanishes: edges of the flag-heavy families (takeover `location_changes_4`, `geo_hop_max`, `currency_transfers_4`, `play_pattern_consistency_4`; family password rates 0.59-1.00) have normal flag rates (estimated share −0.05 to +0.03).
* Flag x continuous-feature pockets (the pipeline's pair scan skips binary columns): no cell above the shuffled-flag noise level (max z 3.0-4.1 vs 4.2-4.5).
* **Keep the 2,148 flags.** No extension has evidence above the F3 bar.

## Round 24 (2026-11-13): "Task 4 should be 0.99" (`dc_r24_t4_test_agreement.py`)
A competitor says other teams get 0.99 on Task 4. Kaggle shows only the combined score, so a per-task test number is either a validation score or a decomposition of the total.
* Our validation: out-of-fold macro-F1 0.9886 (seven members, 31,546 images), i.e. 0.99.
* Test, label-free: the seven members disagree pairwise on 9.4% of test images (4.9-10.8%) vs 1.0-1.6% out of fold; all seven agree on 79.8% (OOF 97.0%); ensemble mean max-probability 0.791 (OOF 0.889). Since disagreement(A, B) <= err(A) + err(B), the average member errs on >= ~4.7% of test images.
* Own uploads: v14 -> v15 changed 919 Task 4 labels and nothing else (T1/T2/T3/T5 0 rows), +0.00159 total = +0.0080 on Task 4. Every changed row is an error in v14 or in v15, so err(v14) + err(v15) >= 919; with err(v14) − err(v15) ≈ 207 (accuracy scale) this gives v15 Task 4 <= ~0.986 even if no change was wrong-to-wrong.
* LB arithmetic (round 18): T4 + T5 ≈ 1.85; T5 ≈ 0.98 (the same label-free pipeline scored 0.983 against the confirmed answer of the first release) -> T4 ≈ 0.87, matching ATC / DoC / AGL 0.87-0.91 (round 11).
* No preprocessing bug: every train and test image is 230 x 120 RGB (headers read), same loader.
* The competitor confirmed the 0.99 is a validation score, the same as our 0.9886: no information about the test images, and no change. The test estimate for Task 4 stays ~0.87 (0.87-0.91 label-free).

## Round 25 (2026-11-13): v16 scored 0.85035; the "Generation-2" brief, first batch (`dd_`, `de_`, `df_`, `dg_`, `dh_r25_*.py`)
**v16 = 0.85035** (v15 0.85032): +0.00003, i.e. about +0.00015 on Task 4 for 275 changed labels (about +4 net correct). Inside the leaderboard's noise; the out-of-fold gain was +0.0003 on full frames and +0.0010 on the 70% zoom.

New acceptance rule from the user for this round: keep a change if its mean out-of-fold gain is positive and it is better in >= 3/5 folds; the 95% CI is reported but is no longer a gate. Every tuned quantity is still chosen nested (4 folds -> 5th), so the reported gains carry no selection bias.

| Task | Change | Nested result vs pipeline | Folds | Verdict (new rule) |
|---|---|---|---|---|
| 2 | 4 x 4 decision matrix M = diag(w) + Delta, coordinate ascent on macro-F1 − lambda ‖Delta‖², lambda 0-0.1 and lambda chosen by an inner split | −0.0001 to +0.0000 (0.7564 -> 0.7564-0.7565), CI [−0.0003, +0.0003] | 2/5 | no: the per-class multipliers already are the macro-F1 plug-in rule; off-diagonal terms move 0.03-0.3% of rows |
| 3 | smooth gate blend x sigmoid((p − tau) / T) x scale | picks tau 0.50, T -> 0, scale 1.000 in every fold = the pipeline's hard median gate | - | no change possible: the hard gate at p = 0.5 is the optimum of the family (45% of rows have 0.3 < p < 0.7 but carry 10% of the error) |
| 4 | stacking head (L2 logistic on 7 x 5 log-probabilities, + entropies + image statistics; trained on full frames or full + zoom) | best of 12 settings: objective +0.0001 [−0.0004, +0.0007], full −0.0004; C = 1: centre-grey −0.007 to −0.013 | 3/5 (best) | no: +0.0001 is the best of 12, and weakly regularised heads lose on views they did not see |
| 5 | Bayes boundary of the six families: flag if P(family | x) > 0.10 (2-component Gaussian mixture per family; and Gaussian family vs kernel-density bulk per gap column) | 0 unflagged accounts above 0.10 (mixture); 1 account at 0.16 (kernel bulk); largest posterior elsewhere 0.02-0.06 | - | no: the gaps are empty, there are no boundary accounts; F3 change +0.0000 |

**Task 4, prior exponent under simulated label shift** (seven members, 200 subsamples of the OOF rows; mean macro-F1 full / zoom70 / centre-grey):

| evaluation set | alpha 0 (v16) | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| training mix (39% 7-Eleven) | **0.9887** / **0.9716** / 0.9609 | 0.9884 / 0.9714 / **0.9621** | 0.9880 / 0.9706 / 0.9619 | 0.9874 / 0.9695 / 0.9611 | 0.9867 / 0.9677 / 0.9587 |
| 7-Eleven subsampled to 20% | 0.9875 / 0.9690 / 0.9567 | 0.9878 / 0.9696 / 0.9595 | 0.9879 / 0.9700 / 0.9605 | **0.9880** / **0.9703** / **0.9613** | 0.9878 / 0.9697 / 0.9606 |
| balanced | 0.9873 / 0.9690 / 0.9563 | 0.9878 / 0.9699 / 0.9596 | 0.9881 / 0.9708 / 0.9610 | **0.9885** / **0.9714** / **0.9623** | 0.9885 / 0.9713 / 0.9623 |

alpha = 0.75 is the best exponent if the test set has about 20% 7-Eleven Knights (better than alpha 0 in 98-100% of the draws) and costs 0.0013-0.0021 if the test mix equals training. On HUD-less frames (centre-grey) alpha 0.25-0.5 gains even at the training mix: degraded frames fall back towards the largest class. `submission_v16_prior_alpha075.csv` (949 Task 4 labels differ from v16; not committed) is ready, to be used only if the TAs allow prior alignment; it is one file to choose, not a second upload to compare.

**v17 = 0.85507: v16 with argmax p * pi^-0.75 for Task 4** (uploaded by the user as `submission_v16_prior_alpha075.csv`; now the pipeline rule, `nb_part2.py` section 7.4).
* +0.00472 over v16 with only 949 Task 4 labels changed = about +0.024 on Task 4, i.e. about +610 net correct among 949 changes: at least ~780 of the changes fixed an error (if none went wrong-to-wrong), so v16 had >= ~780 Task 4 errors and its ceiling was ~0.970; with T5 ≈ 0.98 the estimate moves from ≈ 0.87 to ≈ 0.89-0.90.
* The simulated balanced-mix gain was +0.0005 (full) to +0.005 (HUD hidden); the test gain is five times the largest of them. Test frames are far more ambiguous than OOF frames (ensemble mean max-probability 0.79 vs 0.89), and ambiguous frames fell back to the 39% class.
* Process note for the report: alpha = 0.75 was fixed from training images before the upload (rounds 23 and 25), but v16 and the alpha file were both scored on the leaderboard, so keeping the rule is a leaderboard-informed choice. The TA question on prior alignment is still open; nothing estimated from test predictions (EM, BBSE) is used.

**Round 25, Task 1: peer-group residual features** (`dh_r25_t1_peer.py`; expected value of six performance columns for a legitimate player with the same profile, regressors fitted on legitimate training-fold rows only). LightGBM + 12 residual features alone 0.8354 vs 0.8362 (OOF correlation 0.9962). In the stack (nested cut, pipeline 0.8379): as a 10th member −0.0000 [−0.0004, +0.0004], 4/5 folds (+0.0002 +0.0002 −0.0010 +0.0005 +0.0001); replacing LightGBM −0.0002, 2/5. Mean gain not positive -> not kept under the round 25 rule.
**Round 25, Task 3 new members** (`di_r25_t3_members.py`, blend check still running when this was written): MAE of log1p(y) on spenders: pipeline ratio[hist] LGB tuned 0.1360 (10-seed bag 0.1358), DART 0.1375, Langevin CatBoost 0.1354, peer-ratio features 0.1359.

**Round 22 results (fold 0, `cv_r22_t4_eval.py`): both candidates fail.** Macro-F1 on the 6,310 fold-0 validation images, objective = mean of full and zoom70:

| model | full | zoom70 | centre grey | grid 75% hidden | real dark / low-contrast (833) | objective |
|---|---|---|---|---|---|---|
| cnxt_hud (reference) | 0.9852 | 0.9632 | 0.9538 | 0.7816 | 0.9839 | 0.9742 |
| cnxt_grid (HUD 0.5 + GridMask 0.2) | 0.9811 | 0.9568 | 0.9539 | 0.8918 | 0.9684 | 0.9689 |
| siglip2_hud (SigLIP 2 ViT-B/16, lr 2e-5, layer decay 0.75) | 0.9363 | 0.9084 | 0.8778 | 0.4592 | 0.8853 | 0.9223 |
| seven-member ensemble | 0.9892 | 0.9709 | 0.9607 | 0.7975 | 0.9825 | 0.9801 |
| + cnxt_grid | 0.9885 | 0.9698 | 0.9617 | 0.8451 | 0.9807 | 0.9791 (−0.0010 [−0.0018, −0.0002]) |
| + siglip2_hud | 0.9880 | 0.9699 | 0.9596 | 0.7748 | 0.9828 | 0.9790 (−0.0011 [−0.0020, −0.0003]) |

GridMask buys robustness only to grid occlusion and pays on every natural view. SigLIP 2 at 160 x 320 with this recipe underfits (0.936 alone, 5 points below the CNNs) and is the least sure on the test frames the ensemble is unsure about (agrees with it on 49% of them). No SigLIP folds 1-4; the queue went on to the held-out-cluster runs.

**Round 25, Task 4: member combination under the balanced-prior rule** (`dj_r25_t4_rule_variants.py`, balanced-mix macro-F1, mean of full / zoom70 / centre-grey): mean of probabilities x pi^-alpha (v17) 0.9733 / **0.9741** / 0.9740 at alpha 0.5 / 0.75 / 1; mean of log-probabilities 0.9729 / 0.9737 / 0.9741; each member adjusted and renormalised first 0.9730 / 0.9737 / 0.9742. A plateau of ±0.0005 from alpha 0.75 to 1; the v17 rule stays.

**Round 25, Task 3 new members, final** (`di_r25_t3_members.py`, nested blend, pipeline 0.79458): + DART +0.00000 (blend weight 0.000); + Langevin CatBoost −0.00001 [−0.00005, +0.00002], 2/5 (weight 0.099); + peer-ratio features −0.00002, 0/5 (weight 0.071). None kept.

## Round 26 (2026-11-13): v17 = 0.85507 (top 0.85787); class-prior checks on Tasks 1, 2, 4 (`dk_r26_prior_shift.py`), Swin, Task 3 blend space
**Is there a prior shift in Tasks 1 and 2, as in Task 4? No.** Label-free diagnostics (EM = Saerens et al. 2002; its error on held-out folds in brackets):

| | training shares | mean probability, OOF | mean probability, test | EM estimate of the test shares |
|---|---|---|---|---|
| Task 1 cheater | 0.3493 | 0.3494 | 0.3485 | 0.3477 (±0.004) |
| Task 2 Casual / Grinder / Social / Whale | 0.394 / 0.250 / 0.202 / 0.154 | 0.394 / 0.250 / 0.202 / 0.154 | 0.396 / 0.251 / 0.200 / 0.153 | 0.397 / 0.252 / 0.198 / 0.153 (±0.005) |
| Task 4 (five classes) | 0.394 / 0.140 / 0.177 / 0.188 / 0.101 | 0.376 / 0.145 / 0.180 / 0.190 / 0.110 | 0.183 / 0.229 / 0.159 / 0.257 / 0.172 | calibrated 0.153 / 0.261 / 0.143 / 0.261 / 0.182 (±0.003 in-distribution, 0.04 on HUD-hidden frames); raw 0.083 / 0.290 / 0.122 / 0.282 / 0.223 (±0.025) |

* Tasks 1 and 2: the test sets have the training class mix to within the estimators' error. Task 2 at a balanced 25% mix would gain from prior adjustment (argmax 0.7485, nested multipliers 0.7511, p·π^−0.75 or π^−1 0.7568), but at the mix the test set actually has, the same rules lose (0.7564 -> 0.7550 / 0.7517). The pipeline's multipliers are already the class-weighting for the real mix. No change. (Whale is 15.4% of this year's data, not the 5% of the slide.)
* Task 4: the shift is specific to the images. EM is run as a diagnostic only and is not used: it assumes the frames of each class look the same in train and test, which is false here, and its answer depends on calibration (7-Eleven 8% raw vs 15% calibrated).

**Task 4 exponent** (seven members, OOF; uniform mix = rows weighted 0.2 / π):

| alpha | training mix: full / zoom70 / HUD hidden | uniform mix: full / zoom70 / HUD hidden (mean) | predicted test shares |
|---|---|---|---|
| 0 | 0.9886 / 0.9715 / 0.9608 | 0.9872 / 0.9689 / 0.9562 (0.9708) | 0.170 / 0.250 / 0.151 / 0.253 / 0.175 |
| 0.5 | 0.9879 / 0.9705 / 0.9618 | 0.9881 / 0.9708 / 0.9610 (0.9733) | 0.155 / 0.258 / 0.151 / 0.249 / 0.187 |
| 0.75 (v17) | 0.9873 / 0.9694 / 0.9610 | 0.9885 / 0.9714 / 0.9623 (**0.9741**) | 0.148 / 0.261 / 0.150 / 0.248 / 0.193 |
| 0.85 | 0.9871 / 0.9688 / 0.9601 | 0.9886 / 0.9716 / 0.9623 (**0.9741**) | 0.145 / 0.262 / 0.150 / 0.247 / 0.195 |
| 1.0 | 0.9866 / 0.9675 / 0.9586 | 0.9884 / 0.9713 / 0.9623 (0.9740) | 0.142 / 0.263 / 0.150 / 0.246 / 0.199 |
| 1.15 | 0.9860 / 0.9658 / 0.9565 | 0.9882 / 0.9710 / 0.9617 (0.9737) | 0.139 / 0.264 / 0.150 / 0.245 / 0.202 |

alpha = 1 is the Bayes exponent only for calibrated posteriors. The ensemble is under-confident (label smoothing, Mixup/CutMix: temperature 0.46 on OOF), so its calibrated posterior is p^(1/T) and dividing that by π equals argmax p·π^−T: the exact exponent for in-distribution frames is about 0.46, and frames that are ambiguous for other reasons push the optimum up. On the uniform simulation 0.75-1.0 is a plateau (0.9741 / 0.9741 / 0.9740) and 1.15 is past it. No training-only evidence prefers 1.0 over 0.75; alpha stays 0.75 and is not tuned on the leaderboard.

**Round 26, Task 3: the space predictions are combined in** (`dl_r26_t3_blendspace.py`, nested, pipeline 0.79458 = seeds averaged in log space, members L1-blended in THB). Seeds averaged in THB instead: +0.00001 [+0.00000, +0.00001], 4/5 folds (seed-to-seed SD of mu is small, so the arithmetic and geometric means nearly coincide). Members combined as a weighted geometric mean −0.00002; weighted median −0.00007 (0/5); unweighted mean −0.00104, median −0.00056, geometric mean −0.00117. The fitted L1 blend in THB is the right combination; the +0.00001 is two millionths of the leaderboard score and is not worth a pipeline change on its own.

**Round 26, Task 4: Swin-Tiny as an eighth member, fold 0** (`scripts/train_task4.py --tag swinT_hud --model swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --h 224 --w 448 --lr 1e-4 --p-hud 0.5 --grad-ckpt --folds 0`; `cv_r22_t4_eval.py swinT_hud`; about 20 min per epoch on the 8 GB card). Hierarchical window attention at a higher resolution, the one architecture family not yet in the ensemble.

| fold 0 (6,310 images) | full | zoom70 | HUD hidden | real dark / low contrast | objective |
|---|---|---|---|---|---|
| cnxt_hud alone | 0.9852 | 0.9632 | 0.9538 | 0.9839 | 0.9742 |
| swinT_hud alone | 0.9814 | 0.9594 | 0.9461 | 0.9674 | 0.9704 |
| seven members | 0.9892 | 0.9709 | 0.9607 | 0.9825 | 0.9801 |
| seven + swinT_hud | 0.9889 | 0.9715 | 0.9622 | 0.9807 | 0.9802 |

Adding it: objective +0.0001 [−0.0009, +0.0011] with argmax, full frames −0.0003, real dark frames −0.0019. Under the v17 rule (alpha 0.75) the objective is −0.0002 [−0.0014, +0.0008] at the training mix and −0.0003 [−0.0012, +0.0007] at a uniform mix; 0.2% / 0.3% of labels change. Not a gain under the rule the pipeline uses, so folds 1-4 are not trained (about 17 GPU-hours) and the ensemble stays at seven members. Its only clear advantage is the synthetic 75% grid occlusion (0.870 vs 0.782), which no real view reproduces. With round 22 (GridMask, SigLIP 2) this is the third backbone in a row that adds nothing on fold 0: the seven-member average is saturated on training-distribution frames, and what is left on the test set is the image shift, which more members of the same kind do not address.

**Round 26, Task 2: MLP with periodic (PLR) embeddings** (`dn_r26_t2_plr.py`; Gorishniy et al. 2022; K = 24, sigma 0.3, 3 seeds, settings fixed before the run). Alone 0.7396 (log-loss 0.677), below round 23's plain MLP (0.7494) and LightGBM (0.7559). As a fifth stack member −0.0005 [−0.0011, +0.0001], 0/5 folds; replacing LogReg −0.0005, 1/5. Discard. The features are independent given the class (round 23), so a model that is additive in the features is already close to optimal and higher-frequency feature maps only add variance. "CatBoost feature combinations" needs no run: the pipeline's CatBoost already receives the seven nominal columns as `cat_features` with the default `max_ctr_complexity = 4`, and those columns carry no signal (Cramér's V ≤ 0.008).

**Round 26, Task 1: multi-architecture level-2 stacker** (`dm_r26_t1_l2stack.py`, `do_r26_t1_l2_final.py`). Meta-models on the nine members' logits, settings fixed before the run, 3 seeds each, outputs averaged in logit space; meta-model and cut-off both nested; repeated on three other stacker splits (members fixed). Delta of the nested F2 against the logistic stacker (0.8379 on the pipeline folds):

| level 2 | pipeline folds | split 1 | split 2 | split 3 | mean | folds better |
|---|---|---|---|---|---|---|
| LightGBM (300 trees, 8 leaves) | +0.0006 [+0.0000, +0.0012] | −0.0001 | +0.0002 | +0.0003 | +0.0003 | 5 / 2 / 2 / 4 |
| LightGBM DART | −0.0004 | −0.0010 | −0.0001 | −0.0001 | −0.0004 | 1 / 0 / 2 / 2 |
| CatBoost depth 4 | −0.0000 | +0.0001 | −0.0005 | +0.0004 | −0.0000 | 3 / 3 / 2 / 3 |
| ExtraTrees | −0.0009 | −0.0011 | −0.0008 | −0.0009 | −0.0009 | 0 / 0 / 1 / 0 |
| MLP (32, 16) | +0.0004 | −0.0004 | +0.0002 | +0.0006 | +0.0002 | 5 / 1 / 3 / 3 |
| LR + LightGBM | +0.0005 [+0.0001, +0.0009] | −0.0002 | +0.0002 | +0.0005 | +0.0003 | 5 / 1 / 3 / 4 |
| **LR + LightGBM + CatBoost** | **+0.0006 [+0.0002, +0.0010]** | +0.0002 [−0.0002, +0.0006] | +0.0004 [−0.0000, +0.0009] | +0.0003 [−0.0003, +0.0009] | **+0.0004** | **4 / 4 / 4 / 4** |
| all six | +0.0002 | −0.0000 | +0.0005 | +0.0003 | +0.0003 | 2 / 2 / 4 / 5 |

LR + LightGBM + CatBoost is the only variant that is positive on every split with 4/5 folds each: it passes the round 25 rule (it would not have passed the old CI gate on three of the four splits, and it is the best of eight variants, so the +0.0004 is an optimistic estimate). Fitted as the pipeline would (cut-off on the full OOF: F2 0.8383 -> 0.8387, cut 0.150 -> 0.162), it changes 0.66% of OOF labels and 0.53% of test labels (136 rows: 23 newly flagged, 113 released), so the tree meta-models behave on the averaged test logits as they do out of fold. Expected effect: +0.0004 on Task 1 = about +0.00008 on the leaderboard average.

**v18 built (not uploaded).** The stacker is in the pipeline (`scripts/nb_part2.py` 7.1 and the final refit; caches `t1_l2_trees`, `t1_l2_trees_full`). Master notebook: OOF F2 0.8383 -> 0.8387, cut 0.150 -> 0.162. `submission_v18.csv` differs from v17 in 136 `task1` rows (23 newly flagged, 113 released; flag rate 60.60% -> 60.26%) and in nothing else, exactly as `do_r26_t1_l2_final.py` predicted.

## Round 27 (2026-11-14): v18 = 0.85515; re-tests under the round 25 rule and the overnight queue (`dp_` to `dv_r27_*.py`)
Brief: four items already measured as positive in earlier rounds but not in the pipeline, then an overnight queue (10-seed bags for Tasks 1 and 2, two new kinds of Task 4 member). Acceptance: mean nested gain > 0 and better in >= 4/5 folds; CIs reported, not used as a gate.

**Task 4: class bias on HUD-poor frames** (`ds_r27_t4_occluded_bias.py`). Under the v17 rule the predicted test shares are 0.148 / 0.261 / 0.150 / 0.248 / 0.193; the question was whether FiveN and PUBE Mobile are over-predicted when the HUD is missing. On training images with every class weighted to 20%:

| view | share of predictions (7-Eleven / FiveN / Free Fried / PUBE / Roblock) | precision | recall |
|---|---|---|---|
| full | 0.201 / 0.199 / 0.200 / 0.199 / 0.200 | 0.985 / 0.990 / 0.993 / 0.990 / 0.985 | 0.992 / 0.986 / 0.991 / 0.988 / 0.986 |
| 70% zoom | 0.204 / 0.197 / 0.199 / 0.199 / 0.201 | 0.961 / 0.979 / 0.977 / 0.974 / 0.967 | 0.981 / 0.965 / 0.972 / 0.967 / 0.972 |
| HUD hidden | 0.206 / 0.197 / 0.198 / 0.199 / 0.199 | 0.950 / 0.974 / 0.966 / 0.956 / 0.966 | 0.980 / 0.960 / 0.959 / 0.952 / 0.961 |

No class is over-predicted by more than 0.6 points, and the one that is slightly over-predicted is still 7-Eleven Knights (precision below recall), not FiveN or PUBE. The correction (pbar / 0.2)^−beta, with pbar = balanced mean probability on the HUD-poor views of the training folds, selects beta = 0 in every fold (+0.0000); free class multipliers fitted on those views −0.0000 [−0.0004, +0.0003], 3/5, and −0.0008 at the training mix. So the 26% / 25% predicted for FiveN / PUBE on test is not a bias that training images reproduce; the simpler reading is that the test set contains more of those two games (the EM diagnostic of round 26 says the same). No change.

**Task 4: frozen-feature probe where the ensemble is unsure** (`dq_r27_t4_probe_views.py`, `dr_r27_t4_probe_fusion.py`). Frozen CLIP + DINOv2 + SigLIP 2 features (2,304 numbers) of every training image under the full, 70%-zoom and HUD-hidden views; logistic probe per pipeline fold. Balanced macro-F1 under the v17 rule:

| | full | zoom70 | HUD hidden | mean |
|---|---|---|---|---|
| seven fine-tuned members | 0.9885 | 0.9714 | 0.9623 | 0.9741 |
| probe fitted on full frames (round 11's probe) | 0.9300 | 0.8283 | 0.5869 | 0.7817 |
| probe fitted on the three views | 0.9240 | 0.8917 | 0.8596 | 0.8918 |

* The probe is not better where the ensemble is unsure: on rows with ensemble max-probability < 0.5 its accuracy is 0.40-0.45 against the ensemble's 0.48-0.56, on every view. The premise (web-scale features hold up where fine-tuned ones fail) does not hold on simulated HUD loss.
* Fusion mix = ens if max(ens) >= tau else (1 − lam) ens + lam probe, (tau, lam) nested. Full-frame probe: −0.0003, 1/5. Three-view probe: **+0.0005 [+0.0000, +0.0009], 4/5 folds** (full +0.0004, zoom70 +0.0006, HUD hidden +0.0004; +0.0004 at the training mix); every fold picks lam = 0.1, and tau = 0.85 or "always" give the same score, so the gate does nothing: it is simply a tenth member at 10% weight.
* Where the gain comes from (0.9 ens + 0.1 probe, all 31,546 images): net +15 / +13 / +8 correct labels on full / zoom70 / HUD hidden, almost all from rows with ensemble confidence 0.5-0.7; below 0.5 it changes 28 / 70 / 126 labels for a net +2 / +6 / −2. Test frames sit in the low-confidence bins far more often than training frames, so on test this will change many labels for a small expected net gain.
* Passes the rule, so it goes into the pipeline as a member with fixed weight 0.10, no gate (`scripts/train_task4_probe.py`, caches `t4_probe_fold*.npz`, `t4_probe_full.npz`).

**Round 27, interrupted runs (2026-11-14 04:05).** Six jobs were running at once (Task 2 10-seed bag, Task 1 bag test, Task 1 member test, Task 2 KDE splits, the probe fit, the fold-0 GPU run) and the machine ran out of memory; all six were stopped. Nothing is corrupted (every script resumes from its caches). Partial results, to be completed one job at a time:

* **Task 2, stack blended with the KDE naive Bayes** (`dp_r27_t2_kde_blend.py`; strict version: the density grid spans the fitted rows only, 99.98% of OOF labels identical to round 21's; `task2.kde_nb_cv` reproduces it exactly). Pipeline folds only so far, stack + nested multipliers 0.7564:

  | level 2 | nested macro-F1 | delta [95% CI] | folds |
  |---|---|---|---|
  | 0.5 stack + 0.5 KDE NB | 0.7576 | +0.0011 [+0.0002, +0.0021] | 5/5 |
  | 0.7 stack + 0.3 KDE NB | 0.7569 | +0.0005 [−0.0003, +0.0013] | 4/5 |
  | weight chosen nested (picks 0.3-0.4 for the stack) | 0.7576 | +0.0012 [+0.0001, +0.0022] | 4/5 |
  | geometric 0.5 / 0.5 | 0.7574 | +0.0009 [−0.0000, +0.0019] | 4/5 |
  | KDE NB as a fifth LogReg-stack member | 0.7564 | −0.0001 [−0.0008, +0.0007] | 2/5 |

  Passes on the pipeline folds; the three other stacker splits were not reached.
* **Task 1, extra members under the v18 level 2** (`dt_r27_t1_members.py`), two of four splits: + XGBoost tuned −0.0001 / −0.0001 (2/5, 1/5); + LightGBM reduced −0.0006 / +0.0000 (1/5, 2/5); + both +0.0001 / +0.0002 (1/5, 4/5); + XGBoost tuned + three reduced members +0.0001 / −0.0002 (2/5, 1/5). Nothing passes so far.
* **Task 1, 10-seed bag under the v18 level 2** (`dv_r27_t1_bag_l2.py`): no split finished.
* **Task 2, 10 seeds of LightGBM / XGBoost / CatBoost on the re-released data** (`BAG_DIR=r27_bag by_r14_bag.py cv t2`): only LightGBM seed 42 finished (identical to the pipeline cache, max difference 0). About 7 hours of CPU remain (CatBoost: about 36 minutes per seed).
* **Task 4 probe member** (`scripts/train_task4_probe.py`): folds 0-3 fitted (full-frame accuracy 0.931-0.933), fold 4 and the full fit missing.
* **Task 4, colour-robust ConvNeXt-Tiny** (192x384, greyscale 0.3, colour jitter p 0.8 with 1.6x ranges, HUD dropout 0.6; new options `--p-gray`, `--p-jitter`, `--jitter-gain` in `scripts/train_task4.py`), fold 0: stopped after epoch 2 of 10 (validation macro-F1 0.9474; about 225 s per epoch). DINOv2-B fine-tuning (196x392, lr 1.5e-5, layer decay 0.65) not started.
* **Task 3, seeds averaged in THB**: implemented in `scripts/nb_part2.py` (7.3 and the final refit); notebooks not rebuilt yet.

**Round 27, resumed (2026-11-14 08:10; one GPU job and one CPU job at a time).**

* **Task 2, stack blended with the KDE naive Bayes: the three other stacker splits** (`SPLITS=1,2,3 dp_r27_t2_kde_blend.py`; nested deltas against the stack with nested multipliers, folds better out of 5):

  | level 2 | pipeline folds | split 1 | split 2 | split 3 | mean |
  |---|---|---|---|---|---|
  | 0.5 stack + 0.5 KDE NB | +0.0011, 5/5 | +0.0010, 4/5 | +0.0012, 5/5 | +0.0011, 4/5 | +0.0011 |
  | 0.7 stack + 0.3 KDE NB | +0.0005, 4/5 | +0.0009, 4/5 | +0.0006, 4/5 | +0.0007, 4/5 | +0.0007 |
  | weight chosen nested | +0.0012, 4/5 | +0.0020, 5/5 | +0.0014, 4/5 | +0.0020, 4/5 | +0.0017 |
  | geometric 0.5 / 0.5 | +0.0009, 4/5 | +0.0015, 5/5 | +0.0005, 4/5 | +0.0012, 4/5 | +0.0010 |
  | KDE NB as a fifth stack member | −0.0001, 2/5 | +0.0005, 4/5 | +0.0002, 4/5 | +0.0009, 4/5 | +0.0004 |

  The 95% CI of the fixed 0.5 blend is above zero on every split ([+0.0002, +0.0021], [+0.0001, +0.0020], [+0.0003, +0.0021], [+0.0002, +0.0020]), and of the nested weight too ([+0.0001, +0.0022], [+0.0010, +0.0030], [+0.0004, +0.0024], [+0.0009, +0.0030]). The nested weight is 0.3-0.5 for the stack in every fold of every split. Passes. In the pipeline (`nb_part2.py` 7.2): the KDE weight is chosen on the OOF rows from {0, 0.1, ..., 0.7} together with the class multipliers, the procedure measured in the "weight chosen nested" row; the KDE member's test probabilities come from a fit on all training rows.
* **Task 4 probe member**: fold 4 fitted (full-frame accuracy 0.936) and the full fit done; all six cache files exist, so `HAS_PROBE4` is true in the notebook.
* **Task 4, colour-robust ConvNeXt-Tiny, fold 0** (`cv_r22_t4_eval.py cnxt_col`; the script now also prints the round 27 criterion: uniform class mix, v17 rule, mean over full + zoom70 + HUD hidden). Alone 0.9822 full / 0.9595 zoom70 / 0.9490 HUD hidden (`cnxt_hud`: 0.9852 / 0.9632 / 0.9538); it is better only on synthetic darkening (gamma 3.5: 0.9257 vs 0.9093; gain 0.15: 0.9644 vs 0.9542). Added to the seven members under the v17 rule at a uniform mix: **−0.0009 [−0.0019, +0.0001]** (full −0.0004, zoom70 −0.0003, HUD hidden −0.0021). Fails; folds 1-4 not trained. Stronger colour augmentation costs accuracy on the views that imitate the test frames, and round 18 already found no brightness / colour effect on test confidence.

**v19 built (2026-11-14 09:37; `submission_v19.csv` = `submission.csv`, not uploaded yet).** Master notebook and `task2`, `task3`, `task4` notebooks rebuilt and executed (`task1`, `task5` sources unchanged); the file joined from the task notebooks is identical to the master notebook's file in every column.

| task | change | OOF v18 → v19 | test rows changed vs v18 |
|---|---|---|---|
| 1 | none | 0.8387 | 0 |
| 2 | stacker averaged with the KDE naive Bayes; KDE weight chosen on OOF = 0.6 (in-sample macro-F1 by weight 0 … 0.7: 0.7568, 0.7570, 0.7574, 0.7577, 0.7579, 0.7581, 0.7584, 0.7583); multipliers 0.875 / 1.025 / 1.04 / 1.06 | 0.7568 → 0.7584 (in-sample for weight and multipliers; nested 0.7564 → 0.7576) | 645 (2.49%); predicted shares 0.420 / 0.251 / 0.181 / 0.148 → 0.412 / 0.253 / 0.193 / 0.142 |
| 3 | seeds averaged in THB | 0.7946 → 0.7946 | 12,729 values move; sum of absolute changes 0.016% of the total; zero share 0.5003 unchanged |
| 4 | frozen-feature probe at 10% (probe alone 0.9232 OOF) | under the v17 rule 0.9873 → 0.9879 at the training mix, 0.9885 → 0.9888 at a uniform mix | 340 (1.31%); predicted shares move by at most 0.001 |
| 5 | none | no labels | 0 |

**Round 27, remaining queue items (2026-11-14 10:40).**

* **Task 4, DINOv2-B fine-tuned, fold 0** (`vit_base_patch14_reg4_dinov2.lvd142m`, 196x392, lr 1.5e-5, layer decay 0.65, HUD dropout 0.5, 10 epochs, 89 minutes; `cv_r22_t4_eval.py dinov2_hud`). Alone 0.9403 full / 0.9050 zoom70 / 0.8836 HUD hidden, far below every member (validation macro-F1 was still rising, 0.9352 → 0.9392 in the last epoch: under-trained at this learning rate and decay). Added to the seven members under the v17 rule at a uniform mix: **−0.0003 [−0.0010, +0.0004]** (full −0.0000, zoom70 +0.0000, HUD hidden −0.0011). Fails; folds 1-4 not trained. With `cnxt_col` this closes the "new backbone" item: three candidates since v16 (Swin-T, colour-robust ConvNeXt-T, DINOv2-B) all lower the HUD-hidden score when added.
* **Task 1, extra members under the v18 level 2** (`dt_r27_t1_members.py`), all four splits (nested delta, folds better):

  | added to the nine members | pipeline folds | split 1 | split 2 | split 3 | mean |
  |---|---|---|---|---|---|
  | XGBoost tuned | −0.0001, 2/5 | −0.0001, 1/5 | −0.0004, 0/5 | +0.0002, 4/5 | −0.0001 |
  | LightGBM reduced | −0.0006, 1/5 | +0.0000, 2/5 | −0.0000, 2/5 | +0.0006, 4/5 | +0.0000 |
  | XGBoost tuned + LightGBM reduced | +0.0001, 1/5 | +0.0002, 4/5 | −0.0000, 2/5 | +0.0000, 2/5 | +0.0001 |
  | XGBoost tuned + three reduced members | +0.0001, 2/5 | −0.0002, 1/5 | +0.0001, 3/5 | −0.0001, 3/5 | −0.0000 |

  No set reaches 4/5 folds on more than one split, and no mean is above +0.0001. Fails.
* **Task 1, 10-seed bag under the v18 level 2** (`dv_r27_t1_bag_l2.py`; test-like = the same validation rows with test-like missingness, rows re-weighted to the test's missing-cell rates):

  | members | pipeline folds | split 1 | split 2 | split 3 | mean | test-like, mean |
  |---|---|---|---|---|---|---|
  | bag 3 | −0.0002, 3/5 | −0.0000, 3/5 | +0.0001, 2/5 | +0.0002, 3/5 | +0.0000 | −0.0001 |
  | bag 5 | −0.0001, 3/5 | −0.0002, 1/5 | −0.0000, 2/5 | +0.0003, 4/5 | +0.0000 | −0.0005 |
  | bag 10 | +0.0001, 3/5 | +0.0001, 3/5 | +0.0004, 4/5 | +0.0006, 4/5 | +0.0003 | −0.0002 |

  The 10-seed bag has a positive mean (+0.0003) but is better in only 3/5 folds on two of the four splits, and it is worse under test-like missingness on three of the four (−0.0002, +0.0002, −0.0003, −0.0005). Same picture as round 14 under the logistic stacker (+0.0004, 3/5, test-like −0.0004). Fails the rule (>= 4/5 folds and test-like >= 0); Task 1 stays as in v18.
* **Task 2, 10-seed bag**: running (`BAG_DIR=r27_bag by_r14_bag.py cv t2`, about 7 CPU hours), then `dw_r27_t2_bag_eval.py` scores bags of 3 / 5 / 10 seeds under the v19 level 2.

Expected public-score change of v19 from the nested estimates: Task 2 +0.0012 to +0.0020 and Task 4 +0.0005 on one fifth of the score each, so about +0.0003 to +0.0005 overall, if the gains carry over to test (the Task 4 estimate is on simulated HUD loss, not on test frames).

## Round 28 (2026-11-14): v19 = 0.85623 (top 0.86750); Task 5 re-scan inside the unflagged accounts, Task 4 prior rule variants, DINOv2 re-test, patch-MIL head (`dx_`, `dx2_`, `dy_`, `dz_r28_*.py`)

**v19 = 0.85623, +0.00108 over v18** (snapshot `versions/v19_LB0.85623/`). The nested estimates predicted +0.0003 to +0.0005. Tasks 2 and 4 changed in the same upload, so the split is not identified; if Task 2 gave its nested +0.0012 to +0.0020 (+0.0002 to +0.0004 on the score), Task 4 gave +0.0035 to +0.0045, against +0.0005 on the simulated views. A member built on frozen web-scale features helps test frames more than the HUD-loss simulation says. Top of the leaderboard: 0.86750, a gap of 0.01127 = 0.056 on one task.

**Hypothesis B: a new anomaly family in the re-released Task 5 data that the 2,148 flags miss?** (`dx_r28_t5_unflagged_scan.py`, `dx2_r28_t5_aggregates.py`; label-free, test features only as in the pipeline; every statistic computed inside the 23,741 unflagged accounts so the known families cannot mask anything; every count compared with its own shuffle noise level.)

| scan inside the unflagged accounts | result | noise level / reference |
|---|---|---|
| 1-D density gaps: 174 raw column tails, 200 engineered, 418 trajectory (period differences, curvature, slope, ratios, min / max, range) | largest lambda 23 / 32 / 35, groups of 15-41 accounts; none >= 50 | pipeline threshold 50; the same scan on all accounts: 30 tails >= 50, largest 42,064 |
| pair pockets, engineered features (11,319 pairs) | one pair at z 8.7 (`friend_removes_cv` x `guild_changes_burst`, 126 observed vs 59) | 8.6; the pocket of the pipeline had z above 10.9 |
| pair pockets, trajectory features across behaviours (21,951 pairs, random tie-break) | largest z 5.5 | 5.9 |
| exact repeats across periods | only zeros of `friend_adds` / `friend_removes` (values clipped at 0; 4,469 accounts never remove a friend: 19%, a trait of normal accounts) | 0 repeats in the 22 other behaviours |
| a zero in period 4 only, in >= 2 behaviours | 0 accounts | |
| discrete columns across groups (4 flags, device / refund / guild counts per period and their sums): 141 pairs, 519 triples | largest z 3.1 / 4.1 | 3.7 / 4.6 |
| missing cells per account | SD 2.38, as binomial; 27 accounts with >= 15 missing | 21.7 expected |
| aggregates over the 24 continuous behaviours: within-account volatility, squared level, squared deviation of each period from the other three | SD ratio to the null 0.990-1.001; 15-28 accounts beyond the null's 0.1% / 99.9% quantiles; volatility quantiles equal N(0, 1/sqrt(24)) to 0.01 from 0.1% to 99.9% | 24 expected; top eigenvalue of the cross-behaviour correlations 1.061 vs the independence bound 1.065 |
| multivariate detectors in normal-score space vs a Gaussian-copula null: Mahalanobis, k-NN distance (k = 5, 20, 100), Isolation Forest, LOF (20, 100), 5 principal components, autoencoder | upper tails 2-30 accounts above the null's 99.9% quantile | 24 expected; with the 207 single-IP bots added back: PC1 207 of 207, Isolation Forest 164 |

* The unflagged accounts behave as one population of independent behaviours: no gap, no pocket, no cross-behaviour factor, no excess in any tail. The copula comparison has no power against a 2-feature pocket (adding the 339 pocket accounts back changes nothing), which is why pockets are scanned pairwise; that scan finds nothing new.
* The 192 accounts "closer to the centre than the copula null allows" are the low end of the within-account volatility (aggregate −0.21 against an SD of 0.20) with no gap in the lower tail (lambda 16): a continuum, not a group.
* The friend_removes x guild_changes cell is at the noise level; if it were real it would hold about 67 members among 126 accounts: +0.002 on Task 5 at most. Not added.
* Size argument: for Task 5 to be 0.92 instead of 0.98-0.99 with our flags, about 190 anomalies would have to be missing. A group of that size with no trace in any marginal, pair, trajectory, discrete combination or aggregate cannot be found from the features by anyone. Hypothesis B is not supported; the 2,148 flags stay.
* Leaderboard arithmetic with v19: 5 x 0.85623 = 4.281; T1 0.85 + T2 0.757 + T3 0.795 leaves T4 + T5 = 1.88, i.e. T4 = 0.89 with T5 = 0.99, in line with the label-free Task 4 estimates of round 11 (0.87-0.91). The gap to the top is most plausibly Task 4.

**Task 2, 10-seed bag of LightGBM / XGBoost / CatBoost on the re-released data** (last item of the round 27 queue; `BAG_DIR=r27_bag by_r14_bag.py cv t2`, 206 minutes; `dw_r27_t2_bag_eval.py`). Seed-42 fits equal the pipeline caches exactly. Single-seed argmax macro-F1: LightGBM 0.7557 ± 0.0002, XGBoost 0.7557 ± 0.0002, CatBoost 0.7546 ± 0.0004; 10-seed bags 0.7561 / 0.7561 / 0.7548. Under the level 2 (stacker out of fold, multipliers nested), mean over the pipeline folds and three other splits:

| members | v19 level 2 (0.4 stacker + 0.6 KDE NB) | folds better per split | stacker alone | folds better per split |
|---|---|---|---|---|
| bag 3 | −0.0001 | 2/5 1/5 2/5 1/5 | +0.0000 | 3/5 1/5 3/5 3/5 |
| bag 5 | −0.0003 | 1/5 1/5 3/5 1/5 | −0.0000 | 2/5 2/5 2/5 2/5 |
| bag 10 | −0.0003 | 2/5 0/5 2/5 2/5 | −0.0001 | 1/5 3/5 2/5 3/5 |

Seed noise of a member (0.0002-0.0004) is far below the level-2 resolution, and the stacker already averages three GBDTs with a logistic model and the KDE member. Fails; Task 2 stays as in v19. This closes the round 27 queue: of its nine items, three went into v19 (Task 2 KDE blend, Task 4 probe, Task 3 THB average) and six failed.

**Hypothesis A3: temperature and per-member exponents in the Task 4 prior rule** (`dy_r28_t4_member_alpha.py`; seven members + probe, all training images out of fold, uniform class mix, mean of full / zoom70 / HUD hidden; v19 rule 0.9745; everything nested).

| variant | nested delta [95% CI] | folds | at the training mix |
|---|---|---|---|
| A: one exponent, nested from {0.3 … 2.0} (picks 0.75-1.0) | −0.0003 [−0.0006, −0.0000] | 0/5 | −0.0013 |
| B: per-member temperature (0.52-0.56; EfficientNetV2 0.43), then pi^−alpha (picks 1.0-1.25) | −0.0007 [−0.0010, −0.0003] | 0/5 | −0.0014 |
| C: prior adjustment per member before averaging, one alpha (picks 1.0-1.25) | −0.0002 [−0.0005, +0.0001] | 0/5 | −0.0011 |
| D: per-member exponents, coordinate ascent visiting members first to last | +0.0001 [−0.0002, +0.0004] | 4/5 | −0.0006 |
| D: the same, visiting members last to first | −0.0001 [−0.0003, +0.0002] | 2/5 | −0.0007 |
| E (diagnostic, not usable without the TAs' approval): EM shares from the evaluated rows + Bayes rule | −0.0008 [−0.0012, −0.0004] | 0/5 | −0.0004 |

* A temperature before the prior is not a new rule: argmax normalise(mix^(1/T)) · pi^−a equals argmax mix · pi^−(aT) on every row (checked), so T = 0.46 with a = 1 is alpha = 0.46, inside the family round 26 scanned. In-sample the three-view mean by alpha is 0.9735 (0.46), 0.9745 (0.75), 0.9746 (0.9), 0.9745 (1.0), 0.9738 (1.2): a plateau from 0.75 to 1.0, as before.
* Per-member exponents: the first visiting order gives +0.0001 on 4/5 folds, but the exponents it picks are 1.5 for whichever members are visited first and 0.75 for the rest; reversing the order moves the 1.5s to the other members and the gain becomes −0.0001 (2/5). It is the same plateau reached with eight knobs, not a property of the members. Not adopted.
* EM on the evaluated rows recovers the uniform shares on full and zoomed frames (0.198-0.202) but not on HUD-hidden frames (0.229 for 7-Eleven Knights), and the rule built on it is worse than the fixed exponent on every fold. So even with the TAs' approval a transductive prior estimate would not be expected to beat alpha = 0.75 when frames lose their HUD.
* The rule stays argmax p · pi^−0.75.

**Hypothesis A1: DINOv2-B with a proper ViT fine-tuning recipe, fold 0** (`train_task4.py --tag dinov2_v2 --model vit_base_patch14_reg4_dinov2.lvd142m --h 126 --w 238 --lr 5e-5 --layer-decay 0.8 --warmup-epochs 2 --epochs 15 --p-hud 0.5 --grad-ckpt --folds 0`; new option `--warmup-epochs`; 126x238 keeps the 120x230 aspect ratio and gives 9x17 patches, 220 s per epoch; `cv_r22_t4_eval.py dinov2_v2`). Round 27's run (lr 1.5e-5, decay 0.65, 10 epochs, 196x392) reached 0.9403 and was still rising; this one reaches 0.9869.

| fold 0, macro-F1 | full | zoom70 | HUD hidden | gamma 3.5 | gain 0.15 | 75% grid | real dark / low contrast |
|---|---|---|---|---|---|---|---|
| cnxt_hud (ConvNeXt-T) | 0.9852 | 0.9632 | 0.9538 | 0.9093 | 0.9542 | 0.7816 | 0.9839 |
| cnxtS_hud (ConvNeXt-S, round 20) | 0.9872 | 0.9675 | 0.9593 | | | | |
| **dinov2_v2** | 0.9869 | **0.9740** | 0.9569 | **0.9454** | **0.9673** | **0.8526** | 0.9752 |
| seven members | 0.9892 | 0.9709 | 0.9607 | 0.9270 | 0.9618 | 0.7975 | 0.9825 |
| seven + dinov2_v2 | 0.9893 | 0.9724 | 0.9629 | 0.9295 | 0.9642 | 0.8355 | 0.9838 |

* Alone it beats the seven-member ensemble on zoomed frames and on every synthetic corruption no model was trained on (dark, 75% hidden): the most robust single model so far.
* Added to the seven under the v17 rule at a uniform mix, three views: **+0.0012 [+0.0003, +0.0022]** (full +0.0002, zoom70 +0.0029, HUD hidden +0.0005); at the training mix (full + zoom70) +0.0017 [+0.0008, +0.0028]. First candidate since ConvNeXt-Small whose CI is above zero. Passes; folds 1-4 and the full-data model are training (about 55 minutes each).
* **All five folds** (`ea_r28_t4_eight_members.py`; fold macro-F1 0.9869 / 0.9868 / 0.9872 / 0.9857 / 0.9856, OOF 0.9864; all 31,546 training images out of fold, uniform class mix, v17 rule):

  | | full | zoom70 | HUD hidden | mean |
  |---|---|---|---|---|
  | dinov2_v2 alone | 0.9860 | 0.9725 | 0.9539 | 0.9708 |
  | seven members | 0.9885 | 0.9714 | 0.9623 | 0.9741 |
  | v19: seven + 10% probe | 0.9888 | 0.9721 | 0.9627 | 0.9745 |
  | eight members | 0.9889 | 0.9736 | 0.9637 | 0.9754 |
  | eight + 10% probe | 0.9896 | 0.9743 | 0.9640 | 0.9760 |

  | against v19 | delta [95% CI] | folds | full / zoom70 / HUD hidden | training mix |
  |---|---|---|---|---|
  | eight + 10% probe | **+0.0015 [+0.0011, +0.0019]** | 5/5 | +0.0008 / +0.0023 / +0.0013 | +0.0014 |
  | eight, no probe | +0.0009 [+0.0004, +0.0014] | 5/5 | +0.0001 / +0.0016 / +0.0010 | +0.0009 |
  | eight + probe, probe weight nested (picks 0.1 in every fold) | +0.0015 [+0.0011, +0.0019] | 5/5 | | |
  | eight + 10% probe, weight of the new member nested (picks 4, 4, 3, 3, 4) | **+0.0021 [+0.0013, +0.0028]** | 5/5 | +0.0007 / +0.0039 / +0.0017 | +0.0020 |

  The nested member weight against equal weights: +0.0006 [+0.0001, +0.0011], 5/5 folds (full −0.0001, zoom70 +0.0016, HUD hidden +0.0004); in-sample the gain over v19 by weight is +0.0015 (1), +0.0018 (1.5), +0.0020 (2), +0.0022 (3), +0.0022 (4). The probe keeps its place next to a fine-tuned DINOv2 (with it +0.0015, without it +0.0009).
* **v20 built (2026-11-14 20:36; `submission_v20.csv` = `submission.csv`, not uploaded yet).** `nb_part2.py`: `dinov2_v2` in `T4_BACKBONES`, `T4_MEMBER_W = {"dinov2_v2": 3.0}` (the lower end of the plateau), weighted mean in 7.4 and in the Train+Val refit; master and `task4` notebooks re-executed, the other task notebooks unchanged. OOF in the notebook: equal-weight average 0.9888, member weights 0.9891, + probe 0.9896 (argmax); under the rule 0.9879 → 0.9886 at the training mix and 0.9888 → 0.9895 at a balanced mix. Test: 769 Task 4 labels change against v19 (2.97%); predicted shares 0.147 / 0.262 / 0.149 / 0.248 / 0.194 → 0.142 / 0.271 / 0.144 / 0.244 / 0.198; Tasks 1, 2, 3, 5 identical (0 rows). Expected from the nested estimate alone: +0.0021 on Task 4 = +0.0004 on the score; v19's probe transferred several times its estimate, so the real change may be larger, in either direction.
* **v20 = 0.85948, +0.00325 over v19** (uploaded 2026-11-15; snapshot `versions/v20_LB0.85948/`). Only Task 4 changed, so this is about +0.016 on Task 4 from 769 changed labels: roughly +420 net correct, i.e. at least ~595 of the 769 changes fixed an error. The nested estimate was +0.0021 on Task 4, eight times too small. Two uploads in a row (probe, DINOv2) have now transferred far more than the full / zoom / HUD-hidden proxies predict, in the direction the proxies give. What the proxies rank correctly but under-weight is robustness to content the training set does not cover: on the synthetic corruptions no model trained on (gamma 3.5, 75% hidden) DINOv2 led by 2 to 5 points, while on the three proxy views it led by 0.3. Leaderboard arithmetic: 5 x 0.85948 = 4.297; with T1 0.85, T2 0.757, T3 0.795, T5 0.99 this puts Task 4 at about 0.905. Top of the leaderboard 0.86750: 0.008 away = 0.040 on one task.
* Which of the three changes did it (learning rate 1.5e-5 → 5e-5, layer decay 0.65 → 0.8, 10 → 15 epochs with 2 warm-up, 196x392 → 126x238) is not separated; the round 27 run was under-trained, so the recipe is the likely cause, not the aspect ratio.

**The dinov2_v2 recipe on other ViT-B backbones and the patch-MIL head, fold 0** (2026-11-15; `cv_r22_t4_eval.py` now scores against the v20 ensemble: eight members with DINOv2 at triple weight, + 10% probe on the three-view criterion).

| fold 0, macro-F1 | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | added to v20, three views, uniform mix |
|---|---|---|---|---|---|---|
| v20 members (no probe) | 0.9900 | 0.9763 | 0.9647 | 0.9354 | 0.8588 | |
| **EVA02-Base** (`eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`, 126x238, 300 s per epoch) | 0.9839 | 0.9697 | 0.9452 | 0.9412 | 0.8228 | **+0.0010 [+0.0002, +0.0018]** (full +0.0009, zoom70 +0.0013, HUD hidden +0.0007) |
| SigLIP 2 Base (`vit_base_patch16_siglip_224.v2_webli`, 128x240, 163 s per epoch) | 0.9785 | 0.9611 | 0.9377 | 0.9158 | 0.7596 | −0.0002 [−0.0008, +0.0005] (HUD hidden −0.0011) |
| patch-MIL head on frozen DINOv2 tokens (`dz_r28_t4_patch_mil.py`, 196x392, 392 tokens) | 0.6999 | 0.6515 | 0.6318 | | | −0.0004 to −0.0024 at every weight tried |

* EVA02-Base passes (CI above zero, positive on all three views); folds 1-4 and the full-data model are training (about 6.5 GPU hours). Its weight against DINOv2's triple weight will be chosen nested on all folds before v21 is built.
* SigLIP 2 with the same recipe is better than round 22's run (0.9785 vs 0.975) but still the weakest ViT here and lowers the HUD-hidden score; fails. The recipe was not the only thing wrong with it.
* The MIL head did not converge to anything useful: 0.68 on full frames after 12 epochs with the loss still falling (1.47 → 1.21), far below the pooled logistic probe on the same kind of features (0.92). As run (per-class attention over 392 tokens, token dropout 30%, CutMix / Mixup, HUD dropout, 12 epochs at lr 1e-3) it is an under-fitted head, not a test of the idea; but the fine-tuned DINOv2 already learns from the patch tokens end to end and is 30 points better, so the frozen-token route is closed rather than tuned.

**Task 5 stress test: would the scans see a subtle version of slide 22's four threats?** (`eb_r28_t5_stress.py`, `eb2_r28_t5_pingpong.py`; label-free, test features only.) Slide 22 names account theft ("smart attackers who wait 24 hours between actions"), account-selling rings, bot farms ("gradual botting patterns") and VPN regional-pricing exploiters, and gives "flagging legitimate travelers" and "players who legitimately improve" as the false positives to avoid.

*A. Injection.* n unflagged accounts receive a planted pattern of strength s, in units of the behaviour's within-account SD over the four periods (median 0.27 population SDs; s = 2 is a change an account shows by chance in about 5% of periods); planted values are clipped to the range the unflagged accounts show. Patterns: takeover (period 4: six behaviours, a new device, password change, a move of s x 84 km), sold (from period 3 on: skill, K/D, accuracy, reaction time, friends, guild, device), bot (linear drift over the periods in eight behaviours), vpn (VPN flag, location alternating by s x 84 km, purchases up in the away periods). A scan fires at gap lambda >= 50, pocket z >= 10.9, or >= 45 accounts above the untouched data's 99.9% level (24 expected). Smallest strength at which a scan fires, and what it isolates there:

| pattern | n = 30 | n = 100 | n = 200 | scan that fires first |
|---|---|---|---|---|
| takeover | not up to s = 4 (43 above, 23 planted) | s = 3 (62 above, 42 planted) | s = 3 (gap 30 of 30; pocket z 18) | odd-period-4 aggregate, then pocket |
| sold | s = 4 | s = 3 (49 above, 25 planted) | s = 3 (65 above, 42 planted) | step-at-period-3 and drift aggregates |
| bot | not up to s = 4 | s = 3 (45 above, 24 planted) | s = 3 (73 above, 53 planted) | linear-drift aggregate |
| vpn | s = 4 | s = 3 (85 above, 69 planted) | s = 2 (59 above, 44 planted) | ping-pong location statistic |

  (The "sold" rows of the log also show a gap of lambda 1,974 with 0 planted accounts in it: the accounts whose `friend_adds` is clipped at zero, which cross the scan's 5% size limit when a few accounts move. It is not a detection and is not counted.)
  So a planted group of 100 or more accounts is seen once it moves five to eight behaviours by about 3 within-account SDs (0.8 population SDs), a group of 30 needs about 4, and at s = 2 nothing fires at any size (200 planted takeovers: 34 above the level, 13 of them planted). On the real unflagged accounts every one of these statistics sits at its expected count (round 28 scan: 15-28 against 24). Conclusion: no group of 100+ accounts with one of these four shapes at 3 SDs or more is left; a subtler group lies inside ordinary period-to-period noise and no label-free method can isolate it (at s = 2 the best scan's flags would be 38% planted at 7% recall).

*B. The real unflagged accounts, two patterns the earlier scans did not target.*
* **Ping-pong travel** (location alternating A, B, A, B). Under exchangeable periods the statistic has the same distribution for the true period order and for the two re-orderings. Accounts above the re-orderings' 99% level: 197 for the true order against 155 and 163 (159 expected); excess +38 [+5.5, +73] at 99%, +26.5 [+2.5, +55.5] at 99.5%, +2 at 99.9%, +5 at 95%. A marginal excess of about 30-40 accounts whose alternation is 100-140 km, inside the normal hop range (median 84 km, maximum 335 km). The 197 accounts show nothing else: VPN flag 10.7% (15.1% among all unflagged), password change 4.9% (4.9%), and no purchase, transfer or location-change alternation in phase with the travel (0.69 / 0.73 / 0.67 against 0.70 for all unflagged). An account that only travels back and forth is the slide's "legitimate traveler". Not flagged: if the excess is noise, adding the 197 costs about 0.009 on Task 5; if all 38 were labelled anomalies it would gain about the same, and nothing supports the second reading. The (A, B, A) return over three periods is at 0.332 (1/3 expected), 0.309 among the largest triangles.
* **Accounts moving together** (all four period locations within r km of another account's): 0 pairs up to 20 km, 5 at 40 km against 3.0 after rotating the partner's periods. Homes are spread over the whole globe (13,299 one-degree cells), so there are no shared locations at all.
* **Co-movement rings** (two accounts changing in the same periods: correlation of the 96-number period-deviation profile with the most similar other account, against a block-shuffled null): quantiles identical (99.9%: 0.694 vs 0.692), 25 accounts above the null's 99.9% level (24 expected), none above its maximum. Power: with 5 planted rings of 5 accounts (profile correlation about 0.5) 11 of the 25 members exceed the null's maximum; with 20 rings, 46 of 100.
* **Exact repeats across accounts**: no non-zero value is shared by two accounts in any of 106 continuous columns; no two accounts share their coordinates to 3 decimals; `account_age_days` has no repeated value.
* Task 5 stays at 2,148 flags.

**Three CPU checks on the cached out-of-fold predictions while EVA02 trained** (2026-11-15; one process at a time, one or two threads). All three fail; nothing changes in the pipeline.

* **Task 2, temperature of the KDE naive Bayes before the blend** (`ed_r28_t2_kde_temperature.py`). Idea: the KDE NB multiplies 38 likelihoods as if independent, so it should be over-confident, and flattening it (p^(1/T), T in {1.5, 2, 2.5, 3, 4}) should blend better. The premise does not hold: out of fold the KDE NB is already calibrated (mean top probability 0.778 vs accuracy 0.771; log-loss 0.625 at T = 1, rising to 0.638 / 0.691 / 0.894 at T = 1.5 / 2 / 4); the stacker is equally calibrated (0.771 vs 0.770). This fits round 19 / 23, which found the columns close to independent given the class: then naive Bayes is the right model and its probabilities need no flattening. Nested on four stacker splits (T, KDE weight in {0.3, ..., 0.9} and multipliers from 4 folds; one Powell start instead of five, the same for reference and candidates): mean delta against T = 1 between -0.0003 and +0.0000 for every fixed T and for T chosen nested; at most 4/5 folds on one split, mostly 1-3/5. The in-sample T x w surface varies by 0.0002. Fails.
* **Task 1, standardised level-2 log-odds** (`ee_r28_t1_zscore_l2.py`; the three meta-models re-fitted on the cached member log-odds, because the pipeline caches only the mean of the two tree meta-models). The three log-odds already have almost the same spread (OOF SD 4.88 / 4.59 / 4.55 for LogReg / LightGBM / CatBoost; correlations 0.966-0.996), so standardising moves the weights from 1/3 each to 0.319 / 0.339 / 0.342 and changes 0.09-0.21% of labels. Nested F2 delta per split -0.0001 / -0.0001 / +0.0003 / -0.0001 (mean -0.0000; folds better 2, 1, 3, 2 of 5); test-like +0.0001. Fails.
* **Task 3, zero gate where historical_spending and prev_month_spending are both 0** (`ef_r28_t3_zero_pair.py`; the pipeline's OOF prediction rebuilt nested, NMAE 0.79458, and saved to `outputs/r28/t3_oof_rebuilt.npz` so later Task 3 checks skip the LP). No such row exists: both columns are strictly positive (minimum 0.026 / 0.008 THB, before and after the identity repair; 45 rows have both missing). Nearest well-posed version: the accounts with the lowest repaired historical_spending (bottom 1 / 5 / 10 / 25%; y = 0 in 69% of them against 48% overall). The median gate already sets 90-92% of them to 0, and among the rest the zero share is 25-38%, below 50% in every probability band (at most 44% at P(y>0) + shift in (0.5, 0.6]), so the L1-optimal prediction there is positive and a stricter gate cannot help: nested over group and gate level +0.00000 [-0.00001, +0.00001], 3/5 folds, 12 rows changed. Fails; the gate is already calibrated on these accounts.

**EVA02-Base as a ninth Task 4 member, all five folds** (`ec_r28_t4_ninth_member.py eva02B`, 2026-11-15; every fold model predicts its own validation images under the full, 70%-zoom and HUD-hidden views, h-flip TTA, written to `outputs/r19_views/eva02B_fold{f}.npz`; uniform class mix, v17 rule, mean over the three views). Fold models: plain-argmax macro-F1 0.9839 / 0.9842 / 0.9854 / 0.9841 / 0.9832; the full-data model trained 15 epochs (final loss 0.763).

| full / zoom70 / HUD hidden / mean | |
|---|---|
| eva02B alone | 0.9846 / 0.9690 / 0.9466 / 0.9667 |
| dinov2_v2 alone | 0.9860 / 0.9725 / 0.9539 / 0.9708 |
| v20: eight (DINOv2 x3) + 10% probe | 0.9895 / 0.9757 / 0.9649 / 0.9767 |

| added to v20 | delta [95% CI] | folds better | full / zoom70 / HUD hidden | training mix |
|---|---|---|---|---|
| weight 1 | +0.0004 [+0.0001, +0.0007] | 4/5 | +0.0002 / +0.0006 / +0.0003 | +0.0004 |
| weight 1.5 | +0.0005 [+0.0001, +0.0009] | 5/5 | +0.0004 / +0.0009 / +0.0002 | +0.0005 |
| weight 2 | +0.0006 [+0.0002, +0.0010] | 4/5 | +0.0004 / +0.0012 / +0.0002 | +0.0006 |
| weight 3 | +0.0004 [-0.0002, +0.0009] | 4/5 | +0.0001 / +0.0010 / -0.0000 | +0.0003 |
| weight 4 | -0.0000 [-0.0005, +0.0006] | 3/5 | -0.0001 / +0.0008 / -0.0007 | +0.0001 |
| **weight nested** (picks 2, 2, 1.5, 2, 2) | **+0.0005 [+0.0001, +0.0009]** | **4/5** (+0.0011 +0.0004 +0.0003 +0.0008 -0.0000) | +0.0004 / +0.0011 / +0.0001 | +0.0005 |
| DINOv2 and EVA02 weights nested together (picks (4, 2) once, (3, 2) four times) | +0.0006 [+0.0001, +0.0010] | 4/5 | +0.0004 / +0.0012 / +0.0001 | +0.0005 |
| probe weight nested, EVA02 at 2 (picks 0.1 in every fold) | +0.0006 [+0.0002, +0.0010] | 4/5 | | +0.0006 |

* Passes the rule (mean > 0, 4/5 folds; fold 4 is -0.00004). The nested weight against weight 1 is +0.0001, 3/5 folds, so 2 is the plateau the nested choice supports rather than a proven improvement over 1. The joint grid keeps DINOv2 at 3 (in-sample DINOv2 3 or 4 with EVA02 2 both +0.0006) and the probe at 0.1. Labels changed on the training images 0.38%.
* As with DINOv2, the gain is largest on zoomed frames (+0.0011) and smallest on full frames; EVA02 alone is weaker than DINOv2 on every view but adds a different error pattern (MIM pre-training on IN-22k against DINOv2's self-distillation on LVD-142M).
* Pipeline: `eva02B` in `T4_BACKBONES` (126x238), `T4_MEMBER_W = {"dinov2_v2": 3.0, "eva02B": 2.0}`; the 7.4 member list and the limitations paragraph updated.
* **v21 built (2026-11-15; `submission_v21.csv` = `submission.csv`, not uploaded yet).** Master and `task4` notebooks re-executed (the learning-curve cell now cycles the 8-colour palette, dashed for the ninth backbone); the other task notebooks unchanged. OOF in the notebook: eva02B alone 0.9842 (folds 0.9839-0.9854); equal-weight average 0.9888 → 0.9892, member weights 0.9891 → 0.9893, + probe 0.9896 → 0.9897 (argmax); under the rule 0.9886 → 0.9890 at the training mix and 0.9895 → 0.9899 at a balanced mix. Test: eva02B agrees with the other members' test argmax 88.5-90.9% of the time (DINOv2 90.9%), 18% of its test frames have max-p < 0.7 (OOF 2%). 323 Task 4 labels change against v20 (1.25%), spread over all class pairs (largest: PUBE Mobile → Free Fried 39, FiveN → Roblock 31, Roblock → FiveN 31, Free Fried → PUBE Mobile 31); predicted shares 0.142 / 0.271 / 0.144 / 0.244 / 0.198 → 0.143 / 0.271 / 0.145 / 0.244 / 0.198; Tasks 1, 2, 3, 5 identical (0 rows). Expected from the nested estimate alone: +0.0005 on Task 4 = +0.0001 on the score; the last two Task 4 uploads transferred 3-8 times their estimates.
* **v21 = 0.85939, −0.00009 against v20** (uploaded 2026-11-15; snapshot `versions/v21_LB0.85939/`). Only Task 4 changed, so about −0.0005 on Task 4 from 323 changed labels, against an expected +0.0005. If roughly half of the changed labels fall in the public split and each is close to a coin flip, one SD of the net count is about sqrt(160) = 13 labels, about ±0.001 on Task 4: the result is half an SD below zero and half an SD below the estimate, i.e. noise either way. What it does say: the 3-8x over-transfer of v19 and v20 is not a rule. Those two changes moved 340 and 769 labels in a consistent direction (probe: a different feature family; DINOv2: a much more robust backbone on corrupted frames, +2 to +5 points on gamma 3.5 / 75% hidden); EVA02 alone is below DINOv2 on the three proxy views (mean 0.9667 vs 0.9708) and below the v20 members on the 75%-hidden grid (fold 0: 0.8228 vs 0.8588), though above them on gamma 3.5 (0.9412 vs 0.9354); its gain on the proxies was a diversity gain of +0.0005, not a robustness jump. The pipeline keeps EVA02 at weight 2: it passed the rule fixed before the upload, and the public score is not a selection criterion.
* Round 29 (DINOv2 at 252x476, fold 0) was stopped by the harness for low memory at epoch 3 of 15 (val F1 0.307 / 0.764 / 0.909, the same curve as dinov2_v2's 0.330 / 0.775 / 0.909). Commit reached 45.0 of 46.0 GB with 7.5 GB of VRAM in use (EVA02 at 126x238: 3.4 GB VRAM, ~2 GB commit headroom); without any trainer 30.8 GB is committed while the eight largest processes hold about 8 GB. Not restarted.

## Audit after the TA ruling on test data (2026-11-15)
The TAs answered a classmate's questions: using unlabeled test images to choose a validation split that resembles the test set is
data leakage, and so is comparing test and train image statistics in the EDA to choose augmentation; any pretrained-model licence
is fine; TTA at prediction (flips, zooms, averaging) is allowed.

**Main pipeline (code), checked file by file.** No model, encoder, imputer, scaler or fold split is fitted on test data: Task 1
features standardise with training statistics (`task1.add_features(ref=train)`), the 24-column members fill with the training
median; Task 2 composites and frequency encodings use training rows, the KDE naive Bayes fits training rows only; Task 3 fills
spending from row identities (per row) and imputes `segment` with a classifier trained on training rows; Task 4 normalises with
fixed ImageNet constants, decodes test images only to predict them (`train_task4.py`, `train_task4_probe.py`: the probe's
mean / SD come from training rows); folds are `StratifiedKFold` on training labels. Task 5 has only `test.csv` by design.
Changed in this round so that no check computes statistics on test data:
* `src/audit.py`: adversarial validation (train vs test LightGBM, KS) and the test-image scan removed; the audit now covers the
  training data only (Task 5: its one file). `nb_part1.py` section 3.6 text and calls updated (the Task 1 "test rows are more
  complete, test F2 should be higher" paragraph and the test-brightness remark are gone).
* `nb_part2.py` section 7.4: the "distribution-shift diagnostics" cell (test confidence, fold agreement on test, predicted test class
  mix; figure `t4_shift.png`) replaced by OOF pairwise agreement of the members; the "test-set shift" paragraph replaced by a
  training-data rationale and a note that the test diagnostics were removed; "less confident on test images" (DINOv2-S) and the
  Task 1 "test-like" mentions removed; limitations no longer propose EM label-shift correction.
* `build_task_notebooks.py` (its own copy of the audit cells): the per-task audits call the training-only functions, the Task 1
  overview shows the training missing rate only (it printed the test's too), Task 4 picks the new OOF-agreement cell.
* `experiments/README.md` lists the old scripts that used test data for diagnostics; `cv_r22_t4_eval.py` lost its test diagnostic.

**Decisions that test data influenced (history; still open for the user):**
1. Task 4 balanced-prior rule (alpha 0.75, since v17) and the uniform-class-mix validation used for every Task 4 decision since
   round 25: adopted because the models' predictions on test images showed 17% 7-Eleven Knights against 39% in training (round 23:
   "the reason to switch ... is the grey area"). alpha itself was fixed on training images. Rule text in 7.4 kept as written.
2. HUD dropout (7 of 9 members) and the zoom70 / HUD-hidden validation views: supported by a training-only occlusion study (centre-
   only accuracy 0.919 vs 0.982), but partly motivated by low-confidence test images looking HUD-less (round 19 section 5); the
   7.4 sentence saying so is kept.
3. Low-light augmentation and Mosaic not tried because test confidence was lower in every brightness bin (round 22).
4. Task 1 "test-like" gate (validation rows re-weighted to the test's missing-cell rates): decisive once (round 8, E1a imputed
   LightGBM rejected at +0.0011 OOF); every merged Task 1 change passes on plain OOF alone (round 11 members +0.0016, 5/5 folds).
   Retired: not used for any new decision.

**Round 29, Task 4: 4-view multi-scale TTA** (`eg_r29_t4_tta4.py`; every fold model of the nine v21 members predicts its own
validation images under original, h-flip, central 85% zoom and zoom h-flip, saved separately in `outputs/r29_tta/`; scored on all
31,546 training images out of fold at the training class mix, no synthetic views, no re-weighting; the baseline is the re-computed
2-view mean, which matches the cached probabilities to 0.0001 macro-F1 per member and exactly for the ensemble).

| member, argmax | 2 views | 4 views (w = 0.5) | w = 0.25 | zoom85 pair alone |
|---|---|---|---|---|
| cnxt | 0.9865 | 0.9849 (-0.0016) | 0.9860 (-0.0005) | 0.9787 |
| deit3 | 0.9839 | 0.9839 (+0.0000) | 0.9845 (+0.0007) | 0.9779 |
| cnxt_hud | 0.9845 | 0.9831 (-0.0013) | 0.9844 (-0.0001) | 0.9784 |
| cnxt22k_hud | 0.9832 | 0.9828 (-0.0004) | 0.9834 (+0.0002) | 0.9783 |
| effv2_hud | 0.9816 | 0.9803 (-0.0013) | 0.9815 (-0.0001) | 0.9751 |
| deit3_hud | 0.9814 | 0.9812 (-0.0002) | 0.9822 (+0.0008) | 0.9768 |
| cnxtS_hud | 0.9861 | 0.9843 (-0.0018) | 0.9859 (-0.0002) | 0.9802 |
| dinov2_v2 | 0.9864 | 0.9864 (+0.0000) | 0.9867 (+0.0002) | 0.9828 |
| eva02B | 0.9842 | 0.9845 (+0.0003) | 0.9847 (+0.0006) | 0.9807 |

| v21 ensemble (0.9897 argmax / 0.9890 rule) | argmax | rule alpha 0.75 |
|---|---|---|
| 4 views on DINOv2 + EVA02, w = 0.5 (the brief's candidate) | -0.0003 [-0.0007, +0.0000], 1/5 folds | -0.0003 [-0.0008, +0.0002], 2/5 |
| 4 views on DINOv2 + EVA02, w = 0.25 | -0.0001 [-0.0004, +0.0002], 2/5 | -0.0002 [-0.0006, +0.0002], 1/5 |
| 4 views on all nine, w = 0.5 | -0.0012 [-0.0018, -0.0007], 0/5 | -0.0010 [-0.0017, -0.0005], 0/5 |
| 4 views on all nine, w = 0.25 | -0.0003 [-0.0006, +0.0001], 2/5 | -0.0004 [-0.0008, +0.0000], 1/5 |
| all nine, w nested from {0.25, 0.5} (picks 0.25 in every fold) | -0.0003, 2/5 | -0.0004, 1/5 |

* Fails at every setting. The zoomed views alone are 0.4-0.8 points worse than full frames for every member (they crop away the
  frame edges, where the HUD sits, and the training frames are full screenshots), so averaging them in dilutes the members; the ViTs
  lose least (DINOv2 0.9828, EVA02 0.9807 on zoom alone) and gain at most +0.0006 alone at w = 0.25, which the ensemble does not
  keep. The pipeline stays at original + h-flip. (Test images were not used; whether zoom TTA would help on harder frames cannot be
  judged without them, and judging it from test statistics is what the TA ruling excludes.)

**Team decisions after the audit (2026-11-15, from the user).**
* HUD dropout stays: the user's reading is that the training-image occlusion study (round 19) and the training-image views decide
  it. The provenance (also prompted by HUD-less low-confidence test frames, round 19 section 5) stays recorded above and in a
  one-line note in the notebook's 7.4.
* The balanced-prior rule (alpha 0.75) stays. The notebook, README and this file now give the class-imbalance rationale: 39.4% of
  the training images are 7-Eleven Knights (others 10.1-18.8%), the slides warn "Watch for class imbalance in images", a
  cross-entropy posterior carries the training prior, macro-F1 weighs classes equally, and logit adjustment (Menon et al., ICLR
  2021) removes the prior; alpha fixed on training images (rounds 23 / 25). The historical reason for the switch (round 23: the
  class mix predicted on test images) is not rewritten; the notebook and README carry a one-line provenance note.
* v21 rolled back to v20 because v21 scored lower on the public leaderboard (0.85939 vs 0.85948; inside noise, see the v21 entry):
  `eva02B` removed from `T4_BACKBONES` / `T4_MEMBER_W`, the pipeline is eight members (DINOv2 x3) + 10% probe again. The EVA02
  caches stay in `outputs/preds` and `versions/v21_LB0.85939/` for reference.
* The three CPU checks the user queued for v22 (Task 2 KDE temperature, Task 1 standardised log-odds, Task 3 conditional zero
  gate) are the round 28 checks above (`ed_`, `ee_`, `ef_r28_*.py`): all failed; not re-run.

## Round 30 (2026-11-16, overnight, autonomous): error forensics on training data and new methods
Rules for this round (user): training data only (Task 5: its one file); one GPU job at a time, CPU experiments one at a time at
Idle priority next to the trainer; acceptance (user, 2026-11-16) = pooled nested OOF delta > 0, better in >= 3/5 folds, no fold below
-0.0005; Task 4: mean over the hard views (zoom70, HUD hidden, gamma 3.5, 75% grid) >= +0.0015 with full frames >= -0.0005. The
Task 1 "test-like" gate is not used (it is built from the test's missing-cell rates). No Kaggle upload, no commit.

**Task 1, where the v20 errors sit** (`eh_r30_t1_forensics.py`; level 2 rebuilt from the caches: OOF F2 0.8387 at cut-off 0.162,
recall 0.978, precision 0.535, 709 missed cheaters, 26,981 false alarms).
* No pocket: by missing cells the in-group F2 falls smoothly from 0.863 (0 missing, 1.8% of rows) to 0.823 (8+), recall stays
  0.974-0.987 everywhere; account age, level, reaction time, aim and win-rate quintiles all keep recall >= 0.95.
* The misses sit where cheaters are rare: top reports_received quintile (cheater rate 0.093) holds 53% of the misses at recall 0.764;
  top headshot / K-D quintiles (cheater rate 0.25) hold 26-33% of them. False alarms are spread evenly over the skill quintiles
  (18-20% each) and concentrate in middle account ages (FP rate 0.54-0.61 vs 0.22-0.29 at the extremes) and missing reports (28%).
* 64% of the misses lie within 0.5 log-odds of the cut-off; only 1.4% beyond 2. They are borderline cases of an F2 threshold that
  already trades 27k false alarms for 98% recall, not a missed sub-population. (The "errors predictable from the features" AUC of
  0.99 printed by the script is circular: an error is defined by the model's score, which is itself a function of the features.)

**Task 2, stacker vs KDE naive Bayes and an input-aware gate** (`ei_r30_t2_gate.py`).
* Blend confusions (OOF): Casual<->Grinder 8,311, Casual<->Social 5,809, Social<->Whale 4,020, Grinder<->Whale 2,958.
* The two disagree on only 3.7% of rows; there the stacker is right 43.2%, the KDE NB 44.1%, the blend 46.9%, neither 12.8%. Among
  rows where exactly one is right (3.25%), "the stacker is the right one" is 45-54% in every quintile of missing cells, clip-bound
  count, either entropy or the entropy gap; predictable from the six descriptors with AUC 0.551 only.
* Gated blend (mixture-of-experts gate w(x) = sigmoid(b . z(x)) on the six descriptors, L2-regularised likelihood, multipliers
  after; nested on four stacker splits): delta -0.0002 / -0.0003 / +0.0009 / -0.0001 (pooled +0.0001), folds better 2, 2, 4, 3 of 5,
  worst fold -0.0012 / -0.0014 / -0.0011 / -0.0019. The gate learns "more KDE when the stacker is unsure" (stacker entropy +0.30),
  but the signal is too weak to beat a constant weight. **Fails** (folds below -0.0005 on every split).

**Task 1, residual boosting from the v20 level 2** (`eh2_r30_t1_residual.py`; LightGBM with init_score = the level-2 log-odds and
the 31 raw columns + missing count as inputs, 150 trees at lr 0.02, 15 leaves, 300 rows per leaf, 3 seeds; cut-off from inner OOF):
nested F2 0.8385 -> 0.8378, pooled delta -0.0007 [-0.0012, -0.0002], 0/5 folds (-0.0015 / -0.0006 / -0.0003 / -0.0007 / -0.0006).
The raw features hold no signal the stack has not used; the extra trees only add variance. **Fails.** Together with the forensics:
Task 1's remaining error is label noise around an F2 cut-off, not a structure in the features.

**Task 3, where the absolute error sits** (`ej_r30_t3_forensics.py`; round 28 nested OOF, NMAE 0.79458).
* Whales (segment 3, 14% of rows) carry **79.3%** of the total |y - yhat| (mean |error| 15,350 THB); segments 0-2 carry 1-8% each.
  By size of y: rows with y = 0 carry 32.9% (the gate predicts 0 for 73% of them), y above the 90th percentile of positive spend 44.5%,
  the top 1% alone 18%.
* Identity-filled spending history (30% of rows) is not worse: NMAE in group 0.790 vs 0.797 complete, share of error in proportion.
* Whales almost never stop (y = 0 in 7.1%) and the pipeline never predicts 0 for them; which Whale stops is not predictable
  (AUC 0.575 from P(y > 0), 0.544 from the features + P(y > 0)). 87% of Whales spend more than their historical_spending this month
  (y / hist median 1.60, 90% 2.52, 99% 6.96); the pipeline's own ratio already ranks who surges (AUC 0.765) and the features add
  nothing (0.747).
* Residual L1 boosting from the pipeline prediction (LightGBM l1, init_score = final, features + final + P(y > 0), 200 trees lr 0.02,
  3 seeds; nested on the Task 3 folds): all rows 0.79440 (delta -0.00018, 0/5 folds); Whale rows only 0.79444 (-0.00014, 0/5).
  **Fails.** The Whale error is the spread of a heavy-tailed spend around a median the pipeline already estimates.

**Task 4, the v20 errors on held-out training images** (`ek_r30_t4_errors.py`; v20 members' OOF views from rounds 19-28 + probe,
v17 rule, training class mix; contact sheet `outputs/r30/t4_errors_sheet.png`).
* Errors: full frames 301 (0.95%), zoom70 711 (2.25%), HUD hidden 1,010 (3.20%). 81% of the full-frame errors are also wrong on both
  other views (222 wrong on all three). Recall per class on full frames 0.986-0.993; zoom70 0.970-0.984; HUD hidden 0.957-0.979.
* Most confused (true -> predicted): full: 7-11K -> Roblock 40, PUBE -> Roblock 30, FiveN -> 7-11K 25, FreeF -> PUBE 22; zoom70:
  7-11K -> Roblock 80, FreeF -> PUBE 63, PUBE -> Roblock 57; HUD hidden: FreeF <-> PUBE 95 + 90 (the two shooters look alike once the
  HUD is gone), 7-11K -> Roblock 80, PUBE -> Roblock 78. Roblock is the sink for ambiguous frames despite being the smallest class.
* What the misclassified frames look like: low contrast is the strongest image-level marker (AUC 0.35 for "error"; error rate 2.4% in
  the lowest-contrast decile vs 0.25% in the highest), then darkness (AUC 0.44; 2.2% in the darkest decile); saturation, edge density,
  black rotation corners, noise and flat-area share are near 0.5. In the contact sheet, most of the 40 errors are near-black frames
  with heavy noise, strong colour casts or inverted colours (pink / green / blue), large rotations with black borders, or close-ups of
  a featureless wall / ground; a few are menus or loading screens. Few show the HUD. Many carry no game-specific content at all, so a
  large part of the remaining error looks irreducible.
* The ensemble knows when it is wrong: confidence AUC 0.024 (error rate 9% in the least confident decile, 0.03% in the most);
  member agreement median 0.625 on errors vs 1.0 on correct frames.
* Implication for training methods: the errors are style errors (contrast, brightness, colour statistics) rather than content the
  members never saw, which is what feature-statistics mixing (MixStyle) targets; flat-minimum training (SAM) is the other candidate.

**Task 5, stability of the 2,148 flags under subsampling** (`el_r30_t5_stability.py`; 20 subsamples of 80% of the accounts without
replacement; gap scan + pocket detector re-run on each, pocket features fixed to the full-data scan's `accuracy_std_level`,
`kda_std_level`, n_perm 100). Full data reproduced: gap families 1,809 + pocket 339 = 2,148.
* Scaled group sizes: gap 1,806 +- 23, pocket 345 +- 47 per subsample.
* Gap-family accounts: 1,808 of 1,809 flagged in >= 95% of the subsamples that contain them (one account in < 50%).
* Pocket accounts: 77.6% flagged in >= 95% of their subsamples, 97.9% in >= 75%, all 339 in >= 50% (the cut-off k moves with the
  permutation null, so the pocket's edge is softer than the families').
* Unflagged accounts: none flagged in >= 75% of their subsamples; 2 in >= 50%.
* The flags are not noise: the families are deterministic gaps that re-appear in every subsample, and the pocket's core is stable.
  Nothing to change.

**Task 4, probability-combination rules, fold-0 screen under the hard-view criterion** (`eo_r30_t4_combine_screen.py`; the eight v20
members' fold-0 views from `outputs/r22_views`; v17 rule, uniform mix; temperatures fitted on folds 1-4 only: 0.43-0.55, all members
under-confident as found in round 23). Reference (arithmetic mean, DINOv2 x3): full 0.9903, zoom70 0.9759, HUD hidden 0.9674, gamma
3.5 0.9332, 75% grid 0.8808.

| rule | hard-view mean | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid |
|---|---|---|---|---|---|---|
| geometric mean (logit-space averaging) | -0.0017 | +0.0005 | +0.0003 | -0.0007 | -0.0040 | -0.0025 |
| power mean p = 0.5 | -0.0011 | +0.0001 | +0.0001 | +0.0001 | -0.0026 | -0.0017 |
| power mean p = 2 | +0.0008 | +0.0003 | -0.0004 | -0.0013 | +0.0027 | +0.0022 |
| temperature-calibrated geometric mean | -0.0085 | -0.0003 | -0.0000 | -0.0022 | -0.0088 | -0.0231 |
| temperature-calibrated arithmetic mean | -0.0038 | -0.0007 | +0.0004 | -0.0006 | -0.0043 | -0.0108 |

All fail the screen (+0.0015 needed). Averaging in log space or after calibration lets one confidently wrong member decide on
corrupted frames; the plain probability mean is the most robust combiner, as in rounds 18 and 25. Not re-checked on all folds.

**Task 1, the round 14 10-seed bag re-scored under the 2026-11-16 rule** (`en_r30_t1_bag10_recheck.py`; v20 level 2 = LogReg +
LightGBM + CatBoost meta-models, nested; five decimals; test-like part dropped, test.csv not read). The round 27 check had two folds
printed as -0.0005; thread counts differ here (2 instead of 6), so single folds move by up to 0.0003.

| split | pipeline members | 10-seed bag | delta | folds better | per fold |
|---|---|---|---|---|---|
| pipeline folds | 0.83859 | 0.83872 | +0.00014 | 3/5 | -0.00036 +0.00061 -0.00014 +0.00002 +0.00055 |
| split seed 1 | 0.83835 | 0.83853 | +0.00018 | 3/5 | -0.00012 +0.00009 -0.00024 +0.00063 +0.00057 |
| split seed 2 | 0.83806 | 0.83849 | +0.00043 | 5/5 | +0.00030 +0.00014 +0.00097 +0.00027 +0.00048 |
| split seed 3 | 0.83841 | 0.83859 | +0.00019 | 4/5 | +0.00017 +0.00072 -0.00020 +0.00020 +0.00006 |

Pooled mean +0.00023; worst fold -0.00036. **Passes the 2026-11-16 rule** (it failed the round 27 rule: 4/5 folds on every split).
Caveat: it is a re-test of a candidate discarded before the rule was relaxed; it is pure variance reduction (no tuned quantity),
which keeps the selection risk small, and the expected effect is about +0.00005 on the public score.
* Pipeline (v22 candidate): `nb_part2.py` replaces the eight stochastic Task 1 members by their 10-seed log-odds means (OOF, fold-model
  test and full-data refits; caches `outputs/preds/t1_bag10_<member>(_full).npz`, written from `outputs/r14_bag`, recomputed by the
  notebook if missing), the level-2 caches are renamed `t1_l2_trees_bag10` / `t1_l2_trees_full_bag10`, and the 7.1 text describes it.
* **v22 = 0.85954, +0.00006 over v20** (uploaded 2026-11-16 01:58 via the Kaggle CLI under the user's overnight policy; snapshot
  `versions/v22_LB0.85954/`). Only Task 1 changed (96 labels), so about +0.0003 on Task 1, close to the nested +0.00023. Notebooks
  re-executed (master and `task1`; the others unchanged); OOF F2 0.8387 -> 0.8388 at cut-off 0.163.

**Task 3, a multiplicative scale per segment** (`ep_r30_t3_segment_scale.py`; grid 0.80-1.20, L1-best on 4 folds, applied to the
5th; input = the round 28 nested OOF). In-sample best scales: 1.00 for every segment except segment 0 (1.04, a segment whose mean
y is 74 THB). Nested: every segment -0.00004 (2/5 folds); Whales only 0.00000 (the chosen scale is 1.00 in every fold). **Fails**:
the pipeline's global scale is already L1-calibrated within each segment.

**Task 2, pairwise specialist for Casual <-> Grinder** (`eq_r30_t2_pair_specialist.py`; pairwise coupling, Hastie & Tibshirani
1998: a LightGBM trained on Casual and Grinder rows only, inputs = raw features + the blend's four probabilities, re-splits the
blend's C + G mass; its OOF on the training folds from an inner 4-fold CV so the multipliers see honest probabilities; pipeline
folds). Nested macro-F1 0.7573 -> 0.7561, delta -0.0012, 1/5 folds (-0.0007 -0.0025 -0.0008 +0.0002 -0.0020); accuracy among true
Casual / Grinder rows 0.8082 -> 0.8073. **Fails.** A specialist on the same features learns the same boundary with less data; the
pair is separated by label noise, not by a boundary the four-class models missed (consistent with round 19: calibration is exact).

### Round 30, "Jackpot" CPU loop (user brief of 2026-11-16 02:10; base = v22)

**Task 1, logit-consensus meta-features for the tree meta-models** (`es_r30_t1_l2_consensus.py`; max - min and SD of the nine
member log-odds, mean(24-column) - mean(49-feature), missing-cell count added to the LightGBM / CatBoost meta-models; nested, 4
splits): delta -0.00084 / +0.00012 / -0.00031 / -0.00016 (pooled -0.00030), folds better 0, 3, 2, 1 of 5, worst fold -0.00191.
**Fails.** The meta-models already see the nine log-odds whose spread these features summarise; the extra inputs only add variance.

**Task 3, median bias per segment x identity-repaired group** (`et_r30_t3_group_bias.py`; 10 groups; L1-fitted on 4 folds):
multiplicative (grid 0.80-1.20) -0.00004, 2/5 folds; additive shift on positive predictions -0.00003, 0/5. **Fails.** The
in-sample factors scatter around 1 (0.80-1.10 in the small groups, 1.00 for Whales), i.e. noise, not a systematic back-transform bias.

**Task 5, purity of the pocket's weakest flags** (`eu_r30_t5_pocket_purity.py`; 5 permutation seeds, n_perm 300). k* = 339 under
every seed (SD 0); estimated pocket size 199-201, precision of the 339 flags 0.57 (about 194 estimated members). The last 20 / 30 /
50 flags have estimated marginal precision 0.28 / 0.33 / 0.30, three times the F3 break-even of F / (1 + beta^2) = 0.099 at this
operating point, so dropping them lowers the expected F3 (-0.0018 / -0.0035 / -0.0049). The expected-F3 curve peaks sharply at 339
(0.9883 at 319, 0.9892 at 359). Keep all 339: under F3 a 30%-pure edge still pays.

**Task 2, attribute-weighted KDE naive Bayes and a joint 2-D density** (`ev_r30_t2_kde_variants.py`; per-feature class
log-likelihoods rebuilt out of fold, reproducing the cached KDE NB to 2e-14; candidates replace the KDE NB in the v20 blend, KDE
weight and multipliers nested; 4 stacker splits).
* Weighted NB (w_j in [0, 2], maximum likelihood with a small pull to 1; Zaidi et al. 2013): per split -0.0002 / +0.0003 / +0.0006 /
  -0.0008 (pooled -0.0000), folds better 4, 3, 3, 2 of 5, worst folds -0.0020 / -0.0008 / -0.0003 / -0.0033. The fitted weights
  are all 0.78-0.98 (lowest: item_type_preference_cosmetic, avg_match_length, account_age_days, spending_frequency,
  total_spending_thb): a near-uniform shrinkage of the log-likelihoods, i.e. a temperature, which round 28 already found useless.
* Joint 2-D KDE of avg_monthly_spending x spending_frequency (85% of rows; the rest keep the marginals): log-loss 0.6251 -> 0.6257;
  pooled -0.0001, folds better 1, 2, 4, 1 of 5. The within-Whale intensity factor spreads over seven columns; one pair does not
  capture it, and for the other classes the pair is independent.
* **Both fail.**

### Round 30, Task 3 brief (user, 2026-11-16 02:45; acceptance: pooled > 0, >= 3/5 folds, no fold below -0.0002)
`ex_r30_t3_readout_blend.py`. Everything nested (read-outs, blend, scale on 4 folds): reference 0.79457 (the pipeline-style rebuild
with read-outs tuned on all rows: 0.79458).

| item | nested NMAE | pooled delta | per fold | verdict |
|---|---|---|---|---|
| 1a empirical-quantile mixture median, replacing the log-normal one | 0.79458 | +0.0000038 | +0.0000007 +0.0000132 -0.0000027 +0.0000139 -0.0000056 | passes the letter of the rule; no effect |
| 1b the same, added to the pool | 0.79458 | +0.0000038 | (same) | passes the letter of the rule; no effect |
| 2 segment-conditional L1 blend, shrunk to the global weights (lambda by inner CV: 0.5, 0.5, 0.25, 0.5, 0.25) | 0.79460 | +0.0000295 | +0.0001238 +0.0001427 -0.0002046 +0.0001317 -0.0000447 | **fails** (fold 2 below -0.0002) |

* Item 1: the empirical read-out (residual quantiles of log1p(y) - mu by imputed segment x mu tercile, q = 1 - 0.5 / p') moves
  individual predictions but the exact L1 blend gives the read-out members almost no weight: the raw-THB L1 models carry the blend,
  so the shape of the stage-2 residual distribution barely reaches the final prediction. +0.000004 NMAE is about +0.000001 on the
  public score; not worth a pipeline change.
* Item 2: three folds gain +0.00012-0.00014, one loses 0.00020 and one 0.00004; the per-segment weights are noisy (the Whale group
  dominates every fit) and shrinkage only halves the noise. Narrow fail.
* Item 3: after the identity repair 45 training rows (0.04%) lack both historical_spending and prev_month_spending (the test file
  was not inspected). Their mean |error| is 8,801 THB (all rows 2,714) but they hold 0.14% of the total error: removing all of it
  would raise NMAE by at most +0.00023. An in-fold L1 LightGBM imputes log1p(historical_spending) to a median factor of 1.03 where
  it is known, but mostly through total_transactions x avg_transaction_value, which are missing on exactly those rows (that is why
  the identity failed). Retraining every ratio member for 45 rows was not done; the ceiling is below the noise.

`ez_r30_t3_hurdle_dual.py` (same nested reference, 0.79457):
* **Item 4, within-segment hurdle with slowdown features** (prev / (hist + 1) and its distance outside [0.30, 0.40], purchases per
  account day, within-segment robust z-scores of recency / activity columns; one LightGBM P(y > 0) per imputed segment, tuned
  stage-1 parameters): log-loss 0.5522 (10-seed hurdle) -> 0.5531, worse in every segment (Whales 0.2595 -> 0.2602). The nested
  blend weight is 0 in four folds and 0.25 in one; NMAE delta -0.00001, 2/5 folds. **Fails.** Whales that stop (y = 0) still cost
  56,672 THB each on average, and no feature here (or in ej_r30, AUC 0.54-0.58) tells them apart in advance.
* **Item 5, dual-anchor ratio stage 2** (target log1p(y) - 0.6 log1p(hist) - 0.4 log1p(prev); LightGBM tuned + CatBoost, one seed):
  median |log1p(y) - mu| on spenders 0.1147 / 0.1140 vs 0.1148 for ratio[hist] LGB tuned, i.e. no better anchor (hist and prev move
  together, prev ~ 0.35 hist, so the geometric mix carries the same information). Added to the pool: 0.79455, delta -0.00003,
  3/5 folds (fold 0 -0.00017). **Fails.**
* Task 3 summary after the brief: the read-out shape, segment-specific blends, stage-1 refinements and new anchors all land within
  +-0.00003 of the reference. Task 3 sits at its ceiling for these members; what remains is the irreducible spread of Whale spend.

**Task 1, teacher-student self-training on the 13,184 unlabelled training rows** (`ew_r30_t1_teacher_student.py`; teacher = the
pipeline's LightGBM member on the 4 training folds (its OOF equals the cached member, corr 1.0000); pseudo-labels p > 0.92 -> 1,
p < 0.03 -> 0 on the unlabelled rows: about 1,340 cheaters + 3,460 legitimate per fold (37% of them); student = the same LightGBM
on training folds + pseudo-labelled rows). Student alone vs teacher alone (in-fold cut-offs): +0.0002 / -0.0006 / +0.0003 /
-0.0001 / +0.0001, i.e. no change; corr(student, teacher) 0.994. As a tenth level-2 member (v22 base, nested, 4 splits): delta
-0.00045 / -0.00049 / -0.00022 / -0.00020 (pooled -0.00034), folds better 0, 1, 2, 2 of 5. **Fails.** The confident pseudo-labels
are rows the teacher already gets right, so they add no information near the F2 cut-off; the unlabelled rows look like the labelled
ones (same generator), which is the case where self-training cannot help.

**Round 29, DINOv2 ViT-B/14 at 252x476 (612 patches), fold 0** (`train_task4.py --tag dinov2_r2 --h 252 --w 476 --lr 4e-5
--layer-decay 0.8 --warmup-epochs 2 --epochs 15 --p-hud 0.5 --grad-ckpt`, bs 64, ~980 s per epoch with CPU work alongside; the run
was restarted after the page-file change). Learning curve identical to dinov2_v2 (fold-0 val F1 0.9864 vs 0.9867). Evaluation
(`cv_r22_t4_eval.py dinov2_r2`, v20 base):
* Alone: full 0.9875, zoom70 0.9661, HUD hidden 0.9556, gamma 3.5 0.9407, 75% grid **0.9047** (v20 members together 0.8588).
* Old three-view criterion (with the probe, uniform mix): +0.0001 [-0.0006, +0.0008] added; in place of dinov2_v2 -0.0000;
  alongside at weight 3 +0.0005 [-0.0005, +0.0015]: would have failed.
* Hard-view criterion of 2026-11-16 (members only, rule alpha 0.75, uniform mix): added at weight 1 +0.0043 [+0.0030, +0.0057]
  (full +0.0001; zoom70 -0.0007, HUD hidden +0.0002, gamma 3.5 +0.0017, 75% grid +0.0160); in place of dinov2_v2 +0.0077 (full
  -0.0004; zoom70 -0.0019, HUD -0.0015, gamma -0.0036, grid +0.0379); alongside, both at weight 3, +0.0086 (full +0.0001; zoom70
  -0.0012, HUD +0.0009, gamma +0.0038, grid +0.0308). **Passes**, so folds 1-4 are trained (user rule: a pass goes to all folds).
* Caveat for the decision after all folds: the gain sits almost entirely in the 75% grid view, where twice the resolution gives
  each visible grid cell four times the patches; zoom70 and HUD hidden do not improve. Whether the grid view stands for the test
  frames is an assumption of the criterion, not something the training data can show.

**Task 5, mutual k-NN graph components** (`fa_r30_t5_knn_graph.py`; k = 15 on the 70 robust-scaled detector columns; null = the
same graph on column-shuffled data, 3 draws). Real data: one giant component of 20,894 accounts (3% of them gap-family accounts),
then components of 478, 286, 197 and 29 accounts that are **100% gap-family accounts** (together 55% of all 1,809), and 3,545
fragments of 1-4 accounts (3,951 accounts, 5% gap-family, the usual fringe of a mutual k-NN graph). No component of 5-150 accounts
lies outside the flags (the only one of 20-150 is a takeover sub-family: currency_transfers and location_changes deviations at
period 4, robust z 14 and 11). The shuffled data give 3-5 components of 20-150 and about 30 of 5-19, i.e. more mid-sized fragments
than the real data, so such fragments are not evidence of a group by themselves. The graph re-discovers the planted families from
all columns at once and finds nothing new. Task 5 stays at 2,148 flags.

**Task 3, calibration of the stage-1 P(y > 0)** (`fb_r30_t3_p_calibration.py`; fitted on 4 folds of the 10-seed hurdle's OOF p,
whole pipeline rebuilt nested with the calibrated p): isotonic +0.00001 (2/5 folds), Platt -0.00000 (2/5), isotonic per imputed
segment **+0.00002, 4/5 folds, worst fold -0.00003** (log-loss 0.5522 -> 0.5532, 0.39% of rows cross 0.5). The per-segment variant
passes the letter of the Task 3 rule, but it is one of three variants tried and +0.00002 NMAE is about +0.000004 on the public
score, inside the selection noise. Recorded as a v23 candidate for the user's decision; not merged.
**Decision (user, 2026-11-16 morning): discarded** (noise-level gain; step-function risk of isotonic calibration).

**Task 4, member weights chosen nested on the hard views** (`er_r30_t4_weights_hard.py`; gamma 3.5 / 75% grid views of all eight
members on all folds written by `em_r30_t4_hardviews.py` to `outputs/r30_views`; v20 members only: full 0.9895, zoom70 0.9754,
HUD hidden 0.9639, gamma 3.5 0.9356, 75% grid 0.8836 at a uniform mix under the rule).
* First grid, DINOv2 {2..6} x HUD-dropout members {1, 1.5, 2}: nested picks the corner (6, 2) in every fold; hard mean +0.0012,
  5/5 folds (+0.0008 to +0.0014), full -0.0001.
* Widened grid, DINOv2 {3, 4, 5, 6, 8, 10} x HUD members {1, 1.5, 2, 3}: picks (10, 3) twice and (8, 3) three times; **hard mean
  +0.0014, better in 5/5 folds (+0.0008 +0.0018 +0.0015 +0.0014 +0.0016)**, full -0.0001; per view zoom70 +0.0006, HUD hidden +0.0002,
  gamma 3.5 +0.0014, 75% grid +0.0033. With the 10% probe on its three views (weights 8 / 3): full -0.0002, zoom70 +0.0006, HUD hidden
  +0.0000.
* The direction is consistent: the two plain members (ConvNeXt-T, DeiT-III-S, no HUD dropout) cost robustness on every hard view;
  at (8, 3) they carry 8% of the weight instead of 20%. Passes the generic rule (pooled > 0, >= 3/5 folds, no fold below -0.0005)
  on the hard-view metric; below the +0.0015 "train all folds" bar, which concerns new models.
* **v23 = v22 + these weights** (`T4_MEMBER_W = {dinov2_v2: 8, five HUD members: 3}`), built under the user's overnight policy.
* **v23 = 0.85961, +0.00007 over v22** (uploaded 2026-11-16 05:36 via the Kaggle CLI; snapshot `versions/v23_LB0.85961/`). Only
  Task 4 changed (113 labels), so about +0.00035 on Task 4. Training-data view: full frames -0.0002 under the rule, hard views +0.0014;
  the leaderboard moved with the hard views, as it did for the probe and DINOv2.

### Decisions after v23 (user, 2026-11-16 morning)
* Tasks 1, 2, 3 and 5 stay at their v23 state; the Task 3 per-segment isotonic calibration is discarded.
* DINOv2 252x476 (`dinov2_r2`) keeps training on folds 1-4. Before a full fit and a v24 it is evaluated with v23 as the base, each
  view reported separately: full, zoom70, HUD hidden, gamma 3.5, 75% grid, plus the dark / low-contrast training frames that the
  round 30 forensics singled out. The gain has to reach beyond the 75% grid.
* Next Task 4 fold-0 tests after `dinov2_r2` (126x238, a quarter of the pixels): DINOv2-Large
  (`vit_large_patch14_reg4_dinov2.lvd142m`, lr 3e-5, layer decay 0.85, 15 epochs, HUD dropout 0.5, gradient checkpointing), and
  DINOv2-Base with photometric robustness (MixStyle, or more grey / colour jitter) aimed at the dark, colour-cast and low-contrast
  frames.

## Round 31 (2026-11-16): v23-based Task 4 evaluation and the GPU queue after `dinov2_r2`
* `fc_r31_t4_v23_eval.py cand ...`: scores a candidate against v23 (eight members, DINOv2 x8, HUD-dropout members x3, plain members
  x1; rule, uniform mix, members only) on each view separately and on the **real dark / low-contrast training frames** (lowest
  decile of grey-level contrast or of brightness, from `outputs/r30/t4_errors.npz`; 4,159 frames = 13.2%; v23 scores 0.9746 there
  against 0.9894 on all full frames). Variants: added at weight 2 / 4 / 8, in place of dinov2_v2, and (dinov2_v2, candidate)
  weights in {0, 4, 8} x {0, 2, 4, 8} chosen nested. Works on the folds a candidate has (fold-0 tests).
* **v24 check, fixed before the all-fold run of `dinov2_r2`:** pooled delta > 0 on zoom70, on HUD hidden and on the dark /
  low-contrast frames, full >= -0.0005, on top of the hard-view rule; the queue uses the nested-weight variant for the full-refit
  decision. v23 base numbers: full 0.9894, zoom70 0.9760, HUD hidden 0.9643, gamma 3.5 0.9362, 75% grid 0.8880.
* `em_r30_t4_hardviews.py`: candidates without a checkpoint for some folds get views for the folds they have, scoring left to
  `fc_r31`; ARCH gains `dinov2L` and `dinov2_photo`.
* MixStyle in `train_fold` hooks `model.stages` (CNNs only), so the photometric DINOv2-B run (`dinov2_photo`) uses the other option
  of the brief: greyscale 0.2, colour jitter 0.8 (dinov2_v2 recipe otherwise). DINOv2-L (303M parameters): weights, gradients,
  AdamW states and the EMA copy take about 5.65 GiB in fp32 before activations, so a smoke test (bs 64, gradient checkpointing)
  must show peak reserved memory below 7.3 GiB before the fold-0 run starts.
* Queue (one GPU job at a time): wait for `dinov2_r2` folds 1-4 -> views + v23-based scores -> full refit only if the nested
  variant passes both checks -> smoke test -> DINOv2-L fold 0 (if it fits) -> DINOv2-B photometric fold 0 -> their views and scores.

## Round 31, Task 5 (user brief of 2026-11-16 morning; label-free, Task 5 has only test.csv; CPU at idle priority, 4 threads)
Reference data for item 3: the first release of test.csv (2026-11-03, git commit 99ed4b9), whose anomalies were the id block
P023819-P025888 (2,070 accounts, 259 of them the volatility type outside the gap families). Used only to check a method after it
was chosen label-free; nothing is tuned on it.

**Item 3, shape and cut-off of the dependence pocket** (`ff_r31_t5_pocket_density.py`, `ff2_r31_t5_pocket3d.py`). Scores on the
accounts outside the gap families, in rank space; density ratio = cross-fitted smoothed-histogram ratio in normal-score space
(each account scored by the fit on the other random half; 20 repeats), i.e. the copula density, whose level sets are the
Neyman-Pearson regions against the independence null; 1 - 1/r is the local posterior (Efron's local fdr). Expected F3 counts the
gap families as true, with one common group size G for all scores.
| score (competition data unless marked) | k* | est. precision | expected F3 | first release: F3 vs id block (k*) |
|---|---|---|---|---|
| min-rank, 2 columns (pipeline) | 339 | 0.57 | 0.9785 (G 226) | 0.9830 (k* 323; oracle k 0.9833) |
| Fisher / Stouffer, 2 columns | 247 / 307 | 0.55 / 0.56 | 0.940 / 0.954 (G 261) | 0.9660 / 0.9778 |
| density ratio, 2 columns, h 0.15 / 0.25 / 0.40 | 236 / 301 / 360 | 0.73 / 0.59 / 0.53 | 0.958 / 0.958 / 0.962 (G 261) | 0.9719 / 0.9779 / 0.9783 |
| min-rank, 3 columns (+ performance_stability) | 490 | 0.46 | 0.971 (G 261) | 0.9785 |
| **density ratio, 3 columns, h 0.40** | **263** | **0.85** | **0.9965 (G 226)** | **0.9889** (k* 279; oracle 0.9897) |
* The pocket is a 3-column pattern: among the top 263 the mean rank is kda_std 0.98, accuracy_std 0.97, performance_stability
  0.87 and 0.47-0.53 for every other *_level column; the first release's 259 true members show the same three (0.95 / 0.94 / 0.84).
  The pipeline's pair scan picked only the first two, and its square corner then admits accounts that are high in both by chance.
* Null check: outside the pairwise upper corners the three columns have Spearman -0.06 to -0.07, the value independent uniforms
  give with the same cut (-0.056); among the first release's true normals -0.003 / -0.000 / -0.004. Independence holds.
* In 2 columns the density ratio does not beat min-rank (the corner shape fits there); the gain comes from the third column read
  through the density ratio (min-rank in 3 columns is worse).
* Bandwidth: h 0.30-0.60 gives k* 263-267, expected F3 0.9957-0.9980; first release 0.9882-0.9892. Cross-fitting seeds: k*
  260-268, Jaccard of the flagged sets 0.970-0.996 (first release 0.853-0.970). 80% subsamples: k* 274 +- 30 (min-rank 370 +- 41).
* Against the pipeline's 339: keeps 246, drops 93, adds 17 (263 flags; total 2,072 instead of 2,148). On the first release the same
  operation drops 84 accounts of which 1 is anomalous and adds 24 of which 8 are: precision of the pocket 0.71 -> 0.85.
* The posterior rule (flag where 1 - 1/r > F3 / 10) flags 526-581 accounts and is worse (expected 0.972; first release 0.9796): the
  smoothed ratio spreads the pocket's mass into the neighbouring normal region, so its level is biased up near the pocket; the
  expected-F3 cut-off is the better rule. Bootstrap uncertainty of min-rank's k*: 382 +- 67, flat expected F3 between 340 and 450.
* Size of the gain: +0.006 F3 on the first release (~+0.0012 on the public score if the re-release behaves the same); label-free
  +0.018 expected F3 on the competition data. Not merged: Task 5 is locked at v23 until the user decides.

**Item 1, scans inside yes/no-flag subpopulations and 3-way tail cubes** (`fd_r31_t5_subpop_scan.py`; U = the 23,741 unflagged
accounts; null for a subpopulation S = random subsets of U of the same size, because outside the families the flags are
independent of the features, round 23).
| subpopulation of U | n | B1 1-D gaps: max lambda in S (null max over columns: median / max) | B2 pockets in S: max z / shuffle noise | B3 S vs rest: max z (null median / max) |
|---|---|---|---|---|
| password_changed | 1,100 | 32.1 (24.7 / 32.2) | 5.32 / 5.15 | 6.12 (6.23 / 7.32) |
| vpn_usage | 3,443 | 82.9 (75.1 / 143.7) | 4.85 / 4.79 | 6.34 (6.22 / 6.65) |
| suspicious_login_time | 1,145 | 16.4 (28.7 / 44.7) | 5.48 / 5.17 | 5.55 (6.19 / 6.62) |
| mass_item_sale | 469 | 14.8 (16.0 / 30.5) | 5.17 / 5.12 | 6.58 (6.45 / 8.03) |
| new device in period 4 | 5,417 | 20.4 (128 / 172) | 5.99 / 6.15 | 6.16 (6.14 / 6.62) without device_count columns |
| more guild changes in period 4 than before | 5,353 | 20.4 (125 / 188) | 9.96 / 12.01 | 6.18 (6.35 / 6.72) without guild_changes columns |
* Columns where S's largest gap beats the maximum of its own column's 20 null draws: 20, 14, 13, 8, 13, 9 (expected by chance
  259 / 21 = 12.3 each). The B2 exceedances are 0.06-0.33 z above a 3-shuffle noise level, cells of 15-30 accounts with 10-17 in
  excess; the column-shuffle noise ignores the natural dependence of the real columns, so these are not evidence.
* The first B3 run gave z 19.3 for the guild subgroup on `guild_changes_burst`: the subgroup is defined by guild_changes_4, so its
  own derived columns separate it by construction; without them it is at the null (6.18 vs 6.35 / 6.72). The new-device subgroup
  is at the null with or without device_count columns. (The large null gap maxima of these two rows come from the same defining
  columns, which random subsets do not condition on.)
* 3-way cubes: 56 features (level and period-4 deviation of each behaviour, 2 geo) in 4 categories, 10,144 cross-category triples
  x 8 top / bottom decile corners. Independence z max 4.54 (column-shuffled data 5.13); Kirkwood z (pairwise dependence kept) max
  3.83 (Gaussian copula 4.03). Nothing above the noise.
* No hidden group inside any flag subpopulation and no 3-condition group.

**Item 4, trajectory shape and change points** (`fg_r31_t5_trajectory.py`; 24 continuous behaviours as pooled normal scores,
period medians removed; null = each account's four periods in random order, 10 draws).
| statistic | beyond the null's 99% (real / null) | 99.9% | 99.99% | upper-tail gap lambda (real / null max) |
|---|---|---|---|---|
| accel4 = sudden period-4 change after steady 1-3 | 61 / 237.5 | 5 / 23.8 | 1 / 2.4 | 0 / 27.8 |
| CUSUM break, max over t = 2, 3, 4 | 349 / 237.5 | 27 / 23.8 | 3 / 2.4 | 0 / 0 |
| break at period 3 / period 4 | 247 / 49 vs 237.5 | 23 / 9 vs 23.8 | 4 / 2 vs 2.4 | 0 / 40.6, 0 / 29.1 |
| Mahalanobis of the break vector, t = 3 / t = 4 | 225 / 11 vs 237.5 | 21 / 6 vs 23.8 | 4 / 2 vs 2.4 | 0 / 32.4, 0 / 13.5 |
| direction: steady drift (cos) / ping-pong (-cos) | 85 / 280 vs 237.5 | 4 / 28 vs 23.8 | 0 / 2 vs 2.4 | 0 / 25.3, 15.3 / 16.0 |
* No excess of period-3 or period-4 breaks; the break period of the accounts beyond the null's 99.9% is 15 / 5 / 7 (t = 2 / 3 / 4)
  against 9.0 / 4.2 / 10.6. The top 200 by Mahalanobis distance share no direction (mean pairwise cosine +0.014 at t = 3 and t = 4;
  null +0.002 / +0.021). The small break_max excess at the 99% level sits at t = 2 (period 1 odd), not where a takeover would be.
* Period 4 has a strong deficit of breaks (61 vs 237.5 for accel4): the unflagged accounts change less into period 4 than the
  exchangeable null implies, which makes the period-permuted null conservative for period 4. Step-size check: the median
  |step| in normal scores is 0.351 (1->2), 0.326 (2->3), 0.275 (3->4), smaller into period 4 for 23 of 24 behaviours, and the
  same in all 25,889 accounts (0.344 / 0.321 / 0.272): a property of how the data were generated, not of the flag selection.
* Power: 150 copied accounts shifted by d within-account SDs in 6 of the 24 behaviours: from period 3, d = 1 / 1.5 / 2 put 2 / 5 /
  14 of 150 beyond the null's 99.9% of the Mahalanobis statistic; from period 4, 0 / 1 / 1. Omnibus 24-dimensional statistics
  dilute a 6-behaviour shift; a strong shift in one behaviour is what the 1-D gap scans already catch. So item 4 rules out a
  coherent change-point group, not a weak one.

**Item 2, feature-wise conditional prediction residuals** (`fe_r31_t5_conditional_resid.py`, `fe2_r31_t5_resid_selfnull.py`,
`fe3_r31_t5_resid_surplus.py`; ALSO-style, Paulheim & Meusel 2015: each period-4 behaviour predicted from periods 1-3 of itself,
all periods of the other behaviours, statics and flags; LightGBM Huber, 5-fold out of fold on U; residuals in MAD units).
* The Gaussian-copula null does not calibrate residual tails: real residuals are heavy-tailed (non-linear, heteroscedastic,
  zero-inflated columns), copula residuals near-Gaussian, so 4,119 real accounts exceed the copula's 99.9% quantile of S2 (24
  expected). Not evidence of anything; discarded as a null.
* Self-null (the same models predicting period 3 instead of 4): the period-4 tails are heavier: S2 beyond the period-3 99.9%
  quantile 106 vs 24, |r| > 6 MAD 1,549 vs 776, largest surpluses in purchase_amount_mean, friend_adds / removes, skill_trajectory,
  location_changes. **The first release shows the same surplus (86 vs 24; 1,591 vs 798) and 85 of its 86 surplus accounts are
  normal by the id block.** It is the calmer period 4 (item 4: steps into period 4 are 16-22% smaller), which narrows the residual
  core so the same surprises are larger in MAD units, not a hidden group. On the competition data the surplus accounts share no
  direction (mean pairwise cosine +0.060; period-3 tail +0.082) and have normal flag rates except suspicious_login_time 0.109 vs
  0.051 (about 6 accounts in excess; 0.012 on the first release).
* Upper-tail gaps of the residual scores: largest lambda 28.6 (kda_std), 22.2 (purchase_frequency), 19.0 (economy+), below the
  gap families' lambda >= 50 and within the null maxima of item 4 (13-41).
* Power: 150 copied accounts with purchase_amount_mean_4 and currency_transfers_4 raised to their 85-95th percentile (not extreme
  in either column): 74 of 150 land in the top 1% of the economy residual, 43 in the top 1% of S2. The method can see a
  context-inconsistent group of that size; none is there.
* On the first release the pipeline leaves 24 true anomalies among the unflagged accounts; the residual tails catch 1 of them.

**Round 31 Task 5 summary.** Items 1, 2 and 4 find no group outside the 2,148 flags (each with a null and, for 2 and 4, a power
check). Item 3 finds a better reading of the existing pocket: the cross-fitted 3-column density ratio (accuracy_std,
kda_std, performance_stability) with the expected-F3 cut-off, 263 flags instead of 339, +0.006 F3 on the first release's id block
and +0.018 expected F3 (label-free) on the competition data. Task 5 stays at v23 until the user decides.

### v24 (user decision, 2026-11-16 ~10:30): the 3-column density-ratio pocket in the pipeline
The user approved the item 3 change for Task 5; Tasks 1-4 stay identical to v23. `src/task5.py`:
* `pocket_extra_column`: label-free choice of the third column. Candidates: the other all-period `*_level` columns (more than 50
  distinct values, behaviours other than the pocket's). For each, the provisional (min-rank) pocket's mean rank against what its
  two pocket columns predict (least squares on the ranks, fitted on the other accounts outside the families), z = mean residual /
  (residual SD / sqrt(pocket size)); a column joins when |z| clears the two-sided Bonferroni bound for alpha = 0.001 (4.06 for 20
  candidates); at most one (the ratio is estimated in at most three dimensions).
  Competition data: performance_stability_level z 16.4 (mean rank 0.751 vs 0.495 predicted); every other column |z| <= 1.6.
  First release (reference): the same column, z 16.2; every other |z| <= 1.9.
* `DensityRatio` / `pocket_density_ratio`: the ff_r31 estimator (bins 0.1 SD, h = 0.40, 20 cross-fitting repeats) with the shuffle
  null of `pocket_detector` (300 draws, null points scored with a random half's fit).
* `pocket_refine`: min-rank and density ratio compared at one common group size (the larger estimate); the higher expected F3 is
  kept, cut at its own maximum. Without a further column the min-rank pocket is returned unchanged.
Pipeline result (seed 42): min-rank group 201, k 339, expected F3 0.9788; density ratio group 227, **k 288**, est. precision
0.787, expected F3 0.9970 -> density ratio. Kept 252 of the 339, dropped 87, added 36; **2,097 flags** (1,809 + 288). First
release (reference only): k 282, F3 against the id block 0.9827 -> 0.9892.
Seed sensitivity of the cross-fitting (seeds 42, 0-6): k* 288, 263, 265, 286, 265, 265, 272, 264; Jaccard 0.913-1.000; 263 accounts
in every seed, 292 in any; first-release F3 0.9887-0.9893 over the same seeds. The borderline accounts sit where the expected-F3
curve is flat; the pipeline keeps its fixed seed 42 (no seed chosen by outcome).
The Task 5 notebook was executed alone (master notebook deferred until the GPU queue ends, it runs Grad-CAM on the GPU). An empty
CUDA_VISIBLE_DEVICES does not survive on Windows, so the small autoencoder detector still ran on the GPU next to dinov2_r2 training
(no error; training continued). The autoencoder only enters the ensemble cross-check, not the flags (families + pocket).
`submission_v24.csv`: Tasks 1-4 byte-identical to v23, Task 5 2,097 flags (2,061 shared with v23, 87 only in v23, 36 only in v24),
identical to the standalone check. **Uploaded once with the user's approval (11:59): public LB 0.86135, +0.00174 over v23**
(about +0.009 F3 on Task 5, above the +0.006 seen on the first release's id block). Snapshot `versions/v24_LB0.86135/`;
`submission.csv` = v24.
After the upload (user request): `task5.score_autoencoder` always trains on the CPU (the `device` argument is ignored) and the
notebook no longer passes the GPU. Task 5 notebook re-executed: autoencoder 42.7 s on the CPU, `task5.csv` byte-identical to v24.

## Round 32 (2026-11-16 afternoon): Task 5 deeper on the CPU (user brief after v24 = 0.86135)
v24 committed and merged (`dabcaa8`, PR #6). At about 13:50 Claude Code stopped the background dinov2_r2 training (fold 3, epoch 4;
folds 0-2 done, their predictions saved) and the round-31 GPU queue because the system ran critically low on memory; not
restarted pending the user.

**Item 3, Monte Carlo variance of the pocket cut-off** (`fh_r32_t5_stable_cutoff.py`). `task5.density_ratio_curve` streams the
cross-fitting halvings (one pair of maps at a time; null rows pre-drawn as flat bin indices), so 200 halvings and 1,000 null draws
run in about 6 minutes and 240 MB. Agreement with the old class at 200 halvings: Spearman 0.993 (the rest is Monte Carlo noise in
near-tied bulk scores), top-280 overlap 279. Five master seeds (42, 1-4) per setting:

| setting | competition data: k* | Jaccard min / mean | in every / any seed | expected F3 | first release: k* | Jaccard min / mean | reference F3 (mean) |
|---|---|---|---|---|---|---|---|
| 20 halvings, 300 null draws (v24) | 263, 263, 265, 264, 259 | 0.977 / 0.989 | 259 / 265 | 0.9964-0.9983 | 294, 313, 291, 293, 262 | 0.837 / 0.920 | 0.9878-0.9893 (0.9887) |
| 200, 300 | 260, 265, 265, 265, 265 | 0.981 / 0.993 | 260 / 265 | 0.9980-0.9981 | 291, 298, 299, 291, 305 | 0.954 / 0.975 | 0.9886-0.9893 (0.9890) |
| **200, 1000** | **265, 265, 263, 265, 265** | **0.9925 / 0.997** | 263 / 265 | 0.9978-0.9981 | 287, 297, 295, 296, 293 | 0.966 / 0.982 | 0.9890-0.9895 (0.9892) |

* 200 / 1000 is the only setting with mean Jaccard above the user's 0.995 target on the competition data (the minimum, 0.9925, is
  263 vs 265 accounts). It is now the default of `density_ratio_curve`, `pocket_density_ratio` and `pocket_refine`.
* The stable cut-off on the competition data is k* 263-265, not v24's 288 (seed 42 of the old estimator was the high end of its
  263-288 range). A pipeline rerun would flag about 2,074 accounts (1,809 + about 265): the 23 lowest-ranked of v24's pocket drop.
* No accuracy gain to expect: on the first release the mean reference F3 moves 0.9887 -> 0.9892 (within the seed range of the old
  setting, 0.9878-0.9893), and the first release stays less stable (its est_TP curve is flatter at the cut-off).

**Item 1, every column and every period for the pocket** (`fi_r32_t5_pocket_columns.py`; pocket = `pocket_refine` with the new
defaults: k 265 on the competition data, 287 on the first release).
* A. Regression-adjusted rank shift on all 292 other columns (all derived features incl. flags and statics, every raw period column),
  Bonferroni bound 4.64. **Other behaviours: none** (largest |z| 3.4, device_count_cv; first release 3.8, ip_hash_entropy_2; the
  first release's true pocket members: largest 3.1). There is no 4th dimension, so D (4-D ratio) did not run. Own behaviours: the
  within-account spread of all three (cv, maxjump, burst) is far LOWER than their level predicts (performance_stability_cv z -19.7,
  accuracy_std_cv -15.9, kda_std_cv -10.8; first release the same, -19.7 / -17.5 / -10.5): the group is high in every period.
* B. Persistence. Pocket vs normal accounts at the same level (all three level ranks > 0.8, outside the pocket, 166): periods in
  the top 20% accuracy_std 3.80 vs 2.71, kda_std 3.75 vs 2.73, lowest period rank 0.82 / 0.80 vs 0.69 / 0.68 (performance_stability
  2.66 vs 2.68, no difference). First release (reference), v24 pocket true (240) vs false (47) members: accuracy_std 3.80 vs 2.81
  periods, lowest period rank 0.81 vs 0.64; kda_std 3.82 vs 2.77, 0.82 vs 0.59; the false members are also lower in level (0.84 vs
  0.98, 0.78 vs 0.98). The user's hypothesis holds: real members are persistently high, false members reach the corner with fewer
  high periods.
* C. But no simple per-account aggregate that encodes persistence beats the mean over periods (common group size; first release
  reference F3 / pocket precision):

| aggregate of the three behaviours | competition: group, k, est. precision, expected F3 | first release: k, ref F3, pocket precision |
|---|---|---|
| mean over periods (v24 `_level`) | 224, 265, 0.846, **0.9782** | 287, **0.9895**, **0.836** |
| median over periods | 250, 295, 0.731, 0.9727 | 298, 0.9827, 0.762 |
| mean of per-period normal scores | 210, 300, 0.694, 0.9689 | 303, 0.9829, 0.753 |
| second-lowest period rank (high in >= 3 of 4) | 270, 296, 0.734, 0.9734 | 269, 0.9807, 0.818 |

  (Expected F3 here is at the common group size 270, set by the second-lowest-rank score, hence lower than item 3's values.)
  The mean averages four noisy periods; order statistics keep less of the signal than they gain from persistence.
* A valid way to add persistence on top of the level needs a different null: a level and a spread of the SAME behaviour are
  dependent for normal accounts, so a column-shuffle null would count that dependence as group members. Candidate (not run): a
  cross-fitted classifier density ratio on the 12 per-period columns (3 behaviours x 4 periods) against a null that shuffles whole
  behaviour blocks across accounts (keeps each behaviour's period dependence, removes only the cross-behaviour excess).

**Item 2, multi-column signatures at the edges of the six gap families** (`fj_r32_t5_family_edges.py`; first run in
`outputs/r32/fj_t5_family_edges_first_run.log`). Signature of a family = up to 8 columns of other behaviours whose mean rank in
the family differs from the unflagged accounts U beyond the Bonferroni bound (one per behaviour), Stouffer sum of normal scores.
* Signatures are strong for four families (share of members above U's 99th signature percentile: skill boost + friend purge 1.00,
  bots 1.00, near-inactive 1.00, takeover / VPN 0.63, silent 0.69) and weak for the erratic-stability family (0.15; first release 0.27).
* (1) Missed members at the edges. The first run had two dependence artefacts: the near-inactive family (edge z +15 to +18, 13-25
  above U99) had activity_level_level in its signature (the helper mapped "activity_level_*" to "activity", so the family's own
  behaviour was not excluded; Spearman with the gap column -0.50 to -0.53), and login_count_3 < 14.65 (z +8.8; geo_km_per_login_4 is
  computed from the login count, Spearman -0.23). On the first release those near-inactive edge zones held 0 true anomalies. Fixed
  (longest temporal-group prefix; geo and login_count treated as one behaviour) and rerun: for every gap column (33 on the
  competition data, 31 on the first release) the 1% of U nearest the cut on the bulk side has the signature distribution of the rest
  of U: z -2.3..+1.6 (first release -1.6..+1.1), accounts above U's 99th percentile 0-8 against 2.3 expected, Spearman(signature,
  column) within +-0.02; on the first release these edge zones hold 0-3 true anomalies each (at most 6 accounts in all; 19 missed accounts spread at
  random over U would put about 0.18 in each zone, 5.6 over the 31; the largest, 3 below purchase_frequency_4 > 8.165, is 3 of 226).
* Tail of each signature inside U against a shuffle null of its columns (accounts above U's own 99.9% / 99% quantile): at the null
  for skill boost (24 vs 20.3 +- 4.5), near-inactive (24 vs 26.4), silent (24 vs 23.3), bots (24 vs 17.8 +- 4.3) and erratic
  stability (24 vs 20.5 +- 4.3; 239 vs 221 +- 12 at 99%). The takeover / VPN signature shows 24 vs 1.9, but it contains n_flags,
  the sum of the other flag columns in it (password_changed, mass_item_sale, vpn_usage), so a column-shuffle null is invalid there;
  on the first release its top 0.1% / 1% hold 0 / 1 true anomalies. On the first release the erratic-stability family has no
  login_count gap, so its signature keeps login_count_level and geo_km_per_login_4 (one built from the other): 24 vs 8.2 +- 2.8,
  again a dependence artefact, though its top 0.1% / 1% hold 3 / 6 true anomalies (next bullet).
* (2) False members at the edges. The 10% of each family nearest its cut carry the same signature as the rest (share below U's
  median 0.00-0.02 in both releases); on the first release their precision is 1.000 for every gap column. The gap families have no
  false members to trim; the first release's 47 false flags are all in the pocket (flagged precision 0.978, recall 0.991).
* Where the first release's 19 unflagged true anomalies sit (reference only; 24 before v24's pocket): none has a high pocket score
  (0 above U's 99th percentile; their three pocket-column ranks look normal). 7 of 19 have a family signature above U's 99th
  percentile (about 1 expected), 6 of them the erratic-stability family's (purchase_frequency_level high, login_count_level low,
  account_age_days high), 3 even above 99.9%, but none is near the behavior_stability cut (0 true anomalies in its 1% edge zone).
  They look like erratic-stability members without the low stability value. Not usable without labels: they are 6 of the 238 U
  accounts above that 99th percentile (precision 0.025, far below the F*/10 = 0.1 bar); even U's top 0.1% (about 24 accounts, 3
  true) would sit near the bar, and an excess of 3 in 24 is invisible against Poisson noise (+-5) on the competition data.

**Item 4, classifier density ratio on the whole period profile** (`fk_r32_t5_pocket_12d.py`; user approval of 2026-11-16 ~16:00:
run on the CPU at idle priority, move into the pipeline before v25 only if the expected F3 rises clearly and the first-release
reference F3 is above 0.9895).
* Design. Rows = the accounts outside the gap families (24,080). Per pocket behaviour (accuracy_std, kda_std,
  performance_stability) a block of 8 features from its four periods: normal score of each period (missing stays missing), of the
  period mean, min / max / SD of the period scores. Null = block shuffle (each behaviour's block permuted over the accounts
  independently): keeps every within-behaviour dependence, removes only the dependence across behaviours, the same hypothesis as
  the 3-D null. Ratio = log-odds of a HistGradientBoosting classifier (real rows of a half against 3 block shuffles of the same
  half), cross-fitted over 20 halvings; cut-off from 200 block-shuffle draws scored the same way.
* Checks of the null on the accounts outside the families: about 18% miss at least one period of a behaviour (4.8% of the cells);
  missingness is independent across behaviours; level correlations across behaviours 0.003-0.029, same-period deviation
  correlations about 0. So the block shuffle only removes the pocket.
* A trap: with block features only, greedy trees cannot split. Every block value sits in exactly 1 real and 3 shuffled training
  rows, so every single-feature split has zero gain; the scores were constant and the "group of 3,000" of the first run was a tie
  artefact (early stopping only seemed to work because its validation part breaks the 1 : 3 balance by chance). Fix: six
  across-behaviour columns per row (min / mean / max of the three level scores, min and mean of the per-behaviour lowest period
  score, number of periods in the top 20%), recomputed for every shuffled row.
* Classifier setting picked on the label-free expected F3 of a 6-halving run (group 260): early stopping 0.9828; 100 iterations at
  lr 0.05 0.9864; 150 iterations with 4 leaves 0.9864; **300 iterations at lr 0.03, 8 leaves, 200 rows per leaf 0.9876** (kept).
* Result, 20 halvings / 200 null draws, three master seeds (common group size = the largest estimate of the run):

| score | competition data: group, k, est. precision, expected F3 | first release (reference): k, est. precision, reference F3, pocket precision |
|---|---|---|
| min-rank (2-D) | 201, 339, 0.574, 0.9755 | 380, 0.535, 0.9827, 0.618 |
| density ratio 3-D (v24 pipeline, 200 / 1000) | 224, 265, 0.846, 0.9935 | 287, 0.786, 0.9895, 0.836 |
| **classifier 12-D, seed 42** | 234, **242**, 0.962, **0.9989** | 254, 0.920, **0.9911**, 0.945 |
| classifier 12-D, seed 1 | 233, 249, 0.936, 0.9987 | 247, 0.950, 0.9909, 0.968 |
| classifier 12-D, seed 2 | 233, 249, 0.936, 0.9987 | 246, 0.960, 0.9905, 0.968 |

* Against the 3-D flags on the competition data: 235-240 in both, 7-9 only in the 12-D pocket, 25-30 only in the 3-D pocket
  (Jaccard 0.86-0.88). First release (reference): of the 39-43 accounts only the 3-D pocket flags, 1-2 are true anomalies; of the
  2-6 only the 12-D pocket flags, 0-1. The classifier removes the 3-D pocket's false members (pocket precision 0.84 -> 0.95-0.97)
  at a cost of 1-2 true ones, and its label-free precision estimate (0.92-0.96) matches the reference precision.
* Seed stability at 20 halvings: Jaccard 0.972, 0.972, 1.000 (k 242 / 249 / 249) on the competition data; 0.969-0.980 on the
  first release, where the estimated group size is unstable (295 / 277 / 355: the est_TP curve keeps creeping up far below the
  cut) while the cut-off is not (254 / 247 / 246).
* Decision: by the label-free estimate at the common group size (0.9987-0.9989 against 0.9935, all three seeds) the classifier
  ratio is the better pocket score; the first-release reference agrees (0.9905-0.9911 against 0.9895) and was not used to choose
  anything (features, classifier setting and cut-off all come from the competition data's shuffle-null estimates). Moved into the
  pipeline as a third candidate of `task5.pocket_refine(..., df=...)` (`profile_block`, `profile_ratio_curve`,
  `pocket_profile_ratio`); the pipeline keeps the score with the highest expected F3 at the common group size. The `src` function
  reproduces the experiment exactly at the same seed (scores, order and curve identical at 2 halvings / 20 draws).
  Expected effect: about +0.001 to +0.0015 F3 on Task 5 (reference), about +0.0002 to +0.0003 on the leaderboard mean.

**Round 32 summary.** Item 3: 200 halvings / 1,000 null draws make the pocket cut stable (k* 263-265, Jaccard mean 0.997), now the
`src/task5.py` default; the stable cut is about 265 instead of v24's 288, worth about 0 on the first release (0.9887 -> 0.9892,
inside the old seed range). Item 1: no column of another behaviour is shifted in the pocket (no 4th dimension); the group is
persistently high in every period (and its false members are not), but no simple per-account aggregate uses that better than the
mean over periods. Item 2: the six gap families have neither missed members at their edges nor false members near their cuts; the
first release's 19 missed anomalies are not pocket members, and 6 of them carry the erratic-stability signature without the low
stability value, too few to separate from normal accounts without labels. No submission built (user: keep the src defaults only). Item 4 (added later the same day): a cross-fitted
classifier ratio on the whole period profile against a block-shuffle null beats the 3-D ratio on the label-free estimate
(expected F3 0.9987-0.9989 against 0.9935; first-release reference 0.9905-0.9911 against 0.9895) and is in `task5.pocket_refine`
for v25.

* Seed stability at the pipeline default (40 halvings / 200 draws; `outputs/r32/fk_t5_pocket_12d_reps40.log`, finished after the v25
  upload): seeds 42 / 1 / 2 give k 244 / 242 / 247, estimated group 234 / 233 / 234, precision 0.956 / 0.963 / 0.946, expected F3
  0.9993 / 0.9992 / 0.9993 (3-D ratio 0.9937); Jaccard between seeds 0.992, 0.988, 0.980 (mean 0.987; 20 halvings: 0.981). Below the
  0.995 the 3-D ratio reached with 200 halvings: 2-5 borderline accounts of 244 move with the seed (about +-0.0001 F3). Seed 42 is
  the pipeline's run (k 244, the v25 flags).

* User decision (2026-11-16 ~21:50): keep 40 halvings (k 244, mean Jaccard 0.987); no further halvings.

### v25 (user decision, 2026-11-16 ~19:10: build and upload before the GPU queue ends)
v25 = v24 with the Task 5 pocket from item 4 (pipeline default: 40 halvings, 200 block-shuffle draws, seed 42). `task5/task5.ipynb`
re-executed: `pocket_refine` compares min-rank (k 339, est. precision 0.574, expected F3 0.976), the 3-D ratio (265, 0.846, 0.994)
and the period-profile ratio (244, 0.957, 0.999) at the common group size 234 and keeps the last. 2,053 flags (1,809 + 244);
against v24's 2,097: 46 dropped, 2 added; Tasks 1-4 byte-identical (0 rows changed). Uploaded 20:06 via the Kaggle CLI:
**public 0.86148 (+0.00013 over v24, about +0.0007 F3 on Task 5; expected +0.0002 to +0.0003)**. Snapshot
`versions/v25_LB0.86148/`. The master notebook is still the v23 execution (its Task 5 cells are re-run after the GPU queue).

## Round 33 (2026-11-16 evening): new methods for Tasks 1-3 on the CPU while the GPU trains (user brief of ~16:30)
Brief: methods outside the tree / linear frame, 100% training rows only, CPU at idle priority (2 threads per job, one job at a
time) so that the dinov2_r2 training is not slowed. Acceptance: nested out-of-fold pooled delta > 0, better in >= 3/5 folds, no
fold dropping hard (floor -0.0005; Task 3 -0.0002). Base = v24 (Task 1 F2 0.8388, Task 2 macro-F1 0.7584, Task 3 NMAE 0.7946).

**Task 3** (`fn_r33_t3_meta.py`; reference = the pipeline read-out fully nested as in `ex_r30`, 0.79457; 25 minutes at 2 threads).

| candidate | nested NMAE | delta | folds better | per fold | verdict |
|---|---|---|---|---|---|
| 1a linear mixture-of-experts gate on P(y>0), segment, history and the member predictions, smooth L1 in THB, started at the exact L1 blend | 0.79457 | -0.00000 | 0/5 | all -0.00000 | fail |
| 1b MLP gate (same inputs and start) | 0.79457 | -0.00000 | 2/5 | all within +-0.00001 | fail |
| 1c LightGBM L1 boosted from the blend (`init_score`) | 0.79449 | -0.00008 | 1/5 | -0.00009 +0.00002 -0.00022 -0.00008 -0.00004 | fail |
| 2a per-row sigma (LightGBM on the absolute log residual, out of fold inside the training folds) in the mixture-median read-out, replacing the member | 0.79455 | -0.00002 | 1/5 | -0.00000 +0.00002 -0.00000 -0.00005 -0.00007 | fail |
| 2b the per-row-sigma read-out added to the blend pool | 0.79457 | +0.00000 | 1/5 | all +0.00000 | fail |
| 3 sub-threshold rule for gated rows predicted under 50 THB (zero / floor / scale, chosen on the training folds) | 0.79457 | +0.00000 | 0/5 | all 0 | fail |

* The gates do not move from their start: the exact L1 blend is already the optimum of the smooth-L1 objective in THB, and no
  input (P(y>0), segment, history) says where another member mix would be better. Boosting from the blend only fits noise.
* Per-row sigma: the predicted sigma barely varies (quartiles 0.149 / 0.156 / 0.170) and correlates only 0.13-0.18 with the
  absolute residual of the positive validation rows; the LP gives the added read-out zero weight. The noise is close to
  homoscedastic in log space once the ratio-to-history target is used.
* Sub-threshold rows are 0.24-0.27% of the rows (mean target 16-21 THB against a mean prediction of 24-27 THB); the training
  folds pick "no rule" in all five folds.
* Task 3 stays at its measured ceiling (rounds 13, 17, 30): nothing to add to v25.

**Task 1, item 1: level-2 combiner trained on a soft F2** (`fl_r33_t1_softf2.py`; base = the v22 level 2 nested, F2 0.83872 /
0.83853 / 0.83849 / 0.83859 on the pipeline folds and stacker split seeds 1, 2, 3; 36 minutes at 2 threads, of which 15 for the base).

| candidate | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| A linear soft-F2 on 9 log-odds + 31 missing indicators + count (cut from the other folds) | -0.0022 -0.0024 -0.0017 -0.0016 | 0 0 0 0 | -0.0025 -0.0034 -0.0033 -0.0021 | fail |
| A at its built-in cut (score > 0) | -0.0023 -0.0025 -0.0025 -0.0025 | 1 0 0 0 | -0.0035 -0.0038 -0.0042 -0.0032 | fail |
| B linear + residual MLP (16 tanh units, 3 seeds) | -0.0021 -0.0023 -0.0011 -0.0016 | 0 0 0 0 | -0.0024 -0.0031 -0.0023 -0.0021 | fail |
| C mean of the v22 level 2 and A (standardised) | -0.0009 -0.0006 -0.0004 -0.0006 | 0 0 2 1 | -0.0017 -0.0013 -0.0013 -0.0012 | fail |
| D linear soft-F2 on the nine log-odds only | -0.0018 -0.0015 -0.0013 -0.0012 | 0 0 0 0 | -0.0027 -0.0025 -0.0023 -0.0024 | fail |

* Training on the soft F2 is clearly worse than the log-loss in every fold (-0.0012 to -0.0025): the soft F2 is dominated by the
  rows far from the cut, and a linear combiner loses the tree meta-models' interactions. The F2-optimal rule is a cut on the
  posterior, so the best posterior (log-loss) plus one cut-off is already the right decomposition.
* The exact cut-off search and `optimize_threshold` agree (0.83870 vs 0.83872 etc.), so the cut-off search is not a limit.

**Task 1, items 2 and 3: k-NN label purity in the skill space, RuleFit** (`fo_r33_t1_knn_rulefit.py`).

| candidate | pooled mean delta | folds better | worst fold | verdict |
|---|---|---|---|---|
| 2 k-NN purity on 10 skill columns (normal scores; Euclidean k = 15 / 50 / 150, cosine k = 50; neighbours' cheater rate and rate minus their mean member probability; out of bag in every outer fold) as 8 inputs of the tree meta-models | -0.00029 | 1 2 3 0 | -0.00096 -0.00067 -0.00092 -0.00076 | fail |
| 3 RuleFit (leaves of 200 depth-3 trees + linear terms, L1 logistic; 146-178 of 1,662 terms kept) as a tenth member | +0.00009 | 3 4 3 3 | -0.00029 -0.00016 -0.00072 -0.00031 | fail (fold floor) |

* The neighbours' cheater rate correlates 0.33 with the label, but what the members do not already predict is nil: the
  correlation of (neighbour rate - neighbours' mean member probability) with (label - member mean) is -0.001 to -0.004 on the
  held-out rows. The neighbourhood holds no label information beyond the nine members, in a 10-column skill space as in round 4's
  all-column space.
* RuleFit alone reaches F2 0.8323 and its log-odds correlate 0.94-0.97 with the tree members (0.70 with the logistic member): a
  sparse re-expression of the same trees.

**Task 1, added from the literature: cost-sensitive level 2** (`fq_r33_t1_costsens.py`; the F-beta-optimal rule is a cut on the
posterior, i.e. cost-sensitive classification with class weights (1 - t) : t, Koyejo et al. 2014 / Parambath et al. 2014; every
meta-model of the v22 level 2 fitted with a positive-class weight, so that the weighted log-loss is steepest near the cut).

| candidate | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| A positive weight 5 | -0.00014 +0.00008 +0.00005 +0.00020 | 3 2 3 3 | -0.00097 -0.00018 -0.00054 -0.00022 | fail |
| B positive weight 3 | -0.00008 +0.00013 +0.00034 +0.00007 | 3 2 3 3 | -0.00128 -0.00018 -0.00014 -0.00030 | fail |
| C mean of the v22 level 2 and A (standardised) | -0.00007 +0.00018 +0.00006 +0.00030 | 2 4 3 3 | -0.00078 -0.00023 -0.00020 -0.00022 | fail |

* Means of +0.00005 to +0.00012 with both signs over the folds: re-weighting moves 0.2-0.6% of the labels at random. The
  unweighted posterior is already accurate near the cut (the level 2 is well calibrated there), so there is nothing to re-focus.

**Round 33 summary.** Of the new-method candidates for Tasks 1-3 beyond the tree / linear frame, none passes the fold rule.
Task 1 (soft-F2 combiners, k-NN purity, RuleFit, cost-sensitive level 2) and Task 3 (mixture-of-experts gates, per-row sigma,
sub-threshold rule) confirm the measured ceilings; Task 2's only signal (Dirichlet calibration, positive mean with swinging folds)
led to round 34.

**Task 2** (`fm_r33_t2_gam_dirichlet.py`; reference = the v20 blend nested as in `ev_r30`: 0.7573 / 0.7572 / 0.7564 / 0.7574 on
the pipeline folds and stacker split seeds 1, 2, 3; 108 minutes at 2 threads). Alone (argmax, no multipliers): spline GAM 0.7548
(log-loss 0.6220), GAM + 6 tensor pairs 0.7528 (0.6250), KDE naive Bayes 0.7560 (0.6251).

| candidate | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| 1a spline GAM as a 5th stack member | -0.0003 -0.0001 +0.0001 -0.0006 | 2 3 3 1 | -0.0019 -0.0011 -0.0008 -0.0026 | fail |
| 1a GAM + tensor pairs as a 5th stack member | +0.0001 -0.0004 +0.0008 -0.0006 | 3 3 4 1 | -0.0017 -0.0022 -0.0002 -0.0016 | fail |
| 1b GAM as a third blend term (weight nested) | +0.0004 +0.0002 +0.0008 -0.0000 | 2 2 3 1 | 0 -0.0003 0 -0.0001 | fail (weight 0 picked in most folds) |
| 1b GAM + pairs as a third blend term | +0.0003 +0.0004 +0.0004 -0.0001 | 2 2 3 0 | -0.0002 0 -0.0005 -0.0005 | fail |
| 2a KDE naive Bayes Dirichlet-calibrated | +0.0002 +0.0002 +0.0010 +0.0001 | 4 4 4 3 | -0.0008 -0.0017 -0.0000 -0.0011 | fail (fold floor) |
| 2b both calibrated, log-linear blend | +0.0004 +0.0001 +0.0011 +0.0003 | 4 3 5 3 | -0.0007 -0.0019 +0.0001 -0.0011 | fail (fold floor) |
| 3 Whale factor (PC1 of vip_tier, monthly spending, frequency, rare items) class-conditional KDE, lambda 0.5 | -0.0002 -0.0001 +0.0008 -0.0004 | 3 2 4 1 | -0.0023 -0.0017 -0.0012 -0.0012 | fail |
| 3 Whale factor, lambda 1 | -0.0004 -0.0002 +0.0003 -0.0000 | 2 2 3 3 | -0.0024 -0.0028 -0.0017 -0.0011 | fail |

* The GAM is a fourth view of the same additive structure (the features are independent given the segment, rounds 19 and 23):
  its log-loss equals the naive Bayes's, the tensor pairs make it worse, and the nested blend weight is 0 in most folds.
* Dirichlet calibration is the only candidate with a positive mean on all four splits (+0.0004 / +0.0005) and 3-5 folds better,
  but single folds drop by 0.0007-0.0019, beyond the -0.0005 floor: 20 parameters per model, then a Powell search of the class
  multipliers on a step function. Round 34 tests the two-parameter version and smoother multiplier searches.
* The Whale factor adds a dependence between four spending columns that the stacker's trees already use; no gain.

**Task 2, added from the literature: KDE naive Bayes in a pooled normal-score space** (`fp_r33_t2_kde_zspace.py`; transformation
kernel density estimation, Wand, Marron & Ruppert 1991 / Geenens 2014: no boundary bias, no skewed marginals; the transform is the
same for every class, so its Jacobian cancels in the posterior). Alone (argmax): raw-scale 0.7560 (log-loss 0.6251), z-space with
Silverman's bandwidth 0.7559 (0.6263), 0.7 x Silverman 0.7544 (0.6294), geometric mean of raw and z-space 0.7563 (0.6251);
argmax agreement raw vs z-space 0.985.

| candidate (replaces the KDE naive Bayes in the blend) | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| z-space, Silverman | -0.0001 -0.0002 +0.0002 -0.0002 | 3 1 2 2 | -0.0013 -0.0012 -0.0010 -0.0012 | fail |
| z-space, 0.7 x Silverman | -0.0009 -0.0005 +0.0001 -0.0008 | 0 2 3 0 | -0.0023 -0.0019 -0.0010 -0.0019 | fail |
| geometric mean raw + z-space | -0.0001 -0.0005 +0.0004 -0.0003 | 3 2 3 3 | -0.0012 -0.0023 -0.0011 -0.0018 | fail |

* The raw-scale densities with point masses at the clip bounds are already adequate; the boundary and skewness corrections change
  1.5% of the argmax labels without a gain, and a smaller bandwidth over-fits. The naive Bayes member is at its ceiling.

## Round 34 (2026-11-16 night): fold-to-fold swings of round 33 and further untried methods (user brief of ~20:10)
CPU at idle priority beside the GPU training (epoch 11 took 1,874 s with both CPU jobs suspended for 80% of it, against
1,625-1,653 s with them running: the CPU jobs do not slow the trainer; fold 4 later ran at about 860 s per epoch with three CPU
jobs running). Acceptance as in round 33.

**Task 2** (`fr_r34_t2_calib.py`; reference = the v20 blend nested, 0.7571 / 0.7572 / 0.7568 / 0.7573; 55 minutes at 3 threads).
The pipeline's naive Bayes was rebuilt per feature (max |dp| 8.5e-07 against `t2_kde_nb.npz`).

| candidate | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| 1a(i) log-linear blend, two exponents (alpha/T1, (1-alpha)/T2) by the log-loss | -0.0012 -0.0012 -0.0006 -0.0012 | 0 0 1 1 | -0.0023 -0.0019 -0.0022 -0.0037 | fail |
| 1a(ii) log-linear blend, weight by the macro-F1 | +0.0000 -0.0002 +0.0001 -0.0006 | 3 3 3 3 | -0.0006 -0.0017 -0.0005 -0.0022 | fail |
| 1b one-vs-rest naive Bayes, no clipping (control) | -0.0002 -0.0001 -0.0006 -0.0003 | 2 2 1 2 | -0.0015 -0.0007 -0.0014 -0.0028 | fail |
| 1b winsorised log-likelihood ratios, M nested in (2, 3, 4) | -0.0000 +0.0003 +0.0004 +0.0003 | 3 3 3 3 | -0.0009 -0.0003 -0.0004 -0.0015 | fail |
| 1b winsorised, M = 3 | -0.0001 -0.0006 +0.0000 +0.0003 | 2 1 2 3 | -0.0009 -0.0015 -0.0009 -0.0015 | fail |
| 1c(A) multipliers by coordinate ascent on the macro-F1 averaged over 40 bootstrap resamples | +0.0001 -0.0001 -0.0000 -0.0006 | 3 2 2 2 | -0.0003 -0.0020 -0.0011 -0.0018 | fail |
| 1c(B) multipliers and weight by L-BFGS on a soft macro-F1 (temperature 0.1) | +0.0005 +0.0006 +0.0007 +0.0003 | 4 3 4 2 | -0.0003 -0.0005 -0.0000 -0.0016 | fail (split seed 3) |

* 1a(i): the log-loss puts almost all weight on the stacker (exponents 0.92-0.94 and 0.05-0.07): the naive Bayes is over-confident
  (a product of 34 densities), so a likelihood fit shrinks it away, while the macro-F1 wants it at 0.5-0.6 of a LINEAR blend. A
  geometric blend (1a(ii)) is no better than the linear one, and the weight it picks swings between 0.4 and 0.9.
* 1b: the one-vs-rest form alone is worse than the pipeline's multi-class naive Bayes (0.7484 vs 0.7560 alone) and clipping lowers
  it further alone (M = 2: 0.7421, 0.77% of the ratios clipped; M = 4: 0.7483, 0.13%); in the blend the nested M recovers the
  loss and adds +0.0002 on average (row-bootstrap 95% CI [-0.0003, +0.0008]). No extreme-cell problem to fix.
* 1c: the bootstrap-averaged search is not better than Powell. The soft macro-F1 search is the only candidate of rounds 33-34 with a
  positive delta on all four splits: mean +0.00052, row-bootstrap 95% CI [+0.00005, +0.00105], and it picks the weight 0.6 in 15
  of 20 folds (0.5-0.7 in all; the bootstrap search: 0.4-0.9). It fails the fold rule on split seed 3 (2/5 folds, worst -0.0016), and it is the best of
  15 candidates on these splits, so `fv_r34_t2_soft_confirm.py` re-runs it unchanged on six fresh stacker splits (below).
* Fold-level noise: one fold has 20,800 rows, and the paired fold deltas of candidates that change 1-1.5% of the labels have an
  SD of about 0.001. A true gain of +0.0005 then clears "no fold below -0.0005" on all 20 folds with a probability of a few
  percent; the fold floor decides Task 2 results more than the mean does. Reported so the user can weigh it; the rule is unchanged.

**Task 2, 1c(B) on fresh splits** (`fv_r34_t2_soft_confirm.py`; the candidate unchanged, stacker split seeds 4-9, which no Task 2
experiment had used; 13 minutes).

| split seed | reference | soft search | delta | folds better | per fold |
|---|---|---|---|---|---|
| 4 | 0.7574 | 0.7576 | +0.0002 | 4/5 | +0.0002 +0.0006 +0.0001 -0.0003 +0.0005 |
| 5 | 0.7574 | 0.7577 | +0.0003 | 4/5 | +0.0004 -0.0001 +0.0005 +0.0003 +0.0006 |
| 6 | 0.7568 | 0.7574 | +0.0006 | 4/5 | +0.0006 +0.0006 +0.0020 -0.0010 +0.0008 |
| 7 | 0.7567 | 0.7574 | +0.0007 | 2/5 | -0.0000 +0.0026 -0.0001 +0.0010 -0.0001 |
| 8 | 0.7573 | 0.7577 | +0.0004 | 3/5 | +0.0018 +0.0011 -0.0005 -0.0009 +0.0007 |
| 9 | 0.7563 | 0.7575 | +0.0012 | 5/5 | +0.0008 +0.0019 +0.0006 +0.0010 +0.0019 |

* Mean +0.00058, positive on 6/6 fresh splits (10/10 with the first four); row-bootstrap 95% CI of the mean [+0.00006, +0.00109],
  P(delta > 0) 0.988: the same size as on the selection splits (+0.00052), so no selection effect to speak of. Under the fold rule
  it still fails (seed 7: 2/5 folds; seeds 6 and 8: folds at -0.0010 / -0.0009).
* Why it helps: the soft search picks the weight 0.6 in 27 of 30 folds here (0.7 in 3) and nearly the same log-multipliers in
  every fold (+0.17..+0.24, +0.16..+0.25, +0.23..+0.30), while Powell on the step function picks weights 0.3-0.9 (0.5 in 13 folds,
  0.6 in 8) and log-multipliers from -0.05 to +0.26. The gain is variance removed from the decision rule, and the soft optimum
  sits at larger multipliers for classes 1-3 than the noisy raw optimum.
* **User decision (2026-11-16 ~21:50): for Task 2 a candidate passes when its mean on the fresh splits is positive and the
  row-bootstrap 95% CI lower bound is above 0** (the fold floor is below the fold noise there). 1c(B) passes; it goes into the
  pipeline for v26 (`src/metrics.optimize_class_weights_soft`).
* In the pipeline (prepared for v26, not uploaded): `scripts/nb_part2.py` chooses the KDE weight and the class multipliers with
  `optimize_class_weights_soft` (weights 0.3-0.9, as tested); `task2/task2.ipynb` re-executed. Weight 0.6 (unchanged), multipliers
  Casual 0.842, Grinder 1.030, Social 1.032, Whale 1.097; 172 of 25,889 test labels change (0.66%: Casual -> Grinder 79,
  Casual -> Social 32, Social -> Whale 43, Grinder -> Whale 14, 4 others; predicted Casual share 0.412 -> 0.408). The in-sample
  OOF macro-F1 printed by the notebook is 0.7577 against 0.7584 for the Powell multipliers: Powell's in-sample number is the
  optimistic one (it fits the step function of the rows it is scored on); nested, the soft search is +0.0006 better.

**Task 1, item 2b: bootstrap-smoothed cut-off** (`fs_r34_t1_boot_threshold.py`; 200 resamples of the training folds per fold).

| candidate | pooled mean delta | folds better | worst fold | verdict |
|---|---|---|---|---|
| A mean of the 200 resample optima | -0.00003 | 0 2 3 2 | -0.00063 -0.00016 -0.00041 -0.00035 | fail |
| B median of the resample optima | +0.00003 | 1 3 3 3 | -0.00027 -0.00009 -0.00005 -0.00005 | fail |
| C argmax of the bootstrap-mean F2 curve | +0.00005 | 4 4 2 4 | -0.00005 -0.00002 -0.00100 -0.00013 | fail |

* The cut-off from four folds is already stable: bootstrap SD 0.004-0.012, pipeline cuts 0.160-0.171 over all folds and splits,
  and the smoothed cuts differ from it by 0.000-0.006. The validation fold's own optimum is anywhere in 0.135-0.189, i.e. the
  noise is in the 18,000-row fold being scored, not in the cut-off estimate, so smoothing the estimate cannot help.

**Task 1, item 2a: missingness-pattern and skill-stratified relative features** (`ft_r34_t1_relative.py`; 46 new features, every
statistic fitted inside the training fold: missing counts per column group, group bitmask codes with their frequency and smoothed
cheater rate, and for 16 mechanical / game-sense columns the residual against players in the same decile of `level` and of
`kill_death_ratio`, plus mechanical / game-sense / gap means of those residuals).

| candidate | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| M0 control: three-seed LightGBM on the 49 pipeline features as a tenth member | +0.00009 -0.00003 -0.00001 +0.00033 | 3 3 2 5 | -0.00029 -0.00048 -0.00044 +0.00020 | fail |
| M1 the same LightGBM + the 46 new features as a tenth member | +0.00005 +0.00023 -0.00012 -0.00008 | 3 4 2 2 | -0.00041 -0.00003 -0.00045 -0.00056 | fail |

* Members alone (3 seeds, out of fold): AUC 0.89080 (M0) vs 0.89083 (M1), F2 at the best cut 0.83601 vs 0.83604; per-fold AUC
  difference -0.0003 to +0.0007 with both signs; M1 correlates 0.997 with the pipeline's LightGBM member.
* The missingness features carry nothing: the smoothed cheater rate of a group bitmask has AUC 0.50 on held-out rows (missingness
  is completely at random, as in rounds 8-9). The trees do use the relative features (crosshair_placement against the K-D and
  level strata: 6% and 4% of the gain), but only in place of splits they already had: the mean gap of the stratified residuals has
  AUC 0.437-0.450 against 0.441 for the pipeline's `mech_minus_sense`.
* A tenth member moves 0.2-0.4% of the labels and its mean effect (+0.00002 for M1, +0.00010 for the control) is inside that noise.

**Round 34 summary.** Task 2: the soft macro-F1 search for the blend weight and class multipliers is the one accepted change
(+0.0006 nested on ten stacker splits, six of them fresh; user rule for Task 2: positive mean with the row-bootstrap 95% CI above
0); it is in the pipeline for v26. Temperature-only log-linear blends, winsorised evidence, the bootstrap-averaged search and the
z-space KDE fail. Task 1: bootstrap-smoothed cut-offs and missingness / skill-stratified features fail. Task 4 items (photometric
TTA, DINOv2-L fold 0) run in the GPU queue after the dinov2_r2 training. **User decision (2026-11-16 ~22:00): the tabular
experiments (Tasks 1-3, 5) are closed after these runs; the remaining effort goes to Task 4.**

## Round 35 (2026-11-16 late night): final out-of-the-box check for the tabular tasks (user brief of ~22:25)
Three last hypotheses on the CPU beside the Task 4 GPU queue (Task 1 focal-loss member, Task 2 pair tie-breaker, Task 3 tail
shrinkage); if they fail the tabular side is closed for good. The pending
Task 2 change (soft macro-F1 search) is committed (`2d52b2e`); `submission_v26.csv` = v25 with that Task 2 (172 labels differ, the
other four columns identical) is assembled but not uploaded.

### v26 (user decision, 2026-11-16 22:32: upload the Task 2 change on its own)
Stated before the upload: the expected gain is about +0.0001 on the leaderboard mean, the public split's noise for 172 changed
labels about +-0.0003, so the public score cannot confirm or refute the change, and v26 is kept either way on the nested
evidence. Uploaded 22:33 via the Kaggle CLI: **public 0.86147 (-0.00001 against v25)**. Snapshot `versions/v26_LB0.86147/`;
`submission.csv` = v26. The best public score stays v25 (0.86148), so the final submissions have to be selected by hand.

**Task 3, nested shrinkage of the top 1% of the predictions** (`fw_r35_t3_tail_shrink.py`; y_new = tau + g (y_hat - tau) above
tau = the 99th percentile of the predictions, g on a 0.01 grid chosen on the other four folds; reference = the nested read-out,
NMAE 0.79457).

| variant | delta | folds better | g picked per fold | verdict |
|---|---|---|---|---|
| top 1%, g in [0.85, 1.00] (the brief) | +0.00000 | 0/5 | 1.00 1.00 1.00 1.00 1.00 | fail |
| top 2%, g in [0.85, 1.00] | +0.00000 | 0/5 | 1.00 in every fold | fail |
| top 0.5%, g in [0.85, 1.00] | -0.00001 | 0/5 | 1.00 1.00 1.00 0.99 1.00 | fail |
| top 1%, g free in [0.70, 1.15] | -0.00004 | 0/5 | 1.05 1.00 1.03 1.00 1.04 | fail |
| top 5%, g free in [0.70, 1.15] | -0.00006 | 1/5 | 1.02 1.00 1.01 1.00 1.02 | fail |

* The top 1% of the predictions (1,040 rows; 10.9% of the total absolute error) is not over-predicted: mean prediction 168,855
  against a mean target of 164,726, median target / prediction 1.008, only 5.5% of the rows below half their prediction and none
  above twice. In-sample, any shrinkage loses (g 0.95: -0.00010, 0.90: -0.00026, 0.85: -0.00051); the training folds pick g = 1.00,
  and with a free range slightly above 1. The L1 read-out is already the conditional median up there; the low-spending whales are
  too few to move it.

**Task 1, level-1 LightGBM with an asymmetric focal loss** (`fy_r35_t1_focal.py`; gamma 2, alpha 0.75 on the cheater class, custom
objective with analytic gradient / Hessian (checked numerically; Hessian floored, the focal loss is not convex in the score); the
49 pipeline features + the 15 pairwise ratios of the top-6 skill columns chosen inside each fold (crosshair_placement,
game_sense_score, headshot_percentage, kill_death_ratio, accuracy_score and aiming_smoothness / spray_control_score); 3 seeds;
CatBoost left out: its Python custom objectives take hours on the CPU).

| member as a tenth level-2 input | delta on the four splits | folds better | worst fold | verdict |
|---|---|---|---|---|
| C control: log-loss LightGBM, same features | -0.00048 +0.00022 +0.00011 -0.00016 | 2 4 2 2 | -0.00209 -0.00026 -0.00029 -0.00039 | fail |
| F focal-loss LightGBM | -0.00051 -0.00016 -0.00003 +0.00013 | 0 2 2 4 | -0.00083 -0.00038 -0.00037 -0.00040 | fail |

* Alone the focal member is marginally better (AUC 0.89133 vs 0.89111, F2 at its best cut 0.83693 vs 0.83601), but its ranking is
  the log-loss member's (Spearman 0.996; 0.994 with the pipeline's LightGBM member). Re-weighting towards hard rows changes the
  scale of the scores more than their order, and the level 2 already sets the scale; as a tenth member it only adds noise.

**Task 2, pair-focused naive-Bayes tie-breaker on top of the v26 rule** (`fx_r35_t2_pair_tiebreak.py`; rows whose two best classes
under the v26 rule are within 0.06 (4.5% of the rows) are re-decided by the likelihood ratio of the pair on the n columns that
separate it best on the training folds; the pipeline's per-feature naive-Bayes densities, out of fold).

| variant | brief's four splits: mean delta, folds better, worst fold | fresh splits 4-9: mean delta, folds better | row-bootstrap 95% CI (fresh) | verdict |
|---|---|---|---|---|
| n = 8 columns | -0.00039; 2 0 3 2; -0.0018 | -0.00036; 2 2 2 2 2 2 | [-0.00116, +0.00042] | fail |
| n = 10 columns | -0.00012; 2 2 3 2; -0.0016 | -0.00024; 2 1 3 3 0 3 | [-0.00097, +0.00049] | fail |
| all 34 columns (control) | -0.00008; 2 1 2 2; -0.0018 | -0.00012; 2 1 3 3 1 3 | [-0.00076, +0.00039] | fail |
| n = 10, pairs gated on the training folds | -0.00039; 1 1 1 0; -0.0023 | -0.00024; 1 0 2 0 2 1 | [-0.00050, -0.00002] | fail |

* On the tie rows every rule is right less than half the time: v26 0.439-0.452, the 10-column ratio 0.433-0.441, the whole naive
  Bayes 0.441-0.450 (ten splits). Dropping the columns that do not separate the pair throws away information instead of noise (the features are
  independent given the segment, so every column's evidence counts); the tie rows are decided by label noise.

**Task 5, the period-profile ("12-D") classifier ratio against the 3-D density ratio: final summary** (`fk_r32_t5_pocket_12d.py`,
round 32 item 4).

| pocket score | cut-off k | est. precision | expected F3 (common group size 234) | first release (reference only): F3, pocket precision |
|---|---|---|---|---|
| 3-D density ratio (v24; 200 halvings) | 265 | 0.846 | 0.9937 | 0.9895, 0.836 |
| 12-D classifier ratio (v25; 40 halvings, seeds 42 / 1 / 2) | 244 / 242 / 247 | 0.956 / 0.963 / 0.946 | 0.9993 / 0.9992 / 0.9993 | 0.9905-0.9911, 0.945-0.968 (20 halvings) |

* The 12-D ratio wins on the label-free estimate in every seed, the first-release reference agrees, and the leaderboard moved
  +0.00013 (v24 0.86135 -> v25 0.86148). **Task 5 is locked at the v25 version** (1,809 family accounts + 244 pocket accounts =
  2,053 flags; `task5.pocket_refine(..., df=...)`, 40 halvings, 200 block-shuffle draws, seed 42).

## Round 36 (2026-11-16 night): untried structural and metric-aligned methods for every task (user brief of ~22:40)
Methods already tried and failed in rounds 1-35 are skipped (the brief's list; also the Casual <-> Grinder pair specialist of round
30). CPU jobs at idle priority with 2-4 threads beside the GPU training (fold-4 epochs stayed at 852-873 s with three to four CPU
jobs running). Acceptance as before; for Task 2 also the user's bootstrap rule of 16 Nov 2026.

**Task 4, item 1 (logit-space meta-learner) and item 3 (class multipliers)** (`ga_r36_t4_stacker.py`; the eight v23 members'
out-of-fold views, nested over the five folds, the v17 rule after every candidate; base = v23 weights. dinov2_r2 joins once its
views exist for all folds.)

| candidate | hard-view mean | folds better | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | dark / low-contrast | hard-view rule | v24 check |
|---|---|---|---|---|---|---|---|---|---|---|
| 1a ridge logit stacker (multinomial LR on 40 log-probabilities), C 0.1, fitted on the full view | -0.0156 | 0/5 | +0.0004 | -0.0014 | -0.0026 | -0.0072 | -0.0511 | +0.0015 | fail | fail |
| 1a the same, C 1 | -0.0267 | 0/5 | +0.0001 | -0.0029 | -0.0068 | -0.0073 | -0.0899 | +0.0024 | fail | fail |
| 1b ridge logit stacker, C 0.1, fitted on the five views pooled | +0.0047 | 5/5 | -0.0008 | -0.0002 | -0.0001 | +0.0070 | +0.0119 | -0.0019 | fail (full) | fail |
| 1b the same, C 1 | +0.0045 | 5/5 | -0.0009 | -0.0002 | -0.0002 | +0.0070 | +0.0114 | -0.0024 | fail (full) | fail |
| 1c one non-negative weight per member on the log-probabilities, five views pooled | -0.0028 | 0/5 | -0.0012 | +0.0001 | -0.0019 | -0.0008 | -0.0087 | -0.0030 | fail | fail |
| 3 class multipliers (soft macro-F1, balanced rows) on the v23 rule, fitted on the full view | +0.0004 | 4/5 | -0.0002 | -0.0002 | -0.0008 | +0.0006 | +0.0022 | -0.0010 | fail | fail |
| 3 the same, fitted on the five views pooled | +0.0029 | 5/5 | -0.0002 | -0.0005 | -0.0015 | +0.0044 | +0.0092 | -0.0002 | **PASS** | fail |

* A free meta-learner fitted on full frames over-trusts the members that are best on full frames and collapses on hidden or
  darkened ones (grid -0.05 to -0.09). Fitted on all five views it learns the degraded views, but every gain sits in gamma 3.5
  and the 75% grid, the two synthetic views; full, zoom70, HUD hidden and the real dark frames all lose a little.
* The weighted geometric mean (1c) gives DINOv2 0.52 and three HUD-dropout members the rest, zeroing five members: on these views
  a likelihood fit prefers fewer members than the hard-view-tuned linear weights of v23.
* The multipliers fitted on the pooled views are +0.38..+0.61 in log for every class against 7-Eleven Knights, the same in every
  fold: they lower the largest class's prior further (an effective alpha above 0.75), which helps on the views where the members
  drift towards that class (gamma, grid) and costs on the others. It passes the hard-view rule but not the v24 check the user set
  for Task 4 on the morning of 16 Nov (gain beyond the grid: zoom70, HUD hidden and dark frames must all be > 0). Not adopted.
* Item 2 (entropy-gated TTA): `fu_r34_t4_photo_tta.py` now also scores two geometric views (HUD greyed out, central 85% crop) and
  gated versions that use the transform set only for images whose ensemble entropy on the incoming view is above the 80th / 90th /
  95th percentile of the training full-view entropies. It runs in the GPU queue after the dinov2_r2 evaluation; the CPU scoring
  phase was dry-run on synthetic transform files.

**Task 3, quantile members in the pool and a band-restricted gate calibration** (`gd_r36_t3_quantile_gate.py`; reference = the
pipeline rebuilt fully nested, NMAE 0.79457. The brief's budget was 10 CPU minutes; the run took 49, most of it the nested exact
L1 blends, 7 x 5 HiGHS solves.)

| candidate | nested NMAE | delta | folds better | per fold | verdict |
|---|---|---|---|---|---|
| + LightGBM pinball member q = 0.40 (raw THB) in the L1 pool | 0.79457 | +0.00000 | 0/5 | all 0 | fail |
| + LightGBM pinball member q = 0.45 | 0.79457 | +0.00000 | 0/5 | all 0 | fail |
| + both | 0.79457 | +0.00000 | 0/5 | all 0 | fail |
| logistic recalibration of P(y > 0) inside p in [0.45, 0.65], one fit | 0.79458 | +0.00001 | 4/5 | +0.00001 +0.00001 -0.00000 +0.00001 +0.00002 | passes the letter of the rule |
| the same per imputed segment | 0.79458 | +0.00000 | 4/5 | -0.00004 +0.00002 +0.00002 +0.00001 +0.00002 | passes the letter of the rule |

* The quantile members alone: NMAE 0.78026 (q 0.40) and 0.78429 (q 0.45), coverage 0.558 / 0.577; the exact L1 blend gives them
  weight 0 in every fold (on all rows: ratio[hist] Cat mixture median 0.673, ratio[hist] XGB 0.195, ratio[prev] LGB tuned 0.133). A
  single quantile level is the right read-out only for rows whose p puts the mixture median at that level; the mixture-median
  read-outs of the ratio models already give every row its own level.
* The band holds 16.9% of the rows; its positive rate 0.571 against a mean p of 0.568: P(y > 0) is already calibrated where the
  gate decides. The recalibration moves 0.25-0.29% of the rows across 0.5 and gains +0.00001 NMAE (about +0.000002 on the public
  score). Like round 30's per-segment isotonic (+0.00002, 4/5 folds), it passes the letter of the rule at a noise-level size; the
  user discarded that one for this reason, so this is reported, not merged.

**Interrupted at about 23:45: Claude Code stopped every background job because the system ran critically low on memory** (the
dinov2_r2 fold-4 training at the end of epoch 11 of 15, the GPU queue waiting behind it, and the CPU jobs below). Nothing was
restarted (user rule). dinov2_r2 folds 0-3 are saved (models and out-of-fold predictions); fold 4 has no checkpoint (the trainer
does not resume) and must be trained again from the start, about 3.6 hours. Partial results:

**Task 2, per-class isotonic calibration and pair specialists on the v26 rule** (`ge_r36_t2_iso_specialist.py`; 8 of the 10 splits
finished: the pipeline folds, seeds 1-3 and fresh seeds 4-7).

| candidate | delta per split (pipeline, 1, 2, 3 \| 4, 5, 6, 7) | folds better | verdict |
|---|---|---|---|
| 1a isotonic per class (4 one-vs-all maps per model), linear blend | +0.0001 -0.0003 -0.0003 +0.0000 \| +0.0002 -0.0002 +0.0003 -0.0001 | 2 1 1 2 \| 4 1 3 3 | fail (mean about 0) |
| 1b the same, log-linear blend | +0.0000 -0.0003 -0.0001 +0.0001 \| -0.0000 -0.0001 +0.0003 +0.0001 | 2 0 2 3 \| 2 1 3 3 | fail (mean about 0) |
| 2 specialist Casual <-> Social | 0 on every split | lam = 0 picked in every fold | fail |
| 2 specialist Social <-> Whale | 0 on every split | lam = 0 picked in every fold | fail |

* Base-rule confusion (pipeline folds, symmetric): Casual <-> Grinder 8,344, Casual <-> Social 5,803, Social <-> Whale 4,051,
  Grinder <-> Whale 2,997, Grinder <-> Social 1,906, Casual <-> Whale 789. Casual <-> Grinder was skipped (round 30).
* The specialists are less accurate on their own pair's rows than the four-class blend's pairwise log-odds (Casual <-> Social
  0.8851 vs 0.8882, Social <-> Whale 0.8499 vs 0.8538; correlation 0.98), so the training folds never use them: as in round 30, a
  two-class model on the same features learns the same boundary from less data.
* Isotonic maps change the probabilities but not the decisions the soft search reaches (weight 0.6 in most folds); the mean effect
  over eight splits is about 0. The two missing splits cannot turn either into a pass.

**Task 5, rank fusion of the classifier ratio with a k-NN distance ratio** (`gb_r36_t5_knn_fusion.py`; competition data, seeds 42
and 1, 20 halvings, 100 shared null draws; the first-release reference run was cut off).

| score | seed 42: k, est. precision, expected F3 | seed 1: k, est. precision, expected F3 | seed Jaccard |
|---|---|---|---|
| classifier 12-D alone (v25) | 242, 0.962, **0.9990** | 249, 0.935, **0.9992** | **0.972** |
| k-NN distance ratio alone (k 15 + 50) | 262, 0.737, 0.9786 | 253, 0.769, 0.9802 | 0.705 |
| classifier + k-NN, 1/2 : 1/2 | 329, 0.662, 0.9874 | 284, 0.737, 0.9860 | 0.670 |
| classifier + k-NN, 2/3 : 1/3 | 284, 0.759, 0.9885 | 309, 0.691, 0.9868 | 0.734 |

* The k-NN ratio agrees with the classifier only at the top (Spearman 0.73-0.76 among the classifier's top 300, 0.14-0.21 over
  all accounts): in 12 dimensions with 24,080 points the distance to the 15th / 50th neighbour is a noisy density estimate, and
  averaging ranks with it lowers the precision and the seed stability. Fails the label-free decision rule fixed before the run;
  **Task 5 stays locked at v25** (user, 2026-11-16).

**Task 1, ranking objectives and leaf embeddings** (`gc_r36_t1_rank_leaf.py`): only the members of pipeline fold 0 were finished
(34 minutes per fold at 3 threads): AUC lambdarank (skill-tier x bucket queries) 0.88891, rank:pairwise (skill-tier queries)
0.88833, against 0.89197 for the pipeline's LightGBM member on the same fold. No level-2 result; the leaf-embedding member did not
start.

**Plan A (user, 2026-11-16 ~23:50): dinov2_r2 on its four finished folds, then the DINOv2 TTA test** (one GPU job at a time, no CPU
job beside it; `scratchpad/queue_r36A.py`, logs `outputs/logs/r36A_*.log`). Views of folds 0-3 took 30 minutes; scores against
v23 on the same four folds (`fc_r31_t4_v23_eval.py`, rules fixed before the run).

| dinov2_r2 variant (folds 0-3) | hard-view mean | folds | without the grid | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | dark / low-contrast | hard-view rule | v24 check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alone (dinov2_v2 alone) | | | | 0.9861 (0.9862) | 0.9716 (0.9727) | 0.9565 (0.9534) | 0.9392 (0.9418) | 0.9071 (0.8586) | 0.9699 (0.9699) | | |
| added at weight 2 | +0.0035 | 4/4 | +0.0007 | -0.0002 | -0.0000 | +0.0007 | +0.0014 | +0.0121 | -0.0005 | PASS | fail |
| added at weight 4 | +0.0059 | 4/4 | +0.0011 | -0.0001 | -0.0000 | +0.0009 | +0.0024 | +0.0204 | -0.0008 | PASS | fail |
| added at weight 8 | +0.0085 | 4/4 | +0.0016 | +0.0001 | -0.0004 | +0.0017 | +0.0035 | +0.0294 | -0.0009 | PASS | fail |
| in place of dinov2_v2 (weight 8) | +0.0076 | 4/4 | -0.0009 | -0.0002 | -0.0012 | +0.0014 | -0.0027 | +0.0331 | -0.0014 | PASS | fail |
| weights (dinov2_v2, dinov2_r2) nested: (4, 8) in every fold | +0.0087 | 4/4 | +0.0007 | -0.0000 | -0.0005 | +0.0016 | +0.0010 | +0.0328 | -0.0014 | PASS | fail |

* The higher resolution helps almost only where the frame is cut into a fine grid (alone, 75% grid 0.9071 vs 0.8586); on full
  frames, zoom70 and the real dark / low-contrast frames it is equal or slightly worse than dinov2_v2, and every combination loses
  on zoom70 and on the dark frames (by fold -0.0040 to +0.0019). The hard-view rule passes on the strength of the grid view, which
  is exactly what the user's v24 check (morning of 16 Nov) was set to catch: zoom70, HUD hidden and the dark frames must all gain.
  It fails that check in every variant on four folds; a fifth fold cannot plausibly turn zoom70 and the dark frames positive.
  Recommendation: do not spend 3.6 GPU hours on fold 4; dinov2_r2 is not adopted.

**Task 4, item 2: photometric, geometric and entropy-gated TTA of the DINOv2 member** (`fu_r34_t4_photo_tta.py dinov2_v2`; every
fold model on its own validation images, 5 views x 6 transforms x h-flip, 54 GPU minutes; the transformed member replaces the
plain one at weight 8 in v23; rule fixed before the run: dark / low-contrast frames > 0 in >= 3/5 folds, full >= -0.0005, mean of
zoom70 / HUD hidden / grid >= -0.0005; the gamma 3.5 view is not counted).

| transform set | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | dark / low-contrast | dark by fold | verdict |
|---|---|---|---|---|---|---|---|---|
| id + autocontrast | -0.0001 | +0.0003 | -0.0010 | +0.0032 | -0.0001 | -0.0002 | 0 -19 +14 -14 +8 (x1e-4) | fail |
| id + autogamma | +0.0001 | -0.0002 | -0.0001 | +0.0091 | -0.0013 | +0.0004 | -14 +13 +14 -18 +26 | fail |
| id + gamma 0.5 | -0.0002 | -0.0006 | -0.0004 | +0.0086 | -0.0022 | +0.0000 | -22 +12 +14 -18 +13 | fail |
| all four photometric | -0.0002 | -0.0003 | -0.0004 | +0.0109 | -0.0008 | +0.0005 | -14 +13 +14 -18 +27 | fail |
| id + HUD greyed out | -0.0010 | -0.0017 | +0.0000 | -0.0058 | -0.0119 | -0.0015 | all negative | fail |
| id + central 85% crop | -0.0003 | -0.0013 | +0.0006 | -0.0013 | -0.0011 | +0.0014 | +6 +28 +14 -14 +33 | fail (zoom / HUD / grid mean -0.0006) |
| all six | -0.0006 | -0.0007 | -0.0001 | +0.0044 | -0.0037 | -0.0001 | -8 -8 +14 -18 +15 | fail |
| entropy-gated photometric (above p80, p90 or p95) | -0.0000 | -0.0001 | -0.0008 | +0.0078 | -0.0007 | +0.0003 | -14 +7 +14 -18 +27 | fail |
| entropy-gated all six (p80 / p90 / p95) | -0.0006 | -0.0007 | -0.0001 | +0.0044 | -0.0037 | -0.0001 | -8 -8 +14 -18 +15 | fail |
| set chosen nested (id + crop85 in every fold) | -0.0003 | -0.0013 | +0.0006 | -0.0013 | -0.0011 | +0.0014 | +6 +28 +14 -14 +33 | fail |

* Alone, every transform makes the member WORSE on the real dark / low-contrast frames (argmax accuracy 0.9738 plain; autocontrast
  0.9697, autogamma 0.9687, gamma 0.5 0.9695, crop 0.9666, HUD greyed 0.9255). The fine-tuned DINOv2 has learned the frames as they
  come; re-lighting moves them away from the training distribution. The large gains are on the synthetic gamma 3.5 view only, which
  autogamma inverts by construction (the reason that view is not counted).
* Entropy gating changes nothing: the gated sets equal the ungated ones to four decimals at all three percentiles. With DINOv2 at
  8/32 of the ensemble, only images that are already uncertain (above the 95th entropy percentile) can change label, so a gate at
  the 80th-95th percentile lets through every image whose decision could move anyway.
* The central crop is the only set that helps the real dark frames (+0.0014, 4/5 folds) and the training folds pick it every time,
  but it costs zoom70 (-0.0013) and the grid (-0.0011). Fails; no TTA change for Task 4.

**Round 36 summary.** No candidate passes: Task 1 focal / ranking members (ranking incomplete), Task 2 isotonic calibration and pair
specialists, Task 3 quantile members (the band recalibration passes the letter of the rule at +0.00001 and is not merged, as round
30's), Task 4 meta-learners, class multipliers, photometric / entropy-gated TTA and dinov2_r2 (hard-view rule only through the grid
view; fails the v24 check on four folds), Task 5 k-NN fusion. v26 stays the pipeline state.

## Round 37 (2026-11-17 night): all-in on Task 4 (user brief of ~01:20)
User decisions: dinov2_r2 fold 4 cancelled; the Task 1 ranking / leaf-embedding run cancelled; **Tasks 1, 2, 3, 5 locked at v26**.
Round 35-36 work and the v26 record committed (`989d712`). GPU queue (`scratchpad/queue_r37.py`, one job at a time, no heavy CPU job
beside it): DINOv2-L fold 0 -> views and v23-based scores -> gate -> folds 1-4 + full refit if it passes -> ConvNeXt-Base fold 0.

**Smoke tests** (704 training images, 1 epoch): DINOv2-L at 126x238 with gradient checkpointing peaks at 7.63 GiB at batch 64
(above the 7.3 GiB limit), 7.16 GiB at batch 32 and 6.94 GiB at batch 64 as 2 x 32 with gradient accumulation (new `--accum` option
of `scripts/train_task4.py` / `task4.train_fold(accum=...)`: each optimiser step still sees 64 images, same schedule; accum = 1 runs
exactly as before). Most of the memory is the 300M-parameter model with its AdamW state and EMA copy, not the activations.
ConvNeXt-Base (`convnext_base.fb_in22k_ft_in1k`, weights downloaded from the timm hub, 338 MB) at 160x320, batch 64: 3.82 GiB.
DINOv2-L fold 0 gate, fixed before the run: one fixed-weight variant (added at weight 2, 4 or 8, or in place of dinov2_v2) passes
the hard-view rule and the v24 check on fold 0.

**Queue 3: confidence-adaptive prior exponent** (`gf_r37_t4_adaptive_prior.py`; v23 members' out-of-fold views; alpha = a_lo where
max p >= t, a_hi elsewhere; base alpha 0.75 everywhere).

| rule | uniform mix: hard-view mean (folds) | full | zoom70 | HUD hidden | gamma | grid | dark frames | training mix: full / zoom70 / HUD / dark | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 if max p >= 0.8, else 0.85 | +0.0008 (5/5) | -0.0003 | -0.0003 | -0.0001 | +0.0009 | +0.0027 | -0.0015 | -0.0009 / -0.0008 / -0.0007 / -0.0031 | fail |
| ... else 0.90 | +0.0010 (5/5) | -0.0004 | -0.0005 | -0.0001 | +0.0011 | +0.0036 | -0.0016 | -0.0011 / -0.0013 / -0.0012 / -0.0036 | fail |
| ... else 0.95 | +0.0013 (5/5) | -0.0004 | -0.0006 | -0.0002 | +0.0014 | +0.0046 | -0.0018 | -0.0013 / -0.0017 / -0.0017 / -0.0040 | fail |
| ... else 1.00 | +0.0014 (5/5) | -0.0004 | -0.0005 | -0.0003 | +0.0016 | +0.0048 | -0.0017 | -0.0014 / -0.0019 / -0.0024 / -0.0041 | fail |
| control: 1.00 for every image | identical to the row above | | | | | | | | fail |
| control: 0.50 for every image | -0.0037 (0/5) | -0.0002 | -0.0001 | -0.0002 | -0.0034 | -0.0112 | -0.0012 | +0.0005 / +0.0015 / +0.0017 / +0.0003 | fail |
| (a_lo, a_hi, t) nested: (0.5, 1.0, 0.7-0.8) | +0.0014 (5/5) | -0.0004 | -0.0005 | -0.0003 | +0.0016 | +0.0048 | -0.0017 | -0.0014 / -0.0019 / -0.0024 / -0.0041 | fail |

* For confident images the exponent does not matter: "0.50 above max p 0.8, 1.00 below" gives exactly the labels of "1.00
  everywhere" (95% of full frames have max p >= 0.8, and at that confidence no exponent between 0.5 and 1 flips the argmax). The
  adaptive rule is therefore just a different global alpha, and alpha trades the two class mixes against each other as found in
  round 23: higher alpha helps the uniform mix on gamma / grid only and costs everything at the training mix; lower alpha the
  reverse. No rule improves both mixes; alpha 0.75 stays.

**Session end at about 01:50:** the previous Claude Code session ended and its background jobs stopped with it; the DINOv2-L fold-0
run had finished epoch 0 only (no checkpoint). Not restarted locally: the user moves the DINOv2-L run to Colab
(`colab/t4_colab_train.ipynb`, built by `scripts/make_colab_bundle.py`; fold 0, the same gate scored on Colab with the bundled v23
views, then folds 1-4 + full refit only if it passes). The bundle reproduces the laptop's v23-based scores exactly (checked with
dinov2_r2's views as a stand-in candidate: identical numbers, gate read as fail).

**Is the Task 4 ensemble bigger than it needs to be?** (`gg_r37_t4_ablation.py`; v23 members' out-of-fold views, v17 rule, uniform
class mix; parameter counts approximate.)

| model / ensemble | params (M) | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | dark frames | hard-view mean |
|---|---|---|---|---|---|---|---|---|
| cnxt alone | 28 | 0.9864 | 0.9524 | 0.9252 | 0.9047 | 0.4864 | 0.9695 | 0.8172 |
| deit3 alone | 22 | 0.9828 | 0.9562 | 0.9194 | 0.9190 | 0.8007 | 0.9600 | 0.8988 |
| cnxt_hud alone | 28 | 0.9846 | 0.9631 | 0.9547 | 0.8990 | 0.7925 | 0.9687 | 0.9023 |
| cnxt22k_hud alone | 28 | 0.9841 | 0.9642 | 0.9543 | 0.9215 | 0.7979 | 0.9635 | 0.9095 |
| effv2_hud alone | 21 | 0.9816 | 0.9603 | 0.9406 | 0.8964 | 0.4091 | 0.9571 | 0.8016 |
| deit3_hud alone | 22 | 0.9814 | 0.9622 | 0.9453 | 0.9104 | 0.8407 | 0.9617 | 0.9146 |
| cnxtS_hud alone | 50 | 0.9861 | 0.9668 | 0.9585 | 0.9019 | 0.8213 | 0.9715 | 0.9121 |
| dinov2_v2 alone | 86 | 0.9860 | 0.9725 | 0.9539 | 0.9408 | 0.8606 | 0.9676 | 0.9320 |
| cnxt + dinov2_v2 (equal) | 114 | 0.9896 | 0.9728 | 0.9556 | 0.9376 | 0.8461 | 0.9756 | 0.9280 |
| cnxt_hud + dinov2_v2 (1 : 2) | 114 | 0.9881 | 0.9754 | 0.9613 | 0.9405 | 0.8727 | 0.9698 | 0.9375 |
| five HUD members + dinov2_v2 (v23 weights) | 235 | 0.9889 | 0.9762 | 0.9643 | 0.9371 | 0.8901 | 0.9730 | 0.9419 |
| v23 (eight members) | 285 | 0.9894 | 0.9760 | 0.9643 | 0.9362 | 0.8880 | 0.9746 | 0.9411 |
| v23 without cnxt | 257 | 0.9890 | 0.9762 | 0.9643 | 0.9367 | 0.8897 | 0.9737 | 0.9417 |
| v23 without deit3 | 263 | 0.9890 | 0.9760 | 0.9644 | 0.9364 | 0.8878 | 0.9737 | 0.9412 |
| v23 without cnxt_hud | 257 | 0.9890 | 0.9758 | 0.9639 | 0.9379 | 0.8859 | 0.9735 | 0.9409 |
| v23 without cnxt22k_hud | 257 | 0.9895 | 0.9763 | 0.9640 | 0.9361 | 0.8855 | 0.9738 | 0.9404 |
| v23 without effv2_hud | 264 | 0.9893 | 0.9762 | 0.9643 | 0.9378 | 0.8916 | 0.9746 | 0.9425 |
| v23 without deit3_hud | 263 | 0.9895 | 0.9762 | 0.9644 | 0.9373 | 0.8804 | 0.9746 | 0.9396 |
| v23 without cnxtS_hud | 235 | 0.9889 | 0.9755 | 0.9628 | 0.9383 | 0.8842 | 0.9727 | 0.9402 |
| v23 without dinov2_v2 | 199 | 0.9881 | 0.9725 | 0.9629 | 0.9215 | 0.8599 | 0.9719 | 0.9292 |

* A single ConvNeXt-T (28 M parameters) already reaches 0.9864 on full frames; the eight-member v23 (about 285 M) 0.9894. Two
  models (ConvNeXt-T + DINOv2-B) match v23 on full frames (0.9896) and on the real dark frames (0.9756 vs 0.9746).
* Leaving out any one CNN member changes the hard-view mean by at most 0.0015 (without effv2_hud it even rises, +0.0014; selected
  from eight leave-one-outs, so not a candidate). Only DINOv2 carries weight: without it the hard-view mean drops 0.012 (gamma
  -0.015, grid -0.028).
* The leaderboard history agrees: the large Task 4 gains came from the decision rule (v17, about +0.024 on Task 4) and from the
  self-supervised DINOv2 backbone (v20, about +0.016, eight times what the training-image proxies predicted); adding CNNs of the
  same kind (v16 ConvNeXt-S +0.0002) or another ViT (v21 EVA02-B -0.0005) did not move it. The remaining gap between the training
  out-of-fold score (0.989) and the test score implied by the leaderboard (about 0.93) is a train / test shift that model capacity
  on training-like frames does not close.

**Why is Task 4 lower on the test images than out of fold? (scoping without test-image analysis, user rule of 2026-11-17)**
Sources allowed: training images, the data card / instruction slides, and the public scores of files already uploaded.
* What the slides and data card say: screenshots from live streams, player submissions, forums and support tickets; "different
  maps, phases, weather, UI states"; 144p centre-cropped to 230x120. The leaderboard is a *weighted* mean of the five task scores
  (slide 26: (w1 S1 + ... + w5 S5) / sum w), weights not published, so the test level of Task 4 cannot be pinned down; with equal
  weights the history implies about 0.91-0.93 against 0.989 out of fold.
* Ruled out on training data: near-duplicate leakage inflating the out-of-fold score (round 18-19: no twins; fine-tuned models
  hardly drop on held-out clusters), metadata artefacts (one batch re-encode, random names and order).
* What the leaderboard responded to (Task-4-only uploads; Task 4 deltas assume equal weights):
  - class-prior correction (v17, alpha 0.75): +0.0047 public, about +0.024 on Task 4, from 949 changed labels, i.e. roughly four of
    five changed labels were corrections: the test class mix is far from training's (39% 7-Eleven Knights), and ambiguous test
    frames had defaulted to the largest class (label shift);
  - HUD-independent members (v15, HUD dropout): +0.0016 public, about +0.008, and a self-supervised backbone (v20, DINOv2-B):
    +0.0033 public, about +0.016, eight times what the training-image proxies predicted (covariate shift in content / UI state
    that the HUD-reading shortcut does not survive and that DINOv2 features tolerate);
  - more of the same (v16 ConvNeXt-S +0.00003, v21 EVA02-B -0.00009, v23 re-weighting +0.00007): nothing.
* So the gap is train / test shift (label shift + a content / UI-state shift), not capacity on training-like frames (the ablation
  above: a 28 M-parameter ConvNeXt-T already reaches 0.986 out of fold). The synthetic hard views capture only part of it, which
  is why they under-predicted v20.
* Excluded by the rules: anything that reads the test images or their statistics to adapt (EM prior re-estimation on test
  predictions, Saerens et al. 2002; test-time entropy minimisation, TENT; pseudo-labels; test-based EDA or validation design).
* Practical options ranked by cost (no Colab compute needed for the first three):
  1. WiSE-FT for the DINOv2 member (Wortsman et al., CVPR 2022): interpolate each fold's fine-tuned backbone with the pretrained
     DINOv2 weights (alpha 0.5 / 0.7 / 0.85, fine-tuned head kept); known to recover robustness lost in fine-tuning at no training
     cost; GPU inference only (about 1-1.5 h on the laptop for the five views of all folds);
  2. a linear probe on frozen DINOv2-L features (fine-tuning distorts pretrained features under shift, linear probes do not;
     Kumar et al., ICLR 2022): forward passes only (minutes), replaces or joins the existing 10% probe;
  3. a second DINOv2-B with the LP-FT recipe (probe the head first, then fine-tune; Kumar et al. 2022): about 6 h on the laptop;
  4. DINOv2-L fine-tuning (Colab): the most expensive and the least supported by the evidence above; only if 1-2 show that more
     DINOv2-like robustness helps on the proxies.

**Option 1, WiSE-FT for the DINOv2 member** (`gh_r37_t4_wiseft.py`; user go-ahead 2026-11-17 ~02:05; each fold's fine-tuned
dinov2_v2 interpolated with the pretrained weights at the same 126x238 grid, fine-tuned head kept; 35 GPU minutes, inference only).

| alpha | member alone: full / zoom70 / HUD hidden / gamma / grid / dark frames | in v23 at weight 8: hard-view mean (folds) | full | zoom70 | HUD hidden | gamma | grid | dark frames | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 (v23) | 0.9860 / 0.9725 / 0.9539 / 0.9408 / 0.8606 / 0.9676 | | | | | | | | |
| 0.95 | 0.9858 / 0.9730 / 0.9537 / 0.9440 / 0.8583 / 0.9667 | -0.0001 (3/5) | -0.0001 | -0.0001 | -0.0001 | +0.0010 | -0.0011 | +0.0000 | fail |
| 0.85 | 0.9850 / 0.9723 / 0.9518 / 0.9487 / 0.8501 / 0.9647 | -0.0005 (1/5) | -0.0005 | -0.0005 | +0.0001 | +0.0027 | -0.0044 | -0.0014 | fail |
| 0.70 | 0.9828 / 0.9681 / 0.9450 / 0.9499 / 0.8256 / 0.9593 | -0.0027 (0/5) | -0.0009 | -0.0011 | +0.0004 | +0.0029 | -0.0130 | -0.0018 | fail |
| 0.50 | 0.9720 / 0.9490 / 0.9168 / 0.9365 / 0.7409 / 0.9407 | -0.0084 (0/5) | -0.0009 | -0.0026 | -0.0007 | -0.0023 | -0.0280 | -0.0017 | fail |
| nested | alpha 1.00 picked in all five folds | +0.0000 | | | | | | | fail |

* Moving towards the pretrained weights helps only the synthetic gamma 3.5 view and costs everything else, the real dark /
  low-contrast frames included; the training folds keep the fine-tuned model in every fold. Caveat: this is the variant with the
  fine-tuned head kept (the pretrained DINOv2 has no classifier); the original WiSE-FT also interpolates the head with a zero-shot
  (or linear-probe) head, which needs a probe on frozen features (option 2's computation). Not adopted.

**Task 2 learning curve** (`gj_r37_t2_learning_curve.py`; user request after a competitor reported a higher Task 2 score without
naive Bayes; training rows of each pipeline fold subsampled, stratified; scored on the full held-out folds; argmax, no multipliers).

| training rows per fold | LightGBM macro-F1 / log-loss | KDE NB macro-F1 / log-loss | 0.4 / 0.6 blend macro-F1 / log-loss |
|---|---|---|---|
| 10,400 (12.5%) | 0.7459 / 0.6317 | 0.7486 / 0.6407 | 0.7508 / 0.6166 |
| 20,800 (25%) | 0.7494 / 0.6214 | 0.7512 / 0.6326 | 0.7537 / 0.6107 |
| 41,600 (50%) | 0.7541 / 0.6128 | 0.7549 / 0.6278 | 0.7554 / 0.6066 |
| 62,400 (75%) | 0.7556 / 0.6086 | 0.7562 / 0.6263 | 0.7571 / 0.6050 |
| 83,200 (100%) | 0.7559 / 0.6070 | 0.7560 / 0.6251 | 0.7571 / 0.6044 |

* The curve is flat at the end: the last third of the rows adds +0.0003 (LightGBM), -0.0002 (naive Bayes), 0.0000 (blend) macro-F1
  and 0.0006 log-loss to the blend. Power-law fits put twice the rows at +0.0011 to +0.0023 and the unlimited-data asymptote at
  0.762-0.769 (fits on five points, an upper bound). Even unlimited rows from the same generator stay near 0.76-0.77: the rest is
  label noise, as the round-18 calibration check found.
* So a cross-validated Task 2 macro-F1 clearly above about 0.765 cannot come from a better learner or more data; it would point to
  a different metric (our accuracy is 0.770, weighted-F1 0.769), a single favourable hold-out (our folds range 0.7536-0.7593),
  in-sample tuning of the decision rule, or information outside the training rows.

**Task 3, last round: the "spent nothing this month" error and an age-stratified blend** (`gk_r37_t3_churn.py`; user brief of
2026-11-17 ~03:05, about 30 CPU minutes; reference = the pipeline rebuilt fully nested, 0.79457). Rows predicted to spend that
spent 0 carry 32.9% of the absolute error, 32.7% of it in segments 1-3.

| candidate | nested NMAE | delta | folds better | per fold | verdict |
|---|---|---|---|---|---|
| 1a stage-1 specialist on segment >= 1 rows (slowdown ratios: prev month vs history per 30 days of age, recent vs lifetime daily spend, vip x purchase frequency, premium x days since purchase; 3 seeds) | 0.79456 | -0.00002 | 0/5 | -0.00005 -0.00000 -0.00000 -0.00002 -0.00001 | fail |
| 1b mean log-odds of the specialist and the pipeline on segment >= 1 | 0.79457 | -0.00001 | 2/5 | -0.00003 +0.00000 -0.00002 -0.00000 +0.00000 | fail |
| 2 exact L1 blend per account-age half (missing ages with the older half), shrunk 0.5 to the global weights | 0.79456 | -0.00001 | 2/5 | +0.00000 -0.00001 +0.00000 -0.00003 -0.00001 | fail |

* The specialist ranks the segment 1-3 rows exactly as the pipeline's stage 1 does (AUC 0.7410 vs 0.7412; within segments 1 / 2 /
  3: 0.632 / 0.604 / 0.597 vs 0.633 / 0.605 / 0.598; log-loss 0.5407 vs 0.5404): the slowdown ratios carry nothing the trees did
  not already build from the same columns, and whether a spender skips a month stays close to unpredictable. (Item 2 first crashed
  on the missing account ages, which made both halves empty; rerun with the missing ages in the older half.)
* **Task 3 locked at v26 (user decision of 2026-11-17).**

**Task 1 learning curve and calibration** (`gl_r37_t1_learning_curve.py`; LightGBM with the pipeline's parameters on the 49
features, labelled training rows of each pipeline fold subsampled, scored on the full held-out folds; F2 with the pipeline's
cut-off rule).

| training rows per fold | AUC | log-loss | F2 |
|---|---|---|---|
| 9,082 (12.5%) | 0.87093 | 0.40327 | 0.82156 |
| 18,163 (25%) | 0.88300 | 0.38195 | 0.82967 |
| 36,326 (50%) | 0.88682 | 0.37394 | 0.83366 |
| 54,490 (75%) | 0.88827 | 0.36984 | 0.83520 |
| 72,653 (100%) | 0.88947 | 0.36806 | 0.83492 |

* Flat at the end: the last quarter of the rows adds +0.0012 AUC and no F2 (-0.0003, inside the fold noise); power-law fits put
  twice the rows at +0.0007 AUC / +0.0008 F2 and the asymptotes at AUC 0.890 / F2 0.837 for this learner.
* The pipeline's level 2 is calibrated (expected calibration error 0.0032; deciles of p match the observed cheater rates within
  0.011). If its probabilities were the truth, the best achievable F2 would be 0.8383; the observed best single-cut F2 is 0.8388.
  The remaining errors are exactly the uncertainty the model reports: 56% of the rows sit at 0.1 < p < 0.9, where cheaters and
  legitimate players overlap in every feature; the confident ends are nearly pure (p >= 0.95: 99.8% cheaters; p <= 0.05: 0.16%).
* **Task 1 locked at v26 (user decision of 2026-11-17).** Tasks 1, 2, 3 and 5 are now at the limits the training data allow.

**Option 2, logistic probe on frozen DINOv2-L features** (`gi_r37_t4_probeL.py`; user go-ahead 2026-11-17 ~02:48; frozen
vit_large_patch14_reg4_dinov2.lvd142m at 224 x 224, pooled 1,024-d, image + h-flip; feature extraction on the laptop GPU took 53
minutes (training images under five views, then test images, cached as outputs/preds/t4_feat[_views]_large14_reg4_dinov2.npz); the
existing probe's recipe: C 0.03, fitted on the full / zoom70 / HUD-hidden features, out of fold on the members' folds. The first run
crashed after the extraction (the two extra views clashed with the existing probe's loader); rescored from the caches.)

| probe alone (v17 rule) | full | zoom70 | HUD hidden | dark frames |
|---|---|---|---|---|
| existing (CLIP + DINOv2-B + SigLIP 2, 2,304-d) | 0.9240 | 0.8917 | 0.8596 | 0.8578 |
| DINOv2-L (1,024-d) | 0.9084 | 0.8684 | 0.8342 | 0.8335 |
| existing + DINOv2-L (3,328-d) | **0.9369** | **0.9030** | **0.8788** | **0.8766** |

| in the final average (0.9 x v23 members + 0.1 x probe) vs the existing probe | full | zoom70 | HUD hidden | dark frames | zoom / HUD folds better | verdict |
|---|---|---|---|---|---|---|
| A probe-L instead of the existing probe | -0.0002 | -0.0003 | -0.0001 | +0.0008 | 0/5 | fail |
| **B probe on existing + DINOv2-L features** | **-0.0000** | **+0.0000** | **+0.0008** | **+0.0002** | 3/5 | **passes the rule set before the run** |
| C probe-L, weight nested in {0.1, 0.2, 0.3} (0.2 in four folds) | -0.0002 | -0.0003 | -0.0009 | +0.0004 | 0/5 | fail |

* Frozen DINOv2-L at 224 x 224 alone is weaker than the three base-size backbones together, but it adds to them: the combined probe
  gains +0.013 on full frames, +0.011 zoomed, +0.019 HUD hidden and +0.019 on the real dark frames over the existing probe alone.
* At its 10% share the effect on the ensemble is small (zoom70 positive only in the fifth decimal; dark frames by fold +0.0028 -0.0006
  -0.0016 -0.0002 +0.0006). It passes the rule fixed before the run, at a size the public split cannot resolve. Members + probe-L on
  the synthetic views: gamma 3.5 +0.0029, 75% grid -0.0070 (against members only; the existing probe has no such views).
* No training is needed to use it: adding the DINOv2-L backbone to `scripts/train_task4_probe.py` reuses the cached features (test
  features included) and rewrites the probe files in the same layout; candidate B is exactly that probe.


### v27 prepared (user decision, 2026-11-17 ~07:30: v26 + the probe of candidate B; not uploaded yet)

* `scripts/train_task4_probe.py`: DINOv2-L (vit_large_patch14_reg4_dinov2.lvd142m, 224 x 224) added as a fourth frozen backbone; the
  cached features of `gi_r37_t4_probeL.py` were reused (no GPU work), the probe refitted on the CPU in about 3 minutes
  (`outputs/r37/v27_probe_refit.log`). The v19-v26 probe files are kept in `outputs/preds/probe_v19_v26/` and in
  `versions/v26_LB0.86147/`. The refitted probe reproduces candidate B exactly: out of fold 0.9240 -> 0.9369 (balanced macro-F1,
  v17 rule), accuracy 0.9326 -> 0.9427, folds 0.940-0.945.
* `task4/task4.ipynb` rebuilt (only the probe's description changed) and executed (71 s): probe alone 0.9232 -> 0.9353 (macro-F1 at
  argmax), ensemble + 10% probe 0.9896 -> 0.9895, the v17 rule 0.9884 -> 0.9883 at the training mix and 0.9892 -> 0.9892 at a
  balanced mix; as in the experiment, the change is on the hard views (HUD hidden +0.0008, dark frames +0.0002), not on full frames.
* `submission_v27.csv` (`scripts/make_submission.py --out submission_v27.csv`; sha256 fe7794e4...748f0c): **184 of 25,889 Task 4
  labels change** (0.71%); Tasks 1, 2, 3, 5 identical to v26. `submission.csv` stays v26 until an upload.
  Predicted class shares on the test images, v26 -> v27 (reported only; no test statistic chooses anything): 7-11k 0.1425 -> 0.1416
  (-22), fiveN 0.2718 -> 0.2721 (+8), freefried 0.1438 -> 0.1433 (-12), pube 0.2441 -> 0.2446 (+13), roblock 0.1978 -> 0.1983
  (+13). The 184 moves are spread over every pair of classes (largest: fiveN -> roblock 24, roblock -> fiveN 19, freefried ->
  fiveN 17, 7-11k -> roblock 16): borderline frames that the 10% probe tips, not a shift towards one class.
* Expectation stated before any upload (as for v26): the nested gain is a few ten-thousandths on the hard views and zero on full
  frames, below what the public split resolves for 184 changed labels; v27 is kept on the nested evidence whatever its public
  score (user, 2026-11-17).
* **Uploaded 2026-11-17 08:25 (user's approval; Kaggle CLI, ref 58451173): public 0.86177**, +0.00030 over v26 (0.86147) and +0.00029
  over v25 (0.86148): the best public score so far (top of the LB 0.86750). The size is at the edge of the public split's noise for
  184 changed labels, so it neither confirms nor refutes the nested gain; v27 is kept on the nested evidence, as agreed before the
  upload. Snapshot `versions/v27_LB0.86177/`, `submission.csv` = v27.

## Round 38 (2026-11-17): all-in on Task 4 foundation robustness (user brief after v27 = 0.86177)

Two steps (user): (1) a "super-probe" on frozen features of large backbones from other families, on the laptop; (2) DINOv2-L with
LP-FT and full WiSE-FT, on Colab. Leaderboard at the start: ML003 0.87046, ML008 0.86405, ML007 (us) 0.86177, fourth 0.85209.

**Step 1a, frozen features** (`gm_r38_t4_feat_extract.py`; training images under the five scoring views, test images full frame;
inference only, caches in the probe's layout). The frames are 230 x 120, so inputs above 224 only upsample: the brief's 384 / 448
checkpoints would have cost 3-7x more on the laptop for no new pixels.

| backbone (timm) | input | dim | where | time |
|---|---|---|---|---|
| vit_so400m_patch14_siglip_224.v2_webli (SigLIP 2 SO400M) | 224 | 1,152 | laptop | 77.7 min |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k (position embedding resampled) | 224 | 1,024 | Colab A100 | both under about 30 min |
| convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320 | 320 | 1,536 | Colab A100 | (from the upload to the files) |

The two Colab caches were checked against a laptop recomputation of 128 training images (all five views) and 64 test images: cosine
0.9998 (EVA-02-L, bf16 on a different GPU) and 0.99999 (ConvNeXt-L); shapes, finiteness and the test order were checked too.

**Step 2 code, LP-FT and full WiSE-FT** (for Colab; `colab/t4_colab_train.ipynb`, built by `colab/build_t4_colab_nb.py`):
`src/task4.fit_lp_head` fits the frozen probe's logistic regression (three views, C 0.03) on the pre-logit features of the model at
its fine-tuning input (126 x 238) and folds the standardisation into the model's own head; `train_fold(init_head=...)` starts
fine-tuning (and the EMA) from it; `scripts/train_task4.py --lp-ft` caches the head per fold as `models/t4_<tag>_fold<f>_lphead.pt`.
`gn_r38_t4_wiseft_full.py` interpolates every weight, head included, between the fine-tuned model and theta_0 = pretrained backbone +
LP head (round 37's WiSE-FT had no head for the pretrained end), writes the five views per alpha for `fc_r31_t4_v23_eval.py`, and
with `--member` the member files (val + test probabilities) per alpha. Smoke test with DINOv2-S, `--epochs 0`, fold 0: theta_0's
validation macro-F1 0.8506 equals the trained fold's (the head folding and the EMA start are exact), views of alpha 0 and 0.5 are
identical (as they must be at 0 epochs), member files equal the trainer's to the last digit. Recipe on Colab: lr 2e-5, layer decay
0.85, 2 + 13 epochs, HUD dropout 0.5, 126 x 238, gradient checkpointing below 60 GB; fold-0 gate as in round 37 (alpha = 1 or a
WiSE-FT point must pass the hard-view rule and the v24 check).

**Step 1b, super-probe** (`go_r38_t4_superprobe.py`, 16 min on the laptop GPU; candidates and rule fixed before the run, see its
docstring; reference v27 = members + 10% logistic probe on the first four backbones: full 0.9892, zoom70 0.9763, HUD hidden 0.9656,
dark frames 0.9741).

| probe alone (v17 rule) | full | zoom70 | HUD hidden | dark frames |
|---|---|---|---|---|
| v27: logistic, base4 (3,328-d), C 0.03 | 0.9370 | 0.9031 | 0.8785 | 0.8774 |
| logistic, base4 + new3 (7,040-d), C 0.01 / 0.03 / 0.1 | 0.9489 / 0.9438 / 0.9401 | 0.9203 / 0.9153 / 0.9103 | 0.8971 / 0.8911 / 0.8850 | 0.8990 / 0.8913 / 0.8863 |
| **MLP, base4 + new3** | **0.9724** | **0.9510** | **0.9321** | **0.9342** |
| (diagnostic) SigLIP2-SO400M / EVA-02-L / ConvNeXt-L CLIP alone, logistic | 0.8968 / 0.8366 / 0.8746 | | | |

| against v27 | full | zoom70 | HUD hidden | dark frames | zoom / HUD folds | verdict |
|---|---|---|---|---|---|---|
| S1 logistic, C 0.03, w 0.10 | +0.0001 | +0.0003 | +0.0002 | -0.0001 | 3/5 | fail |
| S2 logistic, (C, w) nested (C 0.01, w 0.15 in four folds) | -0.0000 | +0.0002 | -0.0003 | -0.0003 | 2/5 | fail |
| **S3 MLP, w nested (0.20 in all five folds)** | **+0.0009** | **+0.0009** | **+0.0003** | **+0.0032** | **5/5** | **PASS** |
| (diagnostic) base4 + one new backbone, logistic C 0.03, w 0.10 | +0.0001 to +0.0005 | -0.0002 to +0.0002 | -0.0003 to +0.0003 | -0.0000 to +0.0012 | | all fail |

* With a linear probe the three new backbones add little (each alone is weaker than the four of v27 together); a small MLP on the
  same 7,040 features is the change: the probe alone jumps from 0.937 to 0.972 on full frames and from 0.877 to 0.934 on the real dark
  frames, and at 20% it lifts every view of the ensemble, the dark frames by +0.0032 (by fold +0.0000 +0.0026 +0.0056 +0.0047
  +0.0032). In-sample the MLP's gain grows with its weight (w 0.10: dark +0.0012, HUD -0.0004; 0.15: +0.0022 / -0.0001; 0.20: +0.0032
  / +0.0003); 0.20 is the top of the grid fixed before the run and was not extended.

**v27 -> v28 prepared (user go-ahead 2026-11-17 ~14:05):** `scripts/train_task4_probe.py` with the three new backbones and PROBE = "mlp"
(the S3 code), `PROBE_W4` 0.10 -> 0.20 (`scripts/nb_part2.py`); the v27 probe files are kept in `outputs/preds/probe_v27/` and in
`versions/v27_LB0.86177/`. The refitted probe reproduces S3 exactly (probe alone 0.9724; ensemble full +0.0009, dark +0.0032).
`task4/task4.ipynb` re-executed: rule at the training mix 0.9883 -> 0.9891, at a balanced mix 0.9892 -> 0.9901. `submission_v28.csv`
(sha256 d725001f...83f7ef03): **259 of 25,889 Task 4 labels change** (1.0%), spread over all pairs (largest pube -> freefried 34,
fiveN -> roblock 30, freefried -> pube 25); class shares move by at most 0.05 points; Tasks 1, 2, 3, 5 identical to v27.
On the training images (out of fold, full frame) the same change flips only 51 labels (0.16%): v28 right in 36, v27 right in 11,
both wrong in 4 (net +25); on the dark frames 16 flips, 13 to 1 for v28. The test images flip six times as often, as expected under
the train / test shift (reported only; nothing is chosen from it).
Expectation stated before any upload (equal task weights assumed; the leaderboard's weights are unknown): if the test flips are won
at v27's observed margin (about +0.0003 public for 184 flips), about +0.0004; at the training images' margin (36 : 11), about +0.0010;
the public split's noise for about 260 labels is about +-0.0003. Central guess 0.8622-0.8625, range 0.8617-0.8630. Even if every one of
the 259 flips were a correction, Task 4 would gain about +0.010 and the public score about +0.002 (0.8638): v28 alone cannot reach
0.864.

**Step 2 on Colab, fold-0 gate: FAIL** (A100, 0.61 h; 106 s per epoch against 656 s on the laptop, 6.2x). LP head (theta_0) 0.8955,
LP-FT fold 0 0.9865 (dinov2_v2 0.9869). Alone, on fold 0 against dinov2_v2: full 0.9857 vs 0.9855, zoom70 0.9738 vs 0.9720, gamma 3.5
0.9555 vs 0.9412, 75% grid 0.8729 vs 0.8529, real dark 0.9764 vs 0.9754, but HUD hidden 0.9535 vs 0.9568. In the v23 ensemble every
alpha >= 0.7 passes the hard-view rule (alpha 1 at weight 8: hard mean +0.0063, mostly gamma +0.0145 and grid +0.0094) and none passes
the v24 check: HUD hidden falls whenever zoom70 rises (alpha 1, weight 8: zoom70 +0.0016, HUD hidden -0.0003, dark +0.0006; weight 2:
zoom70 +0.0000, dark +0.0000). alpha 0.5 is worse and theta_0 alone (frozen backbone + LP head) costs on every view. The same pattern
as dinov2_r2 in round 36: gains on the synthetic gamma / grid views, not on HUD-hidden and real dark frames. The notebook stopped as
designed; folds 1-4 were not trained.
* **v28 uploaded 2026-11-17 14:17 (user's approval; Kaggle CLI, ref 58460556): public 0.86192**, +0.00015 over v27: positive, at the
  low end of the range stated before the upload (0.8617-0.8630; central 0.8622-0.8625), i.e. the test flips were won at a smaller
  margin than v27's or the public split's noise went against it. Best public score so far. Snapshot `versions/v28_LB0.86192/`
  (preds pruned to the git-tracked files, as v26 / v27), `submission.csv` = v28, README rows.
* Colab outputs of the fold-0 run stored on the laptop (2026-11-17 ~14:40, from the user's Drive download): checkpoint and LP head in
  `outputs/models/` (git-ignored), `outputs/logs/t4_dinov2L_lpft*`, gate logs in `outputs/r38/`, the five fold-0 view files in
  `outputs/r30_views/dinov2L_lpft*_fold0.npz`, `outputs/preds/t4_dinov2L_lpft_fold0.npz`; the other 48 files in the download were
  byte-identical to the repo's.

## Round 39 (2026-11-17 afternoon): building on the v28 probe (user brief after v28 = 0.86192)

**Step 1, probe heads / weight / gating** (`gp_r39_t4_probe_heads.py`, 17 min on the laptop GPU; candidates and the acceptance through
N fixed before the run, see its docstring; reference v28: full 0.9901, zoom70 0.9772, HUD hidden 0.9659, dark frames 0.9772).
Probes alone (v17 rule; full / zoom70 / HUD hidden / dark): raw MLP (v28) 0.9724 / 0.9510 / 0.9321 / 0.9342; per-backbone-L2 MLP
0.9736 / 0.9510 / 0.9327 / 0.9401; L2 logistic (C 0.01) 0.9494 / 0.9186 / 0.8961 / 0.9005; L2 cosine head 0.8924 / 0.8678 / 0.8246 /
0.8316; mean of the three L2 heads 0.9614 / 0.9339 / 0.9139 / 0.9187. The members' maximum probability is below 0.75 for 3.5% of the
full frames, 7.6% zoomed, 11.5% HUD hidden, 7.1% of the dark frames.

| against v28 (settings nested) | full | zoom70 | HUD hidden | dark | zoom / HUD folds | verdict |
|---|---|---|---|---|---|---|
| R1 raw MLP, w in {0.20, ..., 0.35} (0.35 in 5/5) | +0.0006 | +0.0011 | -0.0000 | +0.0027 | 5/5 | fail (HUD) |
| R2 L2 MLP, w (0.35 in 4/5) | +0.0007 | +0.0011 | +0.0002 | +0.0017 | 5/5 | pass |
| R3 mean of the three L2 heads, w | never chosen | | | | | fail |
| R4 raw MLP gated (0.35 / 0.40 in 4/5) | +0.0004 | +0.0009 | -0.0003 | +0.0008 | 4/5 | fail |
| R5 multi-head gated | never chosen | | | | | fail |
| **N, nested over all 28 settings** (R1 0.35 in four folds, R2 0.35 in one) | **+0.0006** | **+0.0011** | **+0.0001** | **+0.0022** | **5/5** | **PASS** |

* By the procedure fixed before the run, N passing makes the pipeline setting the best one on all five folds: **R1, the v28 probe at
  w 0.35** (in-sample against v28: full +0.0006, zoom70 +0.0011, HUD hidden -0.0000, dark +0.0027). On its own its HUD-hidden delta
  is flat; R2 (L2 MLP at 0.35: +0.0008 / +0.0011 / +0.0002 / +0.0023) is as good within noise but needs a new probe variant.
* The probe weight is again chosen at the top of the grid (0.20 < 0.25 < 0.30 < 0.35 in-sample on zoom70 and the dark frames): the
  frozen-feature probe keeps adding robustness as it gets more say. The multi-head mean is worse than the MLP alone (the logistic and
  cosine heads are far weaker), and confidence gating does not help (w_high 0.50 costs -0.0016 on HUD hidden).

**v28 -> v29 prepared (user go-ahead 2026-11-17, R1 as fixed before the run):** `PROBE_W4` 0.20 -> 0.35 (`scripts/nb_part2.py`; notebook
text in `build_task_notebooks.py`); the probe files are v28's (sha identical to `versions/v28_LB0.86192/`), nothing refitted.
`task4/task4.ipynb` rebuilt and re-executed (log `outputs/r39/v29_task4_nb.log`): OOF argmax 0.9897 -> 0.9901, rule at the training
mix 0.9891 -> 0.9896, at a balanced mix 0.9901 -> 0.9907; the class multipliers stay off in both (nested gain -0.0001 / +0.0004,
below the 0.0005 bar), so the probe weight is the only difference. `submission_v29.csv` (sha256 64cda3ae...ffdc95260): **315 of
25,889 Task 4 labels change** (1.2%), spread over all pairs (largest Free Fried -> PUBE 41, Roblock -> PUBE 28, FiveN -> Roblock 26);
class shares move by at most 0.09 points; Tasks 1, 2, 3, 5 identical to v28. Not uploaded; `submission.csv` is still v28.
On the training images (OOF, full frame) the same change flips 55 labels (0.17%): v29 right 29, v28 right 20, both wrong 6 (net +9,
against +25 for v27 -> v28); dark / low-contrast frames 22 flips, 13 to 7 for v29. The test images flip seven times as often.
Expectation stated before any upload (equal task weights assumed): at the training images' margin (29 : 20) about +0.0004 public; at
the fraction of that margin v28 realised (+0.00015 of a predicted +0.0010, about 15%) about +0.0001; the public split's noise for
about 300 labels is about +-0.0003. Central guess 0.8619-0.8621, range 0.8616-0.8625: a step within noise, kept or not on the nested
evidence as agreed. Base for round 39 step 2 (ViT-H/14 CLIP + Swin-L features from Colab) is v29 -> a pass there becomes v30.

**Step 2, two more backbones for the probe: FAIL** (`gq_r39_t4_probe9.py`, 36 min on the laptop GPU; log `outputs/r39/gq_probe9.log`).
Features of OpenCLIP ViT-H/14 LAION-2B (1,280) and Swin-L IN-22k (1,536) extracted on Colab by the user (`gm_r38_t4_feat_extract.py`,
`TRAIN = False`); checked on the laptop: shapes 31,546 x 5 views / 25,889 test rows, finite, test rows in `test.csv` order, and 128
training images re-embedded on the laptop GPU agree at cosine >= 0.9998 on every view (fp16 storage). Candidates and the rule fixed
before the run (docstring); reference v29 (members + 0.35 x the seven-backbone MLP): full 0.9907, zoom70 0.9783, HUD hidden 0.9659,
dark frames 0.9799. Probes alone (full / zoom70 / HUD hidden / dark): P7 (v29) 0.9724 / 0.9510 / 0.9321 / 0.9342 (reproduces v28's
probe exactly); P9 (+ ViT-H + Swin-L) 0.9755 / 0.9542 / 0.9348 / 0.9384; P8H (+ ViT-H) 0.9738 / 0.9520 / 0.9336 / 0.9344; P8S (+ Swin-L)
0.9754 / 0.9536 / 0.9329 / 0.9406; ViT-H alone 0.9418 / 0.9058 / 0.8802 / 0.8760; Swin-L alone 0.9534 / 0.9218 / 0.8729 / 0.9025.

| against v29 (w nested) | full | zoom70 | HUD hidden | dark | zoom / HUD folds | verdict |
|---|---|---|---|---|---|---|
| A P9, w in {0.30, ..., 0.45} | +0.0001 | -0.0002 | +0.0001 | -0.0015 | 2/5 | fail |
| B P8H | -0.0001 | +0.0000 | -0.0001 | +0.0001 | 3/5 | fail |
| C P8S | +0.0002 | -0.0003 | -0.0003 | -0.0007 | 0/5 | fail |
| W v29 probe at 0.40 / 0.45 | never chosen | | | | | fail |
| **N, nested over all 14 settings** | **+0.0001** | **-0.0001** | **+0.0001** | **-0.0012** | 3/5 | **fail** |

* The stronger probe (alone +0.003 to +0.004 on every view) does not carry into the ensemble: at 35-40% the members already get right
  what the probe now adds, and the per-fold picks scatter (P8H / P9, 0.30-0.40); the dark-frame deltas swing by fold (-0.0043 to
  +0.0020). Nine backbones of frozen features is past the point of returns for this probe.
* The weight alone saturates at 0.35: the v29 probe at 0.40 / 0.45 is worse in-sample on every view (0.45: zoom70 -0.0009, dark
  -0.0012), so 0.35 is the peak of the weight curve, not an edge of the grid. v29 stays the candidate; no v30 from step 2.
* **v29 uploaded 2026-11-17 20:52 (user's approval; Kaggle CLI, ref 58470803): public 0.86197**, +0.00005 over v28, inside the
  central range stated before the upload (0.8619-0.8621). Best public score so far. `submission.csv` = v29, snapshot
  `versions/v29_LB0.86197/` (preds pruned to the git-tracked files, as v26-v28), README rows.

**Step 3 prepared for Colab (user go-ahead 2026-11-17 ~21:00):** `colab/t4_colab_train.ipynb` rebuilt from `colab/build_t4_colab_nb.py`
with `TAG = "dinov2L_hud70"`, `TRAIN = True`, `EXTRACT = []`, `LP_FT = False` (no LP head, no WiSE-FT): DINOv2 ViT-L/14 at 126 x 238,
lr 3e-5, layer decay 0.85, 2 + 13 epochs, HUD dropout 0.70, batch 64, gradient checkpointing below 60 GB. Gate fixed before the run (the
user's rule): fold 0 scored against the v23 members by `fc_r31_t4_v23_eval.py`; folds 1-4 and the full refit only if one fixed-weight
variant passes the v24 check (zoom70, HUD hidden, real dark / low-contrast frames all > 0, full >= -0.0005; the hard-view mean rule is
printed, no longer required). After all folds the member is judged on the laptop against v29 (members + 35% probe).
`em_r30_t4_hardviews.py` knows the new tag (replaces dinov2_v2); bundle `colab/upload/cpe342_code.zip` rebuilt (35,108,374 bytes).
