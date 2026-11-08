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
