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
