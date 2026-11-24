# v7: public leaderboard 0.76224 (best so far)

Snapshot taken 2026-11-06.

v6 (0.76202) with **Task 3 replaced** (`ae_t3_v7.py`):
- missing spending values rebuilt from exact row identities (`t3_fill.py`);
- missing `segment` imputed from the other features by a classifier trained on train rows (`t3_segment.py`, hold-out accuracy 0.9993);
- new ratio-target hurdle members: stage 2 learns log1p(y) - log1p(historical_spending / prev_month_spending);
- notebook recipe: OOF L1 blend weights, zero gate, scale, Train+Val refit averaged 50/50 with the 5-fold models;
- whale-only model dropped (no longer helps).
Task 3 OOF 0.79097 -> 0.79331. Tasks 1, 2, 4, 5 identical to v6 (Tasks 2 and 5 use player_id; README question #6).

Not part of the pipeline (submission.csv is still v3).

submission.csv sha256 starts `f43da882e2a3e74b`
