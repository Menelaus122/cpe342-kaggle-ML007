# v6: public leaderboard 0.76202 (best so far)

Snapshot taken 2026-11-06.

The v5 file (player_id leak test: Task 2 = id block, Task 5 = ids >= 23,819) with **Task 3 replaced** by the
round-2 model (`x3_t3_candidate.py`):
- missing spending values rebuilt from exact row identities (`t3_fill.py`);
- all 8 pool members retrained on the filled features, notebook recipe (OOF L1 weights, zero gate, scale,
  Train+Val refit averaged 50/50 with the 5-fold models);
- whale-only LightGBM (segment 3) mixed 50/50 on whale rows.
Task 3 OOF 0.78917 -> 0.79097. Tasks 1, 2, 4, 5 identical to v5.

Not part of the pipeline (submission.csv is still v3). Whether player_id may be used is open with the TAs
(README question #6).

submission.csv sha256 starts `3624c7017bea14af`
