# v5: public leaderboard 0.76 (as reported)

Snapshot taken 2026-11-05 19:54.

**player_id leak test, not part of the pipeline.** v3's submission with two columns replaced
(`t_make_id_submission.py`):
- Task 2 = id block (ids < 52,000 -> 0, < 84,500 -> 1, < 110,500 -> 2, else 3).
- Task 5 = every id >= 23,819 (2,070 accounts = v3's 1,811 family accounts + 259).
- Tasks 1, 3, 4 identical to v3.

Whether player_id may be used is open with the TAs (README question #6). Until they answer, keep v3 selected
as the final submission on Kaggle.

submission.csv sha256: `37d833f3db5ca52407bc84cd7282c85b13067a22696e4e58e4a499773c3b72cf`
