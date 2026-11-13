# v10: public leaderboard 0.76257 (v9 was 0.76242)

Snapshot taken 2026-11-08. v9 with only the Task 4 column reordered to sorted image-filename order (ANS00001 = s001rpz.jpg,
ANS00002 = s001six.jpg, ...) instead of test.csv row order (`ax_t4_sorted_order.py`; user request). Same predictions,
same class counts. Result: +0.00015 = +0.00075 on the Task 4 score, i.e. still chance level (a correct order would give ~0.91).
So the answer key is in neither test.csv order nor sorted-filename order. The +0.00015 is chance noise (sd of the difference
between two random alignments ~0.0009 on the leaderboard), not an improvement. Tasks 1, 2, 3, 5 identical to v9.

submission.csv sha256: `a5dd9297188da45ff269f03cbb76607defbc699cef9f24bdc9306671b8920cff`
