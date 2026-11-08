"""Task 5: label-free detector for the 'performance volatility' pocket found by y2_t5_pocketscan.py.

Score = Fisher combination of upper-tail ranks of kda_std_level, accuracy_std_level, performance_stability_level:
    S = sum_j -log(1 - u_j)          (u_j = rank percentile among non-family accounts)
Null = the same score after shuffling each column independently (keeps every marginal, destroys any group).
For the k highest-scoring accounts, expected normals = mean permuted count above that score, so the
estimated number of real group members is k - expected, with no labels. k is chosen to maximise the
expected F3 = 10 TP / (9 K + N). The id is used only at the end, to grade the result.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from src import task5  # noqa: E402

FEATS = ["kda_std_level", "accuracy_std_level", "performance_stability_level"]
N_PERM = 300

t5 = task5.load()
X5, _ = task5.build_features(t5)
_, gc_masks = task5.gap_clusters(t5, X5)
fam = gc_masks.any(axis=1).values
idn = t5.player_id.str[1:].astype(int).values
rest = np.where(~fam)[0]

U = X5.loc[~fam, FEATS].rank(pct=True).to_numpy()          # NaN stays NaN
T = -np.log(np.clip(1 - U + 0.5 / len(U), 1e-12, 1))


def score(T):
    return np.nansum(T, axis=1) * len(FEATS) / np.maximum(np.isfinite(T).sum(1), 1)


S = score(T)
order = np.argsort(-S)
Ss = S[order]
rng = np.random.default_rng(0)
null_counts = np.zeros(len(Ss))
for _ in range(N_PERM):
    Tp = np.column_stack([rng.permutation(T[:, j]) for j in range(T.shape[1])])
    Sp = np.sort(score(Tp))
    null_counts += len(Sp) - np.searchsorted(Sp, Ss, side="left")   # permuted accounts with score >= Ss[k]
null_counts /= N_PERM
k = np.arange(1, len(Ss) + 1)
est_tp = np.maximum.accumulate(k - null_counts)                   # estimated group members among the top k
group_size = est_tp[:3000].max()
n_fam = int(fam.sum())
K_est = n_fam + group_size                                          # families assumed all true
f3 = 10 * (n_fam + est_tp) / (9 * K_est + n_fam + k)
k_star = int(np.argmax(f3[:3000])) + 1
print(f"label-free estimates: group size ~{group_size:.0f}; best k = {k_star} "
      f"(estimated members {est_tp[k_star - 1]:.0f}, estimated precision {est_tp[k_star - 1] / k_star:.2f}); "
      f"expected F3 families-only {10 * n_fam / (9 * K_est + n_fam):.3f} -> with the pocket {f3[k_star - 1]:.3f}")
for kk in (100, 200, 300, 400, 600):
    print(f"  top {kk}: estimated members {est_tp[kk - 1]:.0f}, marginal precision of the last 100 "
          f"{(est_tp[kk - 1] - est_tp[max(kk - 101, 0)]) / 100:.2f}")

flag_v = np.zeros(len(fam), bool)
flag_v[rest[order[:k_star]]] = True
# ----- grading with the id (evaluation only)
blk = idn >= 23819
tp = int((flag_v & blk).sum())
print(f"\nid check: {tp} of the {k_star} flagged are in the 259 (precision {tp / k_star:.2f}); "
      f"missed {int(((~fam) & blk).sum()) - tp}")
for name, f in [("families only (v3)", fam), ("families + volatility pocket", fam | flag_v)]:
    TP, N, K = int((f & blk).sum()), int(f.sum()), int(blk.sum())
    print(f"  {name:30s}: {N} flags, F3 if the id block is the truth = {10 * TP / (9 * K + N):.4f}")
pd.DataFrame({"player_id": t5.player_id, "volatility_flag": flag_v.astype(int), "family": fam.astype(int)}).to_csv(
    common.OUT / "t5_volatility_flags.csv", index=False)
