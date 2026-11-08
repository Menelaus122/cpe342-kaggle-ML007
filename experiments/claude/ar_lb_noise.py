"""Insight round 7: how big is leaderboard noise compared with the 0.00055 gap to the top?

The LB is computed on a random subset of the 25,889 test rows (public) and the final ranking on the rest (private).
Two submissions of equal true quality still differ on any finite subset. We measure that spread with our own OOF
predictions as the population: "us" = v8 configuration, "other team" = a close but different model
(Task 1: C2 stack / LightGBM alone; Task 3: v7 blend / quantile read-out). Tasks 2 and 5 are identical for every
team that uses the id rule, so they add no difference. Task 4: if the answer key is misaligned for everyone, every
team scores chance level and the difference between two teams is the noise of two chance scores.
LB weight 0.2 per task (fits every upload so far).
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, _logit, OUT, t3_blend_score  # noqa: E402
from src import task3  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

rng = np.random.default_rng(0)
N_TEST, GAP, B = 25889, 0.00055, 4000

# ------------------------------------------------------------------ Task 1 OOF decisions (nested tau, as in round 6)
tr, te, X, y1, Xt, folds1, _ = load_t1()
base = {n: load_preds(n)["oof"] for n in T1_MODELS}
red = np.load(OUT / "t1_round6b_members.npz")


def stack(oofs):
    S = np.column_stack([_logit(o) for o in oofs]); o = np.zeros(len(y1))
    for a, b in folds1:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y1[a]).predict_proba(S[b])[:, 1]
    return o


def nested(p):
    pred = np.zeros(len(y1), int)
    for a, b in folds1:
        t, _ = optimize_threshold(y1[a], p[a], 2.0, average="binary"); pred[b] = p[b] >= t
    return pred


t1 = {"us (v8 stack)": nested(stack(list(base.values()))),
      "C2 stack": nested(stack(list(base.values()) + [red["lgb"], red["xgb"], red["cat"]])),
      "LightGBM alone": nested(base["t1_lgb"])}

# ------------------------------------------------------------------ Task 3 OOF predictions
tr3, _ = task3.load()
y3 = tr3[task3.TARGET].values.astype(float)
folds3 = make_folds(y3, kind="regression")
v8, v7 = np.load(OUT / "t3_v8.npz"), np.load(OUT / "t3_v7.npz")
t3 = {"us (v8 blend)": t3_blend_score(y3, folds3, v8["P_oof"], v8["gate_oof"])[1],
      "v7 blend": t3_blend_score(y3, folds3, v7["P_oof"], v7["gate_oof"])[1],
      "quantile read-out": np.load(OUT / "t3_quantile_readout.npz")["oof"]}


def f2_counts(yt, pr):
    tp = np.sum(yt & pr); fp = np.sum(~yt & pr); fn = np.sum(yt & ~pr)
    return 5 * tp / (5 * tp + 4 * fn + fp)


def t3_score(yt, pr):
    return 1 / (1 + np.abs(yt - pr).sum() / yt.sum())


print("full-OOF scores (population values):")
yb = y1.astype(bool)
for k, v in t1.items():
    print(f"  T1 {k:<18} F2 {f2_counts(yb, v.astype(bool)):.4f}")
for k, v in t3.items():
    print(f"  T3 {k:<18} score {t3_score(y3, v):.5f}")


def diff_sd(n, comp1, comp3):
    """SD over random test subsets of size n of the LB difference us - other (weight 0.2 per task)."""
    a1, b1 = t1["us (v8 stack)"].astype(bool), t1[comp1].astype(bool)
    a3, b3 = t3["us (v8 blend)"], t3[comp3]
    d = np.empty(B)
    for i in range(B):
        s1 = rng.choice(len(y1), n, replace=False)
        s3 = rng.choice(len(y3), n, replace=False)
        d[i] = 0.2 * (f2_counts(yb[s1], a1[s1]) - f2_counts(yb[s1], b1[s1])) \
            + 0.2 * (t3_score(y3[s3], a3[s3]) - t3_score(y3[s3], b3[s3]))
    return d.std(), d.mean()


# ------------------------------------------------------------------ Task 4 chance level (key misaligned)
p4 = __import__("pandas").read_csv("submission_v8_try.csv")["task4"].values
mix = np.bincount(p4, minlength=5) / len(p4)
print(f"\nT4 our predicted class mix on test: {np.round(mix, 3).tolist()}")


def t4_diff_sd(n, disagree):
    """Two teams whose T4 predictions differ on `disagree` share of images, scored against a shuffled key."""
    d, lvl = np.empty(600), np.empty(600)
    for i in range(600):
        key = rng.permutation(p4)
        other = p4.copy()
        m = rng.random(len(p4)) < disagree
        other[m] = rng.choice(5, m.sum(), p=mix)
        s = rng.choice(len(p4), n, replace=False)
        a = f1_score(key[s], p4[s], average="macro"); b = f1_score(key[s], other[s], average="macro")
        d[i], lvl[i] = 0.2 * (a - b), a
    return d.std(), lvl.mean(), lvl.std()


print("\nLB difference between us and a close competitor, from test-row sampling alone")
print(f"{'public share':>12} {'rows':>6} | {'T1+T3 sd (C2 / v7)':>20} {'T1+T3 sd (LGB / qread)':>23} | {'T4 sd (8% disagree)':>20} | "
      f"{'total sd':>9} | P(gap {GAP} flips on private)")
for f in (0.3, 0.5):
    n_pub, n_prv = int(f * N_TEST), N_TEST - int(f * N_TEST)
    sA, _ = diff_sd(n_pub, "C2 stack", "v7 blend")
    sB, _ = diff_sd(n_pub, "LightGBM alone", "quantile read-out")
    s4, lv, lsd = t4_diff_sd(n_pub, 0.08)
    sA_p, _ = diff_sd(n_prv, "C2 stack", "v7 blend")
    s4_p, _, _ = t4_diff_sd(n_prv, 0.08)
    tot_pub = np.sqrt(sA ** 2 + s4 ** 2)
    tot_prv = np.sqrt(sA_p ** 2 + s4_p ** 2)
    p_flip = norm.cdf(-GAP / np.sqrt(tot_pub ** 2 + tot_prv ** 2))
    print(f"{f:>12.0%} {n_pub:>6} | {sA:>20.5f} {sB:>23.5f} | {s4:>20.5f} | {tot_pub:>9.5f} | {p_flip:.2f}")
print(f"T4 chance-level macro-F1 against a shuffled key: mean {lv:.4f}, sd {lsd:.4f} (on the public rows)")
