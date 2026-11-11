"""Phase 2 (2026-11-10), Task 3: fold stability of the merged pipeline and multi-seed averaging (brief item 3).

Already in place (verified, not re-tested): L1 objectives (median = the NMAE-optimal target), Tweedie as a mean member,
folds stratified on zero / positive-quantile bins of the target (src/modeling.make_folds kind="regression"), hurdle zero gate at
the median cross-over (round 8: every softer or harder gate is worse), Huber not used (it targets something between mean
and median; the metric wants the median).
B0 = pipeline pool (outputs/preds/t3r_*), exact LP blend, zero gate, scale.
C2 = the four ratio stage-2 models that carry ~99% of the blend weight averaged over seeds 42 / 2026 / 777.
C3 = C2 + the stage-1 classifier P(y > 0) averaged over the same seeds.
Acceptance: better on >= 4 of 5 folds, fold std not larger, paired-bootstrap CI above 0.
"""
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, fit_predict  # noqa: E402
from src.utils import load_preds, load_params  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
SEEDS = (42, 2026, 777)
tr, te = task3.load()
X, y, Xt, _ = task3.build_repaired(tr, te)
folds = make_folds(y, kind="regression")
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k
clip0 = lambda v: np.clip(v, 0, None)
half = lambda p_: {**p_, "learning_rate": p_["learning_rate"] / 2}
S1 = half(load_params("task3_stage1"))
R2 = half(load_params("task3_ratio_stage2"))
SPEC = {"ratio[hist] LGB tuned": ("lgb", R2, "historical_spending"), "ratio[prev] LGB tuned": ("lgb", R2, "prev_month_spending"),
        "ratio[hist] LGB": ("lgb", {"learning_rate": 0.05}, "historical_spending"),
        "ratio[prev] LGB": ("lgb", {"learning_rate": 0.05}, "prev_month_spending"),
        "ratio[hist] XGB": ("xgb", {"learning_rate": 0.05}, "historical_spending"),
        "ratio[hist] Cat": ("cat", {"learning_rate": 0.08, "depth": 8}, "historical_spending")}
BAG = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
seed_key = {"lgb": "random_state", "xgb": "random_state", "cat": "random_seed"}

plain = {}
for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False),
                        ("Cat-MAE raw THB", "t3r_cat_raw", False), ("LGB-L1 log1p", "t3r_lgb_log", True),
                        ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    plain[name] = clip0(np.expm1(r["oof"])) if logt else clip0(r["oof"])
h = load_preds("t3r_hurdle")
mu = {"Hurdle": h["mu_oof"], **{t: load_preds(slug3(t))["mu_oof"] for t in SPEC}}


def pool(p, mus):
    m = dict(plain)
    ro = {}
    for tag, mu_ in mus.items():
        ro[tag] = task3.readouts(y, p, mu_)
        for k in ("mixture-median", "hard gate", "E[y] + smearing"):
            m[f"{tag} {k}"] = task3.apply_readout(k, p, mu_, ro[tag])
    gate = (np.clip(p + ro["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
    return np.column_stack(list(m.values())), gate


scales = np.linspace(0.85, 1.15, 61)


def blend(P, gate):
    bo = np.zeros(len(y))
    for a, b in folds:
        bo[b] = P[b] @ task3.l1_blend_weights_exact(y[a], [P[a, j] for j in range(P.shape[1])])
    E = np.abs(y[None, :] - scales[:, None] * (bo * gate)[None, :]).sum(1)
    return bo * scales[int(E.argmin())] * gate


per_fold = lambda f: np.array([nmae(y[fold_id == k], f[fold_id == k]) for k in range(5)])


def boot(a, b, B=1000):
    e1, e0 = np.abs(y - a), np.abs(y - b)
    d = []
    for _ in range(B):
        w = rng.poisson(1.0, len(y)).astype(float)
        sy = w @ y
        d.append(1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy))
    return np.percentile(d, [2.5, 97.5])


f0 = blend(*pool(h["p_oof"], mu))
g0 = per_fold(f0)
log(f"B0: {nmae(y, f0):.5f}, folds {np.round(g0, 4)}, std {g0.std():.5f}")

mu_bag = dict(mu)
for tag in BAG:
    kind, prm, bcol = SPEC[tag]
    acc = [mu[tag]]
    for s in SEEDS[1:]:
        acc.append(task3.ratio_stage2_cv(kind, {**prm, seed_key[kind]: s}, X, y, Xt, folds, bcol, h["mu_oof"], h["mu_test"])["mu_oof"])
    mu_bag[tag] = np.mean(acc, axis=0)
    pos = y > 0
    log(f"  {tag}: stage-2 MAE on spenders, seed 42 {np.mean(np.abs(np.log1p(y[pos]) - mu[tag][pos])):.4f} -> 3 seeds {np.mean(np.abs(np.log1p(y[pos]) - mu_bag[tag][pos])):.4f}")

p_bag = [h["p_oof"]]
pos = (y > 0).astype(int)
for s in SEEDS[1:]:
    p_s = np.zeros(len(y))
    for a, b in folds:
        _, (pv,), _ = fit_predict("lgb", "binary", {**S1, "random_state": s}, X.iloc[a], pos[a], X.iloc[b], pos[b], [X.iloc[b]])
        p_s[b] = pv
    p_bag.append(p_s)
p3 = np.mean(p_bag, axis=0)

rows = []
for name, desc, f in [("C2", "4 main ratio stage-2 models averaged over 3 seeds", blend(*pool(h["p_oof"], mu_bag))),
                      ("C3", "C2 + stage-1 P(y>0) averaged over 3 seeds", blend(*pool(p3, mu_bag)))]:
    g = per_fold(f)
    lo, hi = boot(f, f0)
    better = int((g > g0).sum())
    keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
    rows.append([name, desc, nmae(y, f), nmae(y, f) - nmae(y, f0), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", g.std(), "KEEP" if keep else "discard"])
    log(rows[-1])
np.savez(OUT / "t3_phase2_seedbag.npz", **{f"mu_{slug3(t)}": mu_bag[t] for t in BAG}, p3=p3, f0=f0)
pd.set_option("display.width", 220)
print("\n" + pd.DataFrame(rows, columns=["Exp", "Change", "OOF", "delta", "95% CI", "folds better", "fold std", "Decision"]).to_string(
    index=False, float_format=lambda v: f"{v:.5f}"))
print(f"B0 fold std {g0.std():.5f}")
