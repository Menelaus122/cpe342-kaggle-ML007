"""Round 25, Task 3: a smooth zero gate instead of the hard median gate.

    python dg_r25_t3_gate.py

Pipeline: y_hat = min(blend * scale * 1[p > 0.5], cap), p = bagged P(y > 0) + tuned shift, blend = OOF L1 blend of the level-1 pool
(rebuilt here from the caches exactly as in the master notebook, section 7.3).
Challenger: y_hat = min(blend * scale * sigmoid((p - tau) / T), cap), with (tau, T, scale) chosen to maximise NMAE.
T -> 0 is a hard gate at tau, so the family contains the pipeline rule. Everything chosen is nested (4 folds -> 5th); the pipeline's
own scale is nested the same way for the comparison. Acceptance (round 25): mean nested gain > 0 and >= 3/5 folds; CI reported.
"""
import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t3_train, t3_test = task3.load()
X3, y3, X3_test, _ = task3.build_repaired(t3_train, t3_test)
y3 = np.asarray(y3, float)
folds = make_folds(y3, kind="regression")
clip0 = lambda a: np.clip(a, 0, None)
m3 = {}
for name, key, log in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                       ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    m3[name] = clip0(np.expm1(r["oof"])) if log else clip0(r["oof"])
SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
sfx3 = lambda s: "" if s == 42 else f"_s{s}"
hs = {s: load_preds(f"t3r_hurdle{sfx3(s)}") for s in SEEDS3}
h = {k: np.mean([hs[s][k] for s in SEEDS3], 0) for k in ("p_oof", "mu_oof")}
BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
RATIO3 = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
mu3 = {"Hurdle": h["mu_oof"]}
for tag in RATIO3:
    mu3[tag] = np.mean([load_preds(slug3(tag) + sfx3(s))["mu_oof"] for s in SEEDS3], 0) if tag in BAG3 else load_preds(slug3(tag))["mu_oof"]
RO3 = {}
for tag, mu in mu3.items():
    RO3[tag] = task3.readouts(y3, h["p_oof"], mu)
    for ro_kind in ("mixture-median", "hard gate", "E[y] + smearing"):
        m3[f"{tag} {ro_kind}"] = task3.apply_readout(ro_kind, h["p_oof"], mu, RO3[tag])
P3o = np.column_stack(list(m3.values()))
blend, cap = np.zeros(len(y3)), np.zeros(len(y3))
for tr, va in folds:
    w = task3.l1_blend_weights_exact(y3[tr], [P3o[tr, j] for j in range(P3o.shape[1])])
    blend[va] = P3o[va] @ w
    cap[va] = y3[tr].max()
p = np.clip(h["p_oof"] + RO3["Hurdle"]["shift"], 0, 1)
hard = (p > 0.5).astype(float)
scales = np.linspace(0.85, 1.15, 61)
S_all = float(scales[int(np.argmax([nmae(y3, blend * s * hard) for s in scales]))])
pipe = np.minimum(blend * S_all * hard, cap)
print(f"{len(y3)} rows, {P3o.shape[1]} pool members; pipeline OOF NMAE {nmae(y3, pipe):.5f} (scale {S_all:.3f}); zero share predicted {np.mean(pipe == 0):.3f}, "
      f"true {np.mean(y3 == 0):.3f}")

TAUS = np.round(np.arange(0.30, 0.621, 0.02), 2)
TS = [1e-4, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]
SC2 = np.linspace(0.90, 1.20, 31)
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -50, 50)))
mae = lambda yt, yp: np.abs(yt - yp).sum()


def fit_soft(rows):
    best = (np.inf, None)
    for tau in TAUS:
        for T in TS:
            gb = blend[rows] * sig((p[rows] - tau) / T)
            for s in SC2:
                e = mae(y3[rows], np.minimum(gb * s, cap[rows]))
                if e < best[0]:
                    best = (e, (tau, T, s))
    return best[1]


def fit_hard(rows):
    return float(scales[int(np.argmin([mae(y3[rows], blend[rows] * s * hard[rows]) for s in scales]))])


nest_pipe, nest_soft = np.zeros(len(y3)), np.zeros(len(y3))
for f, (tr, va) in enumerate(folds):
    s = fit_hard(tr)
    nest_pipe[va] = np.minimum(blend[va] * s * hard[va], cap[va])
    tau, T, s2 = fit_soft(tr)
    nest_soft[va] = np.minimum(blend[va] * s2 * sig((p[va] - tau) / T), cap[va])
    print(f"fold {f}: pipeline scale {s:.3f} | soft gate tau {tau:.2f}, T {T}, scale {s2:.3f}", flush=True)
tau, T, s2 = fit_soft(np.arange(len(y3)))
ins = np.minimum(blend * s2 * sig((p - tau) / T), cap)
print(f"in-sample: soft gate tau {tau:.2f}, T {T}, scale {s2:.3f}: NMAE {nmae(y3, ins):.5f} ({nmae(y3, ins) - nmae(y3, pipe):+.5f} vs pipeline in-sample)")
d = nmae(y3, nest_soft) - nmae(y3, nest_pipe)
pf = [nmae(y3[va], nest_soft[va]) - nmae(y3[va], nest_pipe[va]) for _, va in folds]
rng = np.random.default_rng(0)
bs = []
for _ in range(1000):
    i = rng.integers(0, len(y3), len(y3))
    bs.append(nmae(y3[i], nest_soft[i]) - nmae(y3[i], nest_pipe[i]))
print(f"nested: pipeline {nmae(y3, nest_pipe):.5f} | soft gate {nmae(y3, nest_soft):.5f}  delta {d:+.5f} [{np.quantile(bs, .025):+.5f}, {np.quantile(bs, .975):+.5f}] "
      f"folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.5f}' for x in pf)})")
band = (p > 0.3) & (p < 0.7)
print(f"rows with 0.3 < p < 0.7: {band.mean():.3%} of rows, {np.abs(y3 - nest_pipe)[band].sum() / np.abs(y3 - nest_pipe).sum():.3%} of the pipeline's absolute error")
