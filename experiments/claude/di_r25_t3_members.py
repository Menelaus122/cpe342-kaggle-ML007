"""Round 25, Task 3: three new stage-2 members for the level-1 pool (DART, Langevin CatBoost, peer-ratio features).

    python di_r25_t3_members.py

All three are ratio[hist] stage-2 models (target log1p(y) - log1p(historical_spending) on spenders), seed 42, read out through the
pipeline's bagged P(y > 0) with the three hurdle read-outs (mixture median, hard gate, smearing):
  dart     LightGBM DART (drop_rate 0.1, 600 trees, learning rate 0.1; DART has no early stopping), tuned tree shape
  langevin CatBoost MAE depth 8 with stochastic gradient Langevin boosting (langevin=True)
  peer     the tuned LightGBM on X + peer features: log historical / previous-month spending minus the log median of the training
           players in the same daily_login_streak x event_participation_rate decile cell (cell statistics from the fold's training
           rows only, via features only: no target is used)
Pool = the pipeline's 26 members (from the caches) + the new read-outs. Scored as in the notebook, everything nested:
exact L1 blend weights on 4 folds -> 5th, median zero gate, scale chosen on 4 folds, cap. Each member is added alone and all together.
Acceptance (round 25): mean nested gain > 0 and >= 3/5 folds; the CI is reported, not used as a gate.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from common import tuned  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

OUT = Path("experiments/claude/outputs/r25")
OUT.mkdir(parents=True, exist_ok=True)
t3_train, t3_test = task3.load()
X3, y3, X3_test, _ = task3.build_repaired(t3_train, t3_test)
y3 = np.asarray(y3, float)
folds = make_folds(y3, kind="regression")
R2_3 = tuned("task3_ratio_stage2", {"learning_rate": 0.05})
clip0 = lambda a: np.clip(a, 0, None)

# ---------------------------------------------------------------- the pipeline pool (as in dg_r25_t3_gate.py)
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
for tag, mu in mu3.items():
    ro = task3.readouts(y3, h["p_oof"], mu)
    if tag == "Hurdle":
        RO_H = ro
    for k in ("mixture-median", "hard gate", "E[y] + smearing"):
        m3[f"{tag} {k}"] = task3.apply_readout(k, h["p_oof"], mu, ro)
p = np.clip(h["p_oof"] + RO_H["shift"], 0, 1)
gate = (p > 0.5).astype(float)


# ---------------------------------------------------------------- new members
def peer_frames():
    """X with peer features; cell medians from the training rows of each fold are applied inside ratio_stage2_cv through a wrapper."""
    def add(Xa, ref):
        Xa = Xa.copy()
        qs = np.linspace(0.1, 0.9, 9)
        e1, e2 = np.nanquantile(ref["daily_login_streak"], qs), np.nanquantile(ref["event_participation_rate"], qs)
        cell = lambda D: np.searchsorted(e1, D["daily_login_streak"].fillna(ref["daily_login_streak"].median())) * 10 + \
            np.searchsorted(e2, D["event_participation_rate"].fillna(ref["event_participation_rate"].median()))
        cr, ca = cell(ref), cell(Xa)
        for c in ("historical_spending", "prev_month_spending"):
            med = pd.Series(np.log1p(ref[c].values)).groupby(cr).median()
            Xa[f"{c}_vs_peer"] = np.log1p(Xa[c].values) - pd.Series(ca).map(med).fillna(np.nanmedian(np.log1p(ref[c].values))).values
        return Xa
    return add


def peer_cv():
    """ratio_stage2_cv with the peer features recomputed per fold from that fold's training rows."""
    add = peer_frames()
    mu = np.array(hs[42]["mu_oof"], float).copy()
    z, pos, b = np.log1p(y3), y3 > 0, np.log1p(X3["historical_spending"].values)
    okb = np.isfinite(b)
    from src.modeling import fit_predict
    for a, v in folds:
        ref = X3.iloc[a]
        Xa, Xv = add(ref, ref), add(X3.iloc[v], ref)
        ia, iv = pos[a] & okb[a], pos[v] & okb[v]
        _, (rv,), _ = fit_predict("lgb", "regression", {"objective": "regression_l1", **R2_3}, Xa[ia], (z - b)[a][ia], Xv[iv], (z - b)[v][iv], [Xv[okb[v]]])
        mu[v[okb[v]]] = b[v[okb[v]]] + rv
    return mu


NEW = {}
SPEC = {"dart": ("lgb", {**R2_3, "boosting_type": "dart", "drop_rate": 0.1, "n_estimators": 600, "learning_rate": 0.1}),
        "langevin": ("cat", {"learning_rate": 0.08, "depth": 8, "langevin": True})}
for name in ["dart", "langevin", "peer"]:
    f = OUT / f"t3_{name}.npz"
    if f.exists():
        NEW[name] = np.load(f)["mu_oof"]
        continue
    if name == "peer":
        mu = peer_cv()
    else:
        kind, prm = SPEC[name]
        mu = task3.ratio_stage2_cv(kind, prm, X3, y3, X3_test, folds, "historical_spending", hs[42]["mu_oof"], hs[42]["mu_test"])["mu_oof"]
    np.savez(f, mu_oof=mu)
    NEW[name] = mu
    print(f"  {name} fitted", flush=True)

pos = y3 > 0
ref_mu = load_preds("t3r_ratio_hist_lgb_tuned")["mu_oof"]
print(f"MAE of log1p(y) on spenders: pipeline ratio[hist] LGB tuned (seed 42) {np.mean(np.abs(np.log1p(y3[pos]) - ref_mu[pos])):.4f}, 10-seed bag "
      f"{np.mean(np.abs(np.log1p(y3[pos]) - mu3['ratio[hist] LGB tuned'][pos])):.4f} | " + " | ".join(f"{k} {np.mean(np.abs(np.log1p(y3[pos]) - v[pos])):.4f}" for k, v in NEW.items()))
new_cols = {}
for name, mu in NEW.items():
    ro = task3.readouts(y3, h["p_oof"], mu)
    new_cols[name] = [task3.apply_readout(k, h["p_oof"], mu, ro) for k in ("mixture-median", "hard gate", "E[y] + smearing")]
scales = np.linspace(0.85, 1.15, 61)
fold_of = np.zeros(len(y3), int)
for i, (_, va) in enumerate(folds):
    fold_of[va] = i


def nested(P):
    out = np.zeros(len(y3))
    wsum = np.zeros(P.shape[1])
    for tr, va in folds:
        w = task3.l1_blend_weights_exact(y3[tr], [P[tr, j] for j in range(P.shape[1])])
        wsum += w / len(folds)
        btr, bva = P[tr] @ w, P[va] @ w
        s = scales[int(np.argmin([np.abs(y3[tr] - btr * s * gate[tr]).sum() for s in scales]))]
        out[va] = np.minimum(bva * s * gate[va], y3[tr].max())
    return out, wsum


base_P = np.column_stack(list(m3.values()))
base, _ = nested(base_P)
print(f"pipeline pool ({base_P.shape[1]} members), nested blend + gate + nested scale + cap: NMAE {nmae(y3, base):.5f}", flush=True)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y3), len(y3)) for _ in range(1000)]
for label, names in [("+ dart", ["dart"]), ("+ langevin", ["langevin"]), ("+ peer", ["peer"]), ("+ all three", ["dart", "langevin", "peer"])]:
    P = np.column_stack([base_P] + [c for nmn in names for c in new_cols[nmn]])
    out, w = nested(P)
    d = nmae(y3, out) - nmae(y3, base)
    pf = [nmae(y3[fold_of == i], out[fold_of == i]) - nmae(y3[fold_of == i], base[fold_of == i]) for i in range(5)]
    bs = [nmae(y3[i], out[i]) - nmae(y3[i], base[i]) for i in boot]
    print(f"  {label:14s} NMAE {nmae(y3, out):.5f}  delta {d:+.5f} [{np.quantile(bs, .025):+.5f}, {np.quantile(bs, .975):+.5f}]  folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.5f}' for x in pf)})  mean blend weight of the new columns {w[base_P.shape[1]:].sum():.3f}", flush=True)
