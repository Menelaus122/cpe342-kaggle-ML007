"""Round 34, Task 1 item 2a: missingness-pattern features and skill-stratified relative features (training rows only, CPU).

    python ft_r34_t1_relative.py

Earlier rounds: missingness is completely at random (0 of 31 missing indicators informative; indicators + native NaN +0.0002,
discarded in round 9), and the stack loses accuracy on rows with a missing key column. Not tried: (1) the missingness pattern of a
column GROUP as one unit, (2) a player's mechanics relative to players of the same level / K-D.
New features, every statistic fitted on the training fold only (inside each fold of the member's cross-validation):
  missingness   number of missing cells among the mechanical / game-sense / other columns; the group bitmask (mechanical columns +
                reaction time: 9 bits; game-sense columns: 7 bits) as an integer code, its frequency, and its smoothed cheater rate
                (m = 200; for the training rows out of fold over 5 inner folds);
  relative      strata = deciles of `level` and of `kill_death_ratio` (missing -> own stratum); for each mechanical / game-sense
                column and reaction time: (x - stratum median) / (stratum IQR / 1.349); plus, per stratifier, the mean mechanical
                residual, the mean game-sense residual and their difference (reaction time enters the mechanical mean negated).
Members (LightGBM with the pipeline's tuned parameters at the tuned learning rate, early stopping as in the pipeline, 3 seeds,
log-odds mean):
  M0  the 49 pipeline features (control: a three-seed copy of an existing member);
  M1  the 49 features + the new features.
Level 2 = v22 (r33_t1_common.level2) with M0 / M1 as a tenth member, nested on the pipeline folds and stacker split seeds 1, 2, 3.
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_from, log, logit, y
from src import task1
from src.modeling import run_cv

OUT = c.OUT.parent / "r34"
OUT.mkdir(parents=True, exist_ok=True)
raw = c.tr.drop(columns=[col for col in ("player_id", "id", task1.TARGET) if col in c.tr.columns])
X49 = task1.add_features(c.tr, ref=raw)
MECH = task1.MECH_COLS + ["reaction_time_ms"]
SENSE = task1.SENSE_COLS
OTHER = [col for col in c.RAW if col not in MECH + SENSE]
MISS = c.tr[c.RAW].isna()
code = lambda cols: (MISS[cols].values.astype(np.int64) * (1 << np.arange(len(cols)))).sum(1)
STATIC = pd.DataFrame({"miss_n_mech": MISS[MECH].sum(axis=1).values, "miss_n_sense": MISS[SENSE].sum(axis=1).values,
                       "miss_n_other": MISS[OTHER].sum(axis=1).values, "miss_code_mech": code(MECH), "miss_code_sense": code(SENSE)},
                      index=c.tr.index)
RAWV = c.tr[c.RAW]


def rate(codes_fit, y_fit, codes_app, m=200.0):
    s = pd.DataFrame({"c": codes_fit, "y": y_fit}).groupby("c").y.agg(["sum", "count"])
    prior = y_fit.mean()
    r = (s["sum"] + m * prior) / (s["count"] + m)
    return pd.Series(codes_app).map(r).fillna(prior).values, pd.Series(codes_app).map(s["count"]).fillna(0).values / len(codes_fit)


def new_features(a_idx, y_a, frames_idx):
    """New features for each index set in frames_idx, fitted on rows a_idx (labels y_a). The first index set must be a_idx itself:
    its cheater rates are computed out of fold over 5 inner folds."""
    outs = [STATIC.iloc[ix].copy() for ix in frames_idx]
    for g in ("mech", "sense"):
        cf = STATIC[f"miss_code_{g}"].values
        for o, ix in zip(outs[1:], frames_idx[1:]):
            o[f"miss_rate_{g}"], o[f"miss_freq_{g}"] = rate(cf[a_idx], y_a, cf[ix])
        r_in = np.zeros(len(a_idx))
        for u, v in StratifiedKFold(5, shuffle=True, random_state=0).split(a_idx, y_a):
            r_in[v] = rate(cf[a_idx[u]], y_a[u], cf[a_idx[v]])[0]
        outs[0][f"miss_rate_{g}"] = r_in
        outs[0][f"miss_freq_{g}"] = rate(cf[a_idx], y_a, cf[a_idx])[1]
    for strat in ("level", "kill_death_ratio"):
        sv = RAWV[strat].values
        edges = np.nanquantile(sv[a_idx], np.linspace(0.1, 0.9, 9))
        bins = np.where(np.isnan(sv), 10, np.searchsorted(edges, sv))
        res = {}
        for col in MECH + SENSE:
            if col == strat:
                continue
            xv = RAWV[col].values
            g = pd.DataFrame({"b": bins[a_idx], "x": xv[a_idx]}).groupby("b").x
            med, iqr = g.median(), (g.quantile(0.75) - g.quantile(0.25)) / 1.349
            z = (xv - pd.Series(bins).map(med).values) / (pd.Series(bins).map(iqr).values + 1e-9)
            res[col] = z
            for o, ix in zip(outs, frames_idx):
                o[f"rel_{strat[:5]}_{col}"] = z[ix]
        mech = np.nanmean(np.column_stack([-res[k] if k == "reaction_time_ms" else res[k] for k in MECH if k in res]), axis=1)
        sense = np.nanmean(np.column_stack([res[k] for k in SENSE if k in res]), axis=1)
        for o, ix in zip(outs, frames_idx):
            o[f"rel_{strat[:5]}_mech"], o[f"rel_{strat[:5]}_sense"], o[f"rel_{strat[:5]}_gap"] = mech[ix], sense[ix], (mech - sense)[ix]
    return outs


def fold_fn(X_tr, y_tr, others):
    idx = [X_tr.index.values] + [o.index.values for o in others]
    new = new_features(idx[0], np.asarray(y_tr), idx)
    cat = lambda X, N: pd.concat([X, N.set_index(X.index)], axis=1)
    return cat(X_tr, new[0]), [cat(o, n) for o, n in zip(others, new[1:])]


# the pipeline's tuned parameters at the tuned learning rate (the pipeline halves it for its final fits; M0 and M1 share the setting)
P = {**json.load(open("outputs/params/task1_lgb.json")), "n_jobs": c.NT}
f_m = OUT / "t1_relative_members.npz"
if f_m.exists():
    d = np.load(f_m)
    Z0, Z1 = d["z0"], d["z1"]
else:
    z0, z1 = [], []
    with np.errstate(all="ignore"):
        for s in (0, 1, 2):
            ps = {**P, "random_state": s}
            r0 = run_cv("lgb", "binary", ps, X49, y, X49.iloc[:5], c.folds, verbose=False)
            r1 = run_cv("lgb", "binary", ps, X49, y, X49.iloc[:5], c.folds, verbose=False, fold_fn=fold_fn)
            z0.append(logit(r0["oof"]))
            z1.append(logit(r1["oof"]))
            log(f"seed {s}: M0 AUC {roc_auc_score(y, r0['oof']):.5f} (iterations {r0['best_iters']}), M1 AUC {roc_auc_score(y, r1['oof']):.5f} "
                f"(iterations {r1['best_iters']}); M1 top new features by gain: "
                + ", ".join(f"{k} {v:.3f}" for k, v in r1["importance"].items() if k.startswith(("miss_", "rel_")))[:400])
    Z0, Z1 = np.mean(z0, 0), np.mean(z1, 0)
    np.savez(f_m, z0=Z0, z1=Z1)
cut_f2 = lambda z: f2(y, (z >= c.best_cut(y, z)).astype(int))
fo0 = fold_of(c.folds)
log(f"members alone (3 seeds, out of fold): M0 AUC {roc_auc_score(y, Z0):.5f}, F2 at its best cut {cut_f2(Z0):.5f}; M1 AUC {roc_auc_score(y, Z1):.5f}, "
    f"F2 {cut_f2(Z1):.5f}; per-fold AUC M1 - M0: " + " ".join(f"{roc_auc_score(y[fo0 == i], Z1[fo0 == i]) - roc_auc_score(y[fo0 == i], Z0[fo0 == i]):+.5f}"
                                                         for i in range(5))
    + f"; correlation of M1 with the pipeline's LightGBM member {np.corrcoef(Z1, L[:, 0])[0, 1]:.4f}")
base_p = c.base_probs()
base_labels = [labels_from(p, fold_of(fl)) for p, (_, fl) in zip(base_p, SPLITS)]
for name, Z in (("M0 control: three-seed LightGBM on the 49 features as a tenth member", Z0),
                ("M1 LightGBM + missingness and relative features as a tenth member", Z1)):
    S10 = np.column_stack([L, Z])
    cand = []
    for k, (label, fl) in enumerate(SPLITS):
        cand.append(labels_from(c.level2(fl, S_lr=S10, S_tr=S10), fold_of(fl)))
        log(f"{label}: {name.split(' ')[0]} {f2(y, cand[-1]):.5f} (base {f2(y, base_labels[k]):.5f})")
    c.report(name, cand, base_labels)
log("done")
