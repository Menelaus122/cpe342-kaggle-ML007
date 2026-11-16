"""Round 30, Task 3 brief items 1-3: empirical-quantile read-out, segment-conditional L1 blend, unrecovered spending history.
(training rows only; cached members; CPU)

    python ex_r30_t3_readout_blend.py

Members as in scripts/nb_part2.py 7.3 (round 28 rebuild): five raw-THB models + the 10-seed hurdle and six ratio stage-2 tags, each
read out three ways (mixture median, hard gate, mean with smearing), exact non-negative L1 blend (HiGHS), median zero gate
P(y > 0) + shift > 0.5, one scale, cap at the training maximum. The pipeline tunes the read-outs on all OOF rows; here EVERYTHING is
nested (read-outs, blend, scale fitted on 4 folds, applied to the 5th), for the reference and for the candidates alike.
  item 1  empirical-quantile mixture median: for p' = P(y>0) + shift > 0.5 the L1-optimal prediction is the q = 1 - 0.5 / p' quantile
          of y | y > 0; instead of mu + sigma * z_q (log-normal, one sigma) use mu + the empirical q-quantile of the training-fold
          residuals log1p(y) - mu, by group (imputed segment x tercile of mu; groups under 300 positive rows fall back to the segment,
          then to all rows); shift tuned on the training folds. (a) replaces the mixture-median members, (b) is added to the pool.
  item 2  segment-conditional blend: L1 blend weights per group (imputed segment 0-3; rows whose raw historical_spending is missing
          form a fifth group), shrunk towards the global weights, w = (1 - lambda) w_group + lambda w_global, lambda in
          {0, .25, .5, .75, 1} chosen by an inner 4-fold CV inside the training folds.
  item 3  rows whose spending history is still missing after the identity repair (training rows only): count, error share, and how
          well an in-fold L1 LightGBM imputes log1p(historical_spending) where it is known (an upper bound for what an imputed base
          could buy).
Acceptance (user, 2026-11-16, Task 3): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0002.
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import KFold

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t0 = time.time()
OUT = Path("experiments/claude/outputs/r30")
raw = pd.read_csv(DATA_DIRS[3] / "train.csv")                         # training rows only
y = raw[task3.TARGET].values.astype(float)
rep, _ = task3.fill_identities(raw)
Xf = task3.add_features(rep)
Xs, _, seg_acc = task3.impute_segment(Xf, Xf[Xf.segment.isna()].head(5).copy())   # dummy apply-frame of training rows: no test rows are read
seg = Xs["segment"].values.astype(int)
hist_raw_na = raw["historical_spending"].isna().values
folds = make_folds(y, kind="regression")
fo = np.zeros(len(y), int)
for i, (_, va) in enumerate(folds):
    fo[va] = i
print(f"{len(y)} training rows; segment imputation hold-out accuracy {seg_acc:.4f} [{time.time() - t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ item 3 first (cheap)
h_na, p_na = rep["historical_spending"].isna().values, rep["prev_month_spending"].isna().values
fin28 = np.load(OUT.parent / "r28" / "t3_oof_rebuilt.npz")["final"]
err = np.abs(y - fin28)
print(f"\nitem 3: after the identity repair, historical_spending missing in {h_na.sum()} rows, prev_month_spending in {p_na.sum()}, "
      f"both in {(h_na & p_na).sum()} (training rows; the test file is not inspected)")
for name, m in [("hist missing", h_na), ("both missing", h_na & p_na)]:
    print(f"  {name}: {m.sum()} rows, mean |error| {err[m].mean():,.0f} THB (all rows {err.mean():,.0f}), share of the total |error| "
          f"{err[m].sum() / err.sum():.3%}; zeroing their whole error would raise NMAE by at most "
          f"{1 / (1 + (err.sum() - err[m].sum()) / y.sum()) - nmae(y, fin28):+.5f}")
known = ~h_na
F = Xs.drop(columns=[c for c in ("historical_spending", "prev_month_spending", "log_hist", "log_prev") if c in Xs.columns])
F = F[[c for c in F.columns if not any(k in c for k in ("hist", "prev"))]]
th = np.log1p(rep["historical_spending"].values)
o = np.zeros(len(y))
kk = np.where(known)[0]
for a, b in KFold(5, shuffle=True, random_state=0).split(kk):
    m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=4, verbose=-1)
    o[kk[b]] = m.fit(F.iloc[kk[a]], th[kk[a]]).predict(F.iloc[kk[b]])
ae = np.abs(o[kk] - th[kk])
print(f"  in-fold L1 imputer of log1p(historical_spending) from {F.shape[1]} non-spending-history features, where it is known: median "
      f"|error| {np.median(ae):.3f} in log units (a factor of {np.exp(np.median(ae)):.2f}); the identity repair is exact (error 0) "
      f"[{time.time() - t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ members (as the round 28 rebuild)
clip0 = lambda a: np.clip(a, 0, None)
raw_pool = {}
for name, key, log in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                       ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    raw_pool[name] = clip0(np.expm1(r["oof"])) if log else clip0(r["oof"])
SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
sfx3 = lambda s: "" if s == 42 else f"_s{s}"
p_oof = np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["p_oof"] for s in SEEDS3], 0)
BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
RATIO3 = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
MU = {"Hurdle": np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["mu_oof"] for s in SEEDS3], 0)}
for tag in RATIO3:
    MU[tag] = np.mean([load_preds(slug3(tag) + sfx3(s))["mu_oof"] for s in (SEEDS3 if tag in BAG3 else (42,))], 0)
print(f"members loaded [{time.time() - t0:.0f}s]; note: the pipeline averages the read-outs of each seed, here the seed-mean mu is read "
      "out once (the same for reference and candidates)", flush=True)


def emp_tables(tr_rows, mu, shift):
    """Empirical residual quantile tables by group (segment x mu tercile, fallback segment, fallback all) from training rows."""
    pos = tr_rows[y[tr_rows] > 0]
    res = np.log1p(y[pos]) - mu[pos]
    cuts = np.quantile(mu[pos], [1 / 3, 2 / 3])
    grp = lambda idx: seg[idx] * 3 + np.digitize(mu[idx], cuts)
    qs = np.linspace(0.0, 1.0, 401)
    tab_all = np.quantile(res, qs)
    tab = {}
    for g in np.unique(grp(pos)):
        r = res[grp(pos) == g]
        if len(r) >= 300:
            tab[g] = np.quantile(r, qs)
    for s_ in np.unique(seg[pos]):
        r = res[seg[pos] == s_]
        if len(r) >= 300:
            tab[("seg", s_)] = np.quantile(r, qs)
    return dict(qs=qs, tab=tab, all=tab_all, grp=grp)


def emp_readout(rows, mu, p, shift, T):
    pg = np.clip(p[rows] + shift, 1e-6, 1 - 1e-6)
    q = np.clip(1 - 0.5 / pg, 0, 1)
    g, s_ = T["grp"](rows), seg[rows]
    off = np.interp(q, T["qs"], T["all"])
    for v in np.unique(s_):
        if ("seg", v) in T["tab"]:
            m = s_ == v
            off[m] = np.interp(q[m], T["qs"], T["tab"][("seg", v)])
    for v in np.unique(g):
        if v in T["tab"]:
            m = g == v
            off[m] = np.interp(q[m], T["qs"], T["tab"][v])
    return np.where(pg > 0.5, clip0(np.expm1(mu[rows] + off)), 0.0)


def fold_readouts(tr):
    """Read-out parameters of every tag fitted on the training rows, and the empirical tables with their tuned shift."""
    out = {}
    for tag, mu in MU.items():
        ro = task3.readouts(y[tr], p_oof[tr], mu[tr])
        T = emp_tables(tr, mu, ro["shift"])
        best = max(np.linspace(-0.15, 0.15, 13), key=lambda s_: nmae(y[tr], emp_readout(tr, mu, p_oof, s_, T)))
        out[tag] = (ro, T, best)
    return out


def build_pool(tr, va, mode, R):
    """Member predictions for rows tr and va from the fold's read-outs R. mode: 'ref', 'emp_replace', 'emp_add'."""
    cols_tr, cols_va = [v[tr] for v in raw_pool.values()], [v[va] for v in raw_pool.values()]
    for tag, mu in MU.items():
        ro, T, best = R[tag]
        kinds = ["hard gate", "E[y] + smearing"] + ([] if mode == "emp_replace" else ["mixture-median"])
        for k in kinds:
            cols_tr.append(task3.apply_readout(k, p_oof[tr], mu[tr], ro))
            cols_va.append(task3.apply_readout(k, p_oof[va], mu[va], ro))
        if mode != "ref":
            cols_tr.append(emp_readout(tr, mu, p_oof, best, T))
            cols_va.append(emp_readout(va, mu, p_oof, best, T))
    return np.column_stack(cols_tr), np.column_stack(cols_va), R["Hurdle"][0]["shift"]


def mixg(Pm, groups, wk, wg, l_):
    out = np.empty(len(groups))
    for g in np.unique(groups):
        m = groups == g
        out[m] = Pm[m] @ ((1 - l_) * wk.get(g, wg) + l_ * wg)
    return out


SCALES = np.linspace(0.85, 1.15, 61)


def finish(btr, bva, tr, va, shift):
    gate_tr, gate_va = (np.clip(p_oof[tr] + shift, 0, 1) > 0.5), (np.clip(p_oof[va] + shift, 0, 1) > 0.5)
    s = SCALES[int(np.argmin([np.abs(y[tr] - btr * s_ * gate_tr).sum() for s_ in SCALES]))]
    return np.minimum(bva * s * gate_va, y[tr].max())


res = {k: np.zeros(len(y)) for k in ("reference", "item 1a emp. replace", "item 1b emp. added", "item 2 segment blend")}
lam_chosen = []
blend_grp = np.where(hist_raw_na, 4, seg)
for i, (tr, va) in enumerate(folds):
    R = fold_readouts(tr)
    Ptr, Pva, sh = build_pool(tr, va, "ref", R)
    w = task3.l1_blend_weights_exact(y[tr], [Ptr[:, j] for j in range(Ptr.shape[1])])
    res["reference"][va] = finish(Ptr @ w, Pva @ w, tr, va, sh)
    for key, mode in [("item 1a emp. replace", "emp_replace"), ("item 1b emp. added", "emp_add")]:
        Qtr, Qva, sh2 = build_pool(tr, va, mode, R)
        w2 = task3.l1_blend_weights_exact(y[tr], [Qtr[:, j] for j in range(Qtr.shape[1])])
        res[key][va] = finish(Qtr @ w2, Qva @ w2, tr, va, sh2)
    # item 2: lambda by inner 4-fold CV inside the training folds (same reference pool)
    inner = list(KFold(4, shuffle=True, random_state=i).split(tr))
    LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
    inner_err = {l_: 0.0 for l_ in LAMS}
    for ia, ib in inner:
        A, B = tr[ia], tr[ib]
        wg = task3.l1_blend_weights_exact(y[A], [Ptr[ia, j] for j in range(Ptr.shape[1])])
        wk = {g: task3.l1_blend_weights_exact(y[A][blend_grp[A] == g], [Ptr[ia][blend_grp[A] == g, j] for j in range(Ptr.shape[1])])
              for g in np.unique(blend_grp[A])}
        for l_ in LAMS:
            inner_err[l_] += np.abs(y[B] - mixg(Ptr[ib], blend_grp[B], wk, wg, l_)).sum()
    lam = min(LAMS, key=lambda l_: inner_err[l_])
    lam_chosen.append(lam)
    wk = {g: task3.l1_blend_weights_exact(y[tr][blend_grp[tr] == g], [Ptr[blend_grp[tr] == g, j] for j in range(Ptr.shape[1])])
          for g in np.unique(blend_grp[tr])}
    res["item 2 segment blend"][va] = finish(mixg(Ptr, blend_grp[tr], wk, w, lam), mixg(Pva, blend_grp[va], wk, w, lam), tr, va, sh)
    print(f"fold {i}: reference {nmae(y[va], res['reference'][va]):.5f}, 1a {nmae(y[va], res['item 1a emp. replace'][va]):.5f}, "
          f"1b {nmae(y[va], res['item 1b emp. added'][va]):.5f}, 2 {nmae(y[va], res['item 2 segment blend'][va]):.5f} (lambda {lam}) "
          f"[{time.time() - t0:.0f}s, peak {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB]", flush=True)
np.savez(OUT / "t3_nested_variants.npz", **{k.replace(" ", "_").replace(".", ""): v for k, v in res.items()})
ref = res["reference"]
print(f"\nnested reference NMAE {nmae(y, ref):.5f} (pipeline-style, read-outs tuned on all rows: {nmae(y, fin28):.5f})")
for k in [k for k in res if k != "reference"]:
    per = [nmae(y[b], res[k][b]) - nmae(y[b], ref[b]) for _, b in folds]
    d = nmae(y, res[k]) - nmae(y, ref)
    ok = d > 0 and sum(p_ > 0 for p_ in per) >= 3 and min(per) >= -0.0002
    print(f"  {k:22s} {nmae(y, res[k]):.5f}  delta {d:+.5f}  folds +{sum(p_ > 0 for p_ in per)}/5 ({' '.join(f'{p_:+.5f}' for p_ in per)})"
          f"  -> {'PASS' if ok else 'fail'}" + (f"  lambdas {lam_chosen}" if k.startswith("item 2") else ""))
