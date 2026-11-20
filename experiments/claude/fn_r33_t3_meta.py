"""Round 33, Task 3: input-dependent L1 meta-combiner, per-row spread in the mixture median, sub-threshold floor
(training rows only; cached members; CPU).

    python fn_r33_t3_meta.py

Reference = the pipeline rebuilt fully nested as in ex_r30_t3_readout_blend.py: five raw-THB models + the 10-seed hurdle and six
ratio stage-2 tags, each read out three ways (mixture median, hard gate, mean with smearing), exact non-negative L1 blend (HiGHS),
median zero gate, one scale, cap at the training maximum; read-outs, blend and scale fitted on 4 folds, applied to the 5th.
  item 1  input-dependent weights trained on the absolute error in THB (smooth |r| = sqrt(r^2 + eps^2), eps = 1e-4 THB):
          yhat = S * sum_m softmax(g(x))_m P_m, started at the exact L1 blend (bias = log of its weights, S = their sum), with
          x = P(y > 0), the imputed segment, log1p of the repaired historical / previous-month spending (+ missing flags) and
          log1p of the blend. (a) linear gate g, (b) one hidden layer of 16 tanh units; Adam, 400 full-batch steps, weight decay
          1e-2 on the input weights. (c) LightGBM with the L1 objective boosted from the blend (init_score) on x and the member
          predictions: 200 trees, lr 0.02, 15 leaves, 300 rows per leaf.
  item 2  per-row spread in the mixture median: the read-out is expm1(mu + sigma * z_q), q = 1 - 0.5 / p, with ONE sigma per
          member. Here sigma_i = kappa * 1.2533 * E|r_i|, E|r| from a LightGBM fitted on |log1p(y) - mu| of the positive rows
          (mu of ratio[hist] LGB tuned; features: the Task 3 features, mu, P(y > 0)); out of fold inside the training folds
          (4 inner folds) so kappa and the gate shift of every tag are tuned on honest sigma_i. (a) replaces the mixture-median
          members, (b) is added to the pool.
  item 3  sub-threshold rows (gate open, final prediction under 50 THB): one of {zero below c, floor at c, c in 5..50; scale by
          f in 0.5..2} or no change, chosen by the total absolute error on the training folds.
Acceptance (user, 2026-11-16, Task 3): pooled NMAE delta > 0, >= 3/5 folds better, no fold below -0.0002.
"""
import os
import time
from pathlib import Path

NT = int(os.environ.get("R33_THREADS", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
torch.set_num_threads(NT)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
OUT = Path("experiments/claude/outputs/r33")
OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(DATA_DIRS[3] / "train.csv")                         # training rows only
y = raw[task3.TARGET].values.astype(float)
rep, _ = task3.fill_identities(raw)
Xf = task3.add_features(rep)
Xs, _, seg_acc = task3.impute_segment(Xf, Xf[Xf.segment.isna()].head(5).copy())   # dummy apply-frame of training rows: no test rows are read
seg = Xs["segment"].values.astype(int)
folds = make_folds(y, kind="regression")
log(f"{len(y)} training rows, {np.mean(y > 0):.3f} positive; y integer share {np.mean(y == np.round(y)):.3f}, multiples of 10 "
    f"{np.mean(y[y > 0] % 10 == 0):.3f}")

clip0 = lambda a: np.clip(a, 0, None)
raw_pool = {}
for name, key, lg_ in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                       ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    raw_pool[name] = clip0(np.expm1(r["oof"])) if lg_ else clip0(r["oof"])
SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
sfx3 = lambda s: "" if s == 42 else f"_s{s}"
p_oof = np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["p_oof"] for s in SEEDS3], 0)
BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
RATIO3 = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
MU = {"Hurdle": np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["mu_oof"] for s in SEEDS3], 0)}
for tag in RATIO3:
    MU[tag] = np.mean([load_preds(slug3(tag) + sfx3(s))["mu_oof"] for s in (SEEDS3 if tag in BAG3 else (42,))], 0)
log("members loaded")

# context of the gate (item 1)
h, pv = rep["historical_spending"].values.astype(float), rep["prev_month_spending"].values.astype(float)
CTX0 = np.column_stack([p_oof, *[(seg == s).astype(float) for s in range(4)], np.log1p(np.nan_to_num(h)), np.isnan(h).astype(float),
                        np.log1p(np.nan_to_num(pv)), np.isnan(pv).astype(float)])
# features of the spread model (item 2)
FS = Xs.select_dtypes("number").drop(columns=[c for c in (task3.TARGET, "player_id", "id") if c in Xs.columns]).copy()
MU_REF = MU["ratio[hist] LGB tuned"]
FS["mu_ref"], FS["p_pos"] = MU_REF, p_oof
ABS_R = np.abs(np.log1p(y) - MU_REF)
SIGP = dict(objective="l2", n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=NT, verbose=-1)
SCALES = np.linspace(0.85, 1.15, 61)


def sigma_rows(tr, va, seed):
    """E|r| for the training rows (out of fold inside tr) and the validation rows (model fitted on all of tr); positive rows train."""
    s_tr = np.zeros(len(tr))
    for ia, ib in KFold(4, shuffle=True, random_state=seed).split(tr):
        A = tr[ia][y[tr[ia]] > 0]
        s_tr[ib] = lgb.LGBMRegressor(**SIGP, random_state=0).fit(FS.iloc[A], ABS_R[A]).predict(FS.iloc[tr[ib]])
    A = tr[y[tr] > 0]
    s_va = lgb.LGBMRegressor(**SIGP, random_state=0).fit(FS.iloc[A], ABS_R[A]).predict(FS.iloc[va])
    return 1.2533 * np.clip(s_tr, 0.02, None), 1.2533 * np.clip(s_va, 0.02, None)


def tune_het(tr, mu, sig_tr, ro):
    k0 = ro["sigma"] / np.median(sig_tr)
    loss = lambda v: -nmae(y[tr], task3.mixture_median(np.clip(p_oof[tr] + v[1], 0, 1), mu[tr], abs(v[0]) * sig_tr))
    res = minimize(loss, [k0, ro["shift"]], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return abs(res.x[0]), res.x[1]


def build_pool(tr, va, mode, R, het=None):
    """mode: 'ref', 'het_replace', 'het_add'. het = (sig_tr, sig_va, {tag: (kappa, shift)})."""
    cols_tr, cols_va = [v[tr] for v in raw_pool.values()], [v[va] for v in raw_pool.values()]
    for tag, mu in MU.items():
        ro = R[tag]
        kinds = ["hard gate", "E[y] + smearing"] + ([] if mode == "het_replace" else ["mixture-median"])
        for k in kinds:
            cols_tr.append(task3.apply_readout(k, p_oof[tr], mu[tr], ro))
            cols_va.append(task3.apply_readout(k, p_oof[va], mu[va], ro))
        if mode != "ref":
            kap, sh = het[2][tag]
            cols_tr.append(task3.mixture_median(np.clip(p_oof[tr] + sh, 0, 1), mu[tr], kap * het[0]))
            cols_va.append(task3.mixture_median(np.clip(p_oof[va] + sh, 0, 1), mu[va], kap * het[1]))
    return np.column_stack(cols_tr), np.column_stack(cols_va)


def finish(btr, bva, tr, va, shift, both=False):
    gate_tr, gate_va = (np.clip(p_oof[tr] + shift, 0, 1) > 0.5), (np.clip(p_oof[va] + shift, 0, 1) > 0.5)
    s = SCALES[int(np.argmin([np.abs(y[tr] - btr * s_ * gate_tr).sum() for s_ in SCALES]))]
    out = np.minimum(bva * s * gate_va, y[tr].max())
    return (np.minimum(btr * s * gate_tr, y[tr].max()), out, gate_tr, gate_va) if both else out


def gate_fit(Ptr, Pva, ctx_tr, ctx_va, ytr, w, hidden, seed=0):
    """Mixture-of-experts gate started at the exact blend w; returns the gated blends of the training and validation rows."""
    torch.manual_seed(seed)
    mu, sd = ctx_tr.mean(0), ctx_tr.std(0) + 1e-9
    Zt, Zv = torch.tensor((ctx_tr - mu) / sd), torch.tensor((ctx_va - mu) / sd)
    Pt, Pv, yt = torch.tensor(Ptr), torch.tensor(Pva), torch.tensor(ytr)
    m, d = Ptr.shape[1], ctx_tr.shape[1]
    bias = torch.tensor(np.log(w / w.sum() + 1e-3), requires_grad=True)
    S = torch.tensor(float(w.sum()), dtype=torch.float64, requires_grad=True)
    if hidden:
        body = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, m, bias=False)).double()
        torch.nn.init.zeros_(body[2].weight)
    else:
        body = torch.nn.Linear(d, m, bias=False).double()
        torch.nn.init.zeros_(body.weight)
    f = lambda Z, P: S * (torch.softmax(body(Z) + bias, 1) * P).sum(1)
    opt = torch.optim.Adam([{"params": body.parameters(), "weight_decay": 1e-2, "lr": 3e-3}, {"params": [bias], "lr": 3e-3}, {"params": [S], "lr": 1e-3}])
    scale = float(ytr.mean())
    for _ in range(400):
        opt.zero_grad()
        loss = torch.sqrt((yt - f(Zt, Pt)) ** 2 + 1e-8).mean() / scale
        loss.backward()
        opt.step()
    with torch.no_grad():
        return f(Zt, Pt).numpy(), f(Zv, Pv).numpy()


KEYS = ["reference", "1a linear gate", "1b MLP gate", "1c LightGBM L1 from the blend", "2a per-row sigma, replace", "2b per-row sigma, added",
        "3 sub-threshold rule"]
res = {k: np.zeros(len(y)) for k in KEYS}
rules = [("none", None)] + [("zero below", c) for c in (5, 10, 20, 30, 50)] + [("floor at", c) for c in (5, 10, 20, 30, 50)] \
    + [("scale", f) for f in (0.5, 0.75, 1.25, 1.5, 2.0)]


def apply_rule(pred, gate, rule):
    kind, v = rule
    m = gate & (pred < 50)
    if kind == "zero below":
        return np.where(m & (pred < v), 0.0, pred)
    if kind == "floor at":
        return np.where(m, np.maximum(pred, v), pred)
    if kind == "scale":
        return np.where(m, pred * v, pred)
    return pred


picked = []
for i, (tr, va) in enumerate(folds):
    R = {tag: task3.readouts(y[tr], p_oof[tr], mu[tr]) for tag, mu in MU.items()}
    sh = R["Hurdle"]["shift"]
    Ptr, Pva = build_pool(tr, va, "ref", R)
    w = task3.l1_blend_weights_exact(y[tr], [Ptr[:, j] for j in range(Ptr.shape[1])])
    btr, bva = Ptr @ w, Pva @ w
    ftr, fva, gtr, gva = finish(btr, bva, tr, va, sh, both=True)
    res["reference"][va] = fva
    # item 3
    errs = [np.abs(y[tr] - apply_rule(ftr, gtr, r)).sum() for r in rules]
    rule = rules[int(np.argmin(errs))]
    picked.append(rule)
    res["3 sub-threshold rule"][va] = apply_rule(fva, gva, rule)
    sub = gva & (fva < 50)
    # item 1
    ctx_tr, ctx_va = np.column_stack([CTX0[tr], np.log1p(btr)]), np.column_stack([CTX0[va], np.log1p(bva)])
    for key, hid in (("1a linear gate", 0), ("1b MLP gate", 16)):
        gt, gv = gate_fit(Ptr, Pva, ctx_tr, ctx_va, y[tr], w, hid)
        res[key][va] = finish(clip0(gt), clip0(gv), tr, va, sh)
    Ftr, Fva = np.column_stack([ctx_tr, Ptr]), np.column_stack([ctx_va, Pva])
    add = np.zeros(len(va))
    add_tr = np.zeros(len(tr))
    for s in (0, 1, 2):
        m = lgb.LGBMRegressor(objective="l1", n_estimators=200, learning_rate=0.02, num_leaves=15, min_child_samples=300, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, n_jobs=NT, verbose=-1, random_state=s).fit(Ftr, y[tr], init_score=btr)
        add += m.predict(Fva) / 3
        add_tr += m.predict(Ftr) / 3
    res["1c LightGBM L1 from the blend"][va] = finish(clip0(btr + add_tr), clip0(bva + add), tr, va, sh)
    # item 2
    sig_tr, sig_va = sigma_rows(tr, va, i)
    het = (sig_tr, sig_va, {tag: tune_het(tr, mu, sig_tr, R[tag]) for tag, mu in MU.items()})
    for key, mode in (("2a per-row sigma, replace", "het_replace"), ("2b per-row sigma, added", "het_add")):
        Qtr, Qva = build_pool(tr, va, mode, R, het)
        w2 = task3.l1_blend_weights_exact(y[tr], [Qtr[:, j] for j in range(Qtr.shape[1])])
        res[key][va] = finish(Qtr @ w2, Qva @ w2, tr, va, sh)
    pos_va = y[va] > 0
    log(f"fold {i}: " + ", ".join(f"{k.split(' ')[0]} {nmae(y[va], res[k][va]):.5f}" for k in KEYS)
        + f" | rule {rule}; sub-threshold rows {sub.mean():.3%} (mean y {y[va][sub].mean() if sub.any() else 0:.1f}, mean pred "
          f"{fva[sub].mean() if sub.any() else 0:.1f}); sigma_i quartiles {np.round(np.quantile(sig_va, [.25, .5, .75]), 3)}, one sigma "
          f"{R['ratio[hist] LGB tuned']['sigma']:.3f}, kappa {het[2]['ratio[hist] LGB tuned'][0]:.3f}; corr(sigma_i, |r|) on positive validation rows "
          f"{np.corrcoef(sig_va[pos_va], ABS_R[va][pos_va])[0, 1]:.3f}")
np.savez(OUT / "t3_meta_variants.npz", **{k.split(" ")[0]: v for k, v in res.items()})
ref = res["reference"]
print(f"\nnested reference NMAE {nmae(y, ref):.5f}; sub-threshold rules picked per fold: {picked}")
for k in KEYS[1:]:
    per = [nmae(y[b], res[k][b]) - nmae(y[b], ref[b]) for _, b in folds]
    d = nmae(y, res[k]) - nmae(y, ref)
    ok = d > 0 and sum(p_ > 0 for p_ in per) >= 3 and min(per) >= -0.0002
    print(f"  {k:30s} {nmae(y, res[k]):.5f}  delta {d:+.5f}  folds +{sum(p_ > 0 for p_ in per)}/5 ({' '.join(f'{p_:+.5f}' for p_ in per)})"
          f"  -> {'PASS' if ok else 'fail'}")
log("done")
