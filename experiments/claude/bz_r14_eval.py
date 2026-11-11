"""Round 14, step 4: does 10-seed bagging beat the v12 pipeline members? Run after by_r14_bag.py cv <task>.

    python bz_r14_eval.py t1 | t3 | t2

Baseline = the pipeline's cached members. The bag replaces each retrained member by its 10-seed average: log-odds (Task 1), log-probabilities (Task 2),
raw or log scale as in the member itself (Task 3). The stacker, cut-off, class multipliers, L1 blend weights and scale are all re-learned on the bagged OOF, nested.
Acceptance (same rule as earlier rounds): paired-bootstrap 95% CI above 0, >= 4/5 folds better, fold std not above 1.1x; Task 1 also not worse under test-like missingness.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT, T2_MODELS, _logit  # noqa: E402
from by_r14_bag import BAG3, DIR, SEEDS  # noqa: E402
from src import task1, task2, task3  # noqa: E402
from src.metrics import apply_class_weights, fbeta, macro_f1, nmae, optimize_class_weights, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)
load = lambda task, mem, s: np.load(DIR / f"{task}_{mem}_{s}.npz")
fmt_d = lambda v: " ".join(f"{d:+.4f}" for d in v)


def eval_t1():
    tr, te, _ = task1.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task1.build(tr, te)
    folds = make_folds(y)
    fold_id = np.zeros(len(y), int)
    for k, (a, b) in enumerate(folds):
        fold_id[b] = k
    raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
    q_tr, q_te = tr[raw].isna().mean(), te[raw].isna().mean()
    w_tl = np.ones(len(y))
    for c in [c for c in raw if q_te[c] < q_tr[c]]:
        m = tr[c].isna().values
        w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))
    f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")

    def wf2(pred, w):
        tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
        return 5 * tp / (5 * tp + 4 * fn + fp)

    def stack_pred(members, members_eval=None):
        S = np.column_stack([_logit(m) for m in members])
        Se = S if members_eval is None else np.column_stack([_logit(m) for m in members_eval])
        o = np.zeros(len(y))
        for a, b in folds:
            o[b] = LogisticRegression(C=1.0, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
        pred = np.zeros(len(y), int)
        for a, b in folds:
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(S[a], y[a])
            pred[b] = lr.predict_proba(Se[b])[:, 1] >= optimize_threshold(y[a], o[a], 2.0, average="binary")[0]
        return pred

    per_fold = lambda p: np.array([f2(y[fold_id == k], p[fold_id == k]) for k in range(5)])

    def boot(pn, po, w=None, B=1000):
        w = np.ones(len(y)) if w is None else w
        d = []
        for _ in range(B):
            pw = w * rng.poisson(1.0, len(y))
            d.append(wf2(pn, pw) - wf2(po, pw))
        return np.percentile(d, [2.5, 97.5])

    Z = np.load(OUT / "bn_t1_members.npz")
    avg2 = lambda k, s: (Z[f"{k}_42_{s}"] + Z[f"{k}_7_{s}"]) / 2
    keys = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
    cached = {"lgb": "t1_lgb", "xgb": "t1_xgb", "cat": "t1_cat", "et": "t1_et", "lr": "t1_lr", "rf24": "t1_rf24", "lgb24": "t1_lgb24", "xgb24": "t1_xgb24", "cat24": "t1_cat24"}
    b_oof = {k: load_preds(v)["oof"] for k, v in cached.items()}
    z_oof = {**{k: Z[f"{k}_oof"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "oof") for k in ("rf24", "lgb24", "xgb24", "cat24")}}
    z_tl = {**{k: Z[f"{k}_tl"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "tl") for k in ("rf24", "lgb24", "xgb24", "cat24")}}
    bagged = [k for k in keys if k != "lr"]

    def bag(seeds, field):
        out = {k: expit(np.mean([_logit(load("t1", k, s)[field]) for s in seeds], axis=0)) for k in bagged}
        out["lr"] = load_preds("t1_lr")["oof"] if field == "oof" else Z["lr_tl"]
        return [out[k] for k in keys]

    ref = stack_pred([b_oof[k] for k in keys])
    ref_tl = stack_pred([z_oof[k] for k in keys], [z_tl[k] for k in keys])
    fr = per_fold(ref)
    log(f"B0 (v12 members): nested F2 {f2(y, ref):.5f}, folds {np.round(fr, 4).tolist()}, std {fr.std():.5f}; test-like {wf2(ref_tl, w_tl):.5f}")

    print("\nsingle-seed spread: every retrained member with the same one seed, full nested stack")
    single = []
    for s in SEEDS:
        p = stack_pred(bag([s], "oof"))
        single.append(f2(y, p))
        log(f"  seed {s}: nested F2 {single[-1]:.5f}")
    single = np.array(single)
    print(f"single-seed nested F2: mean {single.mean():.5f}, std {single.std(ddof=1):.5f}, min {single.min():.5f}, max {single.max():.5f}", flush=True)

    rows = []
    for n in (3, 5, 10):
        p = stack_pred(bag(SEEDS[:n], "oof"))
        p_tl = stack_pred(bag(SEEDS[:n], "oof"), bag(SEEDS[:n], "tl"))
        fo = per_fold(p)
        lo, hi = boot(p, ref)
        tlo, thi = boot(p_tl, ref_tl, w_tl)
        better, dtl = int((fo > fr).sum()), wf2(p_tl, w_tl) - wf2(ref_tl, w_tl)
        keep = lo > 0 and better >= 4 and fo.std() <= fr.std() * 1.1 and dtl >= 0
        rows.append([f"{n}-seed bag", f2(y, p), f2(y, p) - f2(y, ref), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", fmt_d(fo - fr), fo.std(), f"{dtl:+.5f}", f"[{tlo:+.5f}, {thi:+.5f}]", int((p != ref).sum()), "KEEP" if keep else "discard"])
        log(rows[-1])
    print("\n" + pd.DataFrame(rows, columns=["Variant", "nested F2", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "test-like delta", "test-like CI", "OOF labels changed", "Decision"]).to_string(index=False))


def eval_t2():
    tr, te = task2.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task2.build(tr, te)
    folds = make_folds(y)
    fold_id = np.zeros(len(y), int)
    for k, (a, b) in enumerate(folds):
        fold_id[b] = k

    def stack(oofs):
        S = np.column_stack([np.log(np.clip(o, 1e-6, 1)) for o in oofs])
        o = np.zeros((len(y), 4))
        for a, b in folds:
            o[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
        return o

    def nested(P):
        pred = np.zeros(len(y), int)
        for a, b in folds:
            pred[b] = apply_class_weights(P[b], optimize_class_weights(y[a], P[a])[0])
        return pred

    pf = lambda p: np.array([macro_f1(y[fold_id == k], p[fold_id == k]) for k in range(5)])

    def boot(pn, po, B=300):
        d = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            d.append(macro_f1(y[i], pn[i]) - macro_f1(y[i], po[i]))
        return np.percentile(d, [2.5, 97.5])

    def gmean(k, seeds):
        L = np.mean([np.log(np.clip(load("t2", k, s)["oof"], 1e-6, 1)) for s in seeds], axis=0)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)

    base = {n: load_preds(n)["oof"] for n in T2_MODELS}
    q0 = nested(stack(list(base.values())))
    g0 = pf(q0)
    log(f"B0 (clean pipeline): nested macro-F1 {macro_f1(y, q0):.5f}, folds {np.round(g0, 4).tolist()}, std {g0.std():.5f}")
    rows = []
    for n in (3, 5, 10):
        m = dict(base)
        for k in ("lgb", "xgb", "cat"):
            m[f"t2_{k}"] = gmean(k, SEEDS[:n])
        q = nested(stack(list(m.values())))
        g = pf(q)
        lo, hi = boot(q, q0)
        better = int((g > g0).sum())
        keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
        rows.append([f"{n}-seed bag", macro_f1(y, q), macro_f1(y, q) - macro_f1(y, q0), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", fmt_d(g - g0), g.std(), int((q != q0).sum()), "KEEP" if keep else "discard"])
        log(rows[-1])
    print("\n" + pd.DataFrame(rows, columns=["Variant", "nested macro-F1", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "OOF labels changed", "Decision"]).to_string(index=False))


def eval_t3():
    tr, te = task3.load()
    X, y, Xt, _ = task3.build_repaired(tr, te)
    folds = make_folds(y, kind="regression")
    fold_id = np.zeros(len(y), int)
    for k, (a, b) in enumerate(folds):
        fold_id[b] = k
    clip0 = lambda v: np.clip(v, 0, None)
    slug3 = lambda t: "t3r_" + t.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
    SPEC = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
    bag_of = {"ratio[hist] LGB tuned": "ratio_hist_lgb_tuned", "ratio[prev] LGB tuned": "ratio_prev_lgb_tuned", "ratio[hist] XGB": "ratio_hist_xgb", "ratio[hist] Cat": "ratio_hist_cat"}
    assert set(bag_of.values()) == set(BAG3)
    plain = {}
    for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                            ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
        r = load_preds(key)
        plain[name] = clip0(np.expm1(r["oof"])) if logt else clip0(r["oof"])
    h = load_preds("t3r_hurdle")
    mu0 = {"Hurdle": h["mu_oof"], **{t: load_preds(slug3(t))["mu_oof"] for t in SPEC}}
    scales = np.linspace(0.85, 1.15, 61)

    def final(p, mus):
        m = dict(plain)
        ro = {}
        for tag, mu_ in mus.items():
            ro[tag] = task3.readouts(y, p, mu_)
            for k in ("mixture-median", "hard gate", "E[y] + smearing"):
                m[f"{tag} {k}"] = task3.apply_readout(k, p, mu_, ro[tag])
        P = np.column_stack(list(m.values()))
        gate = (np.clip(p + ro["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
        bo = np.zeros(len(y))
        cap = np.zeros(len(y))
        for a, b in folds:
            bo[b] = P[b] @ task3.l1_blend_weights_exact(y[a], [P[a, j] for j in range(P.shape[1])])
            cap[b] = y[a].max()
        E = [nmae(y, bo * s * gate) for s in scales]
        return np.minimum(bo * scales[int(np.argmax(E))] * gate, cap)

    per_fold = lambda f: np.array([nmae(y[fold_id == k], f[fold_id == k]) for k in range(5)])

    def boot(a, b, B=1000):
        e1, e0 = np.abs(y - a), np.abs(y - b)
        d = []
        for _ in range(B):
            w = rng.poisson(1.0, len(y)).astype(float)
            sy = w @ y
            d.append(1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy))
        return np.percentile(d, [2.5, 97.5])

    f0 = final(h["p_oof"], mu0)
    g0 = per_fold(f0)
    log(f"B0 (v12 members + cap): NMAE {nmae(y, f0):.5f}, folds {np.round(g0, 4).tolist()}, std {g0.std():.5f}")
    rows = []
    for n in (3, 5, 10):
        s_ = SEEDS[:n]
        hs = [load("t3", "hurdle", s) for s in s_]
        p = np.mean([z["p_oof"] for z in hs], axis=0)
        mus = dict(mu0)
        mus["Hurdle"] = np.mean([z["mu_oof"] for z in hs], axis=0)
        for tag, nm in bag_of.items():
            mus[tag] = np.mean([load("t3", nm, s)["mu_oof"] for s in s_], axis=0)
        f = final(p, mus)
        g = per_fold(f)
        lo, hi = boot(f, f0)
        better = int((g > g0).sum())
        keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
        rows.append([f"{n}-seed bag", nmae(y, f), nmae(y, f) - nmae(y, f0), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", " ".join(f"{d:+.5f}" for d in g - g0), g.std(), f.sum() / y.sum(), "KEEP" if keep else "discard"])
        log(rows[-1])
    print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "sum pred / sum y", "Decision"]).to_string(index=False))


if __name__ == "__main__":
    {"t1": eval_t1, "t2": eval_t2, "t3": eval_t3}[sys.argv[1]]()
    log("done")
