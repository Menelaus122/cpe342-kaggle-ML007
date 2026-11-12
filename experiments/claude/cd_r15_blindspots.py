"""Round 15: the five "structural blind spot" checks of the pasted brief, on the current pipeline's cached predictions.

    python cd_r15_blindspots.py t1 t2 t3 t4      # any subset

t1: (item 1) binary vs macro F2 cut-offs, test flips, max-min compromise; (item 5) Gaussian-smoothed and bootstrap-median cut-offs, nested.
t2: (item 3) how the id block decides Task 2, whether flips (label != block) are predictable from features or neighbouring ids,
    and a nested soft-prior rule (features may override the block only above a learned log-odds margin).
t3: (item 2) error by decile of the prediction, nested power calibration a * yhat^b with a zero cut-off, forced 10-30% raw-scale L1 / Huber mixes,
    raw-scale Huber members (LightGBM, CatBoost) offered to the exact L1 blend.
t4: (item 4) chance-level macro-F1 under a key unrelated to the predictions, for our test class mix vs re-weighted mixes (arithmetic only, nothing is built).
Every fit uses training rows only; nested = chosen on four folds, scored on the fifth. Acceptance as before: bootstrap 95% CI > 0, >= 4/5 folds better,
fold std <= 1.1x, and for Task 1 not worse under test-like missingness. No submission is written.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT, T2_MODELS, _logit, stack_multi, tuned  # noqa: E402
from src import task1, task2, task3, task4  # noqa: E402
from src.metrics import apply_class_weights, fbeta, macro_f1, nmae, optimize_class_weights, optimize_threshold  # noqa: E402
from src.modeling import fit_predict, make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
FULL_W = 0.5
blend_full = lambda f, g: (1 - FULL_W) * np.asarray(f) + FULL_W * np.asarray(g)
v12 = pd.read_csv("submission_v12_try.csv")
v13 = pd.read_csv("submission_v13.csv")
fmt = lambda v: " ".join(f"{d:+.5f}" for d in v)


def fold_ids(folds, n):
    f = np.zeros(n, int)
    for k, (_, b) in enumerate(folds):
        f[b] = k
    return f


# ================================================================================= Task 1
def run_t1():
    print("\n" + "=" * 100 + "\nTask 1: item 1 (binary vs macro F2) and item 5 (smoothed / bootstrap cut-offs)\n" + "=" * 100)
    tr, te, _ = task1.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task1.build(tr, te)
    folds = make_folds(y)
    fid = fold_ids(folds, len(y))
    keys = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]
    S = np.column_stack([_logit(load_preds(k)["oof"]) for k in keys])
    St = np.column_stack([_logit(blend_full(load_preds(k)["test"], load_preds(k + "_full")["test"])) for k in keys])
    oof = np.zeros(len(y))
    for a, b in folds:
        oof[b] = LogisticRegression(C=1.0, max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    test = LogisticRegression(C=1.0, max_iter=2000).fit(S, y).predict_proba(St)[:, 1]
    raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
    q_tr, q_te = tr[raw].isna().mean(), te[raw].isna().mean()
    w_tl = np.ones(len(y))
    for c in [c for c in raw if q_te[c] < q_tr[c]]:
        m = tr[c].isna().values
        w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))

    f2b = lambda yt, yp: fbeta(yt, yp, 2, "binary")
    f2m = lambda yt, yp: fbeta(yt, yp, 2, "macro")
    tb, sb = optimize_threshold(y, oof, 2.0, average="binary")
    tm, sm = optimize_threshold(y, oof, 2.0, average="macro")
    assert ((test >= tb).astype(int) == v12.task1.values).all(), "stack rebuild does not reproduce v12 Task 1"
    print(f"code: src/config.py FBETA_AVERAGE = 'binary'; scripts/nb_part2.py 7.1 picks THR1 = optimize_threshold(y1, stack1_oof, 2.0) with that default,")
    print(f"      i.e. positive-class F2 = 5PR/(4P+R). The macro reading is written as a side file only (outputs/format_variants/submission_task1_macro_F2.csv).")
    print(f"rebuilt stack reproduces v12 Task 1 exactly (15,690 ones): True")

    G = np.round(np.arange(0.05, 0.9500001, 0.0005), 4)
    cb = np.array([f2b(y, (oof >= t).astype(int)) for t in G])
    cm = np.array([f2m(y, (oof >= t).astype(int)) for t in G])
    tr_ = G[int(np.argmax(np.minimum(cb, cm)))]
    rows = []
    for nm, t in [("tau_binary (v12)", tb), ("tau_macro", tm), ("tau_robust = argmax min(binary, macro)", tr_)]:
        p = (oof >= t).astype(int)
        pt = (test >= t).astype(int)
        rows.append([nm, t, f2b(y, p), f2m(y, p), int(pt.sum()), pt.mean(), int((pt != v12.task1.values).sum())])
    print(pd.DataFrame(rows, columns=["cut-off", "tau", "OOF F2 binary", "OOF F2 macro", "test 1s", "test flag rate", "test labels flipped vs v12"]).round(5).to_string(index=False))
    vm = pd.read_csv("outputs/format_variants/submission_task1_macro_F2.csv")
    print(f"cross-check: the pipeline's macro side file has {int(vm.task1.sum()):,} ones; {int((vm.task1 != v12.task1).sum()):,} rows differ from v12")
    print(f"cost of each switch if the scorer is the other reading (OOF, x 0.2 for the leaderboard):"
          f"\n  use tau_macro while the scorer is binary: {f2b(y, (oof >= tm).astype(int)) - sb:+.5f} F2 -> LB {0.2 * (f2b(y, (oof >= tm).astype(int)) - sb):+.5f}"
          f"\n  keep tau_binary while the scorer is macro: {f2m(y, (oof >= tb).astype(int)) - sm:+.5f} F2 -> LB {0.2 * (f2m(y, (oof >= tb).astype(int)) - sm):+.5f}"
          f"\n  tau_robust if binary: {f2b(y, (oof >= tr_).astype(int)) - sb:+.5f}; if macro: {f2m(y, (oof >= tr_).astype(int)) - sm:+.5f}")
    print("leaderboard evidence (FINDINGS, round 8): v3 -> v4 moved tau 0.163 -> 0.190 (plus the Task 5 union): observed -0.00234; binary F2 predicts -0.00265 "
          "(off 0.0003), macro F2 predicts -0.00023 (off 0.0021). The scorer reads positive-class F2.")

    # ---------------- item 5: cut-off rules, nested
    def curve(p, yy, w, grid):
        o = np.argsort(p, kind="stable")
        ps, ys, ws = p[o], yy[o], w[o]
        cp = np.concatenate([[0.0], np.cumsum(ws * ys)])
        cn = np.concatenate([[0.0], np.cumsum(ws * (1 - ys))])
        i = np.searchsorted(ps, grid, side="left")
        tp, fp, fn = cp[-1] - cp[i], cn[-1] - cn[i], cp[i]
        return 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1e-12)

    Gf = np.round(np.arange(0.05, 0.6000001, 0.0005), 4)
    rules = {"B0 argmax (pipeline: grid + Brent)": lambda p, yy: optimize_threshold(yy, p, 2.0, average="binary")[0]}
    for sig in (0.005, 0.01, 0.02, 0.04):
        rules[f"Gaussian-smoothed curve, sigma {sig}"] = (lambda s: lambda p, yy: Gf[int(np.argmax(gaussian_filter1d(curve(p, yy, np.ones(len(yy)), Gf), s / 0.0005, mode="nearest")))])(sig)

    def boot_median(p, yy, B=1000):
        out = np.empty(B)
        for j in range(B):
            out[j] = Gf[int(np.argmax(curve(p, yy, rng.poisson(1.0, len(yy)).astype(float), Gf)))]
        return float(np.median(out))
    rules["median of 1,000 bootstrap argmaxes"] = boot_median

    def wf2(pred, w):
        tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
        return 5 * tp / (5 * tp + 4 * fn + fp)

    def boot(pn, po, w, B=1000):
        d = []
        for _ in range(B):
            pw = w * rng.poisson(1.0, len(y))
            d.append(wf2(pn, pw) - wf2(po, pw))
        return np.percentile(d, [2.5, 97.5])

    preds, taus = {}, {}
    for nm, fn in rules.items():
        pr, tt = np.zeros(len(y), int), []
        for a, b in folds:
            t = fn(oof[a], y[a])
            tt.append(t)
            pr[b] = oof[b] >= t
        preds[nm], taus[nm] = pr, tt
        log(f"{nm}: per-fold cut-offs {np.round(tt, 4).tolist()}")
    ref = preds["B0 argmax (pipeline: grid + Brent)"]
    pf = lambda p: np.array([f2b(y[fid == k], p[fid == k]) for k in range(5)])
    g0 = pf(ref)
    rows = []
    for nm, p in preds.items():
        g = pf(p)
        t_full = rules[nm](oof, y)
        lo, hi = boot(p, ref, np.ones(len(y)))
        dtl = wf2(p, w_tl) - wf2(ref, w_tl)
        better = int((g > g0).sum())
        keep = nm != "B0 argmax (pipeline: grid + Brent)" and lo > 0 and better >= 4 and g.std() <= 1.1 * g0.std() and dtl >= 0
        rows.append([nm, f2b(y, p), f2b(y, p) - f2b(y, ref), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", " ".join(f"{d:+.4f}" for d in g - g0), g.std(),
                     f"{dtl:+.5f}", round(t_full, 4), int(((test >= t_full).astype(int) != v12.task1.values).sum()), "KEEP" if keep else ("baseline" if p is ref else "discard")])
    print("\nitem 5, cut-off rules (nested; full-data cut-off and the test labels it would flip vs v12):")
    print(pd.DataFrame(rows, columns=["rule", "nested F2", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "test-like delta",
                                      "full-data tau", "test flips vs v12", "Decision"]).to_string(index=False))
    print(f"\nflatness of the OOF curve near the optimum: F2 within 0.0005 of the best for tau in "
          f"[{G[cb >= cb.max() - 0.0005].min():.4f}, {G[cb >= cb.max() - 0.0005].max():.4f}]; test rows with stack probability in that band: "
          f"{int(((test >= G[cb >= cb.max() - 0.0005].min()) & (test <= G[cb >= cb.max() - 0.0005].max())).sum()):,}")


# ================================================================================= Task 2
def run_t2():
    print("\n" + "=" * 100 + "\nTask 2: item 3 (player_id rule vs features)\n" + "=" * 100)
    tr, te = task2.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task2.build(tr, te)
    folds = make_folds(y)
    fid = fold_ids(folds, len(y))
    CUTS = [52000, 84500, 110500]
    idn = tr.player_id.str[1:].astype(int).values
    idt = te.player_id.str[1:].astype(int).values
    blk, blkt = np.searchsorted(CUTS, idn, side="right"), np.searchsorted(CUTS, idt, side="right")
    assert (blkt == v12.task2.values).all()
    m = [load_preds(n) for n in T2_MODELS]
    so, st = stack_multi([d["oof"] for d in m], [d["test"] for d in m], y, folds)
    clean_t = apply_class_weights(st, optimize_class_weights(y, so)[0])
    print(f"v12/v13 Task 2 = id block for 100.0% of the {len(idt):,} test rows; the feature model decides 0 rows. There is no tie-break: features are never consulted.")
    print(f"the clean feature pipeline disagrees with the block on {int((clean_t != blkt).sum()):,} test rows ({(clean_t != blkt).mean():.2%}); on train OOF the feature argmax "
          f"disagrees with the block on {(so.argmax(1) != blk).mean():.2%}")
    flip = y != blk
    print(f"\ntrain rows with label != block (flips): {flip.sum():,} = {flip.mean():.3%}; block macro-F1 {macro_f1(y, blk):.5f}")
    pb = so[np.arange(len(y)), blk]
    pl = so[np.arange(len(y)), y]
    print(f"feature stack (OOF) on flipped rows: mean P(block class) {pb[flip].mean():.3f}, mean P(true label) {pl[flip].mean():.3f}; "
          f"feature argmax = true label on {(so.argmax(1)[flip] == y[flip]).mean():.3f}, = block on {(so.argmax(1)[flip] == blk[flip]).mean():.3f}")
    print(f"same on non-flipped rows: mean P(block) {pb[~flip].mean():.3f}. AUC of 1 - P(block) for spotting flips: {roc_auc_score(flip, 1 - pb):.4f}")
    # flips as a function of the true label vs the block (is it a random relabelling?)
    ct = pd.crosstab(pd.Series(blk[flip], name="block"), pd.Series(y[flip], name="label"))
    print("flipped rows, block (rows) x label (columns):\n" + ct.to_string())
    o = np.argsort(idn)
    fs = flip[o]
    same_blk = blk[o][1:] == blk[o][:-1]
    p_after = fs[1:][fs[:-1] & same_blk].mean()
    print(f"neighbouring ids (train rows sorted by id, same block): P(flip | previous id flipped) {p_after:.4f} vs base {fs.mean():.4f}; "
          f"P(label = previous id's label | both flipped) {(y[o][1:] == y[o][:-1])[fs[1:] & fs[:-1] & same_blk].mean():.3f}")
    gaps = np.diff(np.sort(idn))
    print(f"train id gaps: median {np.median(gaps):.0f}; test ids fill the gaps (train {len(idn):,} + test {len(idt):,} = {len(idn) + len(idt):,} ids, range {min(idn.min(), idt.min())}-{max(idn.max(), idt.max())})")

    # nested soft prior: switch from the block only if log P_feat(c) - log P_feat(block) > margin
    L = np.log(np.clip(so, 1e-9, 1))
    gain = L.max(1) - L[np.arange(len(y)), blk]
    def rule(margin, rows):
        return np.where(gain[rows] > margin, L[rows].argmax(1), blk[rows])
    M = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, np.inf]
    print("\nin-sample (all rows) macro-F1 by margin (inf = pure block = v12):", {str(mm): round(macro_f1(y, rule(mm, np.arange(len(y)))), 5) for mm in M})
    pred = np.zeros(len(y), int)
    chosen = []
    for a, b in folds:
        sc = [macro_f1(y[a], rule(mm, a)) for mm in M]
        mm = M[int(np.argmax(sc))]
        chosen.append(mm)
        pred[b] = rule(mm, b)
    g0 = np.array([macro_f1(y[fid == k], blk[fid == k]) for k in range(5)])
    g = np.array([macro_f1(y[fid == k], pred[fid == k]) for k in range(5)])
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        d.append(macro_f1(y[i], pred[i]) - macro_f1(y[i], blk[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"nested soft prior: margins chosen {chosen}; macro-F1 {macro_f1(y, pred):.5f} vs block {macro_f1(y, blk):.5f} ({macro_f1(y, pred) - macro_f1(y, blk):+.5f}), "
          f"CI [{lo:+.5f}, {hi:+.5f}], {int((g > g0).sum())}/5, per fold {' '.join(f'{x:+.4f}' for x in g - g0)}")

    # id-neighbour prior from training-fold labels only: label share among the k nearest training-fold ids
    print("\nneighbour-label prior (training-fold labels only; majority label of the k nearest training-fold ids):")
    for kk in (2, 6, 20):
        pr = np.zeros(len(y), int)
        for a, b in folds:
            oa = a[np.argsort(idn[a])]
            ida = idn[oa]
            pos = np.searchsorted(ida, idn[b])
            votes = np.zeros((len(b), 4))
            for off in range(-kk // 2, kk // 2 + 1):
                if off == 0:
                    continue
                j = np.clip(pos + (off if off < 0 else off - 1), 0, len(oa) - 1)
                votes[np.arange(len(b)), y[oa[j]]] += 1
            pr[b] = votes.argmax(1)
        print(f"  k={kk}: neighbour-majority macro-F1 {macro_f1(y, pr):.5f}; agrees with the block on {(pr == blk).mean():.4f}; "
              f"on flipped rows predicts the true label {(pr[flip] == y[flip]).mean():.4f}")


# ================================================================================= Task 3
def run_t3():
    print("\n" + "=" * 100 + "\nTask 3: item 2 (tail errors, power calibration, raw-scale L1 / Huber)\n" + "=" * 100)
    tr, te = task3.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt, _ = task3.build_repaired(tr, te)
    folds = make_folds(y, kind="regression")
    fid = fold_ids(folds, len(y))
    clip0 = lambda v: np.clip(v, 0, None)
    slug3 = lambda t: "t3r_" + t.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
    SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
    sfx = lambda s: "" if s == 42 else f"_s{s}"
    TAGS = ["Hurdle", "ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
    BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
    plain = {}
    for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                            ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
        r = load_preds(key)
        plain[name] = clip0(np.expm1(r["oof"])) if logt else clip0(r["oof"])
    H = [load_preds(f"t3r_hurdle{sfx(s)}") for s in SEEDS3]
    p = np.mean([h["p_oof"] for h in H], axis=0)
    mu = {"Hurdle": np.mean([h["mu_oof"] for h in H], axis=0)}
    for t in TAGS[1:]:
        mu[t] = np.mean([load_preds(slug3(t) + sfx(s))["mu_oof"] for s in SEEDS3], axis=0) if t in BAG3 else load_preds(slug3(t))["mu_oof"]
    names, cols, RO = list(plain), list(plain.values()), {}
    for t in TAGS:
        RO[t] = task3.readouts(y, p, mu[t])
        for rk in ("mixture-median", "hard gate", "E[y] + smearing"):
            names.append(f"{t} {rk}")
            cols.append(task3.apply_readout(rk, p, mu[t], RO[t]))
    P = np.column_stack(cols)
    gate = (np.clip(p + RO["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
    scales = np.linspace(0.85, 1.15, 61)

    def nested_final(Pm):
        bo, cap, W = np.zeros(len(y)), np.zeros(len(y)), []
        for a, b in folds:
            w = task3.l1_blend_weights_exact(y[a], [Pm[a, j] for j in range(Pm.shape[1])])
            W.append(w)
            bo[b] = Pm[b] @ w
            cap[b] = y[a].max()
        sc = scales[int(np.argmax([nmae(y, bo * s * gate) for s in scales]))]
        return np.minimum(bo * sc * gate, cap), cap, np.array(W)

    f0, cap, W0 = nested_final(P)
    log(f"B0 = pipeline (10-seed bag + cap): OOF NMAE {nmae(y, f0):.6f} (pipeline 0.79458)")
    print("raw-scale L1 members' weight in the exact L1 blend, per fold:", {k: np.round(W0[:, j], 3).tolist() for j, k in enumerate(names) if "raw" in k})
    np.savez(OUT / "cd_r15_t3.npz", f0=f0, cap=cap, p=p, gate=gate)

    pf = lambda f: np.array([nmae(y[fid == k], f[fid == k]) for k in range(5)])
    g0 = pf(f0)

    def judge(nm, f, rows):
        g = pf(f)
        e1, e0 = np.abs(y - f), np.abs(y - f0)
        d = []
        for _ in range(1000):
            w = rng.poisson(1.0, len(y)).astype(float)
            sy = w @ y
            d.append(1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy))
        lo, hi = np.percentile(d, [2.5, 97.5])
        better = int((g > g0).sum())
        keep = lo > 0 and better >= 4 and g.std() <= 1.1 * g0.std()
        rows.append([nm, nmae(y, f), nmae(y, f) - nmae(y, f0), f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", fmt(g - g0), g.std(), "KEEP" if keep else "discard"])
        log(rows[-1][:5])

    # ---------------- 2.1 error by decile of the prediction
    tot = np.abs(y - f0).sum()
    zero = f0 == 0
    grp = np.full(len(y), -1)
    grp[~zero] = pd.qcut(f0[~zero], 10, labels=False, duplicates="drop")
    rows = []
    for gk in range(-1, grp.max() + 1):
        mk = grp == gk
        rows.append(["predicted 0" if gk < 0 else f"decile {gk + 1}", int(mk.sum()), f"{f0[mk].min():,.0f}-{f0[mk].max():,.0f}", np.abs(y[mk] - f0[mk]).sum() / tot,
                     y[mk].sum() / y.sum(), f0[mk].sum() / max(y[mk].sum(), 1), (y[mk] == 0).mean(), (y[mk] > f0[mk]).mean(), np.median(y[mk])])
    print("\n2.1 OOF absolute error by group of the prediction (deciles of the positive predictions):")
    print(pd.DataFrame(rows, columns=["group", "rows", "prediction range THB", "share of total |error|", "share of total y", "sum pred / sum y",
                                      "share y = 0", "share y above pred", "median y"]).round(4).to_string(index=False))

    # ---------------- 2.2 nested power calibration
    def apply_pow(theta, f, cp, with_tau):
        la, b = theta[0], theta[1]
        tau = abs(theta[2]) if with_tau else 0.0
        out = np.where(f > tau, np.exp(la) * np.power(np.maximum(f, 1e-9), b), 0.0)
        return np.minimum(np.where(f > 0, out, 0.0), cp)

    def fit_pow(f, yy, cp, with_tau):
        full = (lambda th: th) if with_tau else (lambda th: np.array([th[0], th[1], 0.0]))
        obj = lambda th: np.abs(yy - apply_pow(full(th), f, cp, with_tau)).sum()
        best = None
        for x0 in ([0.0, 1.0, 0.0], [0.5, 0.95, 50.0], [-0.5, 1.05, 200.0]):
            r = minimize(obj, np.array(x0 if with_tau else x0[:2]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        return full(best.x)

    rows = []
    for with_tau in (False, True):
        f, th_all = np.zeros(len(y)), []
        for a, b in folds:
            th = fit_pow(f0[a], y[a], cap[a], with_tau)
            th_all.append([round(float(np.exp(th[0])), 4), round(float(th[1]), 4), round(float(abs(th[2])) if with_tau else 0.0, 1)])
            f[b] = apply_pow(th, f0[b], cap[b], with_tau)
        print(f"power calibration{' + zero cut-off' if with_tau else ''}: per-fold (a, b, tau0) {th_all}")
        judge(f"nested a*yhat^b{' + zero cut-off tau0' if with_tau else ''}", f, rows)

    # ---------------- 2.3 forced raw-scale mixes and raw Huber members
    P3_lgb = tuned("task3_lgb", {"learning_rate": 0.05})
    hub = {}
    for kind in ("lgb", "cat"):
        o = np.zeros(len(y))
        for a, b in folds:
            med = np.median(y[a])
            delta = float(np.quantile(np.abs(y[a] - med), 0.9))
            prm = {**P3_lgb, "objective": "huber", "alpha": delta, "metric": "l1"} if kind == "lgb" else {"learning_rate": 0.08, "depth": 8, "loss_function": f"Huber:delta={delta:.1f}", "eval_metric": "MAE"}
            _, (pv,), _ = fit_predict(kind, "regression", prm, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
            o[b] = clip0(pv)
        hub[f"{'LightGBM' if kind == 'lgb' else 'CatBoost'} Huber raw THB"] = o
        log(f"{kind} Huber raw member alone: NMAE {nmae(y, o):.5f}")
    raw_members = {**{k: plain[k] for k in ("LGB-L1 raw THB", "XGB-L1 raw THB", "Cat-MAE raw THB")}, **hub}
    raw_members["mean of the 3 raw L1 members"] = np.mean([plain[k] for k in ("LGB-L1 raw THB", "XGB-L1 raw THB", "Cat-MAE raw THB")], axis=0)
    for k, v in raw_members.items():
        print(f"  {k}: alone {nmae(y, v):.5f}")
    for k in ("mean of the 3 raw L1 members", "Cat-MAE raw THB", "LightGBM Huber raw THB", "CatBoost Huber raw THB"):
        for w in (0.1, 0.2, 0.3):
            judge(f"forced mix {w:.0%} {k}", np.minimum((1 - w) * f0 + w * raw_members[k], cap), rows)
    f2, _, W2 = nested_final(np.column_stack([P] + list(hub.values())))
    print("Huber members' weight in the exact L1 blend per fold:", {k: np.round(W2[:, P.shape[1] + j], 3).tolist() for j, k in enumerate(hub)})
    judge("exact L1 blend with both Huber members in the pool", f2, rows)
    print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "Decision"]).to_string(index=False))


# ================================================================================= Task 4
def run_t4():
    print("\n" + "=" * 100 + "\nTask 4: item 4 (what a class-mix change can do against a key unrelated to the predictions)\n" + "=" * 100)
    tr4, te4 = task4.load_frames()
    q_train = np.bincount(tr4.label, minlength=5) / len(tr4)
    p_ours = np.bincount(v12.task4, minlength=5) / len(v12)
    print(f"train class shares {np.round(q_train, 4).tolist()}; v12 test prediction shares {np.round(p_ours, 4).tolist()}")
    # if the key is independent of our column (shuffled / mis-aligned), class k gets precision q_k and recall p_k in expectation -> F1_k = 2 p_k q_k / (p_k + q_k)
    ef1 = lambda pp, qq: float(np.mean(2 * pp * qq / np.maximum(pp + qq, 1e-12)))
    for nm, q in [("key mix = our predicted mix", p_ours), ("key mix = train mix", q_train)]:
        print(f"{nm}: expected macro-F1 of v12 {ef1(p_ours, q):.5f}; best possible class mix (p = q) {ef1(q, q):.5f}; uniform mix {ef1(np.full(5, 0.2), q):.5f}; "
              f"all one class {ef1(np.eye(5)[int(np.argmax(q))], q):.5f}")
    n_pub = int(0.5 * len(v12))
    sd = np.std([macro_f1(rng.permutation(v12.task4.values[:n_pub]), v12.task4.values[:n_pub]) for _ in range(300)])
    print(f"spread between random alignments (n = {n_pub:,} rows, assumed public half): SD {sd:.4f} on Task 4 = {0.2 * sd:.5f} on the leaderboard; "
          f"the 0.00067 gap to #1 = {0.00067 / 0.2:.4f} on Task 4 alone = {0.00067 / 0.2 / sd:.1f} SD")


if __name__ == "__main__":
    for part in sys.argv[1:]:
        {"t1": run_t1, "t2": run_t2, "t3": run_t3, "t4": run_t4}[part]()
    log("done")
