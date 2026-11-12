"""Round 16: micro-gain harvesting (pasted brief; the user asked to run all of it).

    python cf_r16_eval.py t2 t5 t3 t1       # any subset; t1 needs by_r14_bag.py full t1 and ce_r16_t1_members.py first

Decision rules. Strict = the rule of rounds 10-15 (95% CI > 0, >= 4/5 folds, fold std <= 1.1x, Task 1 test-like >= 0).
Relaxed (asked for in this round) = mean nested gain > 0 and, for Task 1, test-like gain >= 0. Both are printed; v14 uses the relaxed rule.
Task 1 order fixed before the run: (1) the brief's recipe = 0.88 x 10-seed stack + 0.08 x TabM + 0.04 x spline-LR, Gaussian-smoothed cut-off;
if it fails the relaxed rule, (2) 10-seed stack + smoothed cut-off; then (3) 10-seed stack + argmax cut-off. The first that passes goes into v14.
Task 2 and Task 5 have no labelled way to verify a row-level fix, so they only change if a deterministic contradiction is found.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT, _logit, tuned  # noqa: E402
from by_r14_bag import DIR, SEEDS  # noqa: E402
from src import task1, task2, task3, task5  # noqa: E402
from src.metrics import fbeta, nmae, optimize_threshold  # noqa: E402
from src.modeling import fit_predict, make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
FULL_W = 0.5
blend_full = lambda f, g: (1 - FULL_W) * np.asarray(f) + FULL_W * np.asarray(g)
v13 = pd.read_csv("submission_v13.csv")


def fold_ids(folds, n):
    f = np.zeros(n, int)
    for k, (_, b) in enumerate(folds):
        f[b] = k
    return f


# ================================================================================= Task 2
def run_t2():
    print("\n" + "=" * 100 + "\nTask 2 (item 4): spending by id-block class, looking for deterministic contradictions\n" + "=" * 100)
    tr, te = task2.load()
    CUTS, NAMES = [52000, 84500, 110500], ["Casual", "Grinder", "Social", "Whale"]
    blk = np.searchsorted(CUTS, tr.player_id.str[1:].astype(int).values, side="right")
    blkt = np.searchsorted(CUTS, te.player_id.str[1:].astype(int).values, side="right")
    assert (blkt == v13.task2.values).all()
    y = tr.segment.values
    desc = lambda s: s.describe(percentiles=[0.01, 0.05, 0.5, 0.95]).drop("count")
    for col in ("total_spending_thb", "avg_monthly_spending"):
        print(f"\n--- {col}")
        rows = {}
        for c in range(4):
            rows[f"test, block = {NAMES[c]}"] = desc(te[col][blkt == c])
            rows[f"train, label = {NAMES[c]}"] = desc(tr[col][y == c])
            rows[f"train, block = {NAMES[c]}, label != block"] = desc(tr[col][(blk == c) & (y != c)])
        t = pd.DataFrame(rows).T
        t.insert(0, "rows", [int(m.sum()) for c in range(4) for m in ((blkt == c), (y == c), (blk == c) & (y != c))])
        t.insert(1, "zero or missing", [int(((s == 0) | s.isna()).sum()) for c in range(4)
                                        for s in (te[col][blkt == c], tr[col][y == c], tr[col][(blk == c) & (y != c)])])
        print(t.round(1).to_string())
    s_tr, s_te = tr.total_spending_thb, te.total_spending_thb
    w_min = s_tr[y == 3].min()
    print(f"\nlowest total_spending_thb of a TRUE Whale in train: {w_min}; true Whales with 0 spending: {int((s_tr[y == 3] == 0).sum())}; "
          f"highest of a true Casual: {s_tr[y == 0].max():,.0f}")
    print(f"test rows the block calls Whale with spending below the lowest true Whale in train: {int((s_te[blkt == 3] < w_min).sum())}")
    print("flip rate (label != block) in train by spending decile within each block; a usable contradiction would show a strong trend:")
    rows = []
    for c in range(4):
        m = blk == c
        q = pd.qcut(s_tr[m].rank(method="first"), 10, labels=False)
        rows.append([NAMES[c]] + [round(float((y[m] != c)[q.values == d].mean()), 4) for d in range(10)])
    print(pd.DataFrame(rows, columns=["block"] + [f"d{d + 1}" for d in range(10)]).to_string(index=False))


# ================================================================================= Task 5
def run_t5():
    print("\n" + "=" * 100 + "\nTask 5 (item 3): the most extreme accounts outside the id block, and the F3 arithmetic\n" + "=" * 100)
    import torch
    t5 = task5.load()
    reduce_mem_usage(t5, verbose=False)
    X5, _ = task5.build_features(t5)
    idn = t5.player_id.str[1:].astype(int).values
    blk = idn >= 23819
    assert (blk.astype(int) == v13.task5.values).all()
    _, gc = task5.gap_clusters(t5, X5)
    gap = gc.any(axis=1).values
    clean = pd.read_csv("submission_clean_v13.csv").task5.values == 1
    pocket = clean & ~gap
    feats = ["accuracy_std_level", "kda_std_level"]
    pscore = X5[feats].rank(pct=True).min(axis=1).values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cols = task5.detector_columns(X5)
    Z = task5.detector_matrix(X5, cols, "robust")
    det = {}
    for name, fn in task5.DETECTORS.items():
        det[name] = task5.robust_z(fn(Z, device=device) if name == "AutoEncoder" else fn(Z))
    cons = np.mean([v for k, v in det.items() if k != "LOF"], 0)
    rules = task5.evidence_rules(t5)
    anyrule = rules.any(axis=1).values
    Zabs = pd.DataFrame(np.abs(Z), columns=cols)
    out = ~blk
    print(f"outside the block: {out.sum():,} accounts; pocket flags {int((pocket & out).sum())}; evidence-rule hits {int((anyrule & out).sum())}")
    print(f"consensus score: in-block median {np.median(cons[blk]):.2f}, p10 {np.quantile(cons[blk], 0.1):.2f}, p1 {np.quantile(cons[blk], 0.01):.2f}; "
          f"outside max {cons[out].max():.2f}, p99.9 {np.quantile(cons[out], 0.999):.2f}")
    top = np.where(out)[0][np.argsort(-cons[out])[:25]]
    rows = []
    for i in top:
        zrow = Zabs.iloc[i].sort_values(ascending=False)
        rows.append([t5.player_id.iloc[i], round(float(cons[i]), 2), round(float((cons[blk] < cons[i]).mean()), 3), bool(pocket[i]), round(float(pscore[i]), 3),
                     ",".join(rules.columns[rules.iloc[i].values.astype(bool)]) or "-", ", ".join(f"{k} {v:.0f}" for k, v in zrow.head(3).items())])
    print("\n25 most extreme outside-block accounts by the detector consensus (share of block accounts with a lower consensus; strongest robust-z features):")
    print(pd.DataFrame(rows, columns=["player_id", "consensus", "share of block below", "pocket flag", "pocket score", "evidence rules", "top robust |z| features"]).to_string(index=False))
    fam_cols = {}
    for c in ("geo_km_per_login_4", "geo_hop_max", "device_count_maxjump", "device_count_dev4", "login_count_burst", "login_count_maxjump"):
        v = X5[c].values
        topm = np.nan_to_num(v, nan=-np.inf) >= np.nanquantile(v, 0.999)
        fam_cols[c] = f"{int((topm & blk).sum())}/{int(topm.sum())}"
    print("\nshare of each feature's top 0.1% that lies inside the block (the block is 8.0% of accounts; a generated anomaly type would sit almost entirely inside):")
    print(fam_cols)

    # F3 arithmetic: truth = block + m extra anomalies directly among the k added rows (hit rate h); currently TP 2,070, FN m, FP 0
    f3 = lambda tp, fp, fn: 10 * tp / (10 * tp + 9 * fn + fp)
    rows = []
    for k in (10, 15, 20, 25):
        for h in (0.0, 0.1, 0.3, 0.5):
            m = int(round(h * k))
            before = f3(2070, 0, m)
            after = f3(2070 + m, k - m, 0)
            rows.append([k, h, m, before, after, after - before, 0.2 * (after - before)])
    print("\nF3 if the k most extreme outside-block accounts are added and a share h of them are real anomalies (no other anomalies outside the block):")
    print(pd.DataFrame(rows, columns=["k added", "hit rate h", "real anomalies among them", "F3 now", "F3 after", "delta F3", "delta LB"]).round(5).to_string(index=False))
    gain20, loss20 = f3(2080, 10, 0) - f3(2070, 0, 10), f3(2070, 20, 0) - f3(2070, 0, 0)
    print(f"k = 20: +{gain20:.5f} F3 if half are real, {loss20:+.5f} if none are. Adding them pays only if P(half are real) > {-loss20 / (gain20 - loss20):.2f}; "
          f"there is no labelled data to estimate it, so nothing is changed.")


# ================================================================================= Task 3
def run_t3():
    print("\n" + "=" * 100 + "\nTask 3 (item 2): small forced blends of other model families; cap 250k-350k, nested\n" + "=" * 100)
    tr, te = task3.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt, _ = task3.build_repaired(tr, te)
    folds = make_folds(y, kind="regression")
    fid = fold_ids(folds, len(y))
    clip0 = lambda v: np.clip(v, 0, None)
    slug3 = lambda t: "t3r_" + t.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
    SEEDS3 = SEEDS
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
    cols, names, RO = list(plain.values()), list(plain), {}
    for t in TAGS:
        RO[t] = task3.readouts(y, p, mu[t])
        for rk in ("mixture-median", "hard gate", "E[y] + smearing"):
            names.append(f"{t} {rk}")
            cols.append(task3.apply_readout(rk, p, mu[t], RO[t]))
    P = np.column_stack(cols)
    gate = (np.clip(p + RO["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
    bo, capf = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        bo[b] = P[b] @ task3.l1_blend_weights_exact(y[a], [P[a, j] for j in range(P.shape[1])])
        capf[b] = y[a].max()
    scales = np.linspace(0.85, 1.15, 61)
    sc = scales[int(np.argmax([nmae(y, bo * s * gate) for s in scales]))]
    pre = bo * sc * gate
    f0 = np.minimum(pre, capf)
    log(f"B0 = v13 pipeline: OOF NMAE {nmae(y, f0):.6f}; OOF predictions above 250k before the cap: {int((pre > 250000).sum())}, above 300k: {int((pre > 300000).sum())}; "
        f"training rows with y = 300,000: {int((y == 300000).sum())}, y > 250k: {int((y > 250000).sum())}")
    pf = lambda f: np.array([nmae(y[fid == k], f[fid == k]) for k in range(5)])
    g0 = pf(f0)
    rows = []

    def judge(nm, f):
        g = pf(f)
        e1, e0 = np.abs(y - f), np.abs(y - f0)
        d = []
        for _ in range(1000):
            w = rng.poisson(1.0, len(y)).astype(float)
            sy = w @ y
            d.append(1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy))
        lo, hi = np.percentile(d, [2.5, 97.5])
        delta = nmae(y, f) - nmae(y, f0)
        better = int((g > g0).sum())
        strict = lo > 0 and better >= 4 and g.std() <= 1.1 * g0.std()
        rows.append([nm, nmae(y, f), delta, f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", " ".join(f"{x:+.5f}" for x in g - g0), 0.2 * delta,
                     "KEEP" if strict else "discard", "KEEP" if delta > 0 else "discard"])

    # cap grid, nested
    grid = np.arange(250000, 350001, 10000)
    fc, chosen = np.zeros(len(y)), []
    for k, (a, b) in enumerate(folds):
        c = grid[int(np.argmax([nmae(y[a], np.minimum(pre[a], cc)) for cc in grid]))]
        chosen.append(int(c))
        fc[b] = np.minimum(pre[b], c)
    print(f"cap chosen per fold (grid 250k-350k): {chosen}; in-sample NMAE by cap: " + ", ".join(f"{c // 1000}k {nmae(y, np.minimum(pre, c)):.6f}" for c in grid))
    judge("cap chosen nested from 250k-350k", fc)
    t3v = v13.task3.values
    print(f"v13 test predictions: >= 250k {int((t3v >= 250000).sum())}, >= 280k {int((t3v >= 280000).sum())}, = 300,000 (capped) {int((t3v == 300000).sum())}; "
          f"largest five {np.sort(t3v)[-5:].round(0).tolist()}")

    # distinct families, forced 5-12% (gated so the zero predictions stay zero)
    fams = {"LGB-L1 log1p": plain["LGB-L1 log1p"], "LGB-Tweedie (mean)": plain["LGB-Tweedie (mean)"], "XGB-L1 raw THB": plain["XGB-L1 raw THB"],
            "single-seed ratio[hist] LGB mixture-median": P[:, names.index("ratio[hist] LGB mixture-median")],
            "plain hurdle (no ratio) mixture-median": P[:, names.index("Hurdle mixture-median")]}
    log("training CatBoost raw-THB MAE at depth 6 (alternative depth)")
    o = np.zeros(len(y))
    for a, b in folds:
        _, (pv,), _ = fit_predict("cat", "regression", {"learning_rate": 0.08, "depth": 6, "loss_function": "MAE"}, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
        o[b] = clip0(pv)
    fams["CatBoost MAE raw, depth 6"] = o
    log("training ridge regression on log1p(y) of spenders (non-GBDT stage 2), read out with the bagged stage 1")
    mu_r = np.zeros(len(y))
    Xn = X.select_dtypes("number").values.astype(float)
    for a, b in folds:
        sp = a[y[a] > 0]
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)).fit(Xn[sp], np.log1p(y[sp]))
        mu_r[b] = m.predict(Xn[b])
    ro_r = task3.readouts(y, p, mu_r)
    fams["ridge log1p stage 2 + mixture-median"] = task3.apply_readout("mixture-median", p, mu_r, ro_r)
    for k_, v in fams.items():
        print(f"  {k_}: alone {nmae(y, v):.5f}")
    for k_, v in fams.items():
        for w in (0.05, 0.08, 0.12):
            judge(f"{w:.0%} {k_}", np.minimum((1 - w) * pre + w * v * gate, capf))
    print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "per-fold delta", "expected LB", "strict", "relaxed"]).to_string(index=False))


# ================================================================================= Task 1
def run_t1():
    print("\n" + "=" * 100 + "\nTask 1 (items 1 and 5): 10-seed stack + low-weight TabM / spline-LR + smoothed cut-off\n" + "=" * 100)
    tr, te, _ = task1.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task1.build(tr, te)
    folds = make_folds(y)
    fid = fold_ids(folds, len(y))
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

    keys = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
    Zb = np.load(OUT / "bn_t1_members.npz")
    avg2 = lambda k, s: (Zb[f"{k}_42_{s}"] + Zb[f"{k}_7_{s}"]) / 2
    # baseline (v13) members: pipeline cache for OOF / test, round-11 file for the test-like versions (as in bz_r14_eval.py)
    b_oof = [load_preds(f"t1_{k}")["oof"] for k in keys]
    b_test = [blend_full(load_preds(f"t1_{k}")["test"], load_preds(f"t1_{k}_full")["test"]) for k in keys]
    z_oof = [Zb[f"{k}_oof"] for k in ("lgb", "xgb", "cat", "et", "lr")] + [avg2(f"their_{k[:-2]}", "oof") for k in ("rf24", "lgb24", "xgb24", "cat24")]
    z_tl = [Zb[f"{k}_tl"] for k in ("lgb", "xgb", "cat", "et", "lr")] + [avg2(f"their_{k[:-2]}", "tl") for k in ("rf24", "lgb24", "xgb24", "cat24")]
    # 10-seed bag: mean log-odds over seeds; test input per seed = 0.5 x fold average + 0.5 x Train+Val refit
    lg = lambda arrs: expit(np.mean([_logit(a) for a in arrs], axis=0))
    bag_oof, bag_tl, bag_test = [], [], []
    for k in keys:
        if k == "lr":
            bag_oof.append(load_preds("t1_lr")["oof"]); bag_tl.append(Zb["lr_tl"]); bag_test.append(b_test[keys.index("lr")])
            continue
        cv = [np.load(DIR / f"t1_{k}_{s}.npz") for s in SEEDS]
        fu = [np.load(DIR / f"t1full_{k}_{s}.npz") for s in SEEDS]
        bag_oof.append(lg([z["oof"] for z in cv])); bag_tl.append(lg([z["tl"] for z in cv]))
        bag_test.append(lg([blend_full(z["test"], f["test"]) for z, f in zip(cv, fu)]))
    R = np.load(OUT / "r16_t1_members.npz")
    tabm = (R["tabm_oof"], R["tabm_tl"], blend_full(R["tabm_test"], R["tabm_full"]))
    spl = (R["spline_oof"], R["spline_tl"], blend_full(R["spline_test"], R["spline_full"]))

    def stack(oofs, evals, test=None):
        S = np.column_stack([_logit(m) for m in oofs])
        Se = np.column_stack([_logit(m) for m in evals])
        o, oe = np.zeros(len(y)), np.zeros(len(y))
        for a, b in folds:
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(S[a], y[a])
            o[b], oe[b] = lr.predict_proba(S[b])[:, 1], lr.predict_proba(Se[b])[:, 1]
        t = None if test is None else LogisticRegression(C=1.0, max_iter=3000).fit(S, y).predict_proba(np.column_stack([_logit(m) for m in test]))[:, 1]
        return o, oe, t

    s0, s0_tl_unused, s0_test = stack(b_oof, b_oof, b_test)
    _, s0_tl, _ = stack(z_oof, z_tl)
    s0_tl_oof = stack(z_oof, z_oof)[0]
    sb, sb_tl, sb_test = stack(bag_oof, bag_tl, bag_test)

    Gf = np.round(np.arange(0.05, 0.6000001, 0.0005), 4)

    def curve(p, yy, w=None):
        w = np.ones(len(yy)) if w is None else w
        o = np.argsort(p, kind="stable")
        ps, ys, ws = p[o], yy[o], w[o]
        cp = np.concatenate([[0.0], np.cumsum(ws * ys)]); cn = np.concatenate([[0.0], np.cumsum(ws * (1 - ys))])
        i = np.searchsorted(ps, Gf, side="left")
        tp, fp, fn = cp[-1] - cp[i], cn[-1] - cn[i], cp[i]
        return 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1e-12)

    cut = {"argmax": lambda p, yy: optimize_threshold(yy, p, 2.0, average="binary")[0],
           "Gaussian sigma 0.04": lambda p, yy: Gf[int(np.argmax(gaussian_filter1d(curve(p, yy), 0.04 / 0.0005, mode="nearest")))]}

    def boot_med(p, yy, B=1000):
        return float(np.median([Gf[int(np.argmax(curve(p, yy, rng.poisson(1.0, len(yy)).astype(float))))] for _ in range(B)]))
    cut["bootstrap median"] = boot_med

    def evaluate(p, p_tl, rule, p_tl_sel=None):
        """Nested: cut-off chosen on the training folds' OOF scores, applied to the held-out fold (and to its test-like version)."""
        pred, pred_tl = np.zeros(len(y), int), np.zeros(len(y), int)
        sel = p if p_tl_sel is None else p_tl_sel
        for a, b in folds:
            t = cut[rule](p[a], y[a])
            pred[b] = p[b] >= t
            t2 = cut[rule](sel[a], y[a])
            pred_tl[b] = p_tl[b] >= t2
        return pred, pred_tl

    ref, ref_tl = evaluate(s0, s0_tl, "argmax", s0_tl_oof)
    g0 = np.array([f2(y[fid == k], ref[fid == k]) for k in range(5)])
    log(f"B0 (v13 Task 1): nested F2 {f2(y, ref):.5f}, folds {np.round(g0, 4).tolist()}, std {g0.std():.5f}; test-like {wf2(ref_tl, w_tl):.5f}")
    print(f"members alone (argmax cut-off, OOF): TabM {optimize_threshold(y, tabm[0], 2.0, average='binary')[1]:.5f}, spline-LR {optimize_threshold(y, spl[0], 2.0, average='binary')[1]:.5f}, "
          f"10-seed stack {optimize_threshold(y, sb, 2.0, average='binary')[1]:.5f}; corr(stack, TabM) {np.corrcoef(sb, tabm[0])[0, 1]:.4f}, corr(stack, spline) {np.corrcoef(sb, spl[0])[0, 1]:.4f}")

    def boot(pn, po, w, B=1000):
        d = []
        for _ in range(B):
            pw = w * rng.poisson(1.0, len(y))
            d.append(wf2(pn, pw) - wf2(po, pw))
        return np.percentile(d, [2.5, 97.5])

    variants = {}
    mix = lambda ws, i: ws[0] * [sb, sb_tl, sb_test][i] + ws[1] * tabm[i] + ws[2] * spl[i]
    W_BRIEF = (0.88, 0.08, 0.04)
    variants["(1) brief: 0.88 bag + 0.08 TabM + 0.04 spline, smoothed cut"] = (mix(W_BRIEF, 0), mix(W_BRIEF, 1), mix(W_BRIEF, 2), "Gaussian sigma 0.04")
    variants["(2) 10-seed bag, smoothed cut"] = (sb, sb_tl, sb_test, "Gaussian sigma 0.04")
    variants["(3) 10-seed bag, argmax cut"] = (sb, sb_tl, sb_test, "argmax")
    variants["(4) brief blend, argmax cut"] = (mix(W_BRIEF, 0), mix(W_BRIEF, 1), mix(W_BRIEF, 2), "argmax")
    variants["(5) brief blend, bootstrap-median cut"] = (mix(W_BRIEF, 0), mix(W_BRIEF, 1), mix(W_BRIEF, 2), "bootstrap median")
    for wt, ws in ((0.05, 0.0), (0.10, 0.0), (0.15, 0.0), (0.08, 0.08)):
        W = (1 - wt - ws, wt, ws)
        variants[f"(6) bag + {wt:.0%} TabM + {ws:.0%} spline, smoothed cut"] = (mix(W, 0), mix(W, 1), mix(W, 2), "Gaussian sigma 0.04")
    rows, out = [], {}
    for nm, (p, p_tl, p_te, rule) in variants.items():
        pred, pred_tl = evaluate(p, p_tl, rule)
        g = np.array([f2(y[fid == k], pred[fid == k]) for k in range(5)])
        lo, hi = boot(pred, ref, np.ones(len(y)))
        delta, dtl = f2(y, pred) - f2(y, ref), wf2(pred_tl, w_tl) - wf2(ref_tl, w_tl)
        better = int((g > g0).sum())
        strict = lo > 0 and better >= 4 and g.std() <= 1.1 * g0.std() and dtl >= 0
        relaxed = delta > 0 and dtl >= 0
        t_full = cut[rule](p, y)
        test_lab = (p_te >= t_full).astype(int)
        out[nm] = (test_lab, relaxed, t_full)
        rows.append([nm, f2(y, pred), delta, f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", " ".join(f"{x:+.4f}" for x in g - g0), g.std(), f"{dtl:+.5f}",
                     0.2 * delta, round(t_full, 4), int(test_lab.sum()), int((test_lab != v13.task1.values).sum()), "KEEP" if strict else "discard", "KEEP" if relaxed else "discard"])
        log(f"{nm}: {rows[-1][1]:.5f} ({delta:+.5f}), {better}/5, test-like {dtl:+.5f}, test flips vs v13 {rows[-1][11]}")
    print("\n" + pd.DataFrame(rows, columns=["Variant", "nested F2", "delta", "95% CI", "folds", "per-fold delta", "fold std", "test-like delta", "expected LB",
                                             "full tau", "test 1s", "test flips vs v13", "strict", "relaxed"]).to_string(index=False))
    order = ["(1) brief: 0.88 bag + 0.08 TabM + 0.04 spline, smoothed cut", "(2) 10-seed bag, smoothed cut", "(3) 10-seed bag, argmax cut"]
    pick = next((nm for nm in order if out[nm][1]), None)
    print(f"\npre-registered pick for v14 Task 1: {pick or 'none passes -> Task 1 stays as v13'}")
    if pick:
        lab = out[pick][0]
        np.save(OUT / "r16_t1_pick.npy", lab)
        changed = lab != v13.task1.values
        print(f"test labels changed vs v13: {int(changed.sum())} (0 -> 1: {int((changed & (lab == 1)).sum())}, 1 -> 0: {int((changed & (lab == 0)).sum())}); "
              f"flag rate {v13.task1.mean():.4%} -> {lab.mean():.4%}; full-data cut-off {out[pick][2]:.4f}")
        print("v13 stack probability of the changed rows: " + str(np.round(np.sort(s0_test[changed]), 4).tolist()[:60]))


if __name__ == "__main__":
    for part in sys.argv[1:]:
        {"t1": run_t1, "t2": run_t2, "t3": run_t3, "t5": run_t5}[part]()
    log("done")
