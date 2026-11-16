"""Round 36, Task 2: per-class one-vs-all isotonic calibration before the blend, and confusion-pair specialists on top of the v26 rule
(training rows only, CPU).

    python ge_r36_t2_iso_specialist.py

Base = the v26 rule nested: stacker (multinomial LogReg on four members' log-probabilities, out of fold on the split) blended
linearly with the KDE naive Bayes (out of fold on the pipeline folds); KDE weight and class multipliers by the soft macro-F1 search
(src/metrics.optimize_class_weights_soft) on the four training folds.
item 1  One-vs-all isotonic calibration per class (4 independent monotone maps per model), fitted on the training folds' out-of-fold
        probabilities of the stacker and of the naive Bayes, applied to the held-out fold, rows renormalised; then
          1a  linear blend + soft search (as v26);
          1b  log-linear blend (geometric, weight from the same grid) + soft search.
        (Round 34 tested scalar temperatures and Dirichlet matrices; per-class non-parametric maps are new.)
item 2  Confusion-pair specialists. The pairs are ranked by their symmetric confusion count under the base rule (pipeline folds).
        Casual <-> Grinder is skipped: round 30 (eq_r30_t2_pair_specialist.py) tested a specialist for it and it failed (-0.0012).
        For each of the next two pairs (a, b): a binary LightGBM (400 trees, lr 0.05, 31 leaves, 100 rows per leaf; the 34 naive-Bayes
        input columns + the naive Bayes's 4 log-probabilities) trained on the training-fold rows of the two classes only, out of
        fold on the pipeline folds (like the naive Bayes, reused on every split). On the blended probabilities q the pair's log-odds
        are moved towards the specialist's:  logit_ab(q') = logit_ab(q) + lam * (logit(s) - logit_ab(q)),  q'_a + q'_b = q_a + q_b,
        on every row; lam in {0.25, 0.5, 1} and the multipliers chosen together by the soft macro-F1 on the training folds (lam = 0
        allowed). The KDE weight stays the base's.
Splits: pipeline folds + stacker split seeds 1, 2, 3, then seeds 4-9 (fresh). Rules: the fold rule (pooled > 0, >= 3/5 folds, no fold
below -0.0005) and the user's Task 2 rule of 2026-11-16 (mean delta > 0 and row-bootstrap 95% CI above 0 on the fresh splits).
"""
import os
import time
import warnings

NT = int(os.environ.get("R33_THREADS", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import optimize_class_weights_soft  # noqa: E402
from src.modeling import make_folds  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")
y = tr[task2.TARGET].values
K = 4
NAMES_C = ["Casual", "Grinder", "Social", "Whale"]
softmax = lambda Z: np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
lg = lambda P: np.log(np.clip(P, 1e-12, 1))
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LAMS = (0.25, 0.5, 1.0)
pipe_folds = make_folds(y)


def mf1(yy, pred, w=None):
    cm = np.bincount(yy * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)))


def soft(yy, P):
    """(log-multipliers, soft score) of the soft macro-F1 search."""
    w, sc = optimize_class_weights_soft(yy, P)
    return np.log(w), sc


decide = lambda P, v: np.argmax(lg(P) + v[None, :], 1)


def iso_cal(P, a, b):
    """Per-class isotonic maps fitted on rows a, applied to rows b; renormalised."""
    Q = np.column_stack([IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(P[a, c], (y[a] == c).astype(float)).predict(P[b, c])
                         for c in range(K)])
    return Q / Q.sum(1, keepdims=True)


def couple(Q, sp_logit, pair, lam):
    a_, b_ = pair
    m = Q[:, a_] + Q[:, b_]
    cur = np.log(np.clip(Q[:, a_], 1e-12, 1)) - np.log(np.clip(Q[:, b_], 1e-12, 1))
    new = cur + lam * (sp_logit - cur)
    out = Q.copy()
    out[:, a_] = m / (1 + np.exp(-new))
    out[:, b_] = m - out[:, a_]
    return out


# ------------------------------------------------------------------------------------------- base rule and the confusion pairs
drop = [task2.ID_COL, "id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
R = tr.drop(columns=[c for c in drop if c in tr.columns]).copy()
for c, m in task2.ORDINAL_MAPS.items():
    R[c] = R[c].map(m).astype(float)
FX = np.column_stack([R.values.astype(np.float32), lg(KDE)])
Ps0 = np.zeros((len(y), K))
for a, b in pipe_folds:
    Ps0[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
base0 = np.zeros(len(y), int)
for a, b in pipe_folds:
    sc = {w_: soft(y[a], ((1 - w_) * Ps0 + w_ * KDE)[a]) for w_ in WS}
    wb = max(WS, key=lambda w_: sc[w_][1])
    base0[b] = decide(((1 - wb) * Ps0 + wb * KDE)[b], sc[wb][0])
cm = np.bincount(y * K + base0, minlength=K * K).reshape(K, K)
sym = {(i, j): int(cm[i, j] + cm[j, i]) for i in range(K) for j in range(i + 1, K)}
rank = sorted(sym, key=sym.get, reverse=True)
log("base rule (pipeline folds) confusion, true x predicted:\n" + pd.DataFrame(cm, index=NAMES_C, columns=NAMES_C).to_string()
    + "\nsymmetric confusion per pair: " + ", ".join(f"{NAMES_C[i]}<->{NAMES_C[j]} {sym[(i, j)]}" for i, j in rank))
PAIRS = [pr for pr in rank if pr != (0, 1)][:2]
log(f"specialist pairs (Casual <-> Grinder skipped, failed in round 30): {[f'{NAMES_C[i]}<->{NAMES_C[j]}' for i, j in PAIRS]}")
SPEC = {}
for pr in PAIRS:
    z = np.zeros(len(y))
    for a, b in pipe_folds:
        aa = a[np.isin(y[a], pr)]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=NT, verbose=-1, random_state=0).fit(FX[aa], (y[aa] == pr[0]).astype(int))
        z[b] = m.predict_proba(FX[b])[:, 1]
    SPEC[pr] = np.log(np.clip(z, 1e-6, 1 - 1e-6) / (1 - np.clip(z, 1e-6, 1 - 1e-6)))
    rows = np.isin(y, pr)
    q = (1 - 0.6) * Ps0 + 0.6 * KDE
    cur = lg(q)[:, pr[0]] - lg(q)[:, pr[1]]
    log(f"specialist {NAMES_C[pr[0]]}<->{NAMES_C[pr[1]]}: accuracy on the pair's rows {np.mean((SPEC[pr][rows] > 0) == (y[rows] == pr[0])):.4f} vs the "
        f"blend's pairwise log-odds {np.mean((cur[rows] > 0) == (y[rows] == pr[0])):.4f}; correlation {np.corrcoef(SPEC[pr][rows], cur[rows])[0, 1]:.3f}")

# ------------------------------------------------------------------------------------------------------------- nested comparison
NAMES = ["1a isotonic per class, linear blend", "1b isotonic per class, log-linear blend"] + \
        [f"2 specialist {NAMES_C[i]}<->{NAMES_C[j]}" for i, j in PAIRS]
splits = [("pipeline folds", pipe_folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y)))
                                             for s in range(1, 10)]
summary = {n: [] for n in NAMES}
labs_all = []
for si, (label, fl) in enumerate(splits):
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    labs = {n: np.zeros(len(y), int) for n in ["base"] + NAMES}
    notes = {n: [] for n in NAMES}
    for a, b in fl:
        sc = {w_: soft(y[a], ((1 - w_) * Ps + w_ * KDE)[a]) for w_ in WS}
        wb = max(WS, key=lambda w_: sc[w_][1])
        Pb = (1 - wb) * Ps + wb * KDE
        labs["base"][b] = decide(Pb[b], sc[wb][0])
        # item 1: calibrate on the training folds (rows a), apply to a (for the search) and b
        ab = np.r_[a, b]
        CS, CK = np.zeros_like(Ps), np.zeros_like(KDE)
        CS[ab], CK[ab] = iso_cal(Ps, a, ab), iso_cal(KDE, a, ab)
        for n, blend in ((NAMES[0], lambda w_: (1 - w_) * CS + w_ * CK), (NAMES[1], lambda w_: softmax((1 - w_) * lg(CS) + w_ * lg(CK)))):
            sc2 = {w_: soft(y[a], blend(w_)[a]) for w_ in WS}
            w2 = max(WS, key=lambda w_: sc2[w_][1])
            labs[n][b] = decide(blend(w2)[b], sc2[w2][0])
            notes[n].append(w2)
        # item 2: specialists at the base weight
        for n, pr in zip(NAMES[2:], PAIRS):
            opts = {0.0: sc[wb]}
            opts.update({lam: soft(y[a], couple(Pb, SPEC[pr], pr, lam)[a]) for lam in LAMS})
            lam = max(opts, key=lambda k_: opts[k_][1])
            labs[n][b] = decide(couple(Pb, SPEC[pr], pr, lam)[b], opts[lam][0])
            notes[n].append(lam)
    for n in NAMES:
        per = [mf1(y[b], labs[n][b]) - mf1(y[b], labs["base"][b]) for _, b in fl]
        dl = mf1(y, labs[n]) - mf1(y, labs["base"])
        summary[n].append((dl, sum(x > 0 for x in per), min(per)))
        print(f"{label}: {n:42s} {mf1(y, labs[n]):.4f} vs base {mf1(y, labs['base']):.4f}  delta {dl:+.4f}  folds +{sum(x > 0 for x in per)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in per)})  chosen {notes[n]}", flush=True)
    labs_all.append(labs)
    log(f"{label} done")
rng = np.random.default_rng(0)
W = [rng.multinomial(len(y), np.full(len(y), 1 / len(y))).astype(float) for _ in range(300)]
print()
for n in NAMES:
    for part, sl in (("brief's four splits", slice(0, 4)), ("fresh splits 4-9", slice(4, 10))):
        v_ = summary[n][sl]
        ok = np.mean([r[0] for r in v_]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in v_)
        bd = np.array([np.mean([mf1(y, L_[n], w) - mf1(y, L_["base"], w) for L_ in labs_all[sl]]) for w in W])
        print(f"{n} | {part}: mean delta {np.mean([r[0] for r in v_]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in v_)}; worst fold "
              f"{' '.join(f'{r[2]:+.4f}' for r in v_)} -> fold rule {'PASS' if ok else 'fail'}; row-bootstrap 95% CI "
              f"[{np.quantile(bd, .025):+.5f}, {np.quantile(bd, .975):+.5f}] -> Task 2 rule {'PASS' if np.quantile(bd, .025) > 0 and np.mean(bd) > 0 else 'fail'}",
              flush=True)
log("done")
