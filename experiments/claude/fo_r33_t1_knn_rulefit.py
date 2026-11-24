"""Round 33, Task 1 items 2 and 3: neighbourhood label-purity features in the skill space, and a RuleFit member
(training rows only, CPU).

    python fo_r33_t1_knn_rulefit.py [knn] [rulefit]

item 2  k-NN label purity in a small skill space. Round 4 (E1) put in-fold k-NN target features on ALL columns into a level-1
        LightGBM. Here: 10 skill / aim columns only (kill_death_ratio, headshot_percentage, accuracy_score, spray_control_score,
        reaction_time_ms, game_sense_score, aiming_smoothness, crosshair_placement, damage_per_round, first_blood_rate), each
        mapped to normal scores with the training rows' quantiles (missing -> 0), Euclidean neighbours for k = 15, 50, 150 and cosine
        for k = 50. Per row and k: the neighbours' cheater rate, and that rate minus the neighbours' mean member probability
        (does the neighbourhood cheat more than the models predict?). Out of bag inside every outer fold: a row of the held-out fold
        takes its neighbours from the 4 training folds; a training row of fold j takes them from the 3 training folds other than j,
        so no row ever sees its own label or its fold's. The 8 columns go to the two tree meta-models of the v22 level 2.
item 3  RuleFit (Friedman & Popescu 2008): rules = the leaves of 200 depth-3 LightGBM trees (lr 0.05, 200 rows per leaf) fitted on
        the training fold; sparse L1 logistic regression (liblinear, C = 0.02) on the rule indicators + the standardised raw
        columns (winsorised at 1% / 99%, missing -> median) + missing indicators. Out of fold on the pipeline folds; its log-odds
        become a tenth member of the level 2 (all three meta-models).
Nested on the pipeline folds and stacker split seeds 1, 2, 3. Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each
split, no fold below -0.0005.
"""
import sys

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_from, log, y

what = set(sys.argv[1:]) or {"knn", "rulefit"}
base_p = c.base_probs()
base_labels = [labels_from(p, fold_of(fl)) for p, (_, fl) in zip(base_p, SPLITS)]
log("base level 2 ready: " + ", ".join(f"{f2(y, b):.5f}" for b in base_labels))

if "knn" in what:
    SKILL = ["kill_death_ratio", "headshot_percentage", "accuracy_score", "spray_control_score", "reaction_time_ms", "game_sense_score",
             "aiming_smoothness", "crosshair_placement", "damage_per_round", "first_blood_rate"]
    XS = c.tr[SKILL].values.astype(float)
    PM = expit(L.mean(1))                                                 # mean member probability (out of fold on the pipeline folds)
    KS = (15, 50, 150)

    def normal_scores(ref, rows):
        """Normal scores of `rows` against the empirical distribution of the reference rows; missing -> 0."""
        out = np.zeros((len(rows), XS.shape[1]))
        for j in range(XS.shape[1]):
            v = np.sort(XS[ref, j][~np.isnan(XS[ref, j])])
            x = XS[rows, j]
            u = (np.searchsorted(v, x, side="left") + np.searchsorted(v, x, side="right")) / 2 / len(v)
            out[:, j] = np.where(np.isnan(x), 0.0, norm.ppf(np.clip(u, 0.5 / len(v), 1 - 0.5 / len(v))))
        return out

    def purity(ref, rows):
        """8 features of `rows` from their neighbours among `ref` (disjoint sets)."""
        Zr, Zq = normal_scores(ref, ref), normal_scores(ref, rows)
        idx = NearestNeighbors(n_neighbors=max(KS), algorithm="brute", n_jobs=c.NT).fit(Zr).kneighbors(Zq, return_distance=False)
        yr, pr = y[ref][idx], PM[ref][idx]
        cols = []
        for k in KS:
            cols += [yr[:, :k].mean(1), yr[:, :k].mean(1) - pr[:, :k].mean(1)]
        nrm = lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        ic = NearestNeighbors(n_neighbors=50, algorithm="brute", n_jobs=c.NT).fit(nrm(Zr)).kneighbors(nrm(Zq), return_distance=False)
        cols += [y[ref][ic].mean(1), y[ref][ic].mean(1) - PM[ref][ic].mean(1)]
        return np.column_stack(cols)

    cand = []
    for k, (label, fl) in enumerate(SPLITS):
        fo = fold_of(fl)
        pair = {}
        for i in range(5):
            for j in range(i + 1, 5):
                ref = np.where((fo != i) & (fo != j))[0]
                rows = np.where((fo == i) | (fo == j))[0]
                pair[(i, j)] = (rows, purity(ref, rows))
        extra = []
        for i in range(5):
            E = np.zeros((len(y), 8))
            b = np.where(fo == i)[0]
            E[b] = purity(np.where(fo != i)[0], b)
            for j in range(5):
                if j != i:
                    rows, F = pair[(min(i, j), max(i, j))]
                    m = fo[rows] == j
                    E[rows[m]] = F[m]
            extra.append(E)
        log(f"{label}: purity features done; held-out rows: corr(rate_50, y) {np.mean([np.corrcoef(extra[i][fo == i, 2], y[fo == i])[0, 1] for i in range(5)]):.3f}, "
            f"corr(rate_50 - predicted_50, y - member mean) {np.mean([np.corrcoef(extra[i][fo == i, 3], (y - PM)[fo == i])[0, 1] for i in range(5)]):.4f}")
        cand.append(labels_from(c.level2(fl, extra_by_fold=extra), fo))
        log(f"{label}: level 2 with purity features {f2(y, cand[-1]):.5f} (base {f2(y, base_labels[k]):.5f})")
    c.report("item 2 k-NN purity in the skill space (tree meta-models)", cand, base_labels)

if "rulefit" in what:
    f_rf = c.OUT / "t1_rulefit_oof.npz"
    X = c.tr[c.RAW]
    if f_rf.exists():
        oof = np.load(f_rf)["oof"]
    else:
        oof = np.zeros(len(y))
        for i, (a, b) in enumerate(c.folds):
            gb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, num_leaves=8, min_child_samples=200, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8, n_jobs=c.NT, verbose=-1, random_state=0).fit(X.iloc[a], y[a])
            la, lb = gb.predict(X.iloc[a], pred_leaf=True), gb.predict(X.iloc[b], pred_leaf=True)
            off = np.arange(la.shape[1]) * 8

            def onehot(lv):
                r = np.repeat(np.arange(len(lv)), lv.shape[1])
                return sp.csr_matrix((np.ones(lv.size), (r, (lv + off).ravel())), shape=(len(lv), la.shape[1] * 8))
            Xa, Xb = X.iloc[a].values.astype(float), X.iloc[b].values.astype(float)
            lo, hi, med = np.nanpercentile(Xa, 1, axis=0), np.nanpercentile(Xa, 99, axis=0), np.nanmedian(Xa, axis=0)
            lin = lambda Z: np.clip(np.where(np.isnan(Z), med, Z), lo, hi)
            mu, sd = lin(Xa).mean(0), lin(Xa).std(0) + 1e-9
            # Friedman & Popescu scale the linear terms by 0.4 so that they weigh like a typical rule
            Da = sp.hstack([onehot(la), sp.csr_matrix(0.4 * (lin(Xa) - mu) / sd), sp.csr_matrix(np.isnan(Xa).astype(float))], format="csr")
            Db = sp.hstack([onehot(lb), sp.csr_matrix(0.4 * (lin(Xb) - mu) / sd), sp.csr_matrix(np.isnan(Xb).astype(float))], format="csr")
            m = LogisticRegression(penalty="l1", solver="liblinear", C=0.02, max_iter=300).fit(Da, y[a])
            oof[b] = m.predict_proba(Db)[:, 1]
            nz = m.coef_[0] != 0
            log(f"RuleFit fold {i}: {Da.shape[1]} terms, {int(nz.sum())} non-zero ({int(nz[:la.shape[1] * 8].sum())} rules); F2 at its best cut "
                f"{f2(y[b], (oof[b] >= c.best_cut(y[b], oof[b])).astype(int)):.4f}")
        np.savez(f_rf, oof=oof)
    z = c.logit(oof)
    log(f"RuleFit alone (out of fold, best single cut): F2 {f2(y, (oof >= c.best_cut(y, oof)).astype(int)):.5f}; correlation of its log-odds with "
        f"the members {np.round([np.corrcoef(z, L[:, j])[0, 1] for j in range(L.shape[1])], 3)}")
    S10 = np.column_stack([L, z])
    cand = []
    for k, (label, fl) in enumerate(SPLITS):
        cand.append(labels_from(c.level2(fl, S_lr=S10, S_tr=S10), fold_of(fl)))
        log(f"{label}: level 2 with the RuleFit member {f2(y, cand[-1]):.5f} (base {f2(y, base_labels[k]):.5f})")
    c.report("item 3 RuleFit as a tenth member", cand, base_labels)
log("done")
