"""Round 36, Task 1: ranking-loss level-1 members and a leaf-embedding member (training rows only, CPU).

    python gc_r36_t1_rank_leaf.py [rank] [leaf]

item 1  Ranking objectives. Every member so far minimises a point-wise loss; F2 at the cut depends on how cheaters are ORDERED
        against normal players of similar skill. Members (the 49 pipeline features, early stopping on the validation fold's AUC as
        in the pipeline, 3 seeds, mean of the raw scores):
          RL  LightGBM lambdarank, queries = skill tier (decile of the pipeline's mech_index, cut on the training fold) x random
              buckets of about 200 rows, all pairs of a query used (truncation level = query size), label gain 0 / 1;
          RX  XGBoost rank:pairwise, queries = the ten skill tiers, pairs sampled inside a tier.
        Round 9 (Q2) tried rank:pairwise with ONE query over all rows as a sixth member of the old five-member stack: +0.0009 nested
        (CI [+0.0002, +0.0015]) and was discarded by the test-like gate, which was retired after the TA ruling of 15 Nov 2026;
        tiered queries and lambdarank are new, and the base is now the nine-member v22 level 2.
item 2  Leaf-embedding member (out of fold twice over). Inside each pipeline fold the training rows are halved (A, B). A LightGBM
        (300 trees, 16 leaves) fitted on A gives the leaf indices of B and of the validation rows; the one-hot leaf matrix
        (4,800 columns) is reduced to 16 TruncatedSVD components fitted on B; 64 k-means clusters of the components get a smoothed
        cheater rate (m = 50; out of fold over 5 inner folds for B). A small LightGBM (200 trees, 8 leaves) on the 16 components +
        the cluster rate, fitted on B, scores the validation rows; the roles of A and B are swapped and the two log-odds averaged.
        No row's label is seen by the trees that embed it. (RuleFit, the L1 version on raw leaf indicators, failed in round 33.)
Level 2 = v22 (r33_t1_common.level2) with each member as a tenth member, and RL + RX as tenth and eleventh; nested on the pipeline
folds and stacker split seeds 1, 2, 3. Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below
-0.0005.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_from, log, logit, y
from src import task1

what = set(sys.argv[1:]) or {"rank", "leaf"}
OUT = c.OUT.parent / "r36"
OUT.mkdir(parents=True, exist_ok=True)
raw = c.tr.drop(columns=[col for col in ("player_id", "id", task1.TARGET) if col in c.tr.columns])
X49 = task1.add_features(c.tr, ref=raw)
XV = X49.values.astype(np.float32)
P = {k: v for k, v in json.load(open("outputs/params/task1_lgb.json")).items() if k != "scale_pos_weight"}
LGBR = dict(objective="lambdarank", label_gain=[0, 1], learning_rate=P["learning_rate"], num_leaves=P["num_leaves"],
            min_data_in_leaf=P["min_child_samples"], feature_fraction=P["colsample_bytree"], bagging_fraction=P["subsample"], bagging_freq=1,
            lambda_l1=P["reg_alpha"], lambda_l2=P["reg_lambda"], metric="None", verbose=-1, num_threads=c.NT)
auc_eval = lambda s, d: ("auc", roc_auc_score(d.get_label(), s), True)
cut_f2 = lambda z: f2(y, (z >= c.best_cut(y, z)).astype(int))
members = {}

if "rank" in what:
    f_r = OUT / "t1_rank_members.npz"
    if f_r.exists():
        d = np.load(f_r)
        members["RL lambdarank LightGBM (tier x bucket queries)"], members["RX rank:pairwise XGBoost (tier queries)"] = d["rl"], d["rx"]
    else:
        zl, zx = np.zeros((3, len(y))), np.zeros((3, len(y)))
        mech = X49["mech_index"].values
        for i, (a, b) in enumerate(c.folds):
            edges = np.nanquantile(mech[a], np.linspace(0.1, 0.9, 9))
            tier = np.where(np.isnan(mech), 10, np.searchsorted(edges, mech))
            its = []
            for s in (0, 1, 2):
                rng = np.random.default_rng(100 * i + s)
                # lambdarank: queries = tier x random bucket of about 200 rows; rows sorted by query
                q = tier[a] * 10_000 + rng.permutation(len(a)) % max(1, len(a) // (11 * 200))
                o = np.argsort(q, kind="stable")
                sizes = np.unique(q[o], return_counts=True)[1]
                prm = dict(LGBR, seed=s, bagging_seed=s, feature_fraction_seed=s, lambdarank_truncation_level=int(sizes.max()))
                dtr = lgb.Dataset(XV[a][o], y[a][o], group=sizes)
                dva = lgb.Dataset(XV[b], y[b], group=[len(x) for x in np.array_split(np.arange(len(b)), 8)])   # queries must stay under 10,000 rows
                m = lgb.train(prm, dtr, num_boost_round=6000, valid_sets=[dva], feval=auc_eval, callbacks=[lgb.early_stopping(200, verbose=False)])
                zl[s, b] = m.predict(XV[b], num_iteration=m.best_iteration, raw_score=True)
                its.append(m.best_iteration)
                # rank:pairwise: queries = the skill tiers
                o2 = np.argsort(tier[a], kind="stable")
                mx = xgb.XGBRanker(objective="rank:pairwise", n_estimators=6000, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                                   colsample_bytree=0.6, tree_method="hist", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=4,
                                   eval_metric="auc", early_stopping_rounds=200, n_jobs=c.NT, random_state=s)
                mx.fit(XV[a][o2], y[a][o2], qid=tier[a][o2], eval_set=[(XV[b], y[b])], eval_qid=[np.zeros(len(b), int)], verbose=False)
                zx[s, b] = mx.predict(XV[b])
                its.append(mx.best_iteration)
            log(f"fold {i}: queries per seed about {len(sizes)} (largest {sizes.max()}); best iterations (lambdarank, pairwise per seed) {its}; AUC "
                f"lambdarank {roc_auc_score(y[b], zl[:, b].mean(0)):.5f}, rank:pairwise {roc_auc_score(y[b], zx[:, b].mean(0)):.5f}, the pipeline's "
                f"LightGBM member {roc_auc_score(y[b], L[b, 0]):.5f}")
        members["RL lambdarank LightGBM (tier x bucket queries)"], members["RX rank:pairwise XGBoost (tier queries)"] = zl.mean(0), zx.mean(0)
        np.savez(f_r, rl=zl.mean(0), rx=zx.mean(0))

if "leaf" in what:
    f_l = OUT / "t1_leaf_member.npz"
    if f_l.exists():
        members["LE leaf-embedding member (SVD 16 + cluster rate)"] = np.load(f_l)["z"]
    else:
        z = np.zeros(len(y))
        for i, (a, b) in enumerate(c.folds):
            halves = list(StratifiedKFold(2, shuffle=True, random_state=0).split(a, y[a]))
            parts = []
            for fit_i, emb_i in halves:
                A, B = a[fit_i], a[emb_i]
                g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=16, min_child_samples=100, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.5, n_jobs=c.NT, verbose=-1, random_state=0).fit(XV[A], y[A])
                off = np.arange(300) * 16

                def onehot(rows):
                    lv = g.predict(XV[rows], pred_leaf=True)
                    r = np.repeat(np.arange(len(rows)), lv.shape[1])
                    return sp.csr_matrix((np.ones(lv.size, np.float32), (r, (lv + off).ravel())), shape=(len(rows), 300 * 16))
                svd = TruncatedSVD(16, random_state=0)
                EB, Eb = svd.fit_transform(onehot(B)), svd.transform(onehot(b))
                km = KMeans(64, n_init=3, random_state=0).fit(EB)
                cb, cv = km.labels_, km.predict(Eb)
                prior = y[B].mean()

                def rate(c_fit, y_fit, c_app):
                    s_ = pd.DataFrame({"c": c_fit, "y": y_fit}).groupby("c").y.agg(["sum", "count"])
                    return pd.Series(c_app).map((s_["sum"] + 50 * prior) / (s_["count"] + 50)).fillna(prior).values
                rB = np.zeros(len(B))
                for u, v in StratifiedKFold(5, shuffle=True, random_state=0).split(B, y[B]):
                    rB[v] = rate(cb[u], y[B][u], cb[v])
                head = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1,
                                          n_jobs=c.NT, verbose=-1, random_state=0).fit(np.column_stack([EB, rB]), y[B])
                parts.append(logit(head.predict_proba(np.column_stack([Eb, rate(cb, y[B], cv)]))[:, 1]))
            z[b] = np.mean(parts, 0)
            log(f"fold {i}: leaf-embedding member AUC {roc_auc_score(y[b], z[b]):.5f} (the pipeline's LightGBM member {roc_auc_score(y[b], L[b, 0]):.5f}); "
                f"SVD explained variance {svd.explained_variance_ratio_.sum():.3f}")
        members["LE leaf-embedding member (SVD 16 + cluster rate)"] = z
        np.savez(f_l, z=z)

for n, z in members.items():
    log(f"{n.split(' ')[0]} alone (out of fold): AUC {roc_auc_score(y, z):.5f}, F2 at its best cut {cut_f2(z):.5f}; correlation with the pipeline's "
        f"LightGBM member {np.corrcoef(z, L[:, 0])[0, 1]:.4f} (Spearman {pd.Series(z).corr(pd.Series(L[:, 0]), method='spearman'):.4f})")
base_p = c.base_probs()
base_labels = [labels_from(p, fold_of(fl)) for p, (_, fl) in zip(base_p, SPLITS)]
cands = [(n + " as a tenth member", [z]) for n, z in members.items()]
if "rank" in what:
    cands.append(("RL + RX as tenth and eleventh members", [v for k, v in members.items() if k.startswith("R")]))
for name, zs in cands:
    SX = np.column_stack([L, *zs])
    cand = []
    for k, (label, fl) in enumerate(SPLITS):
        cand.append(labels_from(c.level2(fl, S_lr=SX, S_tr=SX), fold_of(fl)))
        log(f"{label}: {name.split(' ')[0]} {f2(y, cand[-1]):.5f} (base {f2(y, base_labels[k]):.5f})")
    c.report(name, cand, base_labels)
log("done")
