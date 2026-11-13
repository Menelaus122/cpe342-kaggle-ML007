"""Round 14, steps 1 and 2: diagnostic of the uploaded v12 file (Tasks 1, 2, 5; Task 3 is in bx_r14_t3_post.py) and the Task 5 cut-off audit.

Reads cached predictions and existing files, and fits only the unsupervised Task 5 detectors the pipeline already fits on the given test.csv.
Writes no submission. The id block is used as a reference only, never as a training label (that would be pseudo-labelling on test rows).
"""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import _logit, T2_MODELS, stack_multi  # noqa: E402
from src import task1, task2, task5  # noqa: E402
from src.metrics import fbeta, macro_f1, optimize_threshold, optimize_class_weights, apply_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
v12 = pd.read_csv("submission_v12_try.csv")
clean = pd.read_csv("submission_clean_try.csv")
N = len(v12)
print(f"v12 rows {N}\n")

# ========================================================================== Task 1
print("=== Task 1 (positive-class F2) ===")
tr1, te1, nu = task1.load()
reduce_mem_usage(tr1, verbose=False)
X1, y1, Xt1 = task1.build(tr1, te1)
folds1 = make_folds(y1)
keys1 = {"LightGBM": "t1_lgb", "XGBoost": "t1_xgb", "CatBoost": "t1_cat", "ExtraTrees": "t1_et", "LogReg": "t1_lr",
         "RF-24": "t1_rf24", "LightGBM-24": "t1_lgb24", "XGBoost-24": "t1_xgb24", "CatBoost-24": "t1_cat24"}
seeds1 = {"LightGBM": 1, "XGBoost": 1, "CatBoost": 1, "ExtraTrees": 1, "LogReg": "deterministic", "RF-24": 2, "LightGBM-24": 2, "XGBoost-24": 2, "CatBoost-24": 2}
S = np.column_stack([_logit(load_preds(k)["oof"]) for k in keys1.values()])
oof1 = np.zeros(len(y1))
for a, b in folds1:
    oof1[b] = LogisticRegression(C=1.0, max_iter=2000).fit(S[a], y1[a]).predict_proba(S[b])[:, 1]
thr1, s1 = optimize_threshold(y1, oof1, 2.0, average="binary")
nest = np.zeros(len(y1), int)
for a, b in folds1:
    nest[b] = oof1[b] >= optimize_threshold(y1[a], oof1[a], 2.0, average="binary")[0]
coef = LogisticRegression(C=1.0, max_iter=2000).fit(S, y1).coef_.ravel()
print(f"labelled rows {len(y1):,} (unlabelled dropped {nu:,}); prevalence {y1.mean():.4%}")
print(f"OOF F2 at the OOF-optimal cut-off {s1:.5f}; nested (cut-off from the other folds) {fbeta(y1, nest, 2, 'binary'):.5f}")
print(f"cut-off tau = {thr1:.5f}")
print("members, seeds, stacker log-odds weight:")
print(pd.DataFrame({"seeds": seeds1, "stack weight": dict(zip(keys1, coef.round(3)))}).to_string())
print(f"v12 test 1s: {int(v12.task1.sum()):,} = {v12.task1.mean():.3%}; identical to the clean pipeline: {bool((v12.task1 == clean.task1).all())}")

# ========================================================================== Task 2
print("\n=== Task 2 (macro-F1) ===")
tr2, te2 = task2.load()
reduce_mem_usage(tr2, verbose=False)
X2, y2, Xt2 = task2.build(tr2, te2)
folds2 = make_folds(y2)
CUTS = [52000, 84500, 110500]
idn2 = tr2.player_id.str[1:].astype(int).values
blk2 = np.searchsorted(CUTS, idn2, side="right")
pf = [macro_f1(y2[b], blk2[b]) for _, b in folds2]
print(f"how player_id is used: segment = id block, cuts {CUTS}; features are not used at all in the v12 column")
print(f"train: id block = label on {(blk2 == y2).mean():.4%} of {len(y2):,} rows; macro-F1 {macro_f1(y2, blk2):.5f}; per fold {np.round(pf, 4).tolist()}")
m2 = [load_preds(n) for n in T2_MODELS]
so2, st2 = stack_multi([d["oof"] for d in m2], [d["test"] for d in m2], y2, folds2)
w2, s2 = optimize_class_weights(y2, so2)
q = np.zeros(len(y2), int)
for a, b in folds2:
    q[b] = apply_class_weights(so2[b], optimize_class_weights(y2[a], so2[a])[0])
print(f"clean pipeline (no player_id): OOF macro-F1 {s2:.5f}, nested multipliers {macro_f1(y2, q):.5f}; members {T2_MODELS}, one seed each")
print(f"v12 test class counts (0, 1, 2, 3): {np.bincount(v12.task2, minlength=4).tolist()}, shares {np.round(np.bincount(v12.task2, minlength=4) / N, 4).tolist()}")
print(f"clean test class counts: {np.bincount(clean.task2, minlength=4).tolist()}; rows where v12 differs from clean: {int((v12.task2 != clean.task2).sum()):,}")

# ========================================================================== Task 5
print("\n=== Task 5 (F3, no labels) ===")
t5 = task5.load()
reduce_mem_usage(t5, verbose=False)
X5, _ = task5.build_features(t5)
idn = t5.player_id.str[1:].astype(int).values
START = 23819
blk = idn >= START
print(f"ids: min {idn.min()}, max {idn.max()}, unique {len(np.unique(idn))}, contiguous {len(np.unique(idn)) == idn.max() - idn.min() + 1}")
print("mechanism in v12: player_id only, flag = id >= 23,819. Detectors are not combined with it; the id cut is the lowest id of the six gap families.")
print(f"v12 flags {int(v12.task5.sum()):,} = {v12.task5.mean():.3%}; identical to the block: {bool((v12.task5.values == blk).all())}")
_, gc = task5.gap_clusters(t5, X5)
gap = gc.any(axis=1).values
cl = clean.task5.values == 1
print(f"six gap families {gap.sum():,}: outside the block {int((gap & ~blk).sum())}; lowest family id {idn[gap].min()}; 20 lowest family ids {np.sort(idn[gap])[:20].tolist()}")
pocket = cl & ~gap
print(f"clean pipeline flags {cl.sum():,} = families {gap.sum():,} + pocket {pocket.sum()}: pocket in block {int((pocket & blk).sum())}, outside block {int((pocket & ~blk).sum())}")
unseen = blk & ~gap & ~pocket
print(f"block accounts no pipeline detector flags: {unseen.sum()}")
scan, noise = task5.dependence_pockets(X5, ~gap)
sig = scan[scan.z > noise]
feats = sorted(set(sig.a) | set(sig.b))
_, pinfo = task5.pocket_detector(X5, gap, feats)
print(f"pocket features {feats}; cut-off k {pinfo['k']}; estimated members {pinfo['est_tp']:.0f}; estimated false positives {pinfo['k'] - pinfo['est_tp']:.0f}; precision {pinfo['precision']:.3f}")
pscore = X5[feats].rank(pct=True).min(axis=1).values

device = "cuda" if torch.cuda.is_available() else "cpu"
cols = task5.detector_columns(X5)
Z = task5.detector_matrix(X5, cols, "robust")
det = {}
for name, fn in task5.DETECTORS.items():
    det[name] = task5.robust_z(fn(Z, device=device) if name == "AutoEncoder" else fn(Z))
cons = np.mean([v for k, v in det.items() if k != "LOF"], 0)
crank = pd.Series(cons).rank(pct=True).values
rules = task5.evidence_rules(t5).any(axis=1).values
ext = {"impossible travel: km per login, period 4": X5.geo_km_per_login_4.values, "impossible travel: largest hop km": X5.geo_hop_max.values,
       "device switching: device_count_maxjump": X5.device_count_maxjump.values, "device switching: device_count_dev4": X5.device_count_dev4.values,
       "login bursts: login_count_burst": X5.login_count_burst.values, "login bursts: login_count_maxjump": X5.login_count_maxjump.values}

print("\n--- Step 2.1: 500 rows either side of the v12 cut-off (ids), with a far-away control band ---")
bands = {"500 below (23,319-23,818)": (idn >= START - 500) & (idn < START), "500 above (23,819-24,318)": (idn >= START) & (idn < START + 500),
         "control 20,000-20,499": (idn >= 20000) & (idn < 20500), "rest of block (24,319+)": idn >= START + 500}
top8 = cons >= np.quantile(cons, 1 - blk.mean())
rows = []
for nm, m in bands.items():
    rows.append([nm, int(m.sum()), int(gap[m].sum()), int(pocket[m].sum()), rules[m].mean(), np.median(cons[m]), np.mean(crank[m]), top8[m].mean(), (crank[m] >= 0.99).mean(), (pscore[m] >= 0.95).mean()])
print(pd.DataFrame(rows, columns=["band", "rows", "families", "pocket", "evidence-rule rate", "median consensus", "mean consensus rank", "share top 8% consensus", "share top 1% consensus", "share pocket score>=0.95"]).round(4).to_string(index=False))

print("\nfine profile around the cut-off (bins of 100 ids; family and pocket counts, mean consensus rank, share pocket score >= 0.95):")
rows = []
for lo in range(23200, 24400, 100):
    m = (idn >= lo) & (idn < lo + 100)
    rows.append([f"{lo}-{lo + 99}", int(gap[m].sum()), int(pocket[m].sum()), crank[m].mean(), (pscore[m] >= 0.95).mean()])
print(pd.DataFrame(rows, columns=["ids", "families", "pocket", "mean consensus rank", "share pocket>=0.95"]).round(3).to_string(index=False))


def changepoint(v, ids, lo=15000, hi=25500):
    """Single-change-point scan of a score ordered by id: best split by the two-sample t statistic."""
    o = np.argsort(ids)
    x = v[o]
    n = len(x)
    c = np.cumsum(x)
    i0, i1 = int(np.searchsorted(ids[o], lo)), int(np.searchsorted(ids[o], hi))
    best, arg = -1, None
    sd = x.std() + 1e-12
    for i in range(i0, i1):
        m1, m2 = c[i - 1] / i, (c[-1] - c[i - 1]) / (n - i)
        t = abs(m2 - m1) / (sd * np.sqrt(1 / i + 1 / (n - i)))
        if t > best:
            best, arg = t, ids[o][i]
    return int(arg), float(best)


print("\nchange-point of each score along player_id (a boundary lower than 23,819 would show here):")
for nm, v in [("consensus rank", crank), ("family flag", gap.astype(float)), ("pocket flag", pocket.astype(float)), ("pocket score", pscore), ("evidence rules", rules.astype(float))]:
    cp, tstat = changepoint(v, idn)
    print(f"  {nm:16s} change at id {cp:>6}, |t| {tstat:.1f}")

print("\n--- Step 2.2: are confident anomalies suppressed by the id rule, or the reverse? ---")
print(f"consensus rank of the first outside-block account: {int(np.sum(cons > cons[~blk].max())) + 1} (accounts above it are all in the block)")
rows = []
for q in (0.999, 0.995, 0.99):
    r = [f"top {1 - q:.1%}"]
    for nm, v in ext.items():
        t = np.nanquantile(v, q)
        m = v >= t
        r += [f"{int((m & ~blk).sum())}/{int(m.sum())}"]
    rows.append(r)
print("rows above each extreme-feature percentile: outside block / total")
print(pd.DataFrame(rows, columns=["percentile"] + list(ext)).to_string(index=False))
in_b, out_b = cons[blk], cons[~blk]
print(f"consensus: outside block max {out_b.max():.2f}, p99.9 {np.quantile(out_b, .999):.2f}; in block median {np.median(in_b):.2f}, p10 {np.quantile(in_b, .1):.2f}")
print(f"unseen block accounts ({unseen.sum()}): mean consensus rank {crank[unseen].mean():.3f}; pocket-score >= 0.95 share {(pscore[unseen] >= 0.95).mean():.3f} (all rows: {(pscore >= 0.95).mean():.3f})")
print(f"outside-block accounts flagged by an evidence rule: {int((rules & ~blk).sum()):,} of {int((~blk).sum()):,} = {rules[~blk].mean():.3%}; inside block {rules[blk].mean():.3%}")

print("\n--- Step 2.3: F3 curve around the cut-off ---")
F3 = lambda tp, fn, fp: 10 * tp / (10 * tp + 9 * fn + fp) if tp else 0.0
B = int(blk.sum())
print("decision rule: flag a row when P(anomaly) >= F*/10; at F* near 1 the bar is about 0.10 (Lipton et al., 2014)")
rows = []
for d in (0, 50, 100, 250, 500, 1000, 2000):
    r = [d]
    for e in (0, 50, 100, 250, 500, 1000, 2000):   # truth starts e ids below the cut; v12 extended d ids below the cut
        tp = B + min(d, e)
        r.append(F3(tp, B + e - tp, max(d - e, 0)))
    rows.append(r)
print("F3: rows = accounts added directly below the cut; columns = anomalies that really exist directly below the cut (true base rate rises with the column)")
print(pd.DataFrame(rows, columns=["added below cut"] + [f"true extra {e} ({(B + e) / N:.1%})" for e in (0, 50, 100, 250, 500, 1000, 2000)]).round(4).to_string(index=False))
print("\nmoving the cut up (dropping rows from the block), truth = block: ", {d: round(F3(B - d, d, 0), 4) for d in (10, 50, 100, 250)})
print("\ncandidate additions and the hit rate they would need (F* = 1 -> 0.10):")
for nm, m, est in [("145-row pocket flags outside the block", pocket & ~blk, None), ("500 ids directly below", (idn >= START - 500) & (idn < START), None)]:
    k = int(m.sum())
    d0 = 1 - F3(B, 0, k)
    print(f"  {nm}: {k} rows; if none are anomalies F3 falls by {d0:.4f} (LB -{d0 / 5:.5f} at equal weights); break-even true anomalies {max(1, int(np.ceil(k * 0.1)))}")
print(f"\nchance level of the pocket outside the block: permutation-null false positives over the whole pocket cut-off {pinfo['k'] - pinfo['est_tp']:.0f}; pocket flags outside the block {int((pocket & ~blk).sum())}")

LB = 0.76248
T1, T2, T3 = fbeta(y1, nest, 2, "binary"), macro_f1(y2, blk2), 0.79434
print("\nleaderboard arithmetic (equal weights, v12's score only, OOF scores for Tasks 1-3):")
for t4 in (0.19, 0.21, 0.23):
    print(f"  if Task 4 = {t4:.2f}: Task 5 = {5 * LB - T1 - T2 - T3 - t4:.4f}")
