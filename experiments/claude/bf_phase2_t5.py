"""Phase 2 (2026-11-10), Task 5: the brief's consensus-rank ensemble vs our pipeline (6 gap families + dependence pocket).

The pipeline already contains the brief's detectors (src/task5.py: IsolationForest, ECOD, COPOD, LOF, Mahalanobis-MCD,
autoencoder, extreme-value; RobustScaler(10-90) scaling; rank fusion; knee cut-off) as its cross-check "score ensemble".
Here: four consensus variants (robust vs quantile scaling, with / without a PCA subspace keeping 90% of the variance),
cut at the knee and at the pipeline's flag count, scored three ways:
  * agreement with the gap families (label-free),
  * tail stability: Jaccard overlap of the flagged sets for two random seeds (label-free),
  * F3 against the id block (REFERENCE ONLY, as in earlier rounds; never used to choose anything).
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task5  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
t5 = task5.load()
X5, _ = task5.build_features(t5)
cols = task5.detector_columns(X5)
pipe = pd.read_csv("submission.csv").task5.values.astype(bool)
block = (t5.player_id.str[1:].astype(int) >= 23819).values
fam = task5.gap_clusters(t5, X5)
fam = (fam[1] if isinstance(fam, tuple) else fam)
gap = np.asarray(fam.any(axis=1)) if hasattr(fam, "any") else np.asarray(fam)


def f3(pred, truth):
    tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
    return 10 * tp / (10 * tp + 9 * fn + fp)


def consensus(Z, seed):
    S = []
    for name, fn in task5.DETECTORS.items():
        if name == "LOF":                 # the pipeline drops LOF from the fusion (leave-one-out disagreement)
            continue
        if name == "IsolationForest":
            s = task5.score_iforest(Z, seed=seed)
        elif name == "AutoEncoder":
            s = task5.score_autoencoder(Z, seed=seed, device="cuda")
        elif name == "Mahalanobis(MCD)":
            s = task5.score_mahalanobis(Z, seed=seed)
        else:
            s = fn(Z)
        S.append(task5.rank_normalize(s))
    return np.mean(S, axis=0)


def cut(score, k):
    o = np.argsort(-score)
    m = np.zeros(len(score), bool)
    m[o[:k]] = True
    return m


rows = [["pipeline: 6 gap families + dependence pocket", int(pipe.sum()), np.mean(pipe[gap]) if gap.any() else np.nan, "-", f3(pipe, block)]]
for scale in ("robust", "quantile"):
    Z = task5.detector_matrix(X5, cols, scale)
    for sub in ("all columns", "PCA 90%"):
        Zs = Z if sub == "all columns" else PCA(n_components=0.90, random_state=0).fit_transform(Z)
        s1, s2 = consensus(Zs, 42), consensus(Zs, 7)
        srt = np.sort(s1)[::-1]
        knee = task5.knee_point(np.log1p(np.clip(srt - np.median(srt), 0, None)))
        for label, k in (("knee", knee), ("same count as pipeline", int(pipe.sum()))):
            m1, m2 = cut(s1, k), cut(s2, k)
            jac = (m1 & m2).sum() / (m1 | m2).sum()
            rows.append([f"consensus, {scale} scaling, {sub}, cut at {label}", int(k), m1[gap].mean(), f"{jac:.3f}", f3(m1, block)])
            log(rows[-1])
R = pd.DataFrame(rows, columns=["Detector", "flags", "share of gap families caught", "seed stability (Jaccard)", "F3 vs id block (reference only)"])
pd.set_option("display.width", 220)
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
