"""Round 11, Task 4: how accurate are we on TEST-LIKE training images? (covariate-shift importance weighting, Sugiyama 2007)

Last year's 1st-place notebook (public repo) prints its first five test predictions; ours disagree on 2 of 5, unanimously across our 12
models, on dark / noisy images. That questions the label-free estimates (~0.90). Here, with TRAIN labels only:
* a train-vs-test classifier on frozen image embeddings (DINOv2-B and CLIP-B, extracted in bh_*) gives p(test | image);
* each training image gets weight p / (1 - p) (clipped at the 99.5th percentile): the weighted OOF accuracy estimates accuracy on the
  test distribution; accuracy is also shown by decile of test-likeness and by brightness / noise level.
Test images are only used, unlabeled, to fit the train-vs-test classifier (an evaluation tool, as in the Task 1 test-like check).
"""
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401
from src import config as C
from src import task4

train, test = task4.load_frames()
y = train.label.values
oof = {}
for tag in ("cnxt", "deit3"):
    o = np.zeros((len(train), 5))
    for f in range(5):
        z = np.load(C.PRED_DIR / f"t4_{tag}_fold{f}.npz")
        o[z["val_idx"]] = z["val_probs"]
    oof[tag] = o
ens = (oof["cnxt"] + oof["deit3"]) / 2
pred = ens.argmax(1)
print(f"plain OOF: accuracy {np.mean(pred == y):.4f}, macro-F1 {f1_score(y, pred, average='macro'):.4f}")

for tags in (["basep14_reg4_dinov2"], ["basep16_clip_224"], ["basep14_reg4_dinov2", "basep16_clip_224"]):
    Ftr = np.hstack([np.load(C.PRED_DIR / f"t4_feat_{t}.npz")["train"].astype(np.float32) for t in tags])
    Fte = np.hstack([np.load(C.PRED_DIR / f"t4_feat_{t}.npz")["test"].astype(np.float32) for t in tags])
    A = np.vstack([Ftr, Fte]); A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    lab = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
    p = np.zeros(len(lab))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(A, lab):
        p[b] = LogisticRegression(C=0.05, max_iter=2000).fit(A[a], lab[a]).predict_proba(A[b])[:, 1]
    del A
    pt = p[:len(Ftr)]
    w = pt / (1 - pt)
    w = np.minimum(w, np.quantile(w, 0.995))
    ess = w.sum() ** 2 / (w ** 2).sum()
    acc_w = np.sum(w * (pred == y)) / w.sum()
    f1_w = f1_score(y, pred, average="macro", sample_weight=w)
    print(f"\n[{' + '.join(tags)}] train-vs-test AUC {roc_auc_score(lab, p):.3f} | test-like OOF accuracy {acc_w:.4f}, "
          f"macro-F1 {f1_w:.4f} (effective n {ess:,.0f})")
    dec = pd.qcut(pt, 10, labels=False, duplicates="drop")
    print("  accuracy by decile of test-likeness (1 = least, 10 = most):",
          " ".join(f"{np.mean(pred[dec == k] == y[dec == k]):.3f}" for k in range(dec.max() + 1)))
    wc = pd.Series(w).groupby(y).sum() / w.sum()
    print("  test-like class mix (7-11k / fiveN / freefried / pube / roblock):", np.round(wc.values, 3))


def stats(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    hp = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return a.mean(), np.median(np.abs(hp))


with ThreadPoolExecutor(12) as ex:
    s_tr = np.array(list(ex.map(stats, train.path.values)))
    s_te = np.array(list(ex.map(stats, test.path.values)))
for j, nm in enumerate(("brightness", "noise (median |Laplacian|)")):
    edges = np.quantile(np.r_[s_tr[:, j], s_te[:, j]], np.linspace(0, 1, 6))
    btr, bte = np.digitize(s_tr[:, j], edges[1:-1]), np.digitize(s_te[:, j], edges[1:-1])
    print(f"\n{nm}: quintile | share of train | share of test | OOF accuracy in train quintile")
    for k in range(5):
        print(f"  {k + 1}  {np.mean(btr == k):.3f}  {np.mean(bte == k):.3f}  {np.mean(pred[btr == k] == y[btr == k]):.4f}")
