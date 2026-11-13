"""Task 4, round 11: linear probes on frozen foundation features + label-free estimates of TEST accuracy.

Folds = the folds of the cached ConvNeXt/DeiT runs (their val_idx), so every OOF number is comparable.
Test accuracy cannot be measured (no labels, and we must not obtain them), so three published label-free estimators are used.
They only read model predictions on the unlabeled test images; nothing is fitted on test:
* ATC  (Garg et al., ICLR 2022): threshold on confidence chosen so the OOF share above it equals OOF accuracy;
       estimated test accuracy = share of test images above it.
* DoC  (Guillory et al., ICCV 2021): OOF accuracy minus the drop in mean confidence.
* AGL  (Baek et al., NeurIPS 2022, "agreement-on-the-line"): pairwise agreement between models on OOF vs on test lies on a
       line in probit space; the same line maps each model's OOF accuracy to its test accuracy.
"""
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common  # noqa: F401
from src import config as C
from src import task4

train, test = task4.load_frames()
y = train.label.values
folds = [np.load(C.PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]


def cached(tag):
    oof, te = np.zeros((len(train), 5)), []
    for f in range(5):
        z = np.load(C.PRED_DIR / f"t4_{tag}_fold{f}.npz")
        oof[z["val_idx"]] = z["val_probs"]
        te.append(z["test_probs"])
    return oof, np.mean(te, 0)


def probe(Xtr, Xte, Cval):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    oof, te = np.zeros((len(Xtr), 5)), np.zeros((len(Xte), 5))
    for va in folds:
        tr = np.setdiff1d(np.arange(len(Xtr)), va)
        m = LogisticRegression(C=Cval, max_iter=2000).fit(Xtr[tr], y[tr])
        oof[va] = m.predict_proba(Xtr[va])
        te += m.predict_proba(Xte) / 5
    return oof, te


def feats(tag):
    z = np.load(C.PRED_DIR / f"t4_feat_{tag}.npz")
    return z["train"].astype(np.float32), z["test"].astype(np.float32)


M = {"convnext (fine-tuned)": cached("cnxt"), "deit3 (fine-tuned)": cached("deit3")}
F = {}
for tag in ("basep16_clip_224", "basep14_reg4_dinov2", "basep16_siglip_256_256"):
    try:
        F[tag] = feats(tag)
    except FileNotFoundError:
        print("missing features", tag)
# C per backbone = best of the sweep 0.01 / 0.1 / 1.0 (OOF accuracy; first run, outputs/bi_t4_probe_eval.log):
# CLIP 0.8663 / 0.8683 / 0.8614, DINOv2 0.9153 / 0.9099 / 0.9009, SigLIP 0.8751 / 0.8774 / (not reached)
BEST_C = {"basep16_clip_224": 0.1, "basep14_reg4_dinov2": 0.01, "basep16_siglip_256_256": 0.1}
for tag, (a, b) in F.items():
    oof, te = probe(a, b, BEST_C[tag])
    print(f"probe {tag} C={BEST_C[tag]}: OOF acc {(oof.argmax(1) == y).mean():.4f}", flush=True)
    M[f"probe {tag}"] = (oof, te)
if len(F) > 1:
    a = np.hstack([v[0] for v in F.values()])
    b = np.hstack([v[1] for v in F.values()])
    del F
    M["probe all features concatenated"] = probe(a, b, 0.03)
    del a, b
    print(f"probe concatenated C=0.03: OOF acc {(M['probe all features concatenated'][0].argmax(1) == y).mean():.4f}", flush=True)

names = list(M)
acc_id = {n: (M[n][0].argmax(1) == y).mean() for n in names}


def atc(oof, te, acc):
    so, st = oof.max(1), te.max(1)
    return (st >= np.quantile(so, 1 - acc)).mean()


# agreement-on-the-line: fit probit(test agreement) = a * probit(OOF agreement) + b over all model pairs
pa, pb = [], []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pa.append(np.mean(M[names[i]][0].argmax(1) == M[names[j]][0].argmax(1)))
        pb.append(np.mean(M[names[i]][1].argmax(1) == M[names[j]][1].argmax(1)))
pr = lambda p: norm.ppf(np.clip(p, 1e-4, 1 - 1e-4))
slope, icpt = np.polyfit(pr(np.array(pa)), pr(np.array(pb)), 1)
r = np.corrcoef(pr(np.array(pa)), pr(np.array(pb)))[0, 1]
print(f"\nagreement-on-the-line fit over {len(pa)} pairs: slope {slope:.2f}, intercept {icpt:.2f}, r {r:.3f}")

ref = (M[names[0]][1] + M[names[1]][1]).argmax(1)
print(f"\n{'model':48s} {'OOF acc':>8s} {'OOF F1':>7s} {'ATC':>6s} {'DoC':>6s} {'AGL':>6s} {'agree w/ ens (test)':>20s}  test class shares")
for n in names:
    oof, te = M[n]
    a = acc_id[n]
    doc = a - (oof.max(1).mean() - te.max(1).mean())
    agl = norm.cdf(slope * pr(a) + icpt)
    print(f"{n:48s} {a:8.4f} {f1_score(y, oof.argmax(1), average='macro'):7.4f} {atc(oof, te, a):6.3f} {doc:6.3f} {agl:6.3f} "
          f"{np.mean(te.argmax(1) == ref):20.3f}  {np.round(np.bincount(te.argmax(1), minlength=5) / len(te), 3)}")
np.savez(C.PRED_DIR / "t4_round11_probes.npz", **{f"oof_{i}": M[n][0] for i, n in enumerate(names)},
         **{f"test_{i}": M[n][1] for i, n in enumerate(names)}, names=np.array(names))
