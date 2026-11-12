"""Round 18: Task 4 recipes compared under a content shift, using TRAINING images only.

    python ch_r18_t4_shift.py <recipe> [<recipe> ...]      (recipes: see RECIPES; already finished ones are skipped)
    python ch_r18_t4_shift.py report

Why: v14 (first upload with the corrected Task 4 order) scored 0.84873. With Tasks 1-3 at their known levels, Tasks 4 + 5 add up to
about 1.84, and round 11's label-free estimates put Task 4 at 0.87-0.91 on test against 0.99 out-of-fold: the test images show content
the training folds do not (crowds, dark or nearly blank frames). Stratified folds cannot see that, so they cannot rank recipes for it.

Validation used here: k-means (20 clusters per class, seed 0) on frozen DINOv2-B features of the training images, then
StratifiedGroupKFold(5) by cluster, so a validation fold holds whole clusters the model never saw. A frozen DINOv2 probe drops from
0.911 (stratified) to 0.829 accuracy on it, close to its estimated test accuracy (0.76-0.82), so it imitates the test shift.
Every recipe trains on the same fold-0 training rows (paired comparison) and predicts the held-out clusters with h-flip TTA.
No test image is used anywhere. Writes outputs/r18_t4/<recipe>.npz (val_idx, val_probs) and groups.npy.
"""
import sys
import time
from pathlib import Path

import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402

OUT = Path("experiments/claude/outputs/r18_t4")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

RECIPES = {   # name -> (timm model, HxW, extra train_fold kwargs)
    "cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320), {}),                   # current pipeline member
    "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384), {}),     # current pipeline member
    "cnxt_bal": ("convnext_tiny.in12k_ft_in1k", (160, 320), {"class_balanced": True}),
    "eva02_s": ("eva02_small_patch14_336.mim_in22k_ft_in1k", (168, 336), {}),
    "effv2_s": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320), {}),
    "cnxt22k": ("convnext_tiny.fb_in22k", (160, 320), {}),
}


def groups_and_fold(y):
    gp = OUT / "groups.npy"
    if gp.exists():
        g = np.load(gp)
    else:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        F = StandardScaler().fit_transform(np.load("outputs/preds/t4_feat_basep14_reg4_dinov2.npz")["train"].astype(np.float32))
        g = np.zeros(len(y), int)
        for c in range(5):
            i = np.where(y == c)[0]
            g[i] = c * 1000 + KMeans(20, n_init=2, random_state=0).fit_predict(F[i])
        np.save(gp, g)
    from sklearn.model_selection import StratifiedGroupKFold
    tr_idx, va_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g))
    return g, tr_idx, va_idx


def report():
    from sklearn.metrics import f1_score
    train, _ = task4.load_frames()
    y = train.label.values
    rows, P = [], {}
    for name in RECIPES:
        f = OUT / f"{name}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        va, p = z["val_idx"], z["val_probs"]
        P[name] = p
        rows.append((name, (p.argmax(1) == y[va]).mean(), f1_score(y[va], p.argmax(1), average="macro"), float(z["secs"])))
    print(f"held-out clusters: {len(va)} images")
    for n, a, f1, s in rows:
        print(f"  {n:10s} acc {a:.4f}  macro-F1 {f1:.4f}  ({s / 60:.0f} min)")
    names = list(P)
    if "cnxt" in P and "deit3" in P:
        e = (P["cnxt"] + P["deit3"]) / 2
        print(f"  current ensemble cnxt+deit3: acc {(e.argmax(1) == y[va]).mean():.4f}  macro-F1 {f1_score(y[va], e.argmax(1), average='macro'):.4f}")
    if len(names) > 2:
        e = np.mean([P[n] for n in names], 0)
        print(f"  all {len(names)} recipes averaged: acc {(e.argmax(1) == y[va]).mean():.4f}  macro-F1 {f1_score(y[va], e.argmax(1), average='macro'):.4f}")
    # paired bootstrap of each recipe vs cnxt (accuracy)
    if "cnxt" in P:
        rng = np.random.default_rng(0)
        base = P["cnxt"].argmax(1) == y[va]
        for n in names:
            if n == "cnxt":
                continue
            d = (P[n].argmax(1) == y[va]).astype(float) - base
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f"  {n:10s} vs cnxt: acc delta {d.mean():+.4f}  95% CI [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")


def main(names):
    train, _ = task4.load_frames()
    y = train.label.values
    g, tr_idx, va_idx = groups_and_fold(y)
    log(f"fold 0: train {len(tr_idx)} rows, held-out {len(va_idx)} rows in {len(np.unique(g[va_idx]))} clusters")
    arr = None
    for name in names:
        out = OUT / f"{name}.npz"
        if out.exists():
            log(f"{name}: done before, skipped")
            continue
        if arr is None:
            arr = task4.load_uint8(list(train.path))
        model_name, size, kw = RECIPES[name]
        t0 = time.time()
        log(f"{name}: {model_name} {size} {kw or ''}")
        model, hist = task4.train_fold(model_name, arr, y, tr_idx, va_idx, img_size=size, epochs=10, bs=64, lr=2e-4,
                                       seed=42, log=log, **kw)
        vp = task4.predict_array(model, arr[va_idx], size, tta=True)
        np.savez(out, val_idx=va_idx, val_probs=vp, secs=time.time() - t0)
        log(f"{name}: held-out acc {(vp.argmax(1) == y[va_idx]).mean():.4f}")
        del model


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        main(sys.argv[1:])
