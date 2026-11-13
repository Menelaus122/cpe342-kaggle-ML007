"""Round 25, Task 4: (A) the prior exponent under simulated label shift, (B) a stacking head instead of the plain mean.
Training images only (out-of-fold probabilities of the seven v16 members on four views, outputs/r19_views).

    python df_r25_t4_prior_stack.py

A. Rule argmax(p * pi^-alpha). Evaluation sets drawn from the OOF rows (200 draws each, without replacement):
     train mix       : a 60% subsample, class shares as in training (39% 7-Eleven Knights)
     7-Eleven -> 20% : the largest class subsampled until it is 20% of the set, other classes untouched
     balanced        : every class subsampled to the size of the smallest
   Reported: mean macro-F1 per alpha and view, and how often each alpha beats alpha = 0 in the same draw.
B. Stacking head: multinomial logistic regression (L2) on the members' log-probabilities, optionally + entropies and image
   statistics (brightness, contrast, border-vs-centre contrast, bottom-strip contrast), trained on the other four folds and applied
   to the held-out fold on every view. `full+zoom` trains on the full frames and the 70% zoom of the same rows. Compared with the
   plain mean (v16) on the objective (mean of full and zoom70) and on the two views the head never sees (centre_grey, bottom_grey).
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEWS = ["full", "zoom70", "centre_grey", "bottom_grey"]
OBJ = ["full", "zoom70"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
P = {(t, v): np.zeros((len(y), 5)) for t in TAGS for v in VIEWS}
for t in TAGS:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t, v][va] = z[v]
avg = {v: np.mean([P[t, v] for t in TAGS], 0) for v in VIEWS}
pi = np.bincount(y, minlength=5) / len(y)
big = int(pi.argmax())


def mf1(yt, yp):
    cm = np.bincount(yt * 5 + yp, minlength=25).reshape(5, 5)
    return float(np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)))


print("training prior:", dict(zip(T4_CLASS_NAMES, np.round(pi, 3))), "| largest class:", T4_CLASS_NAMES[big])
print("\n== A. prior exponent under simulated label shift (200 draws; mean macro-F1, share of draws better than alpha 0)")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
pred = {(a, v): (avg[v] * pi ** -a).argmax(1) for a in ALPHAS for v in VIEWS}
rng = np.random.default_rng(0)
idx_c = [np.where(y == c)[0] for c in range(5)]
n_other = len(y) - len(idx_c[big])


def draw(kind):
    if kind == "train mix":
        return np.concatenate([rng.choice(i, int(0.6 * len(i)), replace=False) for i in idx_c])
    if kind == "7-Eleven -> 20%":
        return np.concatenate([rng.choice(idx_c[c], int(0.25 * n_other) if c == big else int(0.9 * len(idx_c[c])), replace=False)
                               for c in range(5)])       # 0.25 n_other / (0.9 n_other + 0.25 n_other) ~ 0.217 -> scale below
    m = min(len(i) for i in idx_c)
    return np.concatenate([rng.choice(i, int(0.9 * m), replace=False) for i in idx_c])


for kind in ["train mix", "7-Eleven -> 20%", "balanced"]:
    draws = [draw(kind) for _ in range(200)]
    if kind == "7-Eleven -> 20%":      # exact 20%: big class = 0.25 * (the others kept)
        draws = [np.concatenate([d[y[d] != big], rng.choice(d[y[d] == big], int(0.25 * (y[d] != big).sum()), replace=False)]) for d in draws]
    share = np.mean([(y[d] == big).mean() for d in draws])
    print(f"  {kind} (largest class {share:.1%} of the set, {len(draws[0])} rows)")
    for v in ["full", "zoom70", "centre_grey"]:
        sc = np.array([[mf1(y[d], pred[a, v][d]) for a in ALPHAS] for d in draws])
        print(f"    {v:12s} " + "  ".join(f"a={a:.2f} {sc[:, j].mean():.4f} ({(sc[:, j] > sc[:, 0]).mean():4.0%})" for j, a in enumerate(ALPHAS)))

print("\n== B. stacking head vs plain mean (trained on 4 folds, applied to the 5th)")
arr = task4.load_uint8(train.path.tolist())
g = arr.astype(np.float32).mean(3) / 255                       # (N, 120, 230) grey
H, W = g.shape[1:]
ring = np.ones((H, W), bool)
ring[H // 5:H - H // 5, W // 5:W - W // 5] = False
stats = np.column_stack([g.mean((1, 2)), g.std((1, 2)), g[:, ring].std(1), g[:, ~ring].std(1), g[:, ring].mean(1) - g[:, ~ring].mean(1),
                         g[:, -H // 5:, :].std((1, 2)), g[:, -H // 5:, :].mean((1, 2))])
stats = np.column_stack([stats, stats[:, 2] / (stats[:, 3] + 1e-3)])
del arr, g
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def feats(v, kind):
    L = np.column_stack([np.log(np.clip(P[t, v], 1e-6, 1)) for t in TAGS])
    if kind == "logp":
        return L
    E = np.column_stack([ent(avg[v])] + [ent(P[t, v]) for t in TAGS])
    return np.column_stack([L, E, stats, E[:, :1] * stats])


base = {v: avg[v].argmax(1) for v in VIEWS}
fold_of = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fold_of[va] = f
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
print(f"  plain mean (v16): " + "  ".join(f"{v} {mf1(y, base[v]):.4f}" for v in VIEWS))
for kind in ["logp", "logp+entropy+image stats"]:
    for train_views in [["full"], ["full", "zoom70"]]:
        for C in [0.01, 0.1, 1.0]:
            out = {v: np.zeros(len(y), int) for v in VIEWS}
            for f, va in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(y)), va)
                Xtr = np.vstack([feats(v, kind)[tr] for v in train_views])
                ytr = np.concatenate([y[tr]] * len(train_views))
                sc = StandardScaler().fit(Xtr)
                m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
                for v in VIEWS:
                    out[v][va] = m.predict(sc.transform(feats(v, kind)[va]))
            d = {v: mf1(y, out[v]) - mf1(y, base[v]) for v in VIEWS}
            obj = np.mean([d[v] for v in OBJ])
            pf = [np.mean([mf1(y[fold_of == f], out[v][fold_of == f]) - mf1(y[fold_of == f], base[v][fold_of == f]) for v in OBJ]) for f in range(5)]
            bs = [np.mean([mf1(y[i], out[v][i]) - mf1(y[i], base[v][i]) for v in OBJ]) for i in boot]
            print(f"  {kind:26s} train {'+'.join(train_views):11s} C={C:<5} " + "  ".join(f"{v} {d[v]:+.4f}" for v in VIEWS)
                  + f" | objective {obj:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] folds +{sum(x > 0 for x in pf)}/5", flush=True)
