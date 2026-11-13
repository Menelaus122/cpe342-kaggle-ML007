"""Round 27, Task 4: do frozen web-scale features help where the seven fine-tuned members are unsure? (training images only)

    python dr_r27_t4_probe_fusion.py

Round 11 scored a logistic probe on frozen CLIP + DINOv2 + SigLIP 2 features on full frames only (macro-F1 0.930 vs 0.987 for the
fine-tuned models) and dropped it. Not tested then: the HUD-poor views that imitate the test frames, and using the probe only
where the ensemble is unsure. Here, on the pipeline folds (the fold's training images fit the probe; its validation images are
scored under three views: full, 70% zoom, HUD hidden):
  probe-full  multinomial logistic regression on the concatenated full-frame features (C = 0.03, as in round 11)
  probe-aug   the same, fitted on the full-frame, zoom70 and HUD-hidden features of the training images (three rows per image)
Fusion: mix = ens where max(ens) >= tau, else (1 - lam) * ens + lam * probe; then the v17 rule argmax mix * pi^-0.75.
tau in {0.4, 0.5, 0.6, 0.7, 0.85, always}, lam in {0.1, 0.25, 0.5}. Score: macro-F1 with classes weighted to 20% each, mean over the
three views (the objective of rounds 25-26). (tau, lam) nested: chosen on 4 folds, applied to the 5th.
Acceptance: mean nested gain > 0 and >= 4/5 folds.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
R27 = Path("experiments/claude/outputs/r27")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
FEATS = ["basep16_clip_224", "basep14_reg4_dinov2", "basep16_siglip_256_256"]
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
allr = np.arange(len(y))
ens = {v: np.zeros((len(y), 5)) for v in VIEWS}
for t in TAGS:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            ens[v][va] += z[v] / len(TAGS)
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda idx, pred: f1_score(y[idx], pred, average="macro", sample_weight=wb[idx])

cache = R27 / "t4_probe_views.npz"
if cache.exists():
    z = np.load(cache)
    probe = {k: {v: z[f"{k}_{v}"] for v in VIEWS} for k in ("probe-full", "probe-aug")}
else:
    X = {"full": np.hstack([np.load(PRED_DIR / f"t4_feat_{t}.npz")["train"].astype(np.float32) for t in FEATS])}
    for v in ("zoom70", "centre_grey"):
        X[v] = np.hstack([np.load(R27 / f"t4_featviews_{t}.npz")[v].astype(np.float32) for t in FEATS])
    print("features", {v: a.shape for v, a in X.items()}, flush=True)

    def fit_lr(A, Y, C=0.03, iters=300):
        A, Y = torch.tensor(A, device="cuda"), torch.tensor(Y, device="cuda")
        W = torch.zeros(A.shape[1], 5, device="cuda", requires_grad=True)
        b = torch.zeros(5, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(A @ W + b, Y, reduction="sum") + 0.5 / C * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return W.detach(), b.detach()

    probe = {k: {v: np.zeros((len(y), 5), np.float32) for v in VIEWS} for k in ("probe-full", "probe-aug")}
    for f, va in enumerate(folds):
        tr = np.setdiff1d(allr, va)
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        st = lambda a: (a - mu) / sd
        for k, views in (("probe-full", ["full"]), ("probe-aug", VIEWS)):
            W, b = fit_lr(np.vstack([st(X[v][tr]) for v in views]), np.tile(y[tr], len(views)))
            for v in VIEWS:
                probe[k][v][va] = (torch.tensor(st(X[v][va]), device="cuda") @ W + b).softmax(1).cpu().numpy()
            del W, b
            torch.cuda.empty_cache()
        print(f"fold {f} fitted", flush=True)
    np.savez(cache, **{f"{k}_{v}": probe[k][v] for k in probe for v in VIEWS})

print("macro-F1 at a uniform class mix under the v17 rule (alpha 0.75): full / zoom70 / HUD hidden / mean")
rule = lambda P: (P * pi ** -0.75).argmax(1)
for name, P in [("seven members", ens), ("probe-full", probe["probe-full"]), ("probe-aug", probe["probe-aug"])]:
    s = [bal(allr, rule(P[v])) for v in VIEWS]
    print(f"  {name:14s} " + " / ".join(f"{x:.4f}" for x in s) + f" / {np.mean(s):.4f}")
print("\nwhere the ensemble is unsure (max probability below tau): rows, accuracy of the ensemble / probe-full / probe-aug under the same rule")
for v in VIEWS:
    for tau in (0.5, 0.7, 0.85):
        m = ens[v].max(1) < tau
        acc = lambda P: np.mean(rule(P[v])[m] == y[m])
        print(f"  {v:12s} tau {tau:.2f}: {m.sum():5d} rows ({m.mean():.2%}) | {acc(ens):.3f} / {acc(probe['probe-full']):.3f} / {acc(probe['probe-aug']):.3f}")

TAUS, LAMS = [0.4, 0.5, 0.6, 0.7, 0.85, 9.0], [0.1, 0.25, 0.5]


def fused(k, v, tau, lam):
    m = (ens[v].max(1) < tau)[:, None]
    return rule(np.where(m, (1 - lam) * ens[v] + lam * probe[k][v], ens[v]))


base = {v: rule(ens[v]) for v in VIEWS}
obj = lambda idx, pr: np.mean([bal(idx, pr[v][idx]) for v in VIEWS])
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
for k in ("probe-full", "probe-aug"):
    grid = {(tau, lam): {v: fused(k, v, tau, lam) for v in VIEWS} for tau in TAUS for lam in LAMS}
    print(f"\n{k}: mean balanced macro-F1 over the three views, all folds (baseline {obj(allr, base):.4f}); rows = tau, columns = lam {LAMS}")
    for tau in TAUS:
        print(f"  tau {'always' if tau > 1 else f'{tau:.2f}':>6s}: " + "  ".join(f"{obj(allr, grid[tau, lam]) - obj(allr, base):+.4f}" for lam in LAMS))
    nested = {v: np.zeros(len(y), int) for v in VIEWS}
    chosen = []
    for va in folds:
        tr = np.setdiff1d(allr, va)
        best = max(grid, key=lambda g: obj(tr, grid[g]))
        chosen.append(best)
        for v in VIEWS:
            nested[v][va] = grid[best][v][va]
    d = obj(allr, nested) - obj(allr, base)
    bs = [obj(i, nested) - obj(i, base) for i in boot]
    pf = [obj(va, nested) - obj(va, base) for va in folds]
    per_view = [bal(allr, nested[v]) - bal(allr, base[v]) for v in VIEWS]
    tm = np.mean([f1_score(y, nested[v], average="macro") - f1_score(y, base[v], average="macro") for v in VIEWS])
    print(f"  nested (tau, lam) per fold: {chosen}")
    print(f"  nested delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)}) | "
          f"per view full {per_view[0]:+.4f} zoom70 {per_view[1]:+.4f} HUD hidden {per_view[2]:+.4f} | at the training mix {tm:+.4f}")
