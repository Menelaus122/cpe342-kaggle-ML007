"""Round 39, Task 4, step 1: probe heads on per-backbone L2-normalised features, a larger probe weight, and confidence-gated blending
(training images only; the cached frozen features of the seven v28 backbones; the laptop GPU for the probes, no Colab).

    python gp_r39_t4_probe_heads.py

User brief of 2026-11-17 (after v28 = 0.86192). Reference = v28: v23 members + 0.20 x the S3 MLP probe (z-scored concatenation).
Feature preprocessing (fitted on the training folds' full-frame rows only):
  raw  per-column z-score of the concatenated 7,040 features (v28);
  l2   each backbone's embedding L2-normalised separately (so no backbone dominates by its number of dimensions or its scale), then
       concatenated, centred, and divided by one scalar so that the mean column variance is 1.
Probe heads (fitted on the full / zoom70 / HUD-hidden rows of the training folds, out of fold on the members' folds):
  MLP       the v28 MLP (settings unchanged; seeds 0, 1, 2);
  logistic  sklearn's multinomial objective, C 0.01 (the best C alone in round 38);
  cosine    logits = 16 * cos(w_c, x), x and the class weights L2-normalised; AdamW lr 1e-2, no weight decay, label smoothing 0.1,
            30 epochs of 512, 1 warm-up epoch + cosine; seeds 0, 1, 2;
  multi     the mean of the l2-MLP, l2-logistic and l2-cosine probabilities.
Blending (v17 rule after it): (1 - w) x members + w x probe with w in {0.20, 0.25, 0.30, 0.35}; gated: w_i = w_high for the images
whose members' maximum probability (on that view) is below 0.75, w_base otherwise, (w_base, w_high) in {0.20, ..., 0.35} x {0.40, 0.50}.
Candidates (fixed before the run; each with its settings chosen nested):
  R1 raw MLP (v28's probe), w;  R2 l2 MLP, w;  R3 multi, w;  R4 raw MLP gated;  R5 multi gated;
  N  nested over all 28 settings of R1-R5 together (the honest estimate of "pick the best of everything").
Nested = per held-out fold, the setting with the best mean of the zoom70 / HUD-hidden / dark-frame deltas against v28 on the other four
folds among those with full >= -0.0005 there (none positive: the fold keeps v28).
Acceptance (user's rule against v28, applied to N because N accounts for the choice among the candidates): zoom70, HUD hidden and the
real dark / low-contrast frames all > 0, full >= -0.0005, zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the setting
for the pipeline is the one with the best criterion on all five folds. R1-R5 and the probes alone are reported for information.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402  (BACKBONES of v28, fit_mlp)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIT = ["full", "zoom70", "centre_grey"]
WS, WH = (0.20, 0.25, 0.30, 0.35), (0.40, 0.50)
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)

train, test = task4.load_frames()
y = train.label.values.astype(np.int64)
blocks = []                                                         # per backbone: {view: (n, d) float32}
for name, side, tag in P0.BACKBONES:
    z, zv = np.load(PRED_DIR / f"t4_feat_{tag}.npz"), np.load(PRED_DIR / f"t4_feat_views_{tag}.npz")
    blocks.append({"full": z["train"].astype(np.float32), "zoom70": zv["zoom70"].astype(np.float32), "centre_grey": zv["centre_grey"].astype(np.float32)})
XR = {v: np.hstack([b[v] for b in blocks]) for v in FIT}
XL = {v: np.hstack([b[v] / (np.linalg.norm(b[v], axis=1, keepdims=True) + 1e-6) for b in blocks]) for v in FIT}
del blocks
log(f"{len(P0.BACKBONES)} backbones, {XR['full'].shape[1]} features; device {DEV}")

folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f


def prep(X, tr, kind):
    if kind == "raw":
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
    else:
        mu = X["full"][tr].mean(0)
        sd = np.float32(np.sqrt(((X["full"][tr] - mu) ** 2).mean()))
    return lambda Q: ((Q - mu) / sd).astype(np.float32)


def fit_logistic(A, Y, C=0.01, iters=300):
    A, Y = torch.from_numpy(A).to(DEV), torch.from_numpy(Y).to(DEV)
    W = torch.zeros(A.shape[1], 5, device=DEV, requires_grad=True)
    b = torch.zeros(5, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(A @ W + b, Y, reduction="sum") + 0.5 / C * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    W, b = W.detach(), b.detach()
    return lambda Q: (torch.from_numpy(Q).to(DEV) @ W + b).softmax(1).cpu().numpy()


def fit_cosine(A, Y, seed, s=16.0, epochs=30, bs=512):
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    A, Y = F.normalize(torch.from_numpy(A).to(DEV), dim=1), torch.from_numpy(Y).to(DEV)
    W = nn.Parameter(torch.randn(5, A.shape[1], device=DEV) * 0.01)
    opt = torch.optim.AdamW([W], lr=1e-2, weight_decay=0.0)
    spe = math.ceil(len(Y) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: (t + 1) / spe if t < spe else 0.5 * (1 + math.cos(math.pi * (t - spe) / (epochs * spe - spe))))
    for _ in range(epochs):
        perm = torch.randperm(len(Y), generator=g).to(DEV)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(s * A[idx] @ F.normalize(W, dim=1).T, Y[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    Wn = F.normalize(W.detach(), dim=1)
    return lambda Q: (s * F.normalize(torch.from_numpy(Q).to(DEV), dim=1) @ Wn.T).softmax(1).cpu().numpy()


def oof(X, kind, head):
    out = {v: np.zeros((len(y), 5)) for v in FIT}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        p = prep(X, tr, kind)
        A, Y = np.vstack([p(X[v][tr]) for v in FIT]), np.tile(y[tr], len(FIT))
        if head == "mlp":
            preds = [P0.fit_mlp(A, Y, s, device=DEV) for s in (0, 1, 2)]
        elif head == "logistic":
            preds = [fit_logistic(A, Y)]
        else:
            preds = [fit_cosine(A, Y, s) for s in (0, 1, 2)]
        for v in FIT:
            out[v][va] = np.mean([q(p(X[v][va])) for q in preds], 0)
        del A
    return out


# members (v23 weights), dark frames, scoring
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
M = {v: np.zeros((len(y), 5)) for v in FIT}
for t, w in V23_W.items():
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in FIT:
            M[v][va] += w * (z30 if v in z30.files else z19)[v] / sum(V23_W.values())
low = {v: M[v].max(1) < 0.75 for v in FIT}
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m
print(f"members' maximum probability below 0.75: full {low['full'].mean():.2%}, zoom70 {low['zoom70'].mean():.2%}, "
      f"HUD hidden {low['centre_grey'].mean():.2%} of the training images; dark frames {low['full'][dark].mean():.2%}", flush=True)


def mix(P, w):
    return {v: rule((1 - w) * M[v] + w * P[v]) for v in FIT}


def gated(P, wb_, wh):
    return {v: rule((1 - np.where(low[v], wh, wb_))[:, None] * M[v] + np.where(low[v], wh, wb_)[:, None] * P[v]) for v in FIT}


def alone(P):
    return ", ".join(f"{k} {bal(rule(P[VIEW_OF[k]]), msk(k)):.4f}" for k in SC)


PR = {}
PR["raw MLP"] = oof(XR, "raw", "mlp")
log(f"probe alone, raw MLP (v28): {alone(PR['raw MLP'])}")
for head in ("mlp", "logistic", "cosine"):
    PR[f"l2 {head}"] = oof(XL, "l2", head)
    log(f"probe alone, l2 {head}: {alone(PR[f'l2 {head}'])}")
PR["multi"] = {v: np.mean([PR[f"l2 {h}"][v] for h in ("mlp", "logistic", "cosine")], 0) for v in FIT}
print(f"probe alone, multi (mean of the three l2 heads): {alone(PR['multi'])}", flush=True)
ref = mix(PR["raw MLP"], 0.20)
print("v28 reference (members + 20% raw MLP): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def crit(lab, m):
    d = deltas(lab, m)
    return np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]) if d["full"] >= -0.0005 else -np.inf


def nested(cands):
    out, picks = {v: ref[v].copy() for v in FIT}, []
    for f in range(5):
        sc = {s: crit(lab, fo != f) for s, lab in cands.items()}
        s = max(sc, key=sc.get) if sc and max(sc.values()) > 0 else None
        picks.append(s)
        if s is not None:
            for v in FIT:
                out[v][fo == f] = cands[s][v][fo == f]
    return out, picks


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | "
          f"dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {}
for w in WS:
    C[("R1 raw MLP", w)] = mix(PR["raw MLP"], w)
    C[("R2 l2 MLP", w)] = mix(PR["l2 mlp"], w)
    C[("R3 multi", w)] = mix(PR["multi"], w)
    for wh in WH:
        C[("R4 raw MLP gated", w, wh)] = gated(PR["raw MLP"], w, wh)
        C[("R5 multi gated", w, wh)] = gated(PR["multi"], w, wh)
print(f"\n{len(C)} settings. Candidates against v28:")
for r in ("R1 raw MLP", "R2 l2 MLP", "R3 multi", "R4 raw MLP gated", "R5 multi gated"):
    lab, picks = nested({k: v for k, v in C.items() if k[0] == r})
    report(f"{r}, settings nested; picks {[p[1:] if p else None for p in picks]}", lab)
labN, picksN = nested(C)
okN = report(f"N nested over all {len(C)} settings; picks {picksN}", labN)
best = max(C, key=lambda k: crit(C[k], allm))
print(f"\n-> N {'PASSES' if okN else 'fails'}; the setting with the best criterion on all five folds: {best} "
      f"(in-sample, for the pipeline only if N passes)")
print("\nIn-sample grid against v28 (information only):")
for k, lab in C.items():
    d = deltas(lab)
    print(f"  {k}: " + ", ".join(f"{q} {d[q]:+.4f}" for q in SC))
log("done")
