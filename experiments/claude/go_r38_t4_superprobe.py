"""Round 38, Task 4, step 1b: "super-probe" on frozen features of seven backbones (training images only; GPU for the probes).

    python go_r38_t4_superprobe.py          (after gm_r38_t4_feat_extract.py has written the three new feature caches)

User brief of 2026-11-17 (after v27 = 0.86177). Features (frozen, pooled, image + h-flip, cached):
  v27 probe ("base4", 3,328-d): CLIP ViT-B/16, DINOv2 ViT-B/14, SigLIP 2 ViT-B/16 (768 each), DINOv2 ViT-L/14 (1,024);
  new ("new3", 3,712-d):        SigLIP 2 SO400M/14 at 224 (1,152), EVA-02-L/14 MIM m38m ft in22k-in1k at 224 (1,024),
                                ConvNeXt-L CLIP LAION-2B soup ft in12k-in1k at 320 (1,536).
Probes are fitted, as the v27 probe, on the full / zoom70 / HUD-hidden features of the training folds (three rows per image),
standardised with the training rows' full-frame mean / SD, out of fold on the members' folds:
  logistic  sklearn's multinomial objective (as scripts/train_task4_probe.py), C in {0.01, 0.03, 0.1};
  MLP       fixed before the run, not tuned: dropout 0.2 on the input, one hidden layer of 1,024 (GELU), dropout 0.5, AdamW lr 1e-3,
            weight decay 0.05, 30 epochs of batches of 512, 1 warm-up epoch then cosine, label smoothing 0.1; mean of 3 seeds.
Final average as in the pipeline: (1 - w) x v23 members + w x probe, v17 rule (argmax p * pi^-0.75), uniform class mix.
Reference = v27 (members + 0.10 x the v27 probe: logistic on base4, C 0.03; refitted here with the same code).
Candidates (fixed before the run):
  S1  logistic on base4 + new3, C 0.03, w 0.10 (the v27 recipe with the new features);
  S2  logistic on base4 + new3, (C, w) chosen nested from {0.01, 0.03, 0.1} x {0.10, 0.15, 0.20};
  S3  MLP on base4 + new3, w chosen nested from {0.10, 0.15, 0.20}.
Nested = for each held-out fold, the setting with the best mean of the zoom70 / HUD-hidden / dark-frame deltas on the other four
folds among those with full >= -0.0005 there (none positive: the fold keeps v27).
Rule (user, as round 37): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005, and the mean of the
zoom70 / HUD-hidden deltas better in >= 3/5 folds. If more than one candidate passes, the one with the highest mean of the three
hard-view deltas. Diagnostics (not candidates): each feature set's probe alone; base4 + each new backbone (logistic, C 0.03, w 0.10).
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402  (BACKBONES of v27, features)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEW = {"so400m14_siglip2_224": "SigLIP2-SO400M", "eva02L14_m38m_224": "EVA02-L", "cnxtL_clip_320": "ConvNeXt-L CLIP"}
FIT = ["full", "zoom70", "centre_grey"]
CS, WS = (0.01, 0.03, 0.1), (0.10, 0.15, 0.20)
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)

train, test = task4.load_frames()
y = train.label.values.astype(np.int64)
XB, _ = P0.features(train, test)                                  # v27 probe features (cached): full, zoom70, centre_grey
XN = {}
for tag in NEW:
    z = np.load(PRED_DIR / f"t4_feat_views_{tag}.npz")
    XN[tag] = {v: z[v].astype(np.float32) for v in FIT}
log(f"features: base4 {XB['full'].shape[1]}, " + ", ".join(f"{NEW[t]} {XN[t]['full'].shape[1]}" for t in NEW) + f"; device {DEV}")
cat = lambda parts: {v: np.hstack([p[v] for p in parts]) for v in FIT}

folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f


def fit_logistic(A, Y, C, iters=300):
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


def fit_mlp(A, Y, seed, epochs=30, bs=512):
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    A, Y = torch.from_numpy(A).to(DEV), torch.from_numpy(Y).to(DEV)
    net = nn.Sequential(nn.Dropout(0.2), nn.Linear(A.shape[1], 1024), nn.GELU(), nn.Dropout(0.5), nn.Linear(1024, 5)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    spe = math.ceil(len(Y) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / spe if s < spe else 0.5 * (1 + math.cos(math.pi * (s - spe) / (epochs * spe - spe))))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Y), generator=g).to(DEV)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(net(A[idx]), Y[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def predict(Q):
        with torch.no_grad():
            return net(torch.from_numpy(Q).to(DEV)).softmax(1).cpu().numpy()
    return predict


def oof(X, kind, C=0.03, seeds=(0, 1, 2)):
    """out-of-fold probabilities of a probe on the three views (fitted on the training folds' three views)."""
    out = {v: np.zeros((len(y), 5)) for v in FIT}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        A = np.vstack([(X[v][tr] - mu) / sd for v in FIT]).astype(np.float32)
        Y = np.tile(y[tr], len(FIT))
        preds = [fit_logistic(A, Y, C)] if kind == "logistic" else [fit_mlp(A, Y, s) for s in seeds]
        for v in FIT:
            Q = ((X[v][va] - mu) / sd).astype(np.float32)
            out[v][va] = np.mean([p(Q) for p in preds], 0)
        del A
    return out


# members (v23 weights), dark frames, scoring (as gi_r37_t4_probeL.py)
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
mix = lambda Q, w: {v: rule((1 - w) * M[v] + w * Q[v]) for v in FIT}
alone = lambda Q: ", ".join(f"{k} {bal(rule(Q[VIEW_OF[k]]), msk(k)):.4f}" for k in SC)

base4 = XB
allf = cat([XB] + [XN[t] for t in NEW])
log(f"base4 + new3: {allf['full'].shape[1]} features")
PB = oof(base4, "logistic", 0.03)
ref = mix(PB, 0.10)
print("v27 reference (members + 10% probe on base4): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print(f"probe alone (rule), base4 logistic C 0.03: {alone(PB)}")
LOG = {}
for C in CS:
    LOG[C] = oof(allf, "logistic", C)
    log(f"probe alone, base4 + new3 logistic C {C}: {alone(LOG[C])}")
MLP = oof(allf, "mlp")
log(f"probe alone, base4 + new3 MLP (3 seeds): {alone(MLP)}")


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    hard3 = np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | hard mean {hard3:+.4f} | zoom / HUD by fold "
          f"{' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok, hard3


def nested(cands):
    """cands: {setting: labels}; per held-out fold the setting best on the other four (hard mean, full >= -0.0005), else v27."""
    out, picks = {v: ref[v].copy() for v in FIT}, []
    for f in range(5):
        m = fo != f
        sc = {}
        for s, lab in cands.items():
            d = deltas(lab, m)
            if d["full"] >= -0.0005:
                sc[s] = np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
        s = max(sc, key=sc.get) if sc and max(sc.values()) > 0 else None
        picks.append(s)
        if s is not None:
            for v in FIT:
                out[v][fo == f] = cands[s][v][fo == f]
    return out, picks


print("\nCandidates against v27:")
res = {}
res["S1"] = report("S1 logistic on base4 + new3, C 0.03, w 0.10", mix(LOG[0.03], 0.10))
lab2, p2 = nested({(C, w): mix(LOG[C], w) for C in CS for w in WS})
res["S2"] = report(f"S2 logistic on base4 + new3, (C, w) nested; picks {p2}", lab2)
lab3, p3 = nested({w: mix(MLP, w) for w in WS})
res["S3"] = report(f"S3 MLP on base4 + new3, w nested; picks {p3}", lab3)
passed = [k for k, (ok, _) in res.items() if ok]
print(f"\npassing: {passed or 'none'}" + (f"; chosen by the hard mean: {max(passed, key=lambda k: res[k][1])}" if passed else ""))

print("\nDiagnostics (not candidates):")
for t in NEW:
    Q = oof({v: XN[t][v] for v in FIT}, "logistic", 0.03)
    log(f"  probe alone, {NEW[t]} logistic C 0.03: {alone(Q)}")
    Q = oof(cat([XB, XN[t]]), "logistic", 0.03)
    report(f"base4 + {NEW[t]}, logistic C 0.03, w 0.10", mix(Q, 0.10))
for C in CS:
    for w in WS:
        d = deltas(mix(LOG[C], w))
        print(f"  logistic C {C}, w {w:.2f} (in-sample): " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC))
for w in WS:
    d = deltas(mix(MLP, w))
    print(f"  MLP, w {w:.2f} (in-sample): " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC))
log("done")
