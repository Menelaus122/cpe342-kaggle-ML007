"""Round 37, Task 4, option 2: a logistic probe on frozen DINOv2-L features (training images; GPU feature extraction only).

    python gi_r37_t4_probeL.py

Scoping (FINDINGS, round 37): the test gap is a train / test shift, and fine-tuning distorts pretrained features under shift while a
linear probe on frozen features does not (Kumar et al., ICLR 2022). v19-v26 mix a 10% probe on frozen CLIP + DINOv2-B + SigLIP 2
features (scripts/train_task4_probe.py). Here the same probe recipe on frozen DINOv2-L (vit_large_patch14_reg4_dinov2.lvd142m,
pooled 1,024-d embedding, mean of the image and its h-flip, 224 x 224 input as the existing probe):
  features: training images under five views applied at the native 230x120 frame (full, zoom70, HUD hidden, gamma 3.5, 75% grid);
            test images full frame (inference only; cached for a later pipeline run, not used below);
  probe:    standardised with the training rows' full-frame mean / SD, multinomial logistic regression with C = 0.03 (the existing
            probe's setting, not tuned), fitted on the full / zoom70 / HUD-hidden features of the training folds (three rows per
            image, as the existing probe); out of fold on the members' folds.
Candidates (the probe weight in the final average, as in the pipeline: 0.9 x members + 0.1 x probe; members = v23 weights):
  A  probe-L instead of the existing probe;   B  probe on the existing 2,304 + the 1,024 DINOv2-L features;
  C  probe-L instead of the existing probe at a weight chosen nested from {0.1, 0.2, 0.3}.
The existing probe has views only for full / zoom70 / HUD hidden, so the comparison uses those views and the real dark /
low-contrast frames (full view). Rule fixed before the run: zoom70, HUD hidden and dark frames all > 0, full >= -0.0005, and the
mean of the zoom70 / HUD-hidden deltas better in >= 3/5 folds. Gamma 3.5 / 75% grid are reported against members + no probe.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402  (the existing probe: VIEWS, embed, fit_probe, features)

NAME, SIDE, TAG = "vit_large_patch14_reg4_dinov2.lvd142m", 224, "large14_reg4_dinov2"
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
FIT_VIEWS = ["full", "zoom70", "centre_grey"]


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


P0.VIEWS.update({"dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75})
train, test = task4.load_frames()
y = train.label.values.astype(np.int64)
f_full, f_views = PRED_DIR / f"t4_feat_{TAG}.npz", PRED_DIR / f"t4_feat_views_{TAG}.npz"
if not (f_full.exists() and f_views.exists()):
    import timm
    t0 = time.time()
    model = timm.create_model(NAME, pretrained=True, num_classes=0, img_size=SIDE).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    if not f_views.exists():
        np.savez(f_views, **P0.embed(model, mean, std, SIDE, train.path.values, ALL5))
        print(f"training features (5 views) in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not f_full.exists():
        np.savez(f_full, train=np.load(f_views)["full"], test=P0.embed(model, mean, std, SIDE, test.path.values, ["full"])["full"],
                 test_file_name=np.array(test.file_name.tolist(), dtype="U"))
        print(f"test features in {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    torch.cuda.empty_cache()
ZL = {v: np.load(f_views)[v].astype(np.float32) for v in ALL5}
for k_ in ("dark_g35", "grid75"):                                  # the existing probe's loader expects its own three views
    P0.VIEWS.pop(k_, None)
XB, _ = P0.features(train, test)                                   # existing probe features (cached): full, zoom70, centre_grey
print(f"features: DINOv2-L {ZL['full'].shape}, existing probe {XB['full'].shape}", flush=True)

folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f


def oof_probe(X, views_out):
    out = {v: np.zeros((len(y), 5)) for v in views_out}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        W, b = P0.fit_probe(np.vstack([(X[v][tr] - mu) / sd for v in FIT_VIEWS]), np.tile(y[tr], len(FIT_VIEWS)))
        for v in views_out:
            out[v][va] = (torch.from_numpy((X[v][va] - mu) / sd) @ W + b).softmax(1).numpy()
    return out


torch.set_num_threads(8)
t1 = time.time()
PL = oof_probe(ZL, ALL5)
XBL = {v: np.hstack([XB[v], ZL[v]]) for v in FIT_VIEWS}
PBL = oof_probe(XBL, FIT_VIEWS)
print(f"probes fitted in {(time.time() - t1) / 60:.1f} min", flush=True)

# members (v23 weights) and the existing probe's out-of-fold views
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
M = {v: np.zeros((len(y), 5)) for v in ALL5}
for t, w in V23_W.items():
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in ALL5:
            M[v][va] += w * (z30 if v in z30.files else z19)[v] / sum(V23_W.values())
zp = np.load("experiments/claude/outputs/r27/t4_probe_views.npz")
PB = {v: zp[f"probe-aug_{v}"] for v in FIT_VIEWS}
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC3 = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full", "dark_g35": "dark_g35", "grid75": "grid75"}
msk = lambda k: dark if k == "dark/low-contrast" else allm

print("probe alone (rule): existing (CLIP + DINOv2-B + SigLIP 2) " + ", ".join(f"{k} {bal(rule(PB[VIEW_OF[k]]), msk(k)):.4f}" for k in SC3))
print("                    DINOv2-L                              " + ", ".join(f"{k} {bal(rule(PL[VIEW_OF[k]]), msk(k)):.4f}" for k in SC3)
      + f", dark_g35 {bal(rule(PL['dark_g35']), allm):.4f}, grid75 {bal(rule(PL['grid75']), allm):.4f}")
print("                    existing + DINOv2-L features          " + ", ".join(f"{k} {bal(rule(PBL[VIEW_OF[k]]), msk(k)):.4f}" for k in SC3))
mix = lambda Q, w: {v: rule((1 - w) * M[v] + w * Q[v]) for v in Q}
base = mix(PB, 0.1)
print("v23 members + 10% existing probe: " + ", ".join(f"{k} {bal(base[VIEW_OF[k]], msk(k)):.4f}" for k in SC3), flush=True)


def report(name, lab, ref=base, keys=SC3):
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in keys}
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in keys) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold "
          f"{' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)


report("A probe-L instead of the existing probe (10%)", mix(PL, 0.1))
report("B probe on existing + DINOv2-L features (10%)", mix(PBL, 0.1))
LABW = {w: mix(PL, w) for w in (0.1, 0.2, 0.3)}
nested = {v: base[v].copy() for v in FIT_VIEWS}
picks = []
for f in range(5):
    m = fo != f
    sc = {w: np.mean([bal(LABW[w][v], m) - bal(base[v], m) for v in ("zoom70", "centre_grey")] + [bal(LABW[w]["full"], m & dark) - bal(base["full"], m & dark)])
          for w in LABW if bal(LABW[w]["full"], m) - bal(base["full"], m) >= -0.0005}
    w = max(sc, key=sc.get) if sc and max(sc.values()) > 0 else None
    picks.append(w)
    if w is not None:
        for v in FIT_VIEWS:
            nested[v][fo == f] = LABW[w][v][fo == f]
report("C probe-L instead of the existing probe, weight nested in {0.1, 0.2, 0.3}", nested)
print(f"    nested picks: {picks}")
mo = {v: rule(M[v]) for v in ALL5}
lo = mix(PL, 0.1)
print("  gamma 3.5 / 75% grid (members only -> members + 10% probe-L): "
      + ", ".join(f"{v} {bal(mo[v], allm):.4f} -> {bal(lo[v], allm):.4f} ({bal(lo[v], allm) - bal(mo[v], allm):+.4f})" for v in ("dark_g35", "grid75")))
print("done")
