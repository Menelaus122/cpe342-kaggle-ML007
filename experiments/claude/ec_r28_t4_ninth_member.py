"""Round 28, Task 4: a ninth member on top of the v20 ensemble, all five folds (training images only).

    python ec_r28_t4_ninth_member.py [tag]        (default tag: eva02B)

v20: weighted mean of eight fine-tuned members (DINOv2 ViT-B/14 at weight 3, the other seven at 1), 0.9 * mean + 0.1 * probe,
label = argmax mix * pi^-0.75. Fold 0 of the candidate passed in cv_r22_t4_eval.py. Here every fold model of the candidate predicts
its own validation images under the full, 70%-zoom and HUD-hidden views (h-flip TTA, written to outputs/r19_views), and the
fusion is scored on all 31,546 training images at a uniform class mix, mean over the three views:
    candidate at weight w in {1, 1.5, 2, 3, 4}, fixed and nested (chosen on 4 folds, scored on the 5th)
    the weights of DINOv2 and of the candidate chosen together, nested, from DINOv2 in {2, 3, 4, 6} x candidate in {0, 1, 2, 3, 4}
    the probe weight nested from {0, 0.05, 0.1, 0.15} with the nine members
Acceptance: mean gain > 0 and >= 4/5 folds; the weight used in the pipeline is the one the nested choice supports.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

NEW = sys.argv[1] if len(sys.argv) > 1 else "eva02B"
ARCH = {"eva02B": ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", (126, 238)),
        "siglip2_v2": ("vit_base_patch16_siglip_224.v2_webli", (128, 240))}
V19 = Path("experiments/claude/outputs/r19_views")
R27 = Path("experiments/claude/outputs/r27")
SEVEN = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
DINO = "dinov2_v2"
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
allr = np.arange(len(y))


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}
for f, va in enumerate(folds):
    out = V19 / f"{NEW}_fold{f}.npz"
    if out.exists():
        continue
    name, size = ARCH[NEW]
    assert (np.load(PRED_DIR / f"t4_{NEW}_fold{f}.npz")["val_idx"] == va).all(), "fold rows differ from the pipeline folds"
    m = task4.create_model(name, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{NEW}_fold{f}.pt", map_location="cuda"))
    m.eval()
    arr = task4.load_uint8(list(train.path.values[va]))
    res = {v: [] for v in VIEWS}
    with torch.no_grad():
        for i in range(0, len(arr), 128):
            x0 = task4.to_gpu_batch(arr[i:i + 128], size)
            for v in VIEWS:
                x = FN[v](x0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                               + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                res[v].append(p.cpu().numpy())
    np.savez(out, **{v: np.concatenate(r) for v, r in res.items()})
    print(f"{NEW} fold {f} views written", flush=True)
    del m
    torch.cuda.empty_cache()

P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in SEVEN + [DINO, NEW]}
for t in P:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t][v][va] = z[v]
zp = np.load(R27 / "t4_probe_views.npz")
probe = {v: zp[f"probe-aug_{v}"].astype(float) for v in VIEWS}
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
rule = lambda p: (p * pi ** -0.75).argmax(1)
bal = lambda idx, pred: f1_score(y[idx], pred[idx], average="macro", sample_weight=wb[idx])
obj = lambda idx, pr: np.mean([bal(idx, pr[v]) for v in VIEWS])
S7 = {v: np.sum([P[t][v] for t in SEVEN], 0) for v in VIEWS}


def fused(w_dino, w_new, lam=0.1):
    return {v: rule((1 - lam) * (S7[v] + w_dino * P[DINO][v] + w_new * P[NEW][v]) / (7 + w_dino + w_new) + lam * probe[v]) for v in VIEWS}


base = fused(3.0, 0.0)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
print("uniform class mix, v17 rule: full / zoom70 / HUD hidden / mean")
for name, pr in ((f"{NEW} alone", {v: rule(P[NEW][v]) for v in VIEWS}), (f"{DINO} alone", {v: rule(P[DINO][v]) for v in VIEWS}),
                 ("v20: eight (DINOv2 x3) + 10% probe", base), (f"v20 + {NEW} at weight 1", fused(3.0, 1.0))):
    s = [bal(allr, pr[v]) for v in VIEWS]
    print(f"  {name:36s} " + " / ".join(f"{x:.4f}" for x in s) + f" / {np.mean(s):.4f}")
print(f"  macro-F1 of {NEW} per fold (full frames, plain argmax): " + " ".join(f"{f1_score(y[va], P[NEW]['full'][va].argmax(1), average='macro'):.4f}" for va in folds))


def report(name, pred, ref=None, extra=""):
    ref = base if ref is None else ref
    d = obj(allr, pred) - obj(allr, ref)
    bs = [obj(i, pred) - obj(i, ref) for i in boot]
    pf = [obj(va, pred) - obj(va, ref) for va in folds]
    per = [bal(allr, pred[v]) - bal(allr, ref[v]) for v in VIEWS]
    tm = np.mean([f1_score(y, pred[v], average="macro") - f1_score(y, ref[v], average="macro") for v in VIEWS])
    print(f"  {name}: {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)}) | "
          f"full {per[0]:+.4f} zoom70 {per[1]:+.4f} HUD hidden {per[2]:+.4f} | training mix {tm:+.4f} | labels changed {np.mean([np.mean(pred[v] != ref[v]) for v in VIEWS]):.2%}{extra}")


def nested(cands):
    out, chosen = {v: np.zeros(len(y), int) for v in VIEWS}, []
    for va in folds:
        tr = np.setdiff1d(allr, va)
        best = max(cands, key=lambda k: obj(tr, cands[k]))
        chosen.append(best)
        for v in VIEWS:
            out[v][va] = cands[best][v][va]
    return out, chosen


print("\ndelta against v20:")
WS = [1.0, 1.5, 2.0, 3.0, 4.0]
cw = {w: fused(3.0, w) for w in WS}
for w in WS:
    report(f"{NEW} at weight {w}", cw[w])
nw, ch = nested(cw)
report(f"{NEW} weight nested", nw, extra=f" | chosen {ch}")
report(f"{NEW} weight nested, against weight 1", nw, ref=cw[1.0])
grid = {(wd, wn): fused(wd, wn) for wd in (2.0, 3.0, 4.0, 6.0) for wn in (0.0, 1.0, 2.0, 3.0, 4.0)}
ng, ch = nested(grid)
report(f"DINOv2 and {NEW} weights nested together", ng, extra=f" | chosen (DINOv2, {NEW}) {ch}")
print("  in-sample delta by (DINOv2 weight, candidate weight):")
for wd in (2.0, 3.0, 4.0, 6.0):
    print(f"    DINOv2 {wd}: " + "  ".join(f"{wn}: {obj(allr, grid[wd, wn]) - obj(allr, base):+.4f}" for wn in (0.0, 1.0, 2.0, 3.0, 4.0)))
best_w = max(WS, key=lambda w: obj(allr, cw[w]))
lam_c = {lam: fused(3.0, best_w, lam) for lam in (0.0, 0.05, 0.1, 0.15)}
nl, ch = nested(lam_c)
report(f"probe weight nested ({NEW} at {best_w})", nl, extra=f" | chosen {ch}")
print("  in-sample by probe weight: " + "  ".join(f"{lam}: {obj(allr, lam_c[lam]) - obj(allr, base):+.4f}" for lam in lam_c))
