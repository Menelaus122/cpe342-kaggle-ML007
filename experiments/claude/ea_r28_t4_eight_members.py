"""Round 28, Task 4: the fine-tuned DINOv2-B as an eighth member, all five folds (training images only).

    python ea_r28_t4_eight_members.py [tag]        (default tag: dinov2_v2)

Fold 0 passed (cv_r22_t4_eval.py: +0.0012 [+0.0003, +0.0022] on the three-view uniform-mix objective). Here every fold model
predicts its own validation images under the full, 70%-zoom and HUD-hidden views (h-flip TTA; written to outputs/r19_views like
the other members), and the pipeline's fusion is scored on all 31,546 training images:
    v19:        0.9 * mean(seven) + 0.1 * probe, argmax mix * pi^-0.75
    candidate:  0.9 * mean(seven + new) + 0.1 * probe, same rule
    also:       the eight without the probe; probe weight nested from {0, 0.05, 0.1, 0.15, 0.2}; the new member at double weight
Score: macro-F1 at a uniform class mix, mean over the three views; paired bootstrap; folds better.
Acceptance: mean gain > 0 and >= 4/5 folds.
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

NEW = sys.argv[1] if len(sys.argv) > 1 else "dinov2_v2"
ARCH = {"dinov2_v2": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238))}
V19 = Path("experiments/claude/outputs/r19_views")
R27 = Path("experiments/claude/outputs/r27")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
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

P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in TAGS + [NEW]}
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
mean_of = lambda tags, v, w_new=1.0: (np.sum([P[t][v] for t in tags if t != NEW], 0) + (w_new * P[NEW][v] if NEW in tags else 0)) / (
    len([t for t in tags if t != NEW]) + (w_new if NEW in tags else 0))
fused = lambda tags, lam, w_new=1.0: {v: rule((1 - lam) * mean_of(tags, v, w_new) + lam * probe[v]) for v in VIEWS}
base = fused(TAGS, 0.1)
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
print("uniform class mix, v17 rule: full / zoom70 / HUD hidden / mean")
for name, pr in ((f"{NEW} alone", {v: rule(P[NEW][v]) for v in VIEWS}), ("seven members", fused(TAGS, 0.0)), ("v19: seven + 10% probe", base),
                 ("eight members", fused(TAGS + [NEW], 0.0)), ("eight + 10% probe", fused(TAGS + [NEW], 0.1))):
    s = [bal(allr, pr[v]) for v in VIEWS]
    print(f"  {name:26s} " + " / ".join(f"{x:.4f}" for x in s) + f" / {np.mean(s):.4f}")


def report(name, pred, extra=""):
    d = obj(allr, pred) - obj(allr, base)
    bs = [obj(i, pred) - obj(i, base) for i in boot]
    pf = [obj(va, pred) - obj(va, base) for va in folds]
    per = [bal(allr, pred[v]) - bal(allr, base[v]) for v in VIEWS]
    tm = np.mean([f1_score(y, pred[v], average="macro") - f1_score(y, base[v], average="macro") for v in VIEWS])
    print(f"  {name}: {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)}) | "
          f"full {per[0]:+.4f} zoom70 {per[1]:+.4f} HUD hidden {per[2]:+.4f} | training mix {tm:+.4f} | labels changed {np.mean([np.mean(pred[v] != base[v]) for v in VIEWS]):.2%}{extra}")


print("\ndelta against v19 (seven + 10% probe):")
report("eight + 10% probe (the pipeline with the new member)", fused(TAGS + [NEW], 0.1))
report("eight, no probe", fused(TAGS + [NEW], 0.0))
report("eight + 10% probe, new member at double weight", fused(TAGS + [NEW], 0.1, 2.0))
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2]
nested, chosen = {v: np.zeros(len(y), int) for v in VIEWS}, []
cands = {lam: fused(TAGS + [NEW], lam) for lam in LAMS}
for va in folds:
    tr = np.setdiff1d(allr, va)
    best = max(LAMS, key=lambda lam: obj(tr, cands[lam]))
    chosen.append(best)
    for v in VIEWS:
        nested[v][va] = cands[best][v][va]
report("eight + probe, probe weight nested", nested, f" | chosen {chosen}")
print("  in-sample by probe weight: " + "  ".join(f"{lam}: {obj(allr, cands[lam]) - obj(allr, base):+.4f}" for lam in LAMS))
