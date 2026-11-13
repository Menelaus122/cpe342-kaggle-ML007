"""Round 20: fold-0 test of new Task 4 backbones as a seventh ensemble member, on full frames and HUD-free views. Training images only.

    python cr_r20_t4_candidates.py cnxtS_hud cnxt_hud8

Candidates (scripts/train_task4.py --folds 0, same stratified fold 0 as the pipeline):
  cnxtS_hud   ConvNeXt-Small in12k, HUD dropout p = 0.5   (bigger backbone, same recipe as cnxt_hud)
  cnxt_hud8   ConvNeXt-Tiny in12k, HUD dropout p = 0.8    (stronger HUD dropout)
Each predicts the 6,310 fold-0 validation images under the four views of co_r19_t4_views_allfolds.py (written to
outputs/r19_views/<tag>_fold0.npz so cq_r20_t4_blend.py can use them later).

Pass rule (fixed before the runs): six + candidate (plain mean of seven) beats the v15 six on the objective (mean macro-F1 over full
frames and the 70% zoom) with a paired-bootstrap 95% CI above 0, and loses at most 0.0005 macro-F1 on full frames. Only then are folds
1-4 and the full-data model trained.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR, T4_CLASS_NAMES  # noqa: E402

sys.argv += [] if len(sys.argv) > 1 else ["cnxtS_hud", "cnxt_hud8"]
CANDS = sys.argv[1:]
ARCH = {"cnxtS_hud": ("convnext_small.in12k_ft_in1k", (160, 320)), "cnxt_hud8": ("convnext_tiny.in12k_ft_in1k", (160, 320))}
SIX = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
V19 = Path("experiments/claude/outputs/r19_views")
OBJ = ["full", "zoom70"]



# the four views and the predictor of round 19c (co_r19_t4_views_allfolds.py)
def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def bottom_grey(x):
    x = x.clone()
    x[..., int(x.shape[2] * 0.8):, :] = 0.5
    return x


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


VIEWS = {"full": lambda x: x, "bottom_grey": bottom_grey, "centre_grey": centre_grey, "zoom70": lambda x: zoom(x, 0.7)}


@torch.no_grad()
def predict(model, arr, size):
    out = {v: [] for v in VIEWS}
    for i in range(0, len(arr), 128):
        x0 = task4.to_gpu_batch(arr[i:i + 128], size)
        for v, f in VIEWS.items():
            x = f(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = 0.5 * (model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                           + model(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
            out[v].append(p.cpu())
    return {v: torch.cat(o).numpy() for v, o in out.items()}


train, _ = task4.load_frames()
y = train.label.values
va = np.load(PRED_DIR / "t4_cnxt_fold0.npz")["val_idx"]
yv = y[va]
arr = None
for t in CANDS:
    if (V19 / f"{t}_fold0.npz").exists():
        continue
    assert (np.load(PRED_DIR / f"t4_{t}_fold0.npz")["val_idx"] == va).all()
    arr = task4.load_uint8(list(train.path.values[va])) if arr is None else arr
    name, size = ARCH[t]
    m = task4.create_model(name, pretrained=False).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{t}_fold0.pt", map_location="cuda"))
    m.eval()
    np.savez(V19 / f"{t}_fold0.npz", val_idx=va, **predict(m, arr, size))
    del m
    torch.cuda.empty_cache()

P = {(t, v): np.load(V19 / f"{t}_fold0.npz")[v] for t in SIX + CANDS for v in VIEWS}
mf1 = lambda yy, p: f1_score(yy, p.argmax(1), average="macro")
avg = lambda ts, v: np.mean([P[t, v] for t in ts], 0)
print(f"fold-0 validation images: {len(va)}; macro-F1 per view, objective = mean over {OBJ}")
print(f"{'model':28s} | " + " | ".join(f"{v:>11s}" for v in VIEWS) + " | objective")
rows = {t: [t] for t in ["cnxt_hud"] + CANDS}
rows["v15 six"] = SIX
for c in CANDS:
    rows[f"six + {c}"] = SIX + [c]
    rows[f"six, cnxt_hud -> {c}"] = [t for t in SIX if t != "cnxt_hud"] + [c]
for r, ts in rows.items():
    s = [mf1(yv, avg(ts, v)) for v in VIEWS]
    print(f"{r:28s} | " + " | ".join(f"{x:11.4f}" for x in s) + f" | {np.mean([s[list(VIEWS).index(v)] for v in OBJ]):.4f}")

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(va), len(va)) for _ in range(2000)]
print("\nseven (six + candidate) minus six, paired bootstrap 95% CI:")
for c in CANDS:
    a, b = {v: avg(SIX, v).argmax(1) for v in VIEWS}, {v: avg(SIX + [c], v).argmax(1) for v in VIEWS}
    f = lambda pred, i: f1_score(yv[i], pred[i], average="macro")
    d_obj = [np.mean([f(b[v], i) - f(a[v], i) for v in OBJ]) for i in boot]
    d_full = [f(b["full"], i) - f(a["full"], i) for i in boot]
    pt_obj = np.mean([mf1(yv, avg(SIX + [c], v)) - mf1(yv, avg(SIX, v)) for v in OBJ])
    pt_full = mf1(yv, avg(SIX + [c], "full")) - mf1(yv, avg(SIX, "full"))
    ok = np.quantile(d_obj, .025) > 0 and pt_full >= -0.0005
    print(f"  {c:10s} objective {pt_obj:+.4f} [{np.quantile(d_obj, .025):+.4f}, {np.quantile(d_obj, .975):+.4f}]  "
          f"full {pt_full:+.4f} [{np.quantile(d_full, .025):+.4f}, {np.quantile(d_full, .975):+.4f}]  -> {'PASS' if ok else 'fail'}")
pube = T4_CLASS_NAMES.index("PUBE Mobile")
for t in ["cnxt_hud"] + CANDS:
    print(f"  {t:10s} PUBE recall: " + "  ".join(f"{v} {(P[t, v].argmax(1)[yv == pube] == pube).mean():.3f}" for v in VIEWS))
