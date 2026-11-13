"""Round 21: ConvNeXt-Small HUD as a seventh Task 4 member, scored on all folds (31,546 out-of-fold images), full + HUD-free views.

    python cw_r21_t4_seven.py

Fold 0 alone (cr_r20_t4_candidates.py) gave +0.0008 on the objective with a CI that included 0. The user chose to train all folds for
variance reduction; this decides whether it enters v16. Inclusion rule (fixed before the remaining folds finished): mean out-of-fold
objective gain (macro-F1 averaged over full frames and the 70% zoom) > 0 and full frames not lower by more than 0.0005; the
paired-bootstrap CI and per-fold signs are reported. Writes outputs/r19_views/cnxtS_hud_fold<f>.npz (views as round 19c).
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
SIX = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
NEW = "cnxtS_hud"
OBJ = ["full", "zoom70"]
src = Path("experiments/claude/cr_r20_t4_candidates.py").read_text(encoding="utf-8")
g = {"np": np, "torch": torch, "F": torch.nn.functional, "task4": task4}
exec(src[src.index("def zoom"):src.index("train, _ = task4.load_frames()")], g)   # the round 19c views and predictor
VIEWS, predict = g["VIEWS"], g["predict"]

train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
for f, va in enumerate(folds):
    if (V19 / f"{NEW}_fold{f}.npz").exists():
        continue
    arr = task4.load_uint8(list(train.path.values[va]))
    m = task4.create_model("convnext_small.in12k_ft_in1k", pretrained=False).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{NEW}_fold{f}.pt", map_location="cuda"))
    m.eval()
    np.savez(V19 / f"{NEW}_fold{f}.npz", val_idx=va, **predict(m, arr, (160, 320)))
    print(f"fold {f} predicted", flush=True)
    del m, arr
    torch.cuda.empty_cache()

P = {(t, v): np.zeros((len(y), 5)) for t in SIX + [NEW] for v in VIEWS}
for t in SIX + [NEW]:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t, v][va] = z[v]
avg = lambda ts, v: np.mean([P[t, v] for t in ts], 0)
mf1 = lambda yy, p: f1_score(yy, p, average="macro")
print(f"{len(y)} out-of-fold images; macro-F1 per view")
for name, ts in [(NEW, [NEW]), ("cnxt_hud", ["cnxt_hud"]), ("six (v15)", SIX), ("seven", SIX + [NEW])]:
    print(f"  {name:10s} " + "  ".join(f"{v} {mf1(y, avg(ts, v).argmax(1)):.4f}" for v in VIEWS))
a = {v: avg(SIX, v).argmax(1) for v in VIEWS}
b = {v: avg(SIX + [NEW], v).argmax(1) for v in VIEWS}
rng = np.random.default_rng(0)
d_obj, d_full = [], []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    d = {v: mf1(y[i], b[v][i]) - mf1(y[i], a[v][i]) for v in OBJ}
    d_obj.append(np.mean(list(d.values())))
    d_full.append(d["full"])
pt = {v: mf1(y, b[v]) - mf1(y, a[v]) for v in VIEWS}
pf = [np.mean([mf1(y[va], b[v][va]) - mf1(y[va], a[v][va]) for v in OBJ]) for va in folds]
obj = np.mean([pt[v] for v in OBJ])
print("seven - six: " + "  ".join(f"{v} {pt[v]:+.4f}" for v in VIEWS))
print(f"  objective {obj:+.4f} [{np.quantile(d_obj, .025):+.4f}, {np.quantile(d_obj, .975):+.4f}], better in {sum(x > 0 for x in pf)}/5 folds "
      f"({' '.join(f'{x:+.4f}' for x in pf)}); full {pt['full']:+.4f} [{np.quantile(d_full, .025):+.4f}, {np.quantile(d_full, .975):+.4f}]")
print(f"  -> {'INCLUDE' if obj > 0 and pt['full'] >= -0.0005 else 'do not include'} (rule: mean objective gain > 0, full >= -0.0005)")
