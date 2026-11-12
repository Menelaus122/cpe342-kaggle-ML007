"""Round 19 forensics, Task 4, TRAINING images and files only.

    python cl_r19_forensics_t4.py

  1. file names and row order against the label
  2. file properties: byte size, zip timestamp, JPEG quantisation (quality), chroma subsampling, progressive, metadata blocks
  3. corruption parameters per image (black rotation corners, noise sigma, brightness, contrast, saturation, blur): how were they made,
     do they depend on the class, do they predict our OOF errors
  4. "twins": near-duplicate training images (same source frame under different corruptions), found with an inversion-invariant
     gradient descriptor; how many cross the stratified folds (that would inflate OOF)
  5. which screen regions the fold-0 ConvNeXt relies on (5x5 occlusion on its own validation images; centre vs border masking)
No test image or test file is opened.
"""
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR, T4_CLASS_NAMES  # noqa: E402

rng = np.random.default_rng(0)
train, _ = task4.load_frames()
y = train.label.values
N = len(y)


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))), stats.chi2_contingency(t)[1]


def by_class(v, fmt="{:.3f}"):
    return " | ".join(f"{T4_CLASS_NAMES[k][:8]} " + fmt.format(np.nanmedian(v[y == k])) for k in range(5))


# OOF of the pipeline pair, for "does this property predict our errors"
oof = np.zeros((N, 5))
fold_of = np.zeros(N, int)
for t in ("cnxt", "deit3"):
    for f in range(5):
        z = np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")
        oof[z["val_idx"]] += z["val_probs"] / 2
        fold_of[z["val_idx"]] = f
err = oof.argmax(1) != y
print(f"training images {N}; pipeline OOF errors {err.sum()} ({err.mean():.2%})")

# ======================================================================================== 1. names and order
hdr("1. File names and row order")
stem = train.file_name.str.replace(".jpg", "", regex=False)
print("name lengths:", dict(Counter(stem.str.len())), "| first characters:", dict(Counter(stem.str[0]).most_common(5)))
print("characters used:", "".join(sorted(set("".join(stem)))))
for p in range(stem.str.len().max()):
    v, pv = cramers_v(stem.str[p].fillna("_"), y)
    print(f"  position {p}: Cramer's V with the label {v:.4f} (p {pv:.2g})")
runs = (np.diff(y) != 0).sum() + 1
exp_runs = 1 + 2 * sum(np.bincount(y) * (N - np.bincount(y))) / (2 * N)
print(f"label runs along the csv order: {runs} (random order expects ~{exp_runs:.0f}); class shares by row decile:")
print(pd.crosstab(np.arange(N) * 10 // N, y, normalize="index").round(3).to_string())
num = train.id.str[3:].astype(int)
print(f"id PLY number == row number + 1: {bool((num.values == np.arange(1, N + 1)).all())}")
same_name_test = None  # deliberately not checked against test files

# ======================================================================================== 2. file properties
hdr("2. File properties (zip entries and JPEG headers)")
zf = zipfile.ZipFile(task4.DATA.parent / "task4_image_classification.zip")
info = {i.filename.split("/")[-1]: i for i in zf.infolist() if i.filename.startswith("train/") and i.filename.endswith(".jpg")}
zi = [info[f] for f in train.file_name]
fsize = np.array([i.file_size for i in zi])
ts = pd.to_datetime([f"{d[0]}-{d[1]:02d}-{d[2]:02d} {d[3]:02d}:{d[4]:02d}:{d[5]:02d}" for d in (i.date_time for i in zi)])
tsec = (ts - ts.min()).total_seconds().values
print(f"zip timestamps of training images: {ts.min()} .. {ts.max()} ({len(np.unique(tsec))} distinct seconds)")
print("  median timestamp offset (s) by class:", by_class(tsec, "{:.0f}"))
print(f"  Cramer's V(timestamp minute bin, label) {cramers_v(pd.cut(tsec, 50, labels=False), y)[0]:.4f}; "
      f"rank corr(timestamp, csv row) {stats.spearmanr(tsec, np.arange(N))[0]:+.4f}")
print("  file size (bytes) by class:", by_class(fsize, "{:.0f}"), f"| AUC size -> OOF error {roc_auc_score(err, fsize):.3f}")
qual, sub, prog, extra, dims, modes = [], [], [], Counter(), Counter(), Counter()
for p in train.path.values:
    im = Image.open(p)
    dims[im.size] += 1
    modes[im.mode] += 1
    q = getattr(im, "quantization", None) or {}
    qual.append(np.mean(q[0]) if 0 in q else np.nan)
    try:
        from PIL import JpegImagePlugin
        sub.append(JpegImagePlugin.get_sampling(im))
    except Exception:
        sub.append(-1)
    prog.append(bool(im.info.get("progressive") or im.info.get("progression")))
    for k in im.info:
        extra[k] += 1
qual = np.array(qual)
print(f"  sizes: {dict(dims.most_common(4))}; modes: {dict(modes)}; info keys: {dict(extra)}")
print(f"  luminance quantisation mean (lower = higher quality): unique values {np.unique(np.round(qual, 2))[:12]} ...; by class:", by_class(qual))
print(f"  chroma subsampling codes: {dict(Counter(sub))}; progressive: {dict(Counter(prog))}")
if len(np.unique(np.round(qual, 2))) > 1:
    print(f"  Cramer's V(quality, label) {cramers_v(np.round(qual, 1), y)[0]:.4f}; AUC quality -> OOF error {roc_auc_score(err, np.nan_to_num(qual)):.3f}")

# ======================================================================================== 3. corruption parameters
hdr("3. Corruption parameters per image")
arr = task4.load_uint8(list(train.path))
H, W = arr.shape[1:3]
g = arr.astype(np.float32).mean(-1) / 255
cm = int(min(H, W) * 0.12)
corner = np.concatenate([g[:, :cm, :cm].reshape(N, -1), g[:, :cm, -cm:].reshape(N, -1), g[:, -cm:, :cm].reshape(N, -1), g[:, -cm:, -cm:].reshape(N, -1)], 1)
black_corner = (corner < 0.03).mean(1)
lap = g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
noise = np.sqrt(np.pi / 2) / 6 * np.abs(lap).mean((1, 2)) * 255 / 255   # Immerkaer's noise estimate (grey levels / 255)
blur = lap.var((1, 2))
mx, mn = arr.max(-1).astype(np.float32), arr.min(-1).astype(np.float32)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0).mean((1, 2))
P = pd.DataFrame({"black_corner": black_corner, "noise": noise, "brightness": g.mean((1, 2)), "contrast": g.std((1, 2)), "saturation": sat,
                  "laplacian_var": blur})
print("parameter summary (quantiles 1/10/50/90/99%):")
print(P.quantile([.01, .1, .5, .9, .99]).round(4).to_string())
print("rotated (black corners > 20%):", f"{np.mean(black_corner > 0.2):.3f} of images;",
      "by class:", " | ".join(f"{T4_CLASS_NAMES[k][:8]} {np.mean(black_corner[y == k] > 0.2):.3f}" for k in range(5)))
for c in P:
    a = max(roc_auc_score(err, P[c]), 1 - roc_auc_score(err, P[c]))
    print(f"  {c:14s}: by class {by_class(P[c].values, '{:.3f}')} | Kruskal p {stats.kruskal(*[P[c][y == k] for k in range(5)]).pvalue:.2g} "
          f"| AUC for an OOF error {a:.3f}")
print("rank correlations between parameters (independent draws would give ~0):")
print(P.corr(method="spearman").round(3).to_string())
hist, edges = np.histogram(noise, bins=40)
print("noise sigma histogram (looking for discrete levels):", hist.tolist())

# ======================================================================================== 4. twins
hdr("4. Near-duplicate 'twins' inside train (inversion-invariant gradient descriptor)")
gs = torch.from_numpy(g).unsqueeze(1)
gs = F.interpolate(gs, size=(32, 64), mode="area")
gx = gs[..., :, 1:] - gs[..., :, :-1]
gy = gs[..., 1:, :] - gs[..., :-1, :]
mag = torch.sqrt(gx[..., :-1, :] ** 2 + gy[..., :, :-1] ** 2).flatten(1)
mag = mag - mag.mean(1, keepdim=True)
D = F.normalize(mag, dim=1).cuda().half()
best, best_j = torch.empty(N), torch.empty(N, dtype=torch.long)
for i in range(0, N, 2048):
    s = D[i:i + 2048] @ D.T
    s[torch.arange(s.shape[0]), torch.arange(i, i + s.shape[0])] = -1
    v, j = s.max(1)
    best[i:i + 2048], best_j[i:i + 2048] = v.float().cpu(), j.cpu()
best, best_j = best.numpy(), best_j.numpy()
print("nearest-neighbour cosine similarity quantiles (50/90/99/99.9%):", np.round(np.quantile(best, [.5, .9, .99, .999]), 3).tolist())
for thr in (0.9, 0.95, 0.98):
    m = best > thr
    same = y[best_j[m]] == y[m]
    cross = fold_of[best_j[m]] != fold_of[m]
    print(f"  similarity > {thr}: {m.sum()} images; same label {same.mean() if m.any() else float('nan'):.3f}; "
          f"partner in another fold {cross.mean() if m.any() else float('nan'):.3f}")
m = best > 0.95
if m.sum() >= 10:
    print(f"  OOF accuracy: images with a cross-fold twin {1 - err[m & (fold_of[best_j] != fold_of)].mean():.4f} vs others {1 - err[~m].mean():.4f}")
    ii = np.where(m)[0][:400]
    rel = []
    for i in ii:
        a, b = g[i].ravel(), g[best_j[i]].ravel()
        rel.append((np.corrcoef(a, b)[0, 1], np.corrcoef(a, g[best_j[i]][:, ::-1].ravel())[0, 1], b.mean() / max(a.mean(), 1e-3)))
    rel = np.array(rel)
    print(f"  twin relations (first {len(ii)} pairs): grey corr median {np.median(rel[:, 0]):.3f} (negative = inverted: {np.mean(rel[:, 0] < -0.5):.2f}); "
          f"mirror corr median {np.median(rel[:, 1]):.3f}; brightness ratio quantiles {np.round(np.quantile(rel[:, 2], [.1, .5, .9]), 2).tolist()}")

# ======================================================================================== 5. where the model looks
hdr("5. Screen regions the fold-0 ConvNeXt relies on (its own validation images)")
z = np.load(PRED_DIR / "t4_cnxt_fold0.npz")
va = z["val_idx"]
model = task4.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False).cuda().to(memory_format=torch.channels_last)
model.load_state_dict(torch.load(MODEL_DIR / "t4_cnxt_fold0.pt", map_location="cuda"))
model.eval()
size = (160, 320)


@torch.no_grad()
def probs(images, mask=None):
    out = []
    for i in range(0, len(images), 128):
        x = task4.to_gpu_batch(images[i:i + 128], size)
        if mask is not None:
            x = x * mask + (1 - mask) * 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1).cpu())
    return torch.cat(out).numpy()


sel = rng.choice(va, 1500, replace=False)
yy = y[sel]
base = probs(arr[sel])
print(f"1,500 validation images: accuracy {np.mean(base.argmax(1) == yy):.4f}")
rows_, cols_ = 5, 5
centre = torch.ones(1, 1, *size, device="cuda")
border = torch.zeros(1, 1, *size, device="cuda")
h5, w5 = size[0] // rows_, size[1] // cols_
centre[..., h5:4 * h5, w5:4 * w5] = 0           # mask the inner 3x3 cells
border[..., h5:4 * h5, w5:4 * w5] = 1           # keep only the inner 3x3 cells
for nm, mk in (("centre 3x3 masked (border/HUD only)", centre), ("border ring masked (centre only)", border)):
    p = probs(arr[sel], mk)
    print(f"  {nm:38s}: accuracy {np.mean(p.argmax(1) == yy):.4f}; per class " +
          " | ".join(f"{T4_CLASS_NAMES[k][:8]} {np.mean(p[yy == k].argmax(1) == k):.3f}" for k in range(5)))
drop = np.zeros((5, rows_, cols_))
for r in range(rows_):
    for c in range(cols_):
        mk = torch.ones(1, 1, *size, device="cuda")
        mk[..., r * h5:(r + 1) * h5, c * w5:(c + 1) * w5] = 0
        p = probs(arr[sel], mk)
        for k in range(5):
            drop[k, r, c] = (base[yy == k, k] - p[yy == k, k]).mean()
for k in range(5):
    print(f"  {T4_CLASS_NAMES[k]}: mean drop of P(true class) when one cell of a 5x5 grid is grey (x1000):")
    print("\n".join("      " + " ".join(f"{1000 * v:6.1f}" for v in drop[k, r]) for r in range(rows_)))
