"""Task 4: a probe on frozen foundation features of seven backbones, as a low-weight member (v28: an MLP probe at 20%).

    python scripts/train_task4_probe.py

Pre-trained models are allowed for Task 4. The backbones stay frozen; only the probe is trained, on training images. Test images
are only passed through the frozen backbones and the fitted probe (inference).

v28 (round 38, experiments/claude/go_r38_t4_superprobe.py candidate S3, rule fixed before the run): the features of three more
backbones (SigLIP 2 SO400M/14 at 224, EVA-02-L/14 MIM m38m ft in22k-in1k at 224, ConvNeXt-L CLIP LAION-2B soup ft at 320; caches
from experiments/claude/gm_r38_t4_feat_extract.py) are appended (7,040 numbers per image) and the logistic regression is replaced by
a small MLP (fixed before the run, not tuned: dropout 0.2 on the input, 1,024 GELU units, dropout 0.5, AdamW lr 1e-3, weight decay
0.05, 30 epochs of 512, 1 warm-up epoch + cosine, label smoothing 0.1; mean of seeds 0, 1, 2; GPU). Nested, its weight in the final
average is 0.20 in all five folds; against v27: full +0.0009, zoom70 +0.0009, HUD hidden +0.0003, real dark / low-contrast frames
+0.0032, zoom / HUD better in 5/5 folds. With the logistic regression the new features did not help (S1 / S2 fail). The v27 probe
(logistic, first four backbones) is PROBE = "logistic" with BACKBONES[:4]; its outputs are in versions/v27_LB0.86177/.

Features: pooled embedding, mean of the image and its horizontal flip, 3 x 768 + 1,024 = 3,328 numbers per image. v19-v26 used the
first three backbones (2,304 numbers); DINOv2-L was added for v27 (round 37, experiments/claude/gi_r37_t4_probeL.py candidate B:
with the probe at 10% of the ensemble, HUD hidden +0.0008, real dark / low-contrast frames +0.0002, zoom70 +0.0000, full -0.0000,
zoom70 / HUD hidden better in 3/5 folds; rule fixed before the run). The v19-v26 probe outputs are in versions/v26_LB0.86147/.
  training images: full frame, 70% centre zoom, HUD hidden (borders greyed)  -> the probe sees every training image three times,
                   so it cannot rely on the HUD alone (round 27: 0.86 on HUD-hidden frames vs 0.59 for a full-frame-only probe)
  test images:     full frame
Probe: features standardised with the training rows' full-frame mean / SD, L2-regularised logistic regression (C = 0.03, L-BFGS).
Folds = the folds of the fine-tuned members. Writes outputs/preds/t4_probe_fold{f}.npz (val_idx, val_probs, test_probs) and
t4_probe_full.npz (test_probs), the same layout as scripts/train_task4.py, so the notebook reads it like any other member.
Feature caches: outputs/preds/t4_feat_<backbone>.npz (train, test) and t4_feat_views_<backbone>.npz (zoom70, centre_grey); they are
large and not committed; the script recomputes whatever is missing (about 10 minutes per base backbone on an 8 GB GPU; DINOv2-L about
25 minutes for these three views).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

BACKBONES = [("vit_base_patch16_clip_224.openai", 224, "basep16_clip_224"),
             ("vit_base_patch14_reg4_dinov2.lvd142m", 224, "basep14_reg4_dinov2"),
             ("vit_base_patch16_siglip_256.v2_webli", 256, "basep16_siglip_256_256"),
             ("vit_large_patch14_reg4_dinov2.lvd142m", 224, "large14_reg4_dinov2"),
             ("vit_so400m_patch14_siglip_224.v2_webli", 224, "so400m14_siglip2_224"),            # v28 (round 38)
             ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 224, "eva02L14_m38m_224"),
             ("convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", 320, "cnxtL_clip_320")]
EXTRA_KW = {"eva02L14_m38m_224": {"img_size": 224}}      # timm kwargs when a cache has to be recomputed
C_PROBE = 0.03
PROBE = "mlp"                                             # v19-v27: "logistic"
MLP_SEEDS = (0, 1, 2)


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


VIEWS = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}


@torch.no_grad()
def embed(model, mean, std, side, paths, views):
    feats = {v: [] for v in views}
    for i in range(0, len(paths), 4096):
        arr = task4.load_uint8(list(paths[i:i + 4096]))
        for j in range(0, len(arr), 128):
            x0 = torch.from_numpy(arr[j:j + 128]).cuda().permute(0, 3, 1, 2).float().div_(255)
            for v in views:
                x = (F.interpolate(VIEWS[v](x0), size=(side, side), mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model(x).float() + model(torch.flip(x, dims=[3])).float()
                feats[v].append((f / 2).cpu().numpy().astype(np.float16))
    return {v: np.concatenate(f) for v, f in feats.items()}


def features(train, test):
    """Frozen features of every backbone, concatenated: train under the three views, test full frame."""
    import timm
    X = {v: [] for v in VIEWS}
    Xt = []
    for name, side, tag in BACKBONES:
        f_full, f_views = PRED_DIR / f"t4_feat_{tag}.npz", PRED_DIR / f"t4_feat_views_{tag}.npz"
        if not (f_full.exists() and f_views.exists()):
            t0 = time.time()
            kw = EXTRA_KW.get(tag, {"img_size": side} if "dinov2" in name else {})
            model = timm.create_model(name, pretrained=True, num_classes=0, **kw).cuda().eval()
            cfg = timm.data.resolve_data_config({}, model=model)
            mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
            std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
            if not f_full.exists():
                np.savez(f_full, train=embed(model, mean, std, side, train.path.values, ["full"])["full"],
                         test=embed(model, mean, std, side, test.path.values, ["full"])["full"],
                         test_file_name=np.array(test.file_name.tolist(), dtype="U"))
            if not f_views.exists():
                np.savez(f_views, **embed(model, mean, std, side, train.path.values, ["zoom70", "centre_grey"]))
            del model
            torch.cuda.empty_cache()
            print(f"extracted {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)
        z, zv = np.load(f_full), np.load(f_views)
        assert list(z["test_file_name"]) == test.file_name.tolist(), f"{f_full.name}: test order differs from test.csv"
        X["full"].append(z["train"].astype(np.float32))
        Xt.append(z["test"].astype(np.float32))
        for v in ("zoom70", "centre_grey"):
            X[v].append(zv[v].astype(np.float32))
    return {v: np.hstack(a) for v, a in X.items()}, np.hstack(Xt)


def fit_probe(A, Y, iters=300):
    """Multinomial logistic regression, sklearn's objective: sum of cross-entropies + ||W||^2 / (2 C). CPU, deterministic."""
    A, Y = torch.from_numpy(A), torch.from_numpy(Y)
    W = torch.zeros(A.shape[1], 5, requires_grad=True)
    b = torch.zeros(5, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(A @ W + b, Y, reduction="sum") + 0.5 / C_PROBE * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def fit_mlp(A, Y, seed, epochs=30, bs=512, device="cuda"):
    """The v28 MLP probe (round 38 S3; identical to go_r38_t4_superprobe.fit_mlp). Returns a predict function (numpy -> probabilities)."""
    import math
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    A, Y = torch.from_numpy(A).to(device), torch.from_numpy(Y).to(device)
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(A.shape[1], 1024), torch.nn.GELU(), torch.nn.Dropout(0.5),
                              torch.nn.Linear(1024, 5)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    spe = math.ceil(len(Y) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / spe if s < spe else 0.5 * (1 + math.cos(math.pi * (s - spe) / (epochs * spe - spe))))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Y), generator=g).to(device)
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
            return net(torch.from_numpy(Q).to(device)).softmax(1).cpu().numpy()
    return predict


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    train, test = task4.load_frames()
    y = train.label.values.astype(np.int64)
    X, Xt = features(train, test)
    names = np.array(test.file_name.tolist(), dtype="U")

    def run(tr):
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        A, Y = np.vstack([(X[v][tr] - mu) / sd for v in VIEWS]).astype(np.float32), np.tile(y[tr], len(VIEWS))
        if PROBE == "mlp":
            preds = [fit_mlp(A, Y, s) for s in MLP_SEEDS]
            return lambda Q: np.mean([p_(((Q - mu) / sd).astype(np.float32)) for p_ in preds], 0)
        W, b = fit_probe(A, Y)
        return lambda Q: (torch.from_numpy(((Q - mu) / sd).astype(np.float32)) @ W + b).softmax(1).numpy()

    for f in range(5):
        out = PRED_DIR / f"t4_probe_fold{f}.npz"
        if out.exists():
            continue
        va = np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"]
        predict = run(np.setdiff1d(np.arange(len(y)), va))
        vp = predict(X["full"][va])
        np.savez_compressed(out, val_idx=va, val_probs=vp, test_probs=predict(Xt), test_file_name=names)
        print(f"fold {f}: full-frame accuracy {np.mean(vp.argmax(1) == y[va]):.4f}", flush=True)
    out = PRED_DIR / "t4_probe_full.npz"
    if not out.exists():
        np.savez_compressed(out, test_probs=run(np.arange(len(y)))(Xt), test_file_name=names)
        print("full fit done", flush=True)


if __name__ == "__main__":
    main()
