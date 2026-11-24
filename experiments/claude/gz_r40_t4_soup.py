"""Round 40, Task 4: weight soup of two fine-tuned runs of the same recipe (GPU inference only; no training).

    python gz_r40_t4_soup.py TAG_A TAG_B OUT --model M --h H --w W [--folds 0 1 2 3 4] [--no-full]

User brief of 2026-11-18 (revised round 40): compare, for the DINOv2-L member, seed 1 alone (v30), the mean of the two seeds'
probabilities, and a weight soup (Wortsman et al., ICML 2022: the uniform average of fine-tuned weights that share a pretrained
start). For every fold, the EMA checkpoints outputs/models/t4_<TAG_A>_fold<f>.pt and t4_<TAG_B>_fold<f>.pt are averaged (every float
tensor; other buffers from A) and the averaged model predicts, with h-flip TTA as the pipeline:
  its validation images under the five scoring views (full, zoom70, HUD hidden, gamma 3.5, 75% grid; as em_r30_t4_hardviews.py)
    -> experiments/claude/outputs/r30_views/<OUT>_fold<f>.npz
  its validation images (full) and the test images -> outputs/preds/t4_<OUT>_fold<f>.npz (val_idx, val_probs, test_probs, test_file_name)
and the two full refits are averaged the same way -> outputs/preds/t4_<OUT>_full.npz. Existing outputs are skipped (resumable).
Run on Colab, where the checkpoints are (1.2 GB each). The comparison is gx_r40_t4_dinov2L_seed2_eval.py.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402
from pathlib import Path  # noqa: E402

V30 = Path("experiments/claude/outputs/r30_views")


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey,
      "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75}


def soup(model_name, size, path_a, path_b):
    sa, sb = torch.load(path_a, map_location="cpu"), torch.load(path_b, map_location="cpu")
    assert sa.keys() == sb.keys()
    avg = {k: (0.5 * (sa[k].float() + sb[k].float())).to(sa[k].dtype) if sa[k].is_floating_point() else sa[k] for k in sa}
    m = task4.create_model(model_name, pretrained=False, img_size=size)
    m.load_state_dict(avg)
    return m.cuda().to(memory_format=torch.channels_last).eval()


def views(model, arr, size):
    res = {v: [] for v in FN}
    with torch.no_grad():
        for i in range(0, len(arr), 128):
            x0 = task4.to_gpu_batch(arr[i:i + 128], size)
            for v, fn in FN.items():
                x = fn(x0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = 0.5 * (model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                               + model(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                res[v].append(p.cpu().numpy())
    return {v: np.concatenate(r) for v, r in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-full", action="store_true")
    args = ap.parse_args()
    size = (args.h, args.w)
    train, test = task4.load_frames()
    y = train.label.values
    names = np.array(test.file_name.tolist(), dtype="U")
    for f in args.folds:
        pv, pp = V30 / f"{args.out}_fold{f}.npz", PRED_DIR / f"t4_{args.out}_fold{f}.npz"
        if pv.exists() and pp.exists():
            print(f"fold {f}: cached", flush=True)
            continue
        va = np.load(PRED_DIR / f"t4_{args.tag_a}_fold{f}.npz")["val_idx"]
        assert (va == np.load(PRED_DIR / f"t4_{args.tag_b}_fold{f}.npz")["val_idx"]).all()
        m = soup(args.model, size, MODEL_DIR / f"t4_{args.tag_a}_fold{f}.pt", MODEL_DIR / f"t4_{args.tag_b}_fold{f}.pt")
        vw = views(m, task4.load_uint8(list(train.path.values[va])), size)
        np.savez(pv, **vw)
        tp = task4.predict_paths(m, test.path.values, size, tta=True)
        np.savez_compressed(pp, val_idx=va, val_probs=vw["full"], test_probs=tp, test_file_name=names)
        print(f"fold {f}: soup written; val macro-F1 {f1_score(y[va], vw['full'].argmax(1), average='macro'):.4f}", flush=True)
        del m
        torch.cuda.empty_cache()
    pf = PRED_DIR / f"t4_{args.out}_full.npz"
    if not args.no_full and not pf.exists():
        m = soup(args.model, size, MODEL_DIR / f"t4_{args.tag_a}_full.pt", MODEL_DIR / f"t4_{args.tag_b}_full.pt")
        np.savez_compressed(pf, test_probs=task4.predict_paths(m, test.path.values, size, tta=True), test_file_name=names)
        print("full refit: soup written", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
