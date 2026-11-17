"""Round 38, Task 4, step 2: full WiSE-FT views of an LP-FT member (GPU inference only; training images only).

    python gn_r38_t4_wiseft_full.py TAG --model NAME --h H --w W [--alphas 0 0.5 0.7 0.85] [--folds 0 1 2 3 4] [--member]

For a member trained with `scripts/train_task4.py --lp-ft` (fine-tuning started from a logistic probe head fitted on the frozen
pretrained features; head saved as outputs/models/t4_<TAG>_fold<f>_lphead.pt), WiSE-FT (Wortsman et al., CVPR 2022) interpolates
every weight, the classifier head included, between the two ends of that run:
    theta(alpha) = alpha * theta_fine-tuned + (1 - alpha) * theta_0,   theta_0 = pretrained backbone + LP head.
Round 37's WiSE-FT on dinov2_v2 had no matching head for the pretrained end (it kept the fine-tuned head on every alpha, gh_r37);
with LP-FT, theta_0 is a complete classifier, which is the setting the method was made for. alpha = 1 is the member itself (its
views come from em_r30_t4_hardviews.py), alpha = 0 is the frozen backbone with the LP head.
Each fold's interpolated model predicts its own validation images under the five scoring views (full, zoom70, HUD hidden, gamma 3.5,
75% grid; applied at the model's input size, h-flip averaged, exactly as em_r30_t4_hardviews.py), written to
experiments/claude/outputs/r30_views/<TAG>_wise<alpha x 100>_fold<f>.npz, so fc_r31_t4_v23_eval.py scores each alpha like any
candidate:  python fc_r31_t4_v23_eval.py TAG TAG_wise85 TAG_wise70 TAG_wise50
--member: instead of the views, the member files of each alpha in the layout of scripts/train_task4.py (inference only):
outputs/preds/t4_<TAG>_wise<alpha x 100>_fold<f>.npz (val_idx, full-view val_probs, test_probs; h-flip TTA) and ..._full.npz (test
probabilities of the interpolated full Train+Val refit, from t4_<TAG>_full.pt and its LP head).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
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


def theta(pre, ck, hd):
    """(fine-tuned state dict, theta_0 = pretrained weights with the LP head in the classifier)."""
    ft = torch.load(ck, map_location="cpu")
    assert set(ft) == set(pre), "state-dict keys differ"
    head = torch.load(hd)
    clf = [k for k in pre if k.endswith(".weight") and tuple(pre[k].shape) == tuple(head["weight"].shape)
           and k[:-len(".weight")] + ".bias" in pre and tuple(pre[k[:-len(".weight")] + ".bias"].shape) == (5,)]
    assert len(clf) == 1, f"classifier weight not found uniquely: {clf}"
    t0 = dict(pre)
    t0[clf[0]], t0[clf[0][:-len(".weight")] + ".bias"] = head["weight"].float(), head["bias"].float()
    return ft, t0


def interpolated(name, size, ft, t0, a):
    sd = {k: ((a * v.float() + (1 - a) * t0[k]).to(v.dtype) if v.is_floating_point() else v) for k, v in ft.items()}
    m = task4.create_model(name, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(sd)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--model", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.85])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--member", action="store_true", help="write member files (val + test probabilities) instead of the views")
    args = ap.parse_args()
    size = (args.h, args.w)
    train, _ = task4.load_frames()
    y = train.label.values
    folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
    V30.mkdir(parents=True, exist_ok=True)
    pre = None
    if args.member:
        test = task4.load_frames()[1]
        names = np.array(test.file_name.tolist(), dtype="U")
        for f in args.folds + ["full"]:
            nm = f"fold{f}" if f != "full" else "full"
            ck, hd = MODEL_DIR / f"t4_{args.tag}_{nm}.pt", MODEL_DIR / f"t4_{args.tag}_{nm}_lphead.pt"
            todo = [a for a in args.alphas if not (PRED_DIR / f"t4_{args.tag}_wise{round(a * 100):02d}_{nm}.npz").exists()]
            if not todo or not (ck.exists() and hd.exists()):
                print(f"{nm}: {'done' if not todo else 'no checkpoint / LP head, skipped'}", flush=True)
                continue
            if pre is None:
                pre = {k: v.float() for k, v in task4.create_model(args.model, pretrained=True, img_size=size).state_dict().items()}
            ft, t0 = theta(pre, ck, hd)
            for a in todo:
                m = interpolated(args.model, size, ft, t0, a)
                tp = task4.predict_paths(m, test.path.values, size, tta=True)
                out = PRED_DIR / f"t4_{args.tag}_wise{round(a * 100):02d}_{nm}.npz"
                if f == "full":
                    np.savez_compressed(out, test_probs=tp, test_file_name=names)
                else:
                    va = folds[f]
                    vp = task4.predict_array(m, task4.load_uint8(list(train.path.values[va])), size, tta=True)
                    np.savez_compressed(out, val_idx=va, val_probs=vp, test_probs=tp, test_file_name=names)
                print(f"{out.name} written", flush=True)
                del m
                torch.cuda.empty_cache()
        return
    for f in args.folds:
        ck, hd = MODEL_DIR / f"t4_{args.tag}_fold{f}.pt", MODEL_DIR / f"t4_{args.tag}_fold{f}_lphead.pt"
        todo = [a for a in args.alphas if not (V30 / f"{args.tag}_wise{round(a * 100):02d}_fold{f}.npz").exists()]
        if not todo:
            continue
        if not (ck.exists() and hd.exists()):
            print(f"fold {f}: no checkpoint / LP head yet, skipped", flush=True)
            continue
        if pre is None:
            pre = {k: v.float() for k, v in task4.create_model(args.model, pretrained=True, img_size=size).state_dict().items()}
        ft, t0 = theta(pre, ck, hd)
        arr = task4.load_uint8(list(train.path.values[folds[f]]))
        for a in todo:
            m = interpolated(args.model, size, ft, t0, a)
            res = {v: [] for v in ALL5}
            with torch.no_grad():
                for i in range(0, len(arr), 128):
                    x0 = task4.to_gpu_batch(arr[i:i + 128], size)
                    for v in ALL5:
                        x = FN[v](x0)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                                       + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                        res[v].append(p.cpu().numpy())
            np.savez(V30 / f"{args.tag}_wise{round(a * 100):02d}_fold{f}.npz", **{v: np.concatenate(r) for v, r in res.items()})
            acc = np.mean(np.concatenate(res["full"]).argmax(1) == y[folds[f]])
            print(f"fold {f}, alpha {a:.2f}: views written; full-view accuracy {acc:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        del arr, ft


if __name__ == "__main__":
    main()
