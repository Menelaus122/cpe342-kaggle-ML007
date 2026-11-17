"""Task 4 K-fold fine-tuning runner (GPU). Each fold's OOF/test probabilities and EMA checkpoint
are cached in outputs/ so the master notebook can load them without retraining.

Example:
    python -u scripts/train_task4.py --model convnext_tiny.in12k_ft_in1k --tag cnxt --folds 0 1 2 3 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_DIR, OUT, PRED_DIR, SEED  # noqa: E402
from src import task4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_tiny.in12k_ft_in1k")
    ap.add_argument("--tag", default="cnxt")
    ap.add_argument("--h", type=int, default=160)
    ap.add_argument("--w", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--full", action="store_true",
                    help="after validation: train one final model on ALL training images (Train+Val combined)")
    ap.add_argument("--aug-scale", type=float, nargs=2, default=None,
                    help="zoom range of the affine augmentation (default 0.85 1.15)")
    ap.add_argument("--p-hud", type=float, default=0.0,
                    help="HUD dropout probability (round 19): hide the bottom strip / top + corners / all borders, or zoom into the centre")
    ap.add_argument("--p-grid", type=float, default=0.0, help="heavy occlusion probability (round 22: GridMask / one block hiding 65-85%%)")
    ap.add_argument("--p-gray", type=float, default=None, help="random greyscale probability (default 0.05; round 27: 0.3)")
    ap.add_argument("--p-lowlight", type=float, default=0.0,
                    help="round 40: near-black frame (brightness x0.05-0.30 or gamma 2-4) + Gaussian noise sd 0.05-0.15")
    ap.add_argument("--p-chshuffle", type=float, default=0.0, help="round 40: random RGB channel order")
    ap.add_argument("--train-seed-offset", type=int, default=0,
                    help="round 40: added to the training seeds only (the folds keep SEED), for a second run of a recipe")
    ap.add_argument("--p-jitter", type=float, default=None, help="colour jitter probability (default 0.5)")
    ap.add_argument("--jitter-gain", type=float, default=None,
                    help="widens the brightness / contrast / saturation ranges of the colour jitter (default 1.0; round 27: 1.6)")
    ap.add_argument("--layer-decay", type=float, default=None, help="layer-wise lr decay for ViTs (e.g. 0.75)")
    ap.add_argument("--warmup-epochs", type=int, default=1, help="linear warm-up epochs before the cosine decay (round 28: 2 for ViT-B)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing: same training, less GPU memory (ConvNeXt-Small at batch 64 on an 8 GB GPU)")
    ap.add_argument("--sam-rho", type=float, default=0.0, help="round 30: sharpness-aware minimisation radius (0 = off; 0.05 = Foret et al.)")
    ap.add_argument("--mixstyle", type=int, default=0, help="round 30: MixStyle after the first n stages of a CNN (0 = off)")
    ap.add_argument("--accum", type=int, default=1, help="round 37: gradient accumulation steps (each step still sees --bs images)")
    ap.add_argument("--lp-ft", action="store_true",
                    help="round 38: LP-FT, start fine-tuning from a logistic probe head fitted on the frozen pretrained features of the "
                         "fold's training images (saved as models/t4_<tag>_<fold>_lphead.pt; the WiSE-FT start point)")
    ap.add_argument("--lp-c", type=float, default=0.03, help="LP-FT: inverse L2 strength of the probe head (the frozen probe's C)")
    args = ap.parse_args()
    aug_kw = {**({"scale": tuple(args.aug_scale)} if args.aug_scale else {}), **({"p_hud": args.p_hud} if args.p_hud else {}),
              **({"p_grid": args.p_grid} if args.p_grid else {}), **({"p_gray": args.p_gray} if args.p_gray is not None else {}),
              **({"p_jitter": args.p_jitter} if args.p_jitter is not None else {}),
              **({"jitter_gain": args.jitter_gain} if args.jitter_gain is not None else {}),
              **({"p_lowlight": args.p_lowlight} if args.p_lowlight else {}),
              **({"p_chshuffle": args.p_chshuffle} if args.p_chshuffle else {})} or None

    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_f = open(log_dir / f"t4_{args.tag}.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    train, test = task4.load_frames()
    y = train.label.values
    folds = list(StratifiedKFold(args.n_folds, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    log(f"== {args.model} {args.h}x{args.w} epochs={args.epochs} bs={args.bs} lr={args.lr} aug={aug_kw or 'default'} (GPU augmentation)")
    t0 = time.time()
    arr = task4.load_uint8(list(train.path))
    log(f"decoded {arr.shape} uint8 ({arr.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    size = (args.h, args.w)

    def lp_head(rows, name, va=None):
        """LP-FT: the probe head for these training rows (cached); with va, also the probe's own validation score (theta_0)."""
        if not args.lp_ft:
            return None
        path = MODEL_DIR / f"t4_{args.tag}_{name}_lphead.pt"
        if path.exists():
            return torch.load(path)
        t1 = time.time()
        m = task4.create_model(args.model, img_size=size).cuda().to(memory_format=torch.channels_last)
        head = task4.fit_lp_head(m, arr, y, rows, size, C=args.lp_c)
        torch.save(head, path)
        msg = f"{name}: LP head fitted on {len(rows)} images x 3 views (C {args.lp_c}) in {time.time() - t1:.0f}s"
        if va is not None and len(va):
            vp = task4.predict_array(task4.set_head(m, head), arr[va], size, tta=True)
            msg += f"; pretrained backbone + LP head (theta_0) val macro-F1 {f1_score(y[va], vp.argmax(1), average='macro'):.4f}"
        log(msg)
        del m
        torch.cuda.empty_cache()
        return head
    if args.full:
        out_path = PRED_DIR / f"t4_{args.tag}_full.npz"
        if out_path.exists():
            log("full refit: cached, skipping")
            return
        log(f"full refit on all {len(y)} training images (same recipe, no hold-out)")
        init_head = lp_head(np.arange(len(y)), "full")
        model, hist = task4.train_fold(args.model, arr, y, np.arange(len(y)), [], img_size=size, epochs=args.epochs,
                                       bs=args.bs, lr=args.lr, seed=SEED + 100 + args.train_seed_offset, log=log,
                                       ckpt_path=MODEL_DIR / f"t4_{args.tag}_full.pt", aug_kw=aug_kw, grad_ckpt=args.grad_ckpt,
                                       layer_decay=args.layer_decay, warmup_epochs=args.warmup_epochs,
                                       sam_rho=args.sam_rho, mixstyle_stages=args.mixstyle, accum=args.accum, init_head=init_head)
        tp = task4.predict_paths(model, test.path.values, size, tta=True)
        np.savez_compressed(out_path, test_probs=tp, test_file_name=np.array(test.file_name.tolist(), dtype="U"))
        (log_dir / f"t4_{args.tag}_full_history.json").write_text(json.dumps(hist, indent=1))
        log("full refit: done")
        return
    for f in args.folds:
        out_path = PRED_DIR / f"t4_{args.tag}_fold{f}.npz"
        if out_path.exists():
            log(f"fold {f}: cached, skipping")
            continue
        tr, va = folds[f]
        log(f"fold {f}: train {len(tr)} val {len(va)}")
        init_head = lp_head(tr, f"fold{f}", va)
        model, hist = task4.train_fold(args.model, arr, y, tr, va, img_size=size, epochs=args.epochs, bs=args.bs,
                                       lr=args.lr, seed=SEED + f + args.train_seed_offset, log=log,
                                       ckpt_path=MODEL_DIR / f"t4_{args.tag}_fold{f}.pt", aug_kw=aug_kw, grad_ckpt=args.grad_ckpt,
                                       layer_decay=args.layer_decay, warmup_epochs=args.warmup_epochs,
                                       sam_rho=args.sam_rho, mixstyle_stages=args.mixstyle, accum=args.accum, init_head=init_head)
        vp = task4.predict_array(model, arr[va], size, tta=True)
        tp = task4.predict_paths(model, test.path.values, size, tta=True)
        np.savez_compressed(out_path, val_idx=va, val_probs=vp, test_probs=tp,
                            test_file_name=np.array(test.file_name.tolist(), dtype="U"))
        (log_dir / f"t4_{args.tag}_fold{f}_history.json").write_text(json.dumps(hist, indent=1))
        log(f"fold {f}: TTA macro-F1 {f1_score(y[va], vp.argmax(1), average='macro'):.4f}")
        del model


if __name__ == "__main__":
    main()
