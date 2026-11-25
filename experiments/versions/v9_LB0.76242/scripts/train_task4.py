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
    args = ap.parse_args()
    aug_kw = {"scale": tuple(args.aug_scale)} if args.aug_scale else None

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
    if args.full:
        out_path = PRED_DIR / f"t4_{args.tag}_full.npz"
        if out_path.exists():
            log("full refit: cached, skipping")
            return
        log(f"full refit on all {len(y)} training images (same recipe, no hold-out)")
        model, hist = task4.train_fold(args.model, arr, y, np.arange(len(y)), [], img_size=size, epochs=args.epochs,
                                       bs=args.bs, lr=args.lr, seed=SEED + 100, log=log,
                                       ckpt_path=MODEL_DIR / f"t4_{args.tag}_full.pt", aug_kw=aug_kw)
        tp = task4.predict_paths(model, test.path.values, size, tta=True)
        np.savez_compressed(out_path, test_probs=tp)
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
        model, hist = task4.train_fold(args.model, arr, y, tr, va, img_size=size, epochs=args.epochs, bs=args.bs,
                                       lr=args.lr, seed=SEED + f, log=log,
                                       ckpt_path=MODEL_DIR / f"t4_{args.tag}_fold{f}.pt", aug_kw=aug_kw)
        vp = task4.predict_array(model, arr[va], size, tta=True)
        tp = task4.predict_paths(model, test.path.values, size, tta=True)
        np.savez_compressed(out_path, val_idx=va, val_probs=vp, test_probs=tp)
        (log_dir / f"t4_{args.tag}_fold{f}_history.json").write_text(json.dumps(hist, indent=1))
        log(f"fold {f}: TTA macro-F1 {f1_score(y[va], vp.argmax(1), average='macro'):.4f}")
        del model


if __name__ == "__main__":
    main()
