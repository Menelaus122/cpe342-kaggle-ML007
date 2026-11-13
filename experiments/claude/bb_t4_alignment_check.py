"""Task 4: the TAs confirm the convention we already use (datacard codes 0-4, ANS{i} = row i of test.csv).
So check our own chain end to end:
 1. submission column == argmax of the cached test probabilities, row by row (nothing reordered when writing)
 2. a saved checkpoint, re-run on test.csv rows by their file name, gives the cached probabilities (cache rows = test.csv rows)
 3. the same checkpoint on its held-out training fold, scored against the TEXT labels mapped with the datacard codes
"""
import numpy as np
import pandas as pd
import torch

import common  # noqa: F401
from src import task4
from src import config as C

train, test = task4.load_frames()
sub = pd.read_csv("submission_v9_try.csv")
P = {}
for tag in ("cnxt", "deit3"):
    fold = np.mean([np.load(C.PRED_DIR / f"t4_{tag}_fold{f}.npz")["test_probs"] for f in range(5)], axis=0)
    full = np.load(C.PRED_DIR / f"t4_{tag}_full.npz")["test_probs"]
    P[tag] = 0.5 * fold + 0.5 * full
ens = (P["cnxt"] + P["deit3"]) / 2
print(f"1. v9 task4 == argmax of cached ensemble probabilities: {np.mean(sub.task4.values == ens.argmax(1)):.4f} of rows")
print("   predicted class shares:", dict(zip(C.T4_CLASSES, np.round(np.bincount(sub.task4, minlength=5) / len(sub), 3))))

model = task4.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False).cuda().to(memory_format=torch.channels_last)
model.load_state_dict(torch.load(C.ROOT / "outputs/models/t4_cnxt_fold0.pt", map_location="cuda"))
rows = np.random.default_rng(0).choice(len(test), 600, replace=False)
fresh = task4.predict_paths(model, test.path.values[rows], (160, 320), tta=True)
cached = np.load(C.PRED_DIR / "t4_cnxt_fold0.npz")
print(f"2. fresh fold-0 predictions on 600 test.csv rows vs the cached fold-0 test probabilities: "
      f"same class {np.mean(fresh.argmax(1) == cached['test_probs'][rows].argmax(1)):.3f}, "
      f"max |prob diff| {np.abs(fresh - cached['test_probs'][rows]).max():.4f}; agreement with the v9 column {np.mean(fresh.argmax(1) == sub.task4.values[rows]):.3f}")
va = cached["val_idx"]
vr = np.random.default_rng(1).choice(va, 600, replace=False)
pv = task4.predict_paths(model, train.path.values[vr], (160, 320), tta=True).argmax(1)
lab = train["class"].map({c: i for i, c in enumerate(C.T4_CLASSES)}).values[vr]
print(f"3. fold-0 model on 600 held-out training images, text labels -> datacard codes {C.T4_CLASSES}: accuracy {np.mean(pv == lab):.3f}")
print("   datacard: 0 7-11 Knights, 1 FiveN, 2 Free Fried, 3 PUBE, 4 Roblock | train label values:", sorted(train["class"].unique()))
