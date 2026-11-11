"""Round 13, item 3: can a label-coding mismatch explain Task 4 scoring at chance on the leaderboard?

Uses only the training-set out-of-fold predictions. Each hypothesis is simulated by relabelling our OOF predictions:
identity (datacard codes), the order quoted in the brief, and all 120 permutations of the five codes.
A permutation with f fixed points can only score about f/5 x 0.99, so it is easy to see which ones land near the observed 0.21.
Nothing is uploaded and no test labels are involved.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import config as C  # noqa: E402
from src import task4  # noqa: E402
from src.metrics import macro_f1, optimize_class_weights  # noqa: E402

frames = task4.load_frames()
tr = frames[0]
print("train columns", list(tr.columns), "rows", len(tr))
y = tr.label.values
tags = [t for t in ("cnxt", "deit3", "dino") if all((C.PRED_DIR / f"t4_{t}_fold{f}.npz").exists() for f in range(5))]
oof = {t: np.zeros((len(y), 5)) for t in tags}
rows = []
for t in tags:
    for f in range(5):
        z = np.load(C.PRED_DIR / f"t4_{t}_fold{f}.npz")
        oof[t][z["val_idx"]] = z["val_probs"]
        rows.append([t, f, f1_score(y[z["val_idx"]], z["val_probs"].argmax(1), average="macro")])
pf = pd.DataFrame(rows, columns=["backbone", "fold", "F1"]).pivot(index="backbone", columns="fold", values="F1")
pf["OOF"] = [macro_f1(y, oof[t].argmax(1)) for t in pf.index]
print("\nTask 4 macro-F1 per fold (h-flip TTA included)\n", pf.round(4).to_string())
ens = np.mean(list(oof.values()), 0)
pred = ens.argmax(1)
print(f"\nensemble of {tags}: OOF macro-F1 {macro_f1(y, pred):.4f}, accuracy {(pred == y).mean():.4f}, minimum class F1 "
      f"{f1_score(y, pred, average=None).min():.4f}")
w, s = optimize_class_weights(y, ens)
print(f"with class multipliers (in-sample): {s:.4f}")

print("\nclasses in code order:", dict(enumerate(C.T4_CLASSES)))
brief = {"7-11k": 4, "fiveN": 2, "freefried": 1, "pube": 0, "roblock": 3}
perm_brief = np.array([brief[c] for c in C.T4_CLASSES])
print(f"\nidentity (datacard codes): {macro_f1(y, pred):.4f}")
print(f"order quoted in the brief (0 PUBE, 1 Free Fried, 2 FiveN, 3 Roblock, 4 7-Eleven): our code c is scored as code {perm_brief.tolist()}: "
      f"macro-F1 {macro_f1(y, perm_brief[pred]):.4f}  (fixed points: {(perm_brief == np.arange(5)).sum()})")
res = []
for p in itertools.permutations(range(5)):
    p = np.array(p)
    res.append([int((p == np.arange(5)).sum()), macro_f1(y, p[pred])])
R = pd.DataFrame(res, columns=["fixed points", "macro-F1"])
print("\nall 120 code permutations, grouped by how many classes stay in place\n",
      R.groupby("fixed points")["macro-F1"].agg(["count", "min", "mean", "max"]).round(3).to_string())
rng = np.random.default_rng(0)
sh = [macro_f1(y, rng.permutation(pred)) for _ in range(200)]
print(f"\nshuffled row order (keys do not line up): macro-F1 {np.mean(sh):.4f} +- {np.std(sh):.4f}")
