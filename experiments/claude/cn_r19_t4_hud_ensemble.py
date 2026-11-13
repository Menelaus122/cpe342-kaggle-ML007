"""Round 19c: is the six-backbone Task 4 ensemble (two normal + four HUD-dropout) better than the round-17 pair? Training labels only.

    python cn_r19_t4_hud_ensemble.py

Reads the stratified 5-fold caches written by scripts/train_task4.py (same folds for every tag) and reports:
  * OOF macro-F1 / accuracy per backbone and for several probability averages;
  * paired bootstrap (rows resampled, 2000x) of six vs pair, on macro-F1 and accuracy, plus per-fold wins;
  * the nested check of class multipliers for the six-model blend (chosen on 4 folds, scored on the 5th);
  * how many test labels change vs submission_v14.csv, split into the effect of dropping the v14 multipliers and the effect of the
    new backbones (test inputs blended with the full-data models exactly as the notebook does). No test label is used or inferred.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402

PAIR = ["cnxt", "deit3"]
HUD = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
SIX = PAIR + HUD
FULL_W = 0.5
train, test = task4.load_frames()
y = train.label.values
names = np.array(test.file_name.tolist(), dtype="U")
mf1 = lambda p: f1_score(y, p.argmax(1), average="macro")
acc = lambda p: (p.argmax(1) == y).mean()

oof, tst, fin, folds = {}, {}, {}, None
for t in SIX:
    oof[t], tst[t] = np.zeros((len(y), 5)), np.zeros((len(test), 5))
    fl = []
    for f in range(5):
        z = np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")
        assert (z["test_file_name"] == names).all(), f"{t} fold {f}: test order"
        oof[t][z["val_idx"]] = z["val_probs"]
        tst[t] += z["test_probs"] / 5
        fl.append(z["val_idx"])
    folds = folds or fl
    zf = np.load(PRED_DIR / f"t4_{t}_full.npz")
    assert (zf["test_file_name"] == names).all(), f"{t} full: test order"
    fin[t] = (1 - FULL_W) * tst[t] + FULL_W * zf["test_probs"]

print("per backbone (OOF, stratified 5-fold):")
for t in SIX:
    print(f"  {t:12s} macro-F1 {mf1(oof[t]):.4f}  acc {acc(oof[t]):.4f}")
avg = lambda ts, d=oof: np.mean([d[t] for t in ts], 0)
combos = {"pair (v14)": PAIR, "four HUD only": HUD, "six (planned)": SIX, "cnxt + cnxt_hud": ["cnxt", "cnxt_hud"],
          "pair + cnxt_hud + deit3_hud": PAIR + ["cnxt_hud", "deit3_hud"]}
print("\nprobability averages:")
for k, ts in combos.items():
    print(f"  {k:28s} macro-F1 {mf1(avg(ts)):.4f}  acc {acc(avg(ts)):.4f}")

p2, p6 = avg(PAIR), avg(SIX)
rng = np.random.default_rng(0)
c2, c6 = p2.argmax(1), p6.argmax(1)
bf, ba = [], []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    bf.append(f1_score(y[i], c6[i], average="macro") - f1_score(y[i], c2[i], average="macro"))
    ba.append((c6[i] == y[i]).mean() - (c2[i] == y[i]).mean())
print(f"\nsix - pair: macro-F1 {mf1(p6) - mf1(p2):+.4f} 95% CI [{np.quantile(bf, .025):+.4f}, {np.quantile(bf, .975):+.4f}]"
      f" | acc {acc(p6) - acc(p2):+.4f} 95% CI [{np.quantile(ba, .025):+.4f}, {np.quantile(ba, .975):+.4f}]")
wins = [f1_score(y[v], c6[v], average="macro") - f1_score(y[v], c2[v], average="macro") for v in folds]
print("  per-fold macro-F1 delta:", " ".join(f"{d:+.4f}" for d in wins), f"-> six better in {sum(d > 0 for d in wins)}/5")
print(f"  OOF rows where the two disagree: {(c2 != c6).sum()} (six right {((c6 == y) & (c2 != y)).sum()}, pair right {((c2 == y) & (c6 != y)).sum()})")

W, s_in = optimize_class_weights(y, p6)
nest = np.zeros(len(y), int)
for v in folds:
    tr = np.setdiff1d(np.arange(len(y)), v)
    nest[v] = apply_class_weights(p6[v], optimize_class_weights(y[tr], p6[tr])[0])
gain = f1_score(y, nest, average="macro") - mf1(p6)
print(f"\nclass multipliers on six: in-sample {s_in:.4f}, nested {f1_score(y, nest, average='macro'):.4f} (gain {gain:+.4f}) "
      f"-> {'kept' if gain >= 0.0005 else 'dropped (argmax)'}; W = {np.round(W, 3)}")

v14c = pd.read_csv("submission_v14.csv").task4.values.astype(int)   # integer class codes, rows in test.csv order
pair_fin, six_fin = avg(PAIR, fin), avg(SIX, fin)
pair_arg, six_arg = pair_fin.argmax(1), six_fin.argmax(1)
print(f"\ntest labels ({len(names)} rows):")
print(f"  v14 vs pair argmax (effect of dropping the v14 multipliers): {(v14c != pair_arg).sum()} changed")
print(f"  pair argmax vs six argmax (effect of the new backbones):     {(pair_arg != six_arg).sum()} changed")
print(f"  v14 vs six argmax (total):                                   {(v14c != six_arg).sum()} changed")
print("  predicted class shares  v14:", np.round(np.bincount(v14c, minlength=5) / len(v14c), 4),
      "\n                          six:", np.round(np.bincount(six_arg, minlength=5) / len(six_arg), 4),
      "\n                   train prior:", np.round(np.bincount(y, minlength=5) / len(y), 4))
conf = lambda p: p.max(1)
print(f"  mean max-probability on test (label-free diagnostic): pair {conf(pair_fin).mean():.4f}  six {conf(six_fin).mean():.4f};"
      f" share below 0.6: pair {(conf(pair_fin) < 0.6).mean():.4f}  six {(conf(six_fin) < 0.6).mean():.4f}")
