"""Round 24, Task 4: label-free bound on test accuracy from agreement between our own members.

    python dc_r24_t4_test_agreement.py

For two classifiers A, B on the same images: disagreement(A, B) <= err(A) + err(B). So if two members disagree on d of the test
images, at least one of them errs on >= d / 2. Compared with the same pairs out-of-fold (training images, where errors are known).
Uses only our cached test predictions (no test label, no training).
"""
import itertools

import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from src.config import PRED_DIR  # noqa: E402

TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
oof, tst = {}, {}
for t in TAGS:
    zs = [np.load(PRED_DIR / f"t4_{t}_fold{f}.npz") for f in range(5)]
    n = sum(len(z["val_idx"]) for z in zs)
    p = np.zeros((n, 5))
    for z in zs:
        p[z["val_idx"]] = z["val_probs"]
    oof[t] = p
    tst[t] = 0.5 * np.mean([z["test_probs"] for z in zs], 0) + 0.5 * np.load(PRED_DIR / f"t4_{t}_full.npz")["test_probs"]
ens_o, ens_t = np.mean(list(oof.values()), 0).argmax(1), np.mean(list(tst.values()), 0).argmax(1)
print("pairwise disagreement, OOF | test   (test d/2 = lower bound on the worse member's test error)")
D = []
for a, b in itertools.combinations(TAGS, 2):
    do = (oof[a].argmax(1) != oof[b].argmax(1)).mean()
    dt = (tst[a].argmax(1) != tst[b].argmax(1)).mean()
    D.append(dt)
    print(f"  {a:12s} {b:12s} {do:.4f} | {dt:.4f}")
print(f"mean pairwise disagreement on test {np.mean(D):.4f}, min {np.min(D):.4f}")
print("each member vs the 7-member ensemble, OOF | test:")
for t in TAGS:
    print(f"  {t:12s} {(oof[t].argmax(1) != ens_o).mean():.4f} | {(tst[t].argmax(1) != ens_t).mean():.4f}")
all_o = np.all([oof[t].argmax(1) == ens_o for t in TAGS], 0).mean()
all_t = np.all([tst[t].argmax(1) == ens_t for t in TAGS], 0).mean()
print(f"all seven agree: OOF {all_o:.4f} | test {all_t:.4f}")
print(f"ensemble mean max-prob: OOF {np.mean(list(oof.values()), 0).max(1).mean():.3f} | test {np.mean(list(tst.values()), 0).max(1).mean():.3f}")
