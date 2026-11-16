"""Round 30 (Jackpot loop), Task 3: calibration of the stage-1 P(y > 0) before the gate and the read-outs (training rows only, CPU).

    python fb_r30_t3_p_calibration.py

The mixture-median read-out uses q = 1 - 0.5 / p, so a mis-calibrated p shifts every positive prediction. Candidates, each fitted on
the 4 training folds of the 10-seed hurdle's OOF p and applied to the 5th: isotonic regression; Platt scaling on logit(p); and
isotonic per imputed segment. The calibrated p then drives the whole nested rebuild (gate, read-outs, blend, scale).
Reference: the same nested rebuild with the raw p. Acceptance (user, Task 3): pooled > 0, >= 3/5 folds, no fold below -0.0002.
"""
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import t3_nested as T  # noqa: E402
from src.metrics import nmae  # noqa: E402

t0 = time.time()
D = T.load_all()
y, p, folds = D["y"], D["p"], D["folds"]
seg = D["X"]["segment"].values.astype(int)
t = (y > 0).astype(int)
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
ref = T.nested_final(D, p, D["MU"])
print(f"reference nested NMAE {nmae(y, ref):.5f} [{time.time() - t0:.0f}s]", flush=True)
ll = lambda q, m: -np.mean(np.where(t[m] == 1, np.log(np.clip(q[m], 1e-6, 1)), np.log(np.clip(1 - q[m], 1e-6, 1))))
allm = np.ones(len(y), bool)
for name in ("isotonic", "Platt", "isotonic per segment"):
    pc = p.copy()
    for a, b in folds:
        if name == "isotonic":
            pc[b] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[a], t[a]).predict(p[b])
        elif name == "Platt":
            pc[b] = LogisticRegression(C=1e6).fit(lg(p[a])[:, None], t[a]).predict_proba(lg(p[b])[:, None])[:, 1]
        else:
            for s_ in np.unique(seg):
                aa, bb = a[seg[a] == s_], b[seg[b] == s_]
                pc[bb] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p[aa], t[aa]).predict(p[bb])
    cand = T.nested_final(D, pc, D["MU"])
    print(f"{name}: log-loss {ll(p, allm):.4f} -> {ll(pc, allm):.4f}; share of rows across 0.5 changed {np.mean((p > 0.5) != (pc > 0.5)):.3%}", flush=True)
    print(T.compare(D, ref, cand, name) + f" [{time.time() - t0:.0f}s]", flush=True)
