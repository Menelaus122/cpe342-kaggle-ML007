"""Round 30, Task 4: screen of probability-combination rules on fold 0 under the hard-view criterion (CPU, training images only).

    python eo_r30_t4_combine_screen.py

The eight v20 members' fold-0 predictions under six views (outputs/r22_views: full, zoom70, centre_grey = HUD hidden, dark_g35 =
gamma 3.5, grid75 = 75% hidden grid) are combined by:
  arithmetic mean (v20, DINOv2 x3)            the reference
  geometric mean  (mean of log-probabilities = logit-space averaging), same weights
  power means p = 0.5 and p = 2
  temperature-calibrated geometric mean       each member's temperature fitted on folds 1-4 (its cached OOF full-frame
                                              probabilities, log-loss), i.e. nothing from fold 0 is used to fit it
  calibrated arithmetic mean
Each under the v17 rule (argmax p * pi^-0.75) at a uniform class mix. A rule passes the screen if the hard-view mean (zoom70, HUD
hidden, gamma 3.5, 75% grid) rises by >= +0.0015 with full frames >= -0.0005; a pass is then re-checked on all five folds.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
W = np.array([3.0 if t == "dinov2_v2" else 1.0 for t in V20])
VIEWS = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = VIEWS[1:]
train, _ = task4.load_frames()
y = train.label.values
va = np.load(PRED_DIR / "t4_cnxt_fold0.npz")["val_idx"]
yv = y[va]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[yv]
P = {t: np.load(f"experiments/claude/outputs/r22_views/{t}_fold0.npz") for t in V20}
# temperatures from folds 1-4 only (cached OOF full-frame probabilities, h-flip TTA)
T = {}
for t in V20:
    lp, yy = [], []
    for f in range(1, 5):
        z = np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")
        lp.append(np.log(np.clip(z["val_probs"], 1e-12, 1)))
        yy.append(y[z["val_idx"]])
    lp, yy = np.concatenate(lp), np.concatenate(yy)

    def nll(tau):
        s = lp / tau
        s = s - s.max(1, keepdims=True)
        return -np.mean(s[np.arange(len(yy)), yy] - np.log(np.exp(s).sum(1)))
    T[t] = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded").x
print("temperatures fitted on folds 1-4: " + ", ".join(f"{t} {T[t]:.2f}" for t in V20))


def norm(L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def combine(v, how):
    ps = [np.clip(P[t][v], 1e-12, 1) for t in V20]
    if how == "calibrated arithmetic" or how == "calibrated geometric":
        ps = [norm(np.log(p) / T[t]) for p, t in zip(ps, V20)]
    w = W / W.sum()
    if how in ("arithmetic", "calibrated arithmetic"):
        return sum(wi * p for wi, p in zip(w, ps))
    if how in ("geometric", "calibrated geometric"):
        return norm(sum(wi * np.log(p) for wi, p in zip(w, ps)))
    q = {"power 0.5": 0.5, "power 2": 2.0}[how]
    m = sum(wi * p ** q for wi, p in zip(w, ps)) ** (1 / q)
    return m / m.sum(1, keepdims=True)


score = lambda p: f1_score(yv, (p * pi ** -0.75).argmax(1), average="macro", sample_weight=wb)
base = {v: score(combine(v, "arithmetic")) for v in VIEWS}
print("v20 arithmetic mean, fold 0, uniform mix: " + "  ".join(f"{v} {base[v]:.4f}" for v in VIEWS))
for how in ["geometric", "power 0.5", "power 2", "calibrated geometric", "calibrated arithmetic"]:
    s = {v: score(combine(v, how)) for v in VIEWS}
    hard = np.mean([s[v] - base[v] for v in HARD])
    ok = hard >= 0.0015 and s["full"] - base["full"] >= -0.0005
    print(f"  {how:22s} hard mean {hard:+.4f}  full {s['full'] - base['full']:+.4f} | "
          + "  ".join(f"{v} {s[v] - base[v]:+.4f}" for v in HARD) + f"  -> {'PASS screen' if ok else 'fail'}")
