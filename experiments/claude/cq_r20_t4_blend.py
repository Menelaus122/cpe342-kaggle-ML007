"""Round 20: Task 4 blend rules scored on full frames AND HUD-free views together, with nested selection. Training images only.

    python cq_r20_t4_blend.py

Inputs: out-of-fold probabilities of the six backbones on four views (outputs/r19_views, co_r19_t4_views_allfolds.py) and, if present,
the same views zoomed to their central 75% (outputs/r20_views, cp_r20_t4_z75_views.py). 31,546 images, each predicted only by the
fold model that did not train on it.

Objective (fixed before looking at results): mean of macro-F1 on full frames and macro-F1 on the 70% centre zoom (the HUD-free view
with no grey area, so it does not favour models that saw grey/noise fill in training). The v15 upload fits an even mix of these two
(FINDINGS round 19c). centre_grey and bottom_grey are reported, not optimised.

Rules (L = legacy pair average, H = average of the four HUD-dropout backbones; v15 = mean of six = L/3 + 2H/3):
  group      P = (1 - w) L + w H
  gate       w = w_hi where the legacy pair's max probability < tau (HUD missing -> legacy unsure), else w_lo
  weights    six free non-negative model weights (Powell on the objective)
  crop       any of the above on (1 - b) P(view) + b P(zoom75 of view)
Every rule's parameters are chosen on four folds and applied to the fifth (nested), so the reported gains are out of sample.
Also prints the PUBE Mobile / Free Fried confusion under each view.
"""
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402

V19, V20 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r20_views")
PAIR = ["cnxt", "deit3"]
HUD = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
SIX = PAIR + HUD
VIEWS = ["full", "bottom_grey", "centre_grey", "zoom70"]
OBJ = ["full", "zoom70"]
y = task4.load_frames()[0].label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fold_of = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fold_of[va] = f

M = {v: np.zeros((len(SIX), len(y), 5)) for v in VIEWS}          # M[view][model, row, class]
Z = {v: np.zeros((len(SIX), len(y), 5)) for v in VIEWS}          # zoom75 of the view
have_z = all((V20 / f"{t}_fold{f}.npz").exists() for t in SIX for f in range(5))
for i, t in enumerate(SIX):
    for f, va in enumerate(folds):
        a = np.load(V19 / f"{t}_fold{f}.npz")
        b = np.load(V20 / f"{t}_fold{f}.npz") if have_z else None
        for v in VIEWS:
            M[v][i, va] = a[v]
            if have_z:
                Z[v][i, va] = b[v + "_z75"]
print(f"{len(y)} images out of fold; zoom75 views {'loaded' if have_z else 'NOT available yet (crop rules skipped)'}")
mf1 = lambda yy, p: f1_score(yy, p.argmax(1), average="macro")
pube, ff = T4_CLASS_NAMES.index("PUBE Mobile"), T4_CLASS_NAMES.index("Free Fried")


# ---------------------------------------------------------------- rules: params -> probabilities for one view
def probs(rule, prm, v):
    w6, b = prm.get("w6"), prm.get("b", 0.0)
    base = M[v] if b == 0 else (1 - b) * M[v] + b * Z[v]
    if rule in ("group", "crop_group"):
        return (1 - prm["w"]) * base[:2].mean(0) + prm["w"] * base[2:].mean(0)
    if rule in ("gate", "crop_gate"):
        L, H = base[:2].mean(0), base[2:].mean(0)
        w = np.where(L.max(1) < prm["tau"], prm["w_hi"], prm["w_lo"])[:, None]
        return (1 - w) * L + w * H
    if rule in ("weights", "crop_weights"):
        return np.tensordot(w6 / w6.sum(), base, 1)
    raise ValueError(rule)


def objective(rule, prm, rows):
    return np.mean([mf1(y[rows], probs(rule, prm, v)[rows]) for v in OBJ])


GRIDS = {
    "group": [{"w": w} for w in [0.5, 0.6, 2 / 3, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]],
    "gate": [{"tau": t, "w_lo": lo, "w_hi": hi} for t, lo, hi in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.5, 2 / 3], [0.8, 0.9, 1.0])],
}
if have_z:
    GRIDS["crop_group"] = [{"w": w, "b": b} for w, b in itertools.product([2 / 3, 0.75, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5])]
    GRIDS["crop_gate"] = [{**g, "b": b} for g, b in itertools.product(GRIDS["gate"], [0.1, 0.2, 0.3])]


def fit_weights(rows, b=0.0):
    rule = "crop_weights" if b else "weights"
    f = lambda u: -objective(rule, {"w6": np.exp(u), "b": b}, rows)
    best = min((minimize(f, u0, method="Powell", options={"xtol": 1e-2, "ftol": 1e-5, "maxfev": 400})
                for u0 in [np.zeros(6), np.log([1, 1, 2, 2, 2, 2])]), key=lambda r: r.fun)
    return {"w6": np.exp(best.x), "b": b}


def nested(rule):
    """Choose parameters on four folds, predict the fifth; returns nested probabilities per view and the chosen params."""
    out, chosen = {v: np.zeros((len(y), 5)) for v in VIEWS}, []
    for f in range(5):
        tr, te = fold_of != f, fold_of == f
        if rule in GRIDS:
            prm = max(GRIDS[rule], key=lambda p: objective(rule, p, tr))
        else:
            prm = max((fit_weights(tr, b) for b in ([0.0] if rule == "weights" else [0.2, 0.3])),
                      key=lambda p: objective(rule, p, tr))
        chosen.append(prm)
        for v in VIEWS:
            out[v][te] = probs(rule, prm, v)[te]
    return out, chosen


fmt = lambda p: "{" + ", ".join(f"{k}={np.round(x / x.sum(), 2).tolist() if k == 'w6' else round(float(x), 3)}" for k, x in p.items()) + "}"
base = {v: probs("group", {"w": 2 / 3}, v) for v in VIEWS}       # v15: plain mean of the six
pairp = {v: probs("group", {"w": 0.0}, v) for v in VIEWS}
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]


def report(name, P):
    cells, deltas = [], []
    for v in VIEWS:
        d = mf1(y, P[v]) - mf1(y, base[v])
        cells.append(f"{mf1(y, P[v]):.4f} ({d:+.4f})")
    obj_d = np.array([np.mean([f1_score(y[i], P[v][i].argmax(1), average="macro") - f1_score(y[i], base[v][i].argmax(1), average="macro")
                               for v in OBJ]) for i in boot])
    pf = [np.mean([mf1(y[fold_of == f], P[v][fold_of == f]) - mf1(y[fold_of == f], base[v][fold_of == f]) for v in OBJ]) for f in range(5)]
    print(f"{name:26s} | " + " | ".join(cells) + f" | obj {np.mean(obj_d):+.4f} [{np.quantile(obj_d, .025):+.4f}, {np.quantile(obj_d, .975):+.4f}]"
          f" {sum(d > 0 for d in pf)}/5")


print("\nmacro-F1 out of fold (delta vs v15 = mean of six); obj = mean over full + zoom70, 95% bootstrap CI, folds better")
print(f"{'rule':26s} | " + " | ".join(f"{v:^17s}" for v in VIEWS) + " |")
report("pair (v14 backbones)", pairp)
report("v15: mean of six", base)
report("four HUD only", {v: probs("group", {"w": 1.0}, v) for v in VIEWS})
for rule in list(GRIDS) + ["weights"] + (["crop_weights"] if have_z else []):
    P, ch = nested(rule)
    report(f"nested {rule}", P)
    print(f"{'':26s}   chosen per fold: " + "; ".join(fmt(p) for p in ch))

# in-sample curves (what the nested choices pick from)
print("\nin-sample, group weight w on the HUD four (v15 = 0.667):")
for p in GRIDS["group"]:
    print(f"  w={p['w']:.3f}  " + "  ".join(f"{v} {mf1(y, probs('group', p, v)):.4f}" for v in VIEWS))
if have_z:
    print("in-sample, v15 blend with a zoom75 view at weight b:")
    for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        P = {v: (1 - b) * base[v] + b * ((Z[v][:2].mean(0) + 2 * Z[v][2:].mean(0)) / 3) for v in VIEWS}
        print(f"  b={b:.1f}  " + "  ".join(f"{v} {mf1(y, P[v]):.4f}" for v in VIEWS))

print("\nPUBE Mobile / Free Fried under each view (v15 blend): errors among all errors, and the 2x2 confusion counts")
for v in VIEWS:
    c = base[v].argmax(1)
    err = c != y
    pf_ = err & np.isin(y, [pube, ff]) & np.isin(c, [pube, ff])
    cm = confusion_matrix(y, c, labels=range(5))
    print(f"  {v:12s} errors {err.sum():5d}, PUBE<->Free Fried {pf_.sum():4d} ({pf_.sum() / max(err.sum(), 1):.0%}); "
          f"true PUBE -> FF {cm[pube, ff]}, true FF -> PUBE {cm[ff, pube]}; per-class recall {np.round(cm.diagonal() / cm.sum(1), 3).tolist()}")
print("  classes:", T4_CLASS_NAMES)
