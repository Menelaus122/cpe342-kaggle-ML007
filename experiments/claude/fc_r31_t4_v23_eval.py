"""Round 31, Task 4: a candidate member against the v23 ensemble, view by view (CPU; training images only).

    python fc_r31_t4_v23_eval.py cand [cand ...]     (run em_r30_t4_hardviews.py cand first: it writes the candidate's views)

Base = v23: the eight v20 members weighted as in T4_MEMBER_W (DINOv2 x8, the five HUD-dropout members x3, ConvNeXt-T / DeiT-III x1),
v17 rule (argmax p * pi^-0.75), uniform class mix, members only; the 10% frozen-feature probe on its three views is printed
separately, as in em_r30_t4_hardviews.py.
Views: full, zoom70, HUD hidden (centre_grey), gamma 3.5 (dark_g35), 75% grid (grid75) from outputs/r19_views and outputs/r30_views,
plus the real dark / low-contrast frames: training images in the lowest decile of grey-level contrast or of brightness
(outputs/r30/t4_errors.npz, measured by ek_r30_t4_errors.py on the training images), scored on the full view. When the candidate has
views for only some folds, every score uses those folds only.
Variants: the candidate added at a fixed weight w (dinov2_v2 stays at 8); in place of dinov2_v2 (weight 8); and the pair of weights
(dinov2_v2, candidate) in {0, 4, 8} x {0, 2, 4, 8} chosen nested (on the other folds by the hard-view mean, scored on the held-out
fold; needs at least 3 folds).
Rules: the hard-view rule of 2026-11-16 (hard-view mean >= +0.0015 and full >= -0.0005); the generic rule on the hard-view mean
(pooled > 0, >= 3/5 folds, no fold below -0.0005); and the v24 check the user asked for on 2026-11-16 morning, fixed before the
all-fold run: the gain has to reach beyond the 75% grid, i.e. pooled delta > 0 on zoom70, on HUD hidden and on the dark /
low-contrast frames, with full >= -0.0005 (whether full itself is positive is printed too).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ALL5[1:]
CANDS = sys.argv[1:]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.full((len(y), 5), np.nan) for v in ALL5} for t in V20 + CANDS}
have = {}
for t in P:
    have[t] = []
    for f, va in enumerate(folds):
        if not (V30 / f"{t}_fold{f}.npz").exists():
            continue
        have[t].append(f)
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
    assert t in CANDS or have[t] == list(range(5)), f"{t}: views missing for folds {sorted(set(range(5)) - set(have[t]))}"
zp = np.load("experiments/claude/outputs/r27/t4_probe_views.npz")
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
SCORE = ALL5 + ["dark/low-contrast"]
VIEW_OF = {**{v: v for v in ALL5}, "dark/low-contrast": "full"}


def labels(wmap, probe=False):
    s = sum(wmap.values())
    out = {}
    for v in ALL5:
        p = sum(w * P[t][v] for t, w in wmap.items() if w > 0) / s
        if probe and v in ("full", "zoom70", "centre_grey"):
            p = 0.9 * p + 0.1 * zp[f"probe-aug_{v}"]
        out[v] = (np.nan_to_num(p, nan=0.0) * pi ** -0.75).argmax(1)
    return out


def deltas(lab, base, m):
    """per scored view: balanced macro-F1 delta on the rows in m (the dark / low-contrast entry on the full view)."""
    return {k: bal(lab[VIEW_OF[k]], m & dark if k == "dark/low-contrast" else m) - bal(base[VIEW_OF[k]], m & dark if k == "dark/low-contrast" else m)
            for k in SCORE}


base = labels(V23_W)
base_p = labels(V23_W, probe=True)
allm = np.ones(len(y), bool)
print(f"{len(y)} training images; dark / low-contrast frames (lowest contrast or brightness decile): {dark.sum()} ({dark.mean():.1%})")
print("v23 members (uniform mix, rule): " + ", ".join(f"{k} {bal(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE))
rng = np.random.default_rng(0)


def report(label, lab, F, lab_p=None):
    m = np.isin(fo, F)
    d = deltas(lab, base, m)
    hard = np.mean([d[v] for v in HARD])
    pf = {f: np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in F}
    pdk = {f: bal(lab["full"], (fo == f) & dark) - bal(base["full"], (fo == f) & dark) for f in F}
    idx = np.where(m)[0]
    boots = [rng.choice(idx, len(idx)) for _ in range(200)]
    ci = np.quantile([np.mean([bal(lab[v], b) - bal(base[v], b) for v in HARD]) for b in boots], [0.025, 0.975])
    ok_user = hard >= 0.0015 and d["full"] >= -0.0005
    ok_gen = hard > 0 and sum(p > 0 for p in pf.values()) >= min(3, len(F)) and min(pf.values()) >= -0.0005
    ok_v24 = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    print(f"  {label}")
    print(f"    hard-view mean {hard:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], folds +{sum(p > 0 for p in pf.values())}/{len(F)} "
          f"({' '.join(f'{p:+.4f}' for p in pf.values())}); hard mean without the grid {np.mean([d[v] for v in HARD[:-1]]):+.4f}")
    print("    per view: " + ", ".join(f"{k} {d[k]:+.4f}" for k in SCORE)
          + f" | dark / low-contrast by fold: {' '.join(f'{p:+.4f}' for p in pdk.values())}")
    if lab_p is not None:
        dp = {v: bal(lab_p[v], m) - bal(base_p[v], m) for v in ("full", "zoom70", "centre_grey")}
        print("    with the 10% probe: " + ", ".join(f"{v} {x:+.4f}" for v, x in dp.items()))
    print(f"    -> hard-view rule {'PASS' if ok_user else 'fail'}, generic rule {'PASS' if ok_gen else 'fail'}, "
          f"v24 check (beyond the grid) {'PASS' if ok_v24 else 'fail'}; full {'positive' if d['full'] > 0 else 'not positive'}", flush=True)
    return pf


for c in CANDS:
    F = have[c]
    m = np.isin(fo, F)
    print(f"\ncandidate {c}: views for folds {F}")
    for k in SCORE:
        mm = m & dark if k == "dark/low-contrast" else m
        print(f"  alone, {k}: {bal((np.nan_to_num(P[c][VIEW_OF[k]]) * pi ** -0.75).argmax(1), mm):.4f} "
              f"(dinov2_v2 {bal((P['dinov2_v2'][VIEW_OF[k]] * pi ** -0.75).argmax(1), mm):.4f})")
    for w in (2.0, 4.0, 8.0):
        report(f"added at weight {w:g} (dinov2_v2 8)", labels({**V23_W, c: w}), F, labels({**V23_W, c: w}, probe=True))
    rep = {**V23_W, "dinov2_v2": 0.0, c: 8.0}
    report("in place of dinov2_v2 (weight 8)", labels(rep), F, labels(rep, probe=True))
    if len(F) >= 3:
        grid = [g for g in itertools.product((0.0, 4.0, 8.0), (0.0, 2.0, 4.0, 8.0)) if g != (0.0, 0.0)]
        labs = {g: labels({**V23_W, "dinov2_v2": g[0], c: g[1]}) for g in grid}
        fh = {g: {f: np.mean([bal(labs[g][v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in F} for g in grid}
        nested, chosen = {v: base[v].copy() for v in ALL5}, []
        for f in F:
            g = max(grid, key=lambda g_: np.mean([fh[g_][k] for k in F if k != f]))
            chosen.append(g)
            for v in ALL5:
                nested[v][fo == f] = labs[g][v][fo == f]
        report(f"weights (dinov2_v2, {c}) nested, picks {chosen}", nested, F)
