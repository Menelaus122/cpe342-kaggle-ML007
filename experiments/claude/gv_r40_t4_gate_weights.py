"""Round 40, Task 4, step 1: a smooth brightness / contrast ramp for the DINOv2-L member, and lighter weights for the older members
(training images only; CPU; the caches of round 39).

    python gv_r40_t4_gate_weights.py

User brief of 2026-11-18 (after v30 = 0.86322). Reference = v30: v23 member weights + dinov2L_hud70 at 8, or 2 on images whose grey
mean or grey SD is at or below the 10% quantile of the training frames (thresholds per held-out fold from the other four, as in
gu_r39_t4_gated.py), then 0.65 x members + 0.35 x the seven-backbone MLP probe, v17 rule, balanced class mix; views full / zoom70 /
HUD hidden, each gated on its own image statistics (outputs/r39/t4_view_stats.npz); real dark / low-contrast frames = lowest decile of
contrast or brightness (full view).
Candidates (fixed before the run):
  S1 smooth ramp: r = min(quantile rank of the image's brightness, of its contrast) among the training frames of the other four folds;
     DINOv2-L weight = 2 + 6 * clip((r - lo) / (hi - lo), 0, 1), (lo, hi) in {(0.05, 0.25) (the user's), (0.05, 0.15), (0.10, 0.25)}
  S2 older members (v30's gate kept): (ConvNeXt-T / DeiT-III-S without HUD dropout, each HUD-dropout member) weights in
     {(0, 3), (1, 2), (0, 2)} (v30: (1, 3))
  N  nested over the six settings: per held-out fold, among the settings meeting the rule on the other four folds, the best mean of
     the zoom70 / HUD-hidden / dark deltas; none: the fold keeps v30.
Acceptance (user's rule against v30, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline setting is the one meeting the rule in-sample with the best
criterion on all five folds.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

NEW = "dinov2L_hud70"
PLAIN = ["cnxt", "deit3"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
PW = 0.35

train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [NEW]}
for t in V20 + [NEW]:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in VIEWS:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))

# per-row thresholds / ranks from the other four folds' full frames
HARD = {v: np.zeros(len(y), bool) for v in VIEWS}                    # v30's gate (q 0.10)
RANK = {v: np.zeros(len(y)) for v in VIEWS}                          # min of the brightness / contrast quantile ranks
for f in range(5):
    tr, ho = fo != f, fo == f
    tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
    sb, sc = np.sort(ST["brightness_full"][tr]), np.sort(ST["contrast_full"][tr])
    for v in VIEWS:
        b, c = ST[f"brightness_{v}"][ho], ST[f"contrast_{v}"][ho]
        HARD[v][ho] = (b <= tb) | (c <= tc)
        RANK[v][ho] = np.minimum(np.searchsorted(sb, b, side="right") / len(sb), np.searchsorted(sc, c, side="right") / len(sc))

st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m


def labels(w_plain, w_hud, w_new):
    """w_new: {view: per-row weight of the DINOv2-L member}."""
    wm = {**{t: w_plain for t in PLAIN}, **{t: w_hud for t in HUDM}, "dinov2_v2": 8.0}
    out = {}
    for v in VIEWS:
        s_rest = sum(w * P[t][v] for t, w in wm.items())
        mem = (s_rest + w_new[v][:, None] * P[NEW][v]) / (sum(wm.values()) + w_new[v])[:, None]
        out[v] = rule((1 - PW) * mem + PW * PR[v])
    return out


hard_w = {v: np.where(HARD[v], 2.0, 8.0) for v in VIEWS}
ref = labels(1.0, 3.0, hard_w)
print("v30 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def meets(d):
    return d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005


crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    pfl = [bal(lab["full"], fo == f) - bal(ref["full"], fo == f) for f in range(5)]
    ok = meets(d) and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | full by fold {' '.join(f'{p:+.4f}' for p in pfl)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {}
for lo, hi in ((0.05, 0.25), (0.05, 0.15), (0.10, 0.25)):
    C[("S1 ramp", lo, hi)] = labels(1.0, 3.0, {v: 2.0 + 6.0 * np.clip((RANK[v] - lo) / (hi - lo), 0, 1) for v in VIEWS})
for wp, wh in ((0.0, 3.0), (1.0, 2.0), (0.0, 2.0)):
    C[("S2 weights", wp, wh)] = labels(wp, wh, hard_w)
print(f"\n{len(C)} settings, in-sample against v30:")
for k, lab in C.items():
    report(str(k), lab)
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    inner = {k: deltas(lab, fo != f) for k, lab in C.items()}
    okk = {k: crit(d) for k, d in inner.items() if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
print("\nCandidate:")
okN = report(f"N nested (rule-consistent); picks {picks}", outN)
ins = {k: crit(deltas(lab)) for k, lab in C.items() if meets(deltas(lab))}
best = max(ins, key=ins.get) if ins else None
print(f"\n-> N {'PASSES' if okN else 'fails'}; pipeline setting (meets the rule in-sample, best criterion): {best} (only if N passes)")
