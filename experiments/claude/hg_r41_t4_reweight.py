"""Round 41 step 1, Task 4: re-balance v31's ensemble now that DINOv3-L is its strongest member (CPU; training images only).

    python hg_r41_t4_reweight.py

User brief of 2026-11-18 evening: DINOv3-L alone (0.9925 full / 0.9823 real dark) beats every other member and the probe, but v31
still carries the weights tuned before it existed. Grid fixed before the run (48 settings; v31 = D3 8, gated, probe 0.35, old 1):
  D3     DINOv3-L weight on normal frames            {8, 12, 16, 20}
  DARK   DINOv3-L weight on gated (dark / low-contrast) frames: 'gated' = 2 (as v31) | 'full' = the same as on normal frames
  PW     probe weight in the final mix                {0.35, 0.25, 0.20}
  OLD    scale of the seven older members (ConvNeXt-T, DeiT-III x1; five HUD-dropout members x3) {1, 0.5}
  fixed: DINOv2-B x8, DINOv2-L x8 / x2 gated (as v30 / v31); gate thresholds from the other four folds' training images; v17 rule.
Reference v31 (= he_r40 candidate A). Candidate N: per held-out fold, among the settings meeting the rule on the other four folds
(zoom70, HUD hidden, real dark > 0, full >= -0.0005 against v31), the best mean of the three hard deltas; none: v31.
Acceptance (user's rule against v31, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. Information only: in-sample table, one-factor-at-a-time rows.
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

L2, D3 = "dinov2L_hud70", "dinov3L_lr8e5"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [L2, D3]}
for t in P:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in VIEWS:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
GATE = {v: np.zeros(len(y), bool) for v in VIEWS}
for f in range(5):
    tr, ho = fo != f, fo == f
    tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
    for v in VIEWS:
        GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)
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


def labels(d3, dark_mode, pw, old):
    out = {}
    wm = {**{t: 1.0 * old for t in PLAIN}, **{t: 3.0 * old for t in HUDM}, "dinov2_v2": 8.0}
    gated = {L2: (8.0, 2.0), D3: (d3, 2.0 if dark_mode == "gated" else d3)}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in wm.items())
        den = np.full(len(y), sum(wm.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule((1 - pw) * num / den[:, None] + pw * PR[v])
    return out


GRID = list(itertools.product([8.0, 12.0, 16.0, 20.0], ["gated", "full"], [0.35, 0.25, 0.20], [1.0, 0.5]))
V31 = (8.0, "gated", 0.35, 1.0)
ref = labels(*V31)
print("v31 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)
LAB = {s: labels(*s) for s in GRID}
name = lambda s: f"D3 {s[0]:g} / dark {s[1]} / probe {s[2]:.2f} / old x{s[3]:g}"


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])


def report(nm, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = meets(d) and sum(p > 0 for p in pf) >= 3
    print(f"  {nm}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


D_IN = {s: deltas(LAB[s]) for s in GRID}
print("\nIn-sample (information), all 48 settings, sorted by the criterion; * = meets the pooled rule:")
for s in sorted(GRID, key=lambda s: -crit(D_IN[s])):
    d = D_IN[s]
    print(f"  {'*' if meets(d) else ' '} {name(s):45s} " + " ".join(f"{d[k]:+.4f}" for k in SC) + f" | crit {crit(d):+.4f}")
print("\nOne factor at a time from v31 (information):")
for s in [(12.0, "gated", 0.35, 1.0), (16.0, "gated", 0.35, 1.0), (20.0, "gated", 0.35, 1.0), (8.0, "full", 0.35, 1.0),
          (8.0, "gated", 0.25, 1.0), (8.0, "gated", 0.20, 1.0), (8.0, "gated", 0.35, 0.5)]:
    print(f"  {name(s):45s} " + " ".join(f"{D_IN[s][k]:+.4f}" for k in SC))

outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    okk = {}
    for s in GRID:
        d = deltas(LAB[s], fo != f)
        if meets(d):
            okk[s] = crit(d)
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = LAB[s][v][fo == f]
print("\nNested picks: " + " | ".join(name(s) if s else "v31" for s in picks))
print("Candidate:")
ok = report("N nested over the 48 settings", outN)
ins = {s: crit(D_IN[s]) for s in GRID if meets(D_IN[s])}
best = max(ins, key=ins.get) if ins else None
print(f"\n-> N {'PASSES' if ok else 'fails'}; pipeline setting if N passes: {name(best) if best else None}")
if best:
    report(f"information: the in-sample best on all five folds ({name(best)})", LAB[best])
