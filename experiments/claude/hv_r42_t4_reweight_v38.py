"""Round 42, Task 4: one nested re-weighting round around v38 (CPU; training images only).

    python hv_r42_t4_reweight_v38.py

User brief of 2026-11-21 (after v38 = 0.86891; one run, grid fixed before the run): does moving weight onto the new ViT-H+ 192 x 384
member gain more, as the re-weightings of v32 / v37 did? Grid (2 x 2 x 2 x 3 x 2 = 48 settings; v38 = H384 40, H144 40, D384 20,
PW 0.30, OLD 1 is in it):
  H384   ViT-H+ 192x384 (dinov3Hp_384) weight on every frame        {40, 60}
  H144   ViT-H+ 144x272 (dinov3Hp_144) weight on every frame        {20, 40}
  D384   DINOv3-L 192x384 (dinov3L_384) weight on every frame       {10, 20}
  PW     probe weight in the final mix                              {0.20, 0.25, 0.30}
  OLD    the older seven relative to v38 (ConvNeXt-T / DeiT-III 0.25, HUD members 0.75)   {x1, x0.5}
  fixed at v38: DINOv2-B x8, DINOv2-L x8 / x2 gated, DINOv3-L-144 x10, C-RADIOv3-L x10 / x2 gated; gate = v30's Q10 brightness /
  contrast gate with thresholds from the other four folds; v17 rule; balanced class mix.
Reference v38. Candidate N: per held-out fold, among the settings other than v38 meeting the rule on the other four folds (zoom70,
HUD hidden, real dark > 0, full >= -0.0005 against v38), the best mean of the three hard deltas; none: v38.
Acceptance (user's rule against v38, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline setting is the one meeting the rule in-sample with the best
criterion. One run; no extension of this grid afterwards.
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

L2, D3, D384, RAD, HP, HP384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144", "dinov3Hp_384"
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
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [L2, D3, D384, RAD, HP, HP384]}
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


def labels(h384, h144, d384, pw, old):
    w = {**{t: 0.25 * old for t in PLAIN}, **{t: 0.75 * old for t in HUDM}, "dinov2_v2": 8.0, D3: 10.0, D384: d384, HP: h144, HP384: h384}
    gated = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}
    out = {}
    for v in VIEWS:
        num = sum(wt * P[t][v] for t, wt in w.items())
        den = np.full(len(y), sum(w.values()))
        for t, (wn, wd) in gated.items():
            ww = np.where(GATE[v], wd, wn)
            num, den = num + ww[:, None] * P[t][v], den + ww
        out[v] = rule((1 - pw) * num / den[:, None] + pw * PR[v])
    return out


V38 = (40.0, 40.0, 20.0, 0.30, 1.0)
GRID = list(itertools.product((40.0, 60.0), (20.0, 40.0), (10.0, 20.0), (0.20, 0.25, 0.30), (1.0, 0.5)))
assert V38 in GRID and len(GRID) == 48
ref = labels(*V38)
print("v38 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)
name = lambda s: f"H384 {s[0]:g} H144 {s[1]:g} D384 {s[2]:g} PW {s[3]:.2f} OLD x{s[4]:g}"
# per-fold scores of every setting, so the nested choice and the reports reuse them
LAB = {s: labels(*s) for s in GRID}
print(f"{len(GRID)} settings computed", flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def report(nm, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {nm}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    okk = {s: crit(d) for s, d in ((s, deltas(LAB[s], fo != f)) for s in GRID if s != V38) if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = LAB[s][v][fo == f]
print("\nNested choice per held-out fold:")
for f, s in enumerate(picks):
    print(f"  fold {f}: {name(s) if s else 'v38 (no setting meets the rule)'}")
print("Candidate (decision):")
ok = report("N nested over the 48-setting grid", outN)
ins = sorted(((crit(deltas(LAB[s])), s) for s in GRID if s != V38 and meets(deltas(LAB[s]))), reverse=True)
print("\nInformation: top in-sample settings meeting the rule")
for c, s in ins[:5]:
    report(f"{name(s)} (criterion {c:+.4f})", LAB[s])
print("Information: one factor at a time from v38")
for i, alt in enumerate([(60.0,), (20.0,), (10.0,), (0.20, 0.25), (0.5,)]):
    for a in alt:
        s = list(V38)
        s[i] = a
        report(name(tuple(s)), LAB[tuple(s)])
best = ins[0][1] if ins else None
EDGE = {"H384": (0, 60.0), "H144": (1, 20.0), "D384": (2, 10.0), "PW": (3, 0.20), "OLD": (4, 0.5)}   # the far end of each factor
edge = None if best is None else [nm for nm, (i, far) in EDGE.items() if best[i] == far]
print(f"\n-> N {'PASSES' if ok else 'fails'}; pipeline setting if N passes: {name(best) if best else None}"
      + (f" (at the far grid edge in: {', '.join(edge)})" if edge else ""))
