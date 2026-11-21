"""Round 42, Task 4: nested check of the balanced-prior exponent alpha (argmax p * pi^-alpha) on the v38 / v39 ensemble (CPU;
training images only).

    python hw_r42_t4_alpha_v38.py

User brief of 2026-11-21: alpha = 0.75 (v17 rule) was last checked at v30 (round 40), before DINOv3-L, C-RADIOv3-L and both ViT-H+
members changed the ensemble's calibration. Grid fixed before the run: alpha in {0.65, 0.70, 0.75, 0.80, 0.85}; member weights,
gate and probe weight fixed at v38 (the dinov3Hp_384 full refit of v39 changes only the test input, not these out-of-fold scores).
Reference v38 (alpha 0.75). Candidate N: per held-out fold, among the alphas other than 0.75 meeting the rule on the other four folds
(zoom70, HUD hidden, real dark > 0, full >= -0.0005 against v38), the best mean of the three hard deltas; none: 0.75.
Acceptance (user's rule against v38, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline alpha is the one meeting the rule in-sample with the best
criterion. One run.
"""
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
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m


def probs(v):
    w = {**{t: 0.25 for t in PLAIN}, **{t: 0.75 for t in HUDM}, "dinov2_v2": 8.0, D3: 10.0, D384: 20.0, HP: 40.0, HP384: 40.0}
    gated = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}
    num = sum(wt * P[t][v] for t, wt in w.items())
    den = np.full(len(y), sum(w.values()))
    for t, (wn, wd) in gated.items():
        ww = np.where(GATE[v], wd, wn)
        num, den = num + ww[:, None] * P[t][v], den + ww
    return 0.70 * num / den[:, None] + 0.30 * PR[v]


MIX = {v: probs(v) for v in VIEWS}


def labels(alpha):
    return {v: (MIX[v] * pi ** -alpha).argmax(1) for v in VIEWS}


V38 = 0.75
GRID = [0.65, 0.70, 0.75, 0.80, 0.85]
ref = labels(V38)
print("v38 reference (alpha 0.75): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)
name = lambda s: f"alpha {s:.2f}"
LAB = {s: labels(s) for s in GRID}


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
    print(f"  fold {f}: {name(s) if s is not None else 'alpha 0.75 (no alpha meets the rule)'}")
print("Candidate (decision):")
ok = report("N nested over alpha", outN)
ins = sorted(((crit(deltas(LAB[s])), s) for s in GRID if s != V38 and meets(deltas(LAB[s]))), reverse=True)
print("\nInformation: every alpha against v38 (in-sample)")
for s in GRID:
    if s != V38:
        report(name(s), LAB[s])
best = ins[0][1] if ins else None
print(f"\n-> N {'PASSES' if ok else 'fails'}; pipeline alpha if N passes: {name(best) if best is not None else None}")
