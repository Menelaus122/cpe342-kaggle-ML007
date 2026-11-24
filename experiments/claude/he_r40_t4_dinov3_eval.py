"""Round 40, Task 4: the DINOv3-L member (dinov3L_lr8e5, five folds) against v30 (CPU; training images only).

    python he_r40_t4_dinov3_eval.py

Written 2026-11-18 after fold 0 only (alone 0.9925; fold-0 check against v30 in hd_r40_t4_dinov3_fold0_v30.py: R and A pass) and
BEFORE folds 1-4 were trained. Candidates (fixed now; the v30 pipeline otherwise unchanged: v23 members, brightness / contrast gate with
thresholds from the other four folds, 0.65 x members + 0.35 x the seven-backbone MLP probe, v17 rule, balanced class mix):
  R  dinov3L_lr8e5 in place of dinov2L_hud70 (x8, x2 on gated frames)
  A  dinov3L_lr8e5 next to dinov2L_hud70 (both x8, x2 on gated frames)
  N  per held-out fold, among R / A meeting the rule on the other four folds, the best mean of the three hard deltas; none: v30.
Acceptance (user's rule against v30, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline variant is the one meeting the rule in-sample with the
best criterion on all five folds. Information only: the member alone; U = R without the gate (x8 on every frame).
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

L2, D3 = "dinov2L_hud70", "dinov3L_lr8e5"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 1.0 for t in PLAIN}, **{t: 3.0 for t in HUDM}, "dinov2_v2": 8.0}
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
missing = [f for f in range(5) if not (V30 / f"{D3}_fold{f}.npz").exists() or not (PRED_DIR / f"t4_{D3}_fold{f}.npz").exists()]
assert not missing, f"{D3}: folds {missing} missing"
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
for f, va in enumerate(folds):
    z = np.load(PRED_DIR / f"t4_{D3}_fold{f}.npz")
    assert (z["val_idx"] == va).all() and (z["val_probs"].argmax(1) == P[D3]["full"][va].argmax(1)).mean() > 0.995
    print(f"fold {f}: {D3} macro-F1 (TTA) {f1_score(y[va], z['val_probs'].argmax(1), average='macro'):.4f}")
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


def labels(gated):
    """gated: {tag: (weight, weight on gated frames)} on top of the v23 members."""
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(y), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule(0.65 * num / den[:, None] + 0.35 * PR[v])
    return out


ref = labels({L2: (8.0, 2.0)})
print("v30 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
for t in (L2, D3):
    print(f"{t} alone: " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC))


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {"R": labels({D3: (8.0, 2.0)}), "A": labels({L2: (8.0, 2.0), D3: (8.0, 2.0)})}
print("\nCandidates against v30 (in-sample):")
report("R: dinov3L in place of DINOv2-L (x8 / x2 gated)", C["R"])
report("A: dinov3L next to DINOv2-L (both x8 / x2 gated)", C["A"])
meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    okk = {k: crit(d) for k, d in ((k, deltas(lab, fo != f)) for k, lab in C.items()) if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
print("Candidate:")
ok = report(f"N nested over {list(C)}; picks {picks}", outN)
print("Information:")
report("U: dinov3L in place of DINOv2-L, no gate (x8 everywhere)", labels({D3: (8.0, 8.0)}))
ins = {k: crit(deltas(lab)) for k, lab in C.items() if meets(deltas(lab))}
print(f"\n-> N {'PASSES' if ok else 'fails'}; pipeline variant if N passes: {max(ins, key=ins.get) if ins else None}")
