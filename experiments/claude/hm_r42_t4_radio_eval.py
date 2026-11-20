"""Round 42 part B, Task 4: cradio3L_384 (fine-tuned C-RADIOv3-L at 192 x 384) on folds 1-4 against v33 (CPU; training images only).

    python hm_r42_t4_radio_eval.py

Written 2026-11-19 ~23:30 Thai time, after the user had run folds 1-4 on Colab (12:07-16:14 UTC; no full refit) and BEFORE any result
of folds 1-4 was looked at (no log read, no file downloaded). Fold 0 was seen twice (hj_r42: R / H / A fail on the dark frames;
hk_r42: the gated variants, proposed after that, give dark exactly 0), so fold 0 is excluded from the decision: everything below is
judged on folds 1-4 only (fold 0 is printed as information).
Reference v33 (as hi_r41 / hl_r42): older seven x0.5 (ConvNeXt-T / DeiT-III 0.5, HUD members 1.5), DINOv2-B 8, DINOv2-L 8 / 2 on gated
frames, both DINOv3-L members x20 on every frame, 0.65 x members + 0.35 x P7, v17 rule, balanced class mix; gate = v30's Q10
brightness / contrast gate with thresholds from the other four folds.
Candidates (the three of hj_r42 and the four of hk_r42; nothing new):
  R    in place of dinov3L_384 (x20)          H    added x10          A    added x20
  G20/2, G20/0, G10/2, G10/0                  added x20 or x10 on normal frames, x2 or x0 on gated frames
  N    per held-out fold f in 1-4, among the candidates meeting the rule on the other three of folds 1-4, the best mean of the three
       hard deltas; none: v33.
Acceptance (user's rule, unchanged: dark > 0), on N pooled over folds 1-4: zoom70, HUD hidden and the real dark / low-contrast frames
all > 0, full >= -0.0005, zoom70 / HUD-hidden mean better in >= 3 of the 4 folds. If N passes, the pipeline variant is the candidate
meeting the rule on folds 1-4 with the best criterion. Information: the member alone per fold; each candidate on folds 1-4 and on all
five folds.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

CAND, L2, D3, D384 = "cradio3L_384", "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 0.5 for t in PLAIN}, **{t: 1.5 for t in HUDM}, "dinov2_v2": 8.0, D3: 20.0}
PW = 0.35
VIEWS = ["full", "zoom70", "centre_grey"]
EVAL = [1, 2, 3, 4]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
missing = [f for f in range(5) if not (V30 / f"{CAND}_fold{f}.npz").exists() or not (PRED_DIR / f"t4_{CAND}_fold{f}.npz").exists()]
assert not missing, f"{CAND}: folds {missing} missing"
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [L2, D3, D384, CAND]}
for t in P:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in VIEWS:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
for f, va in enumerate(folds):
    z = np.load(PRED_DIR / f"t4_{CAND}_fold{f}.npz")
    assert (z["val_idx"] == va).all() and (z["val_probs"].argmax(1) == P[CAND]["full"][va].argmax(1)).mean() > 0.995
    print(f"fold {f}: {CAND} macro-F1 (TTA) {f1_score(y[va], z['val_probs'].argmax(1), average='macro'):.4f}"
          + ("  (seen before; not in the decision)" if f not in EVAL else ""))
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
ev = np.isin(fo, EVAL)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m: m & dark if k == "dark/low-contrast" else m


def labels(gated):
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(y), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule((1 - PW) * num / den[:, None] + PW * PR[v])
    return out


base = {L2: (8.0, 2.0), D384: (20.0, 20.0)}
ref = labels(base)
print("v33 reference, folds 1-4: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k, ev)):.4f}" for k in SC))
print(f"{CAND} alone, folds 1-4: " + ", ".join(f"{k} {bal(rule(P[CAND][VIEW_OF[k]]), msk(k, ev)):.4f}" for k in SC)
      + " | dinov3L_384 alone: " + ", ".join(f"{k} {bal(rule(P[D384][VIEW_OF[k]]), msk(k, ev)):.4f}" for k in SC))


def deltas(lab, m):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def report(name, lab, m=ev, fl=EVAL):
    d = deltas(lab, m)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in fl]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in fl]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | labels changed (full) {(lab['full'] != ref['full'])[m].sum()}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {"R": labels({L2: (8.0, 2.0), CAND: (20.0, 20.0)}),
     "H": labels({**base, CAND: (10.0, 10.0)}),
     "A": labels({**base, CAND: (20.0, 20.0)}),
     "G20/2": labels({**base, CAND: (20.0, 2.0)}),
     "G20/0": labels({**base, CAND: (20.0, 0.0)}),
     "G10/2": labels({**base, CAND: (10.0, 2.0)}),
     "G10/0": labels({**base, CAND: (10.0, 0.0)})}
print("\nCandidates against v33, folds 1-4 (in-sample):")
for k, lab in C.items():
    report(k, lab)
meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in EVAL:
    other = np.isin(fo, [g for g in EVAL if g != f])
    okk = {k: crit(d) for k, d in ((k, deltas(lab, other)) for k, lab in C.items()) if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
print("Candidate (decision):")
ok = report(f"N nested over {list(C)} on folds 1-4; picks {picks}", outN)
print("Information: each candidate on all five folds (fold 0 included)")
for k, lab in C.items():
    report(k, lab, m=allm, fl=list(range(5)))
ins = {k: crit(deltas(lab, ev)) for k, lab in C.items() if meets(deltas(lab, ev))}
print(f"\n-> N {'PASSES' if ok else 'fails'} on folds 1-4; pipeline variant if N passes: {max(ins, key=ins.get) if ins else None}")
