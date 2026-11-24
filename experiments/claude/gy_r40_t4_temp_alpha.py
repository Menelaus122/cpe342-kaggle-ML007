"""Round 40 (revised brief), Task 4, part 1: prior exponent alpha and probe temperature for v30 (training images only; CPU).

    python gy_r40_t4_temp_alpha.py

User brief of 2026-11-18 (after v30 = 0.86322): the v17 rule (argmax p * pi^-alpha, alpha = 0.75) and the mixing were set when the
ensemble had no MLP probe and no DINOv2-L; re-check temperature scaling and alpha, nested on v30's OOF, under the four-view rule.
Note fixed before the run: one temperature T on the final mixture changes nothing in the labels beyond alpha (argmax of log p / T -
alpha log pi = argmax of log p - alpha T log pi), so the temperature is applied where it changes the mixture: to the probe's
probabilities before they are mixed with the members' (p_probe^(1/T), renormalised; members at T = 1).
Settings (fixed before the run): alpha in {0.5, 0.625, 0.75, 0.875, 1.0} x T_probe in {0.5, 1.0, 2.0}; v30 = (0.75, 1.0).
Reference v30 as gv_r40_t4_gate_weights.py (gate thresholds per held-out fold; views full / zoom70 / HUD hidden on their own
statistics; probe 0.35; balanced class mix; real dark / low-contrast frames = lowest decile, full view).
Selection N: per held-out fold, among the settings meeting the rule on the other four folds, the best mean of the zoom70 / HUD-hidden
/ dark deltas; none: v30. Acceptance (user's rule against v30, on N): zoom70, HUD hidden and dark all > 0, full >= -0.0005, zoom70 /
HUD-hidden mean better in >= 3/5 folds.
Caution: the balanced-mix score assumes a balanced test class mix, under which alpha near 1 is Bayes-optimal; v17 set 0.75 as a hedge
between the training mix (alpha 0) and a balanced one. The macro-F1 at the training class mix is printed for every setting, so a
move of alpha that only reflects the assumed mix is visible.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

NEW = "dinov2L_hud70"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 1.0 for t in PLAIN}, **{t: 3.0 for t in HUDM}, "dinov2_v2": 8.0}
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")

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
MEM = {}
for v in VIEWS:
    gate = np.zeros(len(y), bool)
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        gate[ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)
    wn = np.where(gate, 2.0, 8.0)
    MEM[v] = (sum(w * P[t][v] for t, w in WM.items()) + wn[:, None] * P[NEW][v]) / (sum(WM.values()) + wn)[:, None]

st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m


def temper(p, T):
    if T == 1.0:
        return p
    q = np.exp(np.log(np.clip(p, 1e-12, 1)) / T)
    return q / q.sum(1, keepdims=True)


def labels(alpha, T):
    return {v: ((0.65 * MEM[v] + 0.35 * temper(PR[v], T)) * pi ** -alpha).argmax(1) for v in VIEWS}


ref = labels(0.75, 1.0)
print("v30 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC)
      + f" | training mix, full {f1_score(y, ref['full'], average='macro'):.4f}", flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def meets(d):
    return d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005


crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
C = {(a, T): labels(a, T) for a in (0.5, 0.625, 0.75, 0.875, 1.0) for T in (0.5, 1.0, 2.0) if (a, T) != (0.75, 1.0)}
print(f"\n{len(C)} settings (alpha, T_probe), in-sample against v30 (balanced mix) | training-mix macro-F1 of full frames:")
for k, lab in C.items():
    d = deltas(lab)
    print(f"  {k}: " + ", ".join(f"{q} {d[q]:+.4f}" for q in SC) + f" | train-mix full {f1_score(y, lab['full'], average='macro'):.4f}"
          + ("  meets the rule" if meets(d) else ""), flush=True)
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    inner = {k: deltas(lab, fo != f) for k, lab in C.items()}
    okk = {k: crit(d) for k, d in inner.items() if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
d = deltas(outN)
pf = [np.mean([bal(outN[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
pdk = [bal(outN["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
okN = meets(d) and sum(p > 0 for p in pf) >= 3
print(f"\nN nested (rule-consistent); picks {picks}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
      + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
      + f"\n    -> {'PASS' if okN else 'fail'}")
ins = {k: crit(deltas(lab)) for k, lab in C.items() if meets(deltas(lab))}
print(f"-> pipeline setting if N passes: {max(ins, key=ins.get) if ins else None}")
