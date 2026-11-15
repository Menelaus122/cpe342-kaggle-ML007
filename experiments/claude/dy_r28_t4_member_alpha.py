"""Round 28, Task 4: temperature and per-member prior exponents under the simulated label shift (training images only).

    python dy_r28_t4_member_alpha.py

v19 rule: mix = 0.9 * mean of the seven members + 0.1 * probe, label = argmax mix * pi^-0.75 (pi = training class shares).
Score: macro-F1 with every class weighted to 20% (the label-shift simulation of rounds 25-27), mean over the full, 70%-zoom and
HUD-hidden views of all training images (each predicted out of fold). Everything tuned is nested: chosen on 4 folds, scored on the 5th.
  A  one temperature before the prior: argmax normalise(mix^(1/T)) * pi^-a  ==  argmax mix * pi^-(a T), so (T, a) is the same
     one-parameter family as the exponent alpha that round 26 scanned; shown by a nested choice of alpha from a wider grid
  B  per-member temperature T_m (fitted by log-loss on the 4 folds' full frames), members averaged after calibration, then pi^-alpha
     with alpha nested (calibrated posteriors: alpha = 1 would be the Bayes rule)
  C  prior adjustment per member before averaging: mean_m normalise(p_m * pi^-alpha), one alpha, nested
  D  per-member exponents alpha_m (coordinate ascent from 0.75 on the 4 folds), members averaged after adjustment
  E  diagnostic only, NOT usable on test without the TAs' approval: class shares estimated by EM (Saerens et al. 2002) from the
     evaluated rows' own calibrated probabilities, then the Bayes rule with those shares
Acceptance: mean nested gain > 0 and >= 4/5 folds.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
R27 = Path("experiments/claude/outputs/r27")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
allr = np.arange(len(y))
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in TAGS}
for t in TAGS:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t][v][va] = z[v]
zp = np.load(R27 / "t4_probe_views.npz")
probe = {v: zp[f"probe-aug_{v}"].astype(float) for v in VIEWS}
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
norm = lambda q: q / q.sum(1, keepdims=True)
bal = lambda idx, pred: f1_score(y[idx], pred[idx], average="macro", sample_weight=wb[idx])
obj = lambda idx, pr: np.mean([bal(idx, pr[v]) for v in VIEWS])
fuse = lambda members, v: 0.9 * np.mean(members, 0) + 0.1 * probe[v]
base = {v: (fuse([P[t][v] for t in TAGS], v) * pi ** -0.75).argmax(1) for v in VIEWS}
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
print(f"v19 rule, uniform class mix: full {bal(allr, base['full']):.4f} zoom70 {bal(allr, base['zoom70']):.4f} HUD hidden {bal(allr, base['centre_grey']):.4f} "
      f"mean {obj(allr, base):.4f}")


def report(name, pred, chosen=""):
    d = obj(allr, pred) - obj(allr, base)
    bs = [obj(i, pred) - obj(i, base) for i in boot]
    pf = [obj(va, pred) - obj(va, base) for va in folds]
    per = [bal(allr, pred[v]) - bal(allr, base[v]) for v in VIEWS]
    tm = np.mean([f1_score(y, pred[v], average="macro") - f1_score(y, base[v], average="macro") for v in VIEWS])
    print(f"  {name}: {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)}) | "
          f"full {per[0]:+.4f} zoom70 {per[1]:+.4f} HUD hidden {per[2]:+.4f} | training mix {tm:+.4f} | labels changed {np.mean([np.mean(pred[v] != base[v]) for v in VIEWS]):.2%}"
          + (f"\n      chosen per fold: {chosen}" if chosen else ""))


def nested(make, grid):
    """make(param, fit_rows) -> {view: labels for all rows}; param chosen on the 4 training folds."""
    out, chosen = {v: np.zeros(len(y), int) for v in VIEWS}, []
    for va in folds:
        tr = np.setdiff1d(allr, va)
        cands = {g: make(g, tr) for g in grid}
        best = max(grid, key=lambda g: obj(tr, cands[g]))
        chosen.append(best)
        for v in VIEWS:
            out[v][va] = cands[best][v][va]
    return out, chosen


print("\nA  one exponent on the fused probabilities (equivalent to a temperature before the prior), nested from a wider grid")
ALPHAS = [0.3, 0.46, 0.6, 0.75, 0.9, 1.0, 1.2, 1.5, 2.0]
MIX = {v: fuse([P[t][v] for t in TAGS], v) for v in VIEWS}
print("   all rows, mean over views by alpha: " + "  ".join(f"{a}: {obj(allr, {v: (MIX[v] * pi ** -a).argmax(1) for v in VIEWS}):.4f}" for a in ALPHAS))
pred, ch = nested(lambda a, tr: {v: (MIX[v] * pi ** -a).argmax(1) for v in VIEWS}, ALPHAS)
report("alpha nested", pred, ch)
T, a = 0.46, 1.0
same = all(np.array_equal((norm(MIX[v] ** (1 / T)) * pi ** -a).argmax(1), (MIX[v] * pi ** -(a * T)).argmax(1)) for v in VIEWS)
print(f"   check: argmax normalise(mix^(1/0.46)) * pi^-1 equals argmax mix * pi^-0.46 on every row: {same}")


def temperature(p, yy):
    lp = np.log(np.clip(p, 1e-12, 1))

    def nll(t):
        z = lp / t
        z = z - z.max(1, keepdims=True)
        return -np.mean(z[np.arange(len(yy)), yy] - np.log(np.exp(z).sum(1)))
    return minimize_scalar(nll, bounds=(0.2, 3.0), method="bounded").x


def calibrated(tr):
    Ts = {t: temperature(P[t]["full"][tr], y[tr]) for t in TAGS}
    Tp = temperature(probe["full"][tr], y[tr])
    mix = {v: 0.9 * np.mean([norm(P[t][v] ** (1 / Ts[t])) for t in TAGS], 0) + 0.1 * norm(probe[v] ** (1 / Tp)) for v in VIEWS}
    return mix, Ts


print("\nB  per-member temperature (log-loss on the training folds' full frames), then pi^-alpha, alpha nested")
ALPHAS_B = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
out, chosen = {v: np.zeros(len(y), int) for v in VIEWS}, []
for va in folds:
    tr = np.setdiff1d(allr, va)
    mix, Ts = calibrated(tr)
    cands = {a: {v: (mix[v] * pi ** -a).argmax(1) for v in VIEWS} for a in ALPHAS_B}
    best = max(ALPHAS_B, key=lambda a: obj(tr, cands[a]))
    chosen.append((best, [round(Ts[t], 2) for t in TAGS]))
    for v in VIEWS:
        out[v][va] = cands[best][v][va]
report("calibrated members + alpha", out, chosen)

print("\nC  prior adjustment per member before averaging (one alpha, nested)")
ALPHAS_C = [0.5, 0.75, 1.0, 1.25, 1.5]
per_member = lambda al: {v: (0.9 * np.mean([norm(P[t][v] * pi ** -al[i]) for i, t in enumerate(TAGS)], 0) + 0.1 * norm(probe[v] * pi ** -al[-1])).argmax(1)
                         for v in VIEWS}
pred, ch = nested(lambda a, tr: per_member([a] * 8), ALPHAS_C)
report("per-member adjustment, one alpha", pred, ch)

print("\nD  per-member exponents alpha_m (seven members + probe; coordinate ascent on the training folds), nested")
GRID_D = [0.0, 0.5, 0.75, 1.0, 1.5]
for order_name, order in (("members visited first to last", list(range(8))), ("members visited last to first", list(range(7, -1, -1)))):
    out, chosen = {v: np.zeros(len(y), int) for v in VIEWS}, []
    for va in folds:
        tr = np.setdiff1d(allr, va)
        al = [0.75] * 8
        best = obj(tr, per_member(al))
        for _ in range(2):
            for i in order:
                for g in GRID_D:
                    trial = al.copy()
                    trial[i] = g
                    s = obj(tr, per_member(trial))
                    if s > best + 1e-6:
                        al, best = trial, s
        chosen.append(al)
        lab = per_member(al)
        for v in VIEWS:
            out[v][va] = lab[v][va]
    report(f"per-member alpha_m, {order_name}", out, chosen)

print("\nE  diagnostic only (needs the TAs' approval before any use on test): EM class shares from the evaluated rows' own probabilities")


def em_shares(q, w, iters=200):
    """q: posteriors under the training prior pi; returns the class shares maximising the weighted likelihood of the rows."""
    s = pi.copy()
    for _ in range(iters):
        r = norm(q * (s / pi))
        s_new = np.average(r, axis=0, weights=w)
        if np.abs(s_new - s).max() < 1e-7:
            break
        s = s_new
    return s


out, est = {v: np.zeros(len(y), int) for v in VIEWS}, []
for va in folds:
    tr = np.setdiff1d(allr, va)
    mix, _ = calibrated(tr)
    for v in VIEWS:
        s = em_shares(mix[v][va], wb[va])                       # the fold's rows at a uniform class mix (weights), labels unused
        out[v][va] = (mix[v][va] * (s / pi)).argmax(1)
        est.append(np.round(s, 3).tolist())
report("EM shares + Bayes rule (calibrated members)", out)
print("   EM estimates of the (truly uniform, 0.2 each) shares, fold 0: full", est[0], "zoom70", est[1], "HUD hidden", est[2])
