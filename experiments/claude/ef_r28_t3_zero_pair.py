"""Round 28, Task 3: a conditional zero gate for accounts whose repaired historical_spending and prev_month_spending are both 0
(OOF caches only, CPU, one thread).

    python ef_r28_t3_zero_pair.py

The pipeline's out-of-fold Task 3 prediction is rebuilt from the cached members exactly as in scripts/nb_part2.py 7.3 (seeds read
out in THB and averaged, exact non-negative L1 blend fitted on 4 folds, median zero gate P(y>0) + shift <= 0.5 -> 0, scale, cap);
the scale is chosen on the 4 training folds as well (dl_r26_t3_blendspace.py). On the rows with both repaired spending columns
equal to 0 ("zero pair"):
  1  what the target and the prediction look like there
  2  rule A: predictions <= c on zero-pair rows are set to 0, c in {0 (off), 10, 25, 50, 100, 250, 500, 1000, 2500, all}
     rule B: a stricter gate on zero-pair rows, 0 where P(y>0) + shift <= g, g in {0.5 (off), 0.55, ..., 0.9, 1 (all)}
     c and g chosen on 4 folds by the absolute error, applied to the 5th (nested). Acceptance: mean gain > 0 and >= 4/5 folds.
For the L1 loss the best constant in a group is its median, so zeroing helps exactly where more than half the rows are 0.
First run: no row has either column equal to 0 (both are strictly positive, minimum 0.026 / 0.008 THB; 45 rows have both missing).
  3  the nearest well-posed version: the accounts with the lowest repaired historical_spending (bottom 1 / 5 / 10 / 25%; the two
     columns move together, prev ~ 0.35 x hist) and the 45 rows with both missing; rule B with the group and g chosen together, nested.
The rebuilt out-of-fold prediction is saved to outputs/r28/t3_oof_rebuilt.npz (the exact L1 blend's LP peaks at about 650 MB).
"""
import time
from pathlib import Path

import numpy as np
import psutil

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t0 = time.time()
t3_train, t3_test = task3.load()
X3, y3, _, _ = task3.build_repaired(t3_train, t3_test)
y3 = np.asarray(y3, float)
raw_zero = ((t3_train["historical_spending"] == 0) & (t3_train["prev_month_spending"] == 0)).values
ZP = ((X3["historical_spending"] == 0) & (X3["prev_month_spending"] == 0)).values
HIST = X3["historical_spending"].values.astype(float)
BOTH_NA = (X3["historical_spending"].isna() & X3["prev_month_spending"].isna()).values
del t3_train, t3_test, X3
folds = make_folds(y3, kind="regression")
fo = np.zeros(len(y3), int)
for i, (_, va) in enumerate(folds):
    fo[va] = i
R28 = Path("experiments/claude/outputs/r28")
R28.mkdir(parents=True, exist_ok=True)
CACHE = R28 / "t3_oof_rebuilt.npz"
if CACHE.exists():
    z = np.load(CACHE)
    final, p_oof, shift = z["final"], z["p_oof"], float(z["shift"])
else:
    clip0 = lambda a: np.clip(a, 0, None)
    pool = {}
    for name, key, log in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                           ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
        r = load_preds(key)
        pool[name] = clip0(np.expm1(r["oof"])) if log else clip0(r["oof"])
    SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
    sfx3 = lambda s: "" if s == 42 else f"_s{s}"
    hs = {s: load_preds(f"t3r_hurdle{sfx3(s)}") for s in SEEDS3}
    p_oof = np.mean([hs[s]["p_oof"] for s in SEEDS3], 0)
    BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
    RATIO3 = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
    slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
    mus = {"Hurdle": [hs[s]["mu_oof"] for s in SEEDS3]}
    for tag in RATIO3:
        mus[tag] = [load_preds(slug3(tag) + sfx3(s))["mu_oof"] for s in SEEDS3] if tag in BAG3 else [load_preds(slug3(tag))["mu_oof"]]
    del hs
    for tag, ms in mus.items():
        ro = task3.readouts(y3, p_oof, np.mean(ms, 0))
        if tag == "Hurdle":
            shift = ro["shift"]
        for k in ("mixture-median", "hard gate", "E[y] + smearing"):
            pool[f"{tag} {k}"] = np.mean([task3.apply_readout(k, p_oof, m, ro) for m in ms], 0)
    del mus
    pg = np.clip(p_oof + shift, 0, 1)
    gate = (pg > 0.5).astype(float)
    scales = np.linspace(0.85, 1.15, 61)
    P = np.column_stack(list(pool.values()))
    del pool
    final = np.zeros(len(y3))
    for f, (tr, va) in enumerate(folds):
        w = task3.l1_blend_weights_exact(y3[tr], [P[tr, j] for j in range(P.shape[1])])
        btr, bva = P[tr] @ w, P[va] @ w
        s = scales[int(np.argmin([np.abs(y3[tr] - btr * s_ * gate[tr]).sum() for s_ in scales]))]
        final[va] = np.minimum(bva * s * gate[va], y3[tr].max())
    np.savez(CACHE, final=final, p_oof=p_oof, shift=shift)
pg = np.clip(p_oof + shift, 0, 1)
print(f"{len(y3)} rows; pipeline Task 3 rebuilt out of fold, nested: NMAE score {nmae(y3, final):.5f} [{time.time() - t0:.0f}s]")

print(f"\n1. zero pair (both repaired spending columns = 0): {ZP.sum()} rows ({ZP.mean():.2%}); before the identity repair {raw_zero.sum()} rows")
for name, m in (("zero pair", ZP), ("all other rows", ~ZP)):
    pos = final[m] > 0
    print(f"  {name}: y = 0 in {np.mean(y3[m] == 0):.2%}; prediction 0 in {np.mean(~pos):.2%}; median y {np.median(y3[m]):,.0f}, mean y {y3[m].mean():,.0f}; "
          f"share of the total |error| {np.abs(y3 - final)[m].sum() / np.abs(y3 - final).sum():.2%}")
    if pos.any():
        yp, fp = y3[m][pos], final[m][pos]
        print(f"    rows with a positive prediction: {pos.sum()}; y = 0 among them {np.mean(yp == 0):.2%}; prediction quantiles 10/50/90% "
              f"{np.quantile(fp, .1):,.0f} / {np.median(fp):,.0f} / {np.quantile(fp, .9):,.0f}; median y {np.median(yp):,.0f}")
if ZP.any():
    zpos = ZP & (final > 0)
    print("  zero-pair rows with a positive prediction, by prediction band: rows | y = 0 share | median y")
    for lo, hi in ((0, 50), (50, 250), (250, 1000), (1000, 5000), (5000, np.inf)):
        b = zpos & (final > lo) & (final <= hi)
        if b.sum():
            print(f"    ({lo:,.0f}, {hi:,.0f}]: {b.sum():6d} | {np.mean(y3[b] == 0):.2%} | {np.median(y3[b]):,.0f}")
    print("  zero-pair rows by gate probability P(y>0) + shift: rows | y = 0 share")
    for lo, hi in ((0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        b = ZP & (pg > lo) & (pg <= hi)
        if b.sum():
            print(f"    ({lo:.1f}, {hi:.1f}]: {b.sum():6d} | {np.mean(y3[b] == 0):.2%}")

CS = [0, 10, 25, 50, 100, 250, 500, 1000, 2500, np.inf]
GS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
rule = {**{("A", c): np.where(ZP & (final <= c), 0.0, final) for c in CS}, **{("B", g): np.where(ZP & (pg <= g), 0.0, final) for g in GS}}
def boots(n, B):
    """The same B bootstrap draws for every candidate, generated one at a time (no B x n index matrix in memory)."""
    r = np.random.default_rng(0)
    for _ in range(B):
        yield r.integers(0, n, n).astype(np.int32)




def nested(keys):
    out, ch = final.copy(), []
    for i in range(5):
        t_ = fo != i
        best = min(keys, key=lambda k: np.abs(y3[t_] - rule[k][t_]).sum())
        ch.append(best)
        out[fo == i] = rule[best][fo == i]
    return out, ch


print("\n2. delta in NMAE score against the pipeline (positive = better)")
print("  in-sample by threshold: A " + "  ".join(f"{c}: {nmae(y3, rule['A', c]) - nmae(y3, final):+.5f}" for c in CS))
print("                          B " + "  ".join(f"{g}: {nmae(y3, rule['B', g]) - nmae(y3, final):+.5f}" for g in GS))
for name, keys in (("rule A (amount cut) nested", [("A", c) for c in CS]), ("rule B (stricter gate) nested", [("B", g) for g in GS])):
    o, ch = nested(keys)
    pf = [nmae(y3[fo == i], o[fo == i]) - nmae(y3[fo == i], final[fo == i]) for i in range(5)]
    bs = [nmae(y3[b], o[b]) - nmae(y3[b], final[b]) for b in boots(len(y3), 1000)]
    print(f"  {name}: {nmae(y3, o) - nmae(y3, final):+.5f} [{np.quantile(bs, .025):+.5f}, {np.quantile(bs, .975):+.5f}]  folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.5f}' for x in pf)})  rows changed {np.sum(o != final)}  chosen {ch}")

print("\n3. low spending history: group = bottom q of repaired historical_spending (fold-free quantile of a feature), or both columns missing")
GROUPS = {f"bottom {q:.0%}": HIST <= np.nanquantile(HIST, q) for q in (0.01, 0.05, 0.10, 0.25)}
GROUPS["both missing"] = BOTH_NA
for name, m in GROUPS.items():
    pos = m & (final > 0)
    print(f"  {name:12s} rows {m.sum():6d}; y = 0 {np.mean(y3[m] == 0):.2%}; prediction 0 {np.mean(final[m] == 0):.2%}; positive predictions {pos.sum()}, "
          f"y = 0 among them {np.mean(y3[pos] == 0) if pos.any() else float('nan'):.2%}")
    for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
        b = m & (pg > lo) & (pg <= hi)
        if b.sum() >= 20:
            print(f"      P(y>0) + shift in ({lo:.1f}, {hi:.1f}]: {b.sum():6d} rows, y = 0 {np.mean(y3[b] == 0):.2%}")
for name, m in GROUPS.items():
    for g in GS[1:]:
        rule[name, g] = np.where(m & (pg <= g), 0.0, final)
keys = [("B", 0.5)] + [(name, g) for name in GROUPS for g in GS[1:]]
o, ch = nested(keys)
pf = [nmae(y3[fo == i], o[fo == i]) - nmae(y3[fo == i], final[fo == i]) for i in range(5)]
bs = [nmae(y3[b], o[b]) - nmae(y3[b], final[b]) for b in boots(len(y3), 1000)]
print(f"  group and g nested: {nmae(y3, o) - nmae(y3, final):+.5f} [{np.quantile(bs, .025):+.5f}, {np.quantile(bs, .975):+.5f}]  folds +{sum(x > 0 for x in pf)}/5 "
      f"({' '.join(f'{x:+.5f}' for x in pf)})  rows changed {np.sum(o != final)}  chosen {[k for k in ch]}")
print("  in-sample, best g per group: " + "  ".join(f"{name}: {max(nmae(y3, rule[name, g]) - nmae(y3, final) for g in GS[1:]):+.5f}" for name in GROUPS))
print(f"peak working set {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB; {time.time() - t0:.0f}s")
