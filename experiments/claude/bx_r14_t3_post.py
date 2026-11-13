"""Round 14, steps 1 (Task 3 row) and 3: global zero-snap and scale on top of the v12 pipeline.

B0 = pipeline with the cap: exact L1 blend (nested), median zero gate, scale 1.000, cap at the training folds' maximum (0.79434).
Round 13 already searched alpha on a grid and tau on nine values; this adds a fine tau grid, a per-fold fixed-tau table, and the brief's
joint (tau, alpha) Nelder-Mead fit. Every choice is made on the training folds only and applied to the held-out fold.
Acceptance: paired-bootstrap CI above 0, >= 4/5 folds, fold std not above 1.1x.
"""
from pathlib import Path

from scipy.optimize import minimize

_src = open(Path(__file__).with_name("bv_t3_scale_whale.py"), encoding="utf-8").read()
exec(_src.split("bo, inner = fold_blend(P)")[0].split('"""', 2)[2])

bo, inner = fold_blend(P)
log("nested blend done")
capf = {k: y[a].max() for k, (a, b) in enumerate(folds)}


def cap_oof(v):
    out = v.copy()
    for k, (a, b) in enumerate(folds):
        out[b] = np.minimum(v[b], capf[k])
    return out


raw0 = bo * gate
pred0 = cap_oof(raw0)
s0 = nmae(y, pred0)
pf = lambda v: np.array([nmae(y[b], v[b]) for _, b in folds])
g0 = pf(pred0)
v12 = pd.read_csv("submission_v12_try.csv").task3.values

print("=== Task 3 row of the v12 audit ===")
print(f"exact OOF NMAE 1 / (1 + sum|y - yhat| / sum y) = {s0:.5f}; folds {np.round(g0, 4).tolist()}; fold std {g0.std():.5f}")
print(f"training rows {len(y):,}; zeros in train.csv: {(y == 0).mean():.3%}; predicted zeros OOF: {(pred0 == 0).mean():.3%}; predicted zeros in v12 test: {(v12 == 0).mean():.3%}")
print(f"sum of predictions: OOF {pred0.sum():,.0f} vs actual {y.sum():,.0f} (ratio {pred0.sum() / y.sum():.4f}); mean per row OOF {pred0.mean():,.1f}, actual {y.mean():,.1f}")
print(f"v12 test: sum {v12.sum():,.2f}; mean per row {v12.mean():,.2f}; max {v12.max():,.0f}", flush=True)

print("\n=== small positive predictions ===")
edges = [0, 1, 10, 25, 50, 100, 200, 400, 800]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    mo = (pred0 > lo) & (pred0 <= hi) if lo else (pred0 > 0) & (pred0 <= hi)
    mt = (v12 > lo) & (v12 <= hi) if lo else (v12 > 0) & (v12 <= hi)
    snap = np.abs(y[mo]).sum() - np.abs(y[mo] - pred0[mo]).sum()
    rows.append([f"({lo}, {hi}]", int(mo.sum()), int(mt.sum()), (y[mo] == 0).mean() if mo.sum() else np.nan, np.median(y[mo]) if mo.sum() else np.nan, -snap])
print(pd.DataFrame(rows, columns=["prediction range THB", "OOF rows", "test rows", "share y = 0 (OOF)", "median y (OOF)", "THB saved in OOF if snapped to 0 (negative = worse)"]).round(3).to_string(index=False), flush=True)

print("\n=== fixed zero-snap thresholds, no refit (descriptive) ===")
rows = []
for t in (1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500):
    v = np.where(pred0 < t, 0.0, pred0)
    g = pf(v)
    rows.append([t, int(((pred0 > 0) & (pred0 < t)).sum()), nmae(y, v), nmae(y, v) - s0, int((g > g0).sum()), int((g < g0).sum())])
print(pd.DataFrame(rows, columns=["tau", "rows snapped OOF", "NMAE", "delta", "folds better", "folds worse"]).round(6).to_string(index=False), flush=True)


def boot(pn, po, B=300):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        T = y[i].sum()
        d.append(1 / (1 + np.abs(y[i] - pn[i]).sum() / T) - 1 / (1 + np.abs(y[i] - po[i]).sum() / T))
    return np.percentile(d, [2.5, 97.5])


rows = []


def judge(name, v):
    s, g = nmae(y, v), pf(v)
    lo, hi = boot(v, pred0)
    better = int((g > g0).sum())
    keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
    rows.append([name, s, s - s0, f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", " ".join(f"{d:+.5f}" for d in g - g0), g.std(), "KEEP" if keep else "discard"])
    log(f"{name}: {s:.5f} ({s - s0:+.5f}) CI {rows[-1][3]}, {better}/5, per-fold {rows[-1][5]} -> {rows[-1][-1]}")


def apply_post(q, tau, alpha, cap):
    return np.minimum(np.where(q < tau, 0.0, alpha * q), cap)


TAUS = np.r_[np.arange(0, 100, 2), np.arange(100, 1001, 10)].astype(float)
v1, v2 = np.zeros(len(y)), np.zeros(len(y))
ch1, ch2 = [], []
for k, (a, b) in enumerate(folds):
    sa, sb, cap = inner[k] * gate[a], bo[b] * gate[b], capf[k]
    ya = y[a]
    # Z1: zero-snap only, fine grid
    tau = TAUS[int(np.argmax([nmae(ya, apply_post(sa, t, 1.0, cap)) for t in TAUS]))]
    v1[b] = apply_post(sb, tau, 1.0, cap)
    ch1.append(float(tau))
    # Z2: joint (tau, alpha), Nelder-Mead from several starts
    obj = lambda x: -nmae(ya, apply_post(sa, max(x[0], 0.0), x[1], cap))
    best = None
    for x0 in [(tau + 1.0, 1.0), (1.0, 1.0), (50.0, 1.0), (150.0, 1.02), (300.0, 0.98), (25.0, 0.99)]:
        r = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-8, "maxiter": 400, "initial_simplex": [x0, (x0[0] + 20.0, x0[1]), (x0[0], x0[1] + 0.01)]})
        if best is None or r.fun < best.fun:
            best = r
    t2, a2 = max(best.x[0], 0.0), best.x[1]
    v2[b] = apply_post(sb, t2, a2, cap)
    ch2.append([round(float(t2), 2), round(float(a2), 4), round(-float(best.fun) - nmae(ya, np.minimum(sa, cap)), 6)])
print("\nZ1 chosen tau per fold:", ch1)
print("Z2 chosen (tau, alpha, gain on the training rows) per fold:", ch2, flush=True)
judge("Z1 nested zero-snap only (fine grid)", v1)
judge("Z2 nested joint (tau, alpha), Nelder-Mead", v2)

pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "per-fold delta", "fold std", "Decision"]).to_string(index=False))
print(f"B0 fold std {g0.std():.5f}")
np.savez(common.OUT / "bx_r14_t3_post.npz", pred0=pred0, v1=v1, v2=v2)
log("done")
