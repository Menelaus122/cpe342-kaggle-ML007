"""Round 36, Task 3 (light, about 10 CPU minutes): direct quantile members in the L1 blend pool, and a gate calibration restricted to
the boundary band (training rows only).

    python gd_r36_t3_quantile_gate.py

item 1  The pool holds raw-THB L1 / Tweedie members and ratio stage-2 models read out through the mixture median (the
        (1 - 0.5 / p) quantile of the mixture). New: LightGBM members trained directly on the pinball loss of the raw THB target,
        q = 0.40 and q = 0.45 (the brief), the pipeline's tuned LightGBM settings with the objective replaced, out of fold on the
        pipeline folds with early stopping on the validation fold (as every pipeline member). They enter the exact non-negative L1
        blend (HiGHS) next to the 26 existing columns; everything after the blend (gate, scale, cap) is as in the pipeline, fully
        nested (t3_nested.nested_final). Round 6 (ak_t3_quantile.py) tried a 12-level quantile read-out of the old pipeline
        (+0.00007); quantile members inside the current pool are new.
item 2  Recalibration of the stage-1 P(y > 0) only inside the band where the median gate decides, p in [0.45, 0.65]: a logistic
        fit of the label on logit(p) using the training folds' rows of the band, applied to the held-out rows of the band (p
        outside the band unchanged; the calibrated p is clipped into the band so no row jumps across it). Variants: one fit for
        all rows; one fit per imputed segment. Round 30 calibrated p over the whole range (isotonic, Platt, isotonic per segment:
        +0.00001 / -0.00000 / +0.00002); the band restriction is new.
Reference = the pipeline rebuilt fully nested from the cached members (NMAE 0.79457).
Acceptance (user, 2026-11-16, Task 3): pooled delta > 0, >= 3/5 folds better, no fold below -0.0002.
"""
import os
import time

NT = int(os.environ.get("R33_THREADS", 3))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import t3_nested as T  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import run_cv  # noqa: E402
from src.utils import load_params  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
D = T.load_all()
y, p, folds = D["y"], D["p"], D["folds"]
X = D["X"].select_dtypes("number").drop(columns=[c for c in (task3.TARGET, "player_id", "id") if c in D["X"].columns])
seg = D["X"]["segment"].values.astype(int)
ref = T.nested_final(D, p, D["MU"])
log(f"{len(y)} training rows, {X.shape[1]} features; reference nested NMAE {nmae(y, ref):.5f}")

# item 1: quantile members
from pathlib import Path  # noqa: E402

OUT = Path("experiments/claude/outputs/r36")
OUT.mkdir(parents=True, exist_ok=True)
f_q = OUT / "t3_quantile_members.npz"
P3 = dict(load_params("task3_lgb") or {})
P3["learning_rate"] = max(P3.get("learning_rate", 0.05), 0.05)          # coarser than the pipeline's final fits: the CPU budget is 10 minutes
P3.update(n_estimators=3000, n_jobs=NT)
if f_q.exists():
    d = np.load(f_q)
    QM = {q: d[f"q{q}"] for q in (0.40, 0.45)}
else:
    QM = {}
    for q in (0.40, 0.45):
        r = run_cv("lgb", "regression", {**P3, "objective": "quantile", "alpha": q, "metric": "quantile"}, X, y, X.iloc[:5], folds,
                   early_stopping=100, verbose=False)
        QM[q] = np.clip(r["oof"], 0, None)
        log(f"quantile member q = {q}: best iterations {r['best_iters']}; alone NMAE {nmae(y, QM[q]):.5f}; share predicted 0 "
            f"{np.mean(QM[q] < 0.5):.3f}; coverage P(y <= pred) {np.mean(y <= QM[q]):.3f}")
    np.savez(f_q, **{f"q{q}": v for q, v in QM.items()})


def with_members(extra):
    D2 = dict(D)
    D2["raw_pool"] = {**D["raw_pool"], **extra}
    return D2


for name, extra in (("+ quantile member q = 0.40", {"LGB quantile 0.40": QM[0.40]}), ("+ quantile member q = 0.45", {"LGB quantile 0.45": QM[0.45]}),
                    ("+ both quantile members", {"LGB quantile 0.40": QM[0.40], "LGB quantile 0.45": QM[0.45]})):
    cand = T.nested_final(with_members(extra), p, D["MU"])
    print(T.compare(D, ref, cand, name), flush=True)
# blend weight the quantile members receive (fit on all rows, for the record)
D2 = with_members({"LGB quantile 0.40": QM[0.40], "LGB quantile 0.45": QM[0.45]})
cols = list(D2["raw_pool"].values())
for tag, mu in D["MU"].items():
    ro = task3.readouts(y, p, mu)
    cols += [task3.apply_readout(k, p, mu, ro) for k in ("mixture-median", "hard gate", "E[y] + smearing")]
w = task3.l1_blend_weights_exact(y, cols)
names = list(D2["raw_pool"]) + [f"{t} / {k}" for t in D["MU"] for k in ("mixture-median", "hard gate", "E[y] + smearing")]
log("L1 blend weights on all rows (record only): " + ", ".join(f"{n} {v:.3f}" for n, v in zip(names, w) if v > 0.005))

# item 2: band-restricted recalibration of p
t = (y > 0).astype(int)
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
LO, HI = 0.45, 0.65
band = (p >= LO) & (p <= HI)
log(f"band p in [{LO}, {HI}]: {band.mean():.3%} of the rows; positive rate there {t[band].mean():.3f} vs mean p {p[band].mean():.3f}")
for name, per_seg in (("band recalibration, one fit", False), ("band recalibration, per segment", True)):
    pc = p.copy()
    for a, b in folds:
        groups = np.unique(seg) if per_seg else [None]
        for s_ in groups:
            aa = a[band[a] & ((seg[a] == s_) if per_seg else True)]
            bb = b[band[b] & ((seg[b] == s_) if per_seg else True)]
            if len(aa) < 200 or not len(bb):
                continue
            m = LogisticRegression(C=1e6).fit(lg(p[aa])[:, None], t[aa])
            pc[bb] = np.clip(m.predict_proba(lg(p[bb])[:, None])[:, 1], LO, HI)
    cand = T.nested_final(D, pc, D["MU"])
    print(T.compare(D, ref, cand, name) + f"  (rows crossing 0.5: {np.mean((p > 0.5) != (pc > 0.5)):.3%})", flush=True)
log("done")
