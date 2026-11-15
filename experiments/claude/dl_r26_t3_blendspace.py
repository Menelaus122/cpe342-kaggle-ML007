"""Round 26, Task 3: in which space are predictions combined (the slide's "common mistakes in target scaling")?

    python dl_r26_t3_blendspace.py

What the pipeline does (master notebook 7.3, src/task3.py):
  * seeds: the 10 seeds of each stage-2 model are averaged in LOG space (mean of mu = predicted median of log1p(y)), then one
    read-out (mixture median / hard gate / smearing) maps mu to THB. For a median target this is the geometric mean of the seeds.
  * members: the 26 read-outs are blended in THB with exact non-negative L1 weights fitted out of fold, then gate, scale, cap.
Alternatives, everything nested (weights and scale on 4 folds -> 5th), same gate and cap:
  A  seeds averaged in THB (each seed read out separately, then the arithmetic mean), same L1 blend
  B  members combined in log space with the pipeline's weights (normalised): expm1(sum w log1p(P))  = weighted geometric mean
  C  weighted median of the members in THB (pipeline weights)
  D  unweighted mean / median of the members in THB; unweighted geometric mean
"""
import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t3_train, t3_test = task3.load()
X3, y3, X3_test, _ = task3.build_repaired(t3_train, t3_test)
y3 = np.asarray(y3, float)
folds = make_folds(y3, kind="regression")
clip0 = lambda a: np.clip(a, 0, None)
direct = {}
for name, key, log in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                       ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    direct[name] = clip0(np.expm1(r["oof"])) if log else clip0(r["oof"])
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
KINDS = ("mixture-median", "hard gate", "E[y] + smearing")
pool_log, pool_thb = dict(direct), dict(direct)
for tag, ms in mus.items():
    mu = np.mean(ms, 0)
    ro = task3.readouts(y3, p_oof, mu)
    if tag == "Hurdle":
        shift = ro["shift"]
    for k in KINDS:
        pool_log[f"{tag} {k}"] = task3.apply_readout(k, p_oof, mu, ro)                               # pipeline: seeds in log space
        pool_thb[f"{tag} {k}"] = np.mean([task3.apply_readout(k, p_oof, m, ro) for m in ms], 0)      # A: seeds in THB
gate = (np.clip(p_oof + shift, 0, 1) > 0.5).astype(float)
scales = np.linspace(0.85, 1.15, 61)
names = list(pool_log)
spread = np.std([np.asarray(m) for m in mus["ratio[hist] LGB tuned"]], 0)
print(f"{len(y3)} rows; seed-to-seed SD of mu (ratio[hist] LGB tuned): median {np.median(spread):.4f}, p99 {np.quantile(spread, .99):.4f} "
      f"(log units; arithmetic vs geometric mean differ by about SD^2 / 2 = {np.median(spread) ** 2 / 2:.1e} relative)")
print("member NMAE, seeds in log space vs in THB (bagged members only):")
for n in names:
    if n.split(" ")[0] in ("Hurdle",) or any(n.startswith(t) for t in BAG3):
        print(f"  {n:38s} {nmae(y3, pool_log[n]):.5f} | {nmae(y3, pool_thb[n]):.5f} ({nmae(y3, pool_thb[n]) - nmae(y3, pool_log[n]):+.5f})")


def wmedian(P, w):
    o = np.argsort(P, axis=1)
    Ps, ws = np.take_along_axis(P, o, 1), w[o]
    c = np.cumsum(ws, 1)
    j = (c >= 0.5 * c[:, -1:]).argmax(1)
    return Ps[np.arange(len(P)), j]


def finish(btr, bva, tr, va):
    s = scales[int(np.argmin([np.abs(y3[tr] - btr * s * gate[tr]).sum() for s in scales]))]
    return np.minimum(bva * s * gate[va], y3[tr].max())


PL = np.column_stack([pool_log[n] for n in names])
PT = np.column_stack([pool_thb[n] for n in names])
LL = np.log1p(PL)
out = {k: np.zeros(len(y3)) for k in ["pipeline", "A seeds in THB", "B weighted geometric", "C weighted median", "D mean", "D median", "D geometric mean"]}
for f, (tr, va) in enumerate(folds):
    w = task3.l1_blend_weights_exact(y3[tr], [PL[tr, j] for j in range(PL.shape[1])])
    out["pipeline"][va] = finish(PL[tr] @ w, PL[va] @ w, tr, va)
    wn = w / w.sum()
    out["B weighted geometric"][va] = finish(np.expm1(LL[tr] @ wn), np.expm1(LL[va] @ wn), tr, va)
    out["C weighted median"][va] = finish(wmedian(PL[tr], w), wmedian(PL[va], w), tr, va)
    out["D mean"][va] = finish(PL[tr].mean(1), PL[va].mean(1), tr, va)
    out["D median"][va] = finish(np.median(PL[tr], 1), np.median(PL[va], 1), tr, va)
    out["D geometric mean"][va] = finish(np.expm1(LL[tr].mean(1)), np.expm1(LL[va].mean(1)), tr, va)
    wt = task3.l1_blend_weights_exact(y3[tr], [PT[tr, j] for j in range(PT.shape[1])])
    out["A seeds in THB"][va] = finish(PT[tr] @ wt, PT[va] @ wt, tr, va)
    print(f"fold {f} done (weight sum {w.sum():.4f})", flush=True)
fold_of = np.zeros(len(y3), int)
for i, (_, va) in enumerate(folds):
    fold_of[va] = i
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y3), len(y3)) for _ in range(1000)]
base = out["pipeline"]
print(f"\npipeline (seeds in log space, members L1-blended in THB), nested: NMAE {nmae(y3, base):.5f}")
for k, o in out.items():
    if k == "pipeline":
        continue
    pf = [nmae(y3[fold_of == i], o[fold_of == i]) - nmae(y3[fold_of == i], base[fold_of == i]) for i in range(5)]
    bs = [nmae(y3[i], o[i]) - nmae(y3[i], base[i]) for i in boot]
    print(f"  {k:22s} NMAE {nmae(y3, o):.5f}  delta {nmae(y3, o) - nmae(y3, base):+.5f} [{np.quantile(bs, .025):+.5f}, {np.quantile(bs, .975):+.5f}]  folds +{sum(x > 0 for x in pf)}/5")
