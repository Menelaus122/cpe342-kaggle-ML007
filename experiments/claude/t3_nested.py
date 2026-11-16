"""Round 30 helper: the Task 3 pipeline rebuilt fully nested from the cached members (training rows only).

Members as scripts/nb_part2.py 7.3: five raw-THB models + the 10-seed hurdle and six ratio stage-2 tags (seed-mean mu), each read out
three ways (mixture median, hard gate, mean with smearing); exact non-negative L1 blend (HiGHS); median zero gate P(y>0) + shift > 0.5;
one scale; cap at the training maximum. Read-outs, blend and scale are fitted on 4 folds and applied to the 5th.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task3  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_params, load_preds  # noqa: E402

SEEDS3 = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
sfx3 = lambda s: "" if s == 42 else f"_s{s}"
BAG3 = ("ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] XGB", "ratio[hist] Cat")
RATIO3 = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
clip0 = lambda a: np.clip(a, 0, None)
SCALES = np.linspace(0.85, 1.15, 61)


def load_all():
    raw = pd.read_csv(DATA_DIRS[3] / "train.csv")
    y = raw[task3.TARGET].values.astype(float)
    rep, _ = task3.fill_identities(raw)
    Xf = task3.add_features(rep)
    X, _, _ = task3.impute_segment(Xf, Xf[Xf.segment.isna()].head(5).copy())     # dummy apply-frame of training rows
    folds = make_folds(y, kind="regression")
    raw_pool = {}
    for name, key, log in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False),
                           ("Cat-MAE raw THB", "t3r_cat_raw", False), ("LGB-L1 log1p", "t3r_lgb_log", True),
                           ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
        r = load_preds(key)
        raw_pool[name] = clip0(np.expm1(r["oof"])) if log else clip0(r["oof"])
    p_oof = np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["p_oof"] for s in SEEDS3], 0)
    MU = {"Hurdle": np.mean([load_preds(f"t3r_hurdle{sfx3(s)}")["mu_oof"] for s in SEEDS3], 0)}
    for tag in RATIO3:
        MU[tag] = np.mean([load_preds(slug3(tag) + sfx3(s))["mu_oof"] for s in (SEEDS3 if tag in BAG3 else (42,))], 0)
    return dict(raw=raw, y=y, X=X, rep=rep, folds=folds, raw_pool=raw_pool, p=p_oof, MU=MU)


def nested_final(D, p, MU, log=None):
    """Fully nested OOF prediction for hurdle probabilities p and the stage-2 dict MU."""
    y, folds = D["y"], D["folds"]
    final = np.zeros(len(y))
    for i, (tr, va) in enumerate(folds):
        cols_tr, cols_va = [v[tr] for v in D["raw_pool"].values()], [v[va] for v in D["raw_pool"].values()]
        shift = 0.0
        for tag, mu in MU.items():
            ro = task3.readouts(y[tr], p[tr], mu[tr])
            if tag == "Hurdle":
                shift = ro["shift"]
            for k in ("mixture-median", "hard gate", "E[y] + smearing"):
                cols_tr.append(task3.apply_readout(k, p[tr], mu[tr], ro))
                cols_va.append(task3.apply_readout(k, p[va], mu[va], ro))
        Ptr, Pva = np.column_stack(cols_tr), np.column_stack(cols_va)
        w = task3.l1_blend_weights_exact(y[tr], [Ptr[:, j] for j in range(Ptr.shape[1])])
        gtr, gva = np.clip(p[tr] + shift, 0, 1) > 0.5, np.clip(p[va] + shift, 0, 1) > 0.5
        s = SCALES[int(np.argmin([np.abs(y[tr] - Ptr @ w * s_ * gtr).sum() for s_ in SCALES]))]
        final[va] = np.minimum(Pva @ w * s * gva, y[tr].max())
        if log:
            log(f"    nested fold {i}: NMAE {nmae(y[va], final[va]):.5f}")
    return final


def compare(D, ref, cand, label, floor=-0.0002):
    y, folds = D["y"], D["folds"]
    per = [nmae(y[b], cand[b]) - nmae(y[b], ref[b]) for _, b in folds]
    d = nmae(y, cand) - nmae(y, ref)
    ok = d > 0 and sum(x > 0 for x in per) >= 3 and min(per) >= floor
    return (f"  {label:34s} {nmae(y, cand):.5f} vs {nmae(y, ref):.5f}  delta {d:+.5f}  folds +{sum(x > 0 for x in per)}/5 "
            f"({' '.join(f'{x:+.5f}' for x in per)})  -> {'PASS' if ok else 'fail'}")
