from common import *
tr, te, X, y, Xt, folds = load_t2()
base = {n: load_preds(n) for n in T2_MODELS}
new = {n: dict(np.load(OUT / f"{n}.npz")) for n in ["t2_lgb_factor", "t2_nb16", "t2_stumps_raw", "t2_stumps_FE"]}
allm = {**base, **new}
def run(names):
    s, so, st, w = t2_score([allm[n]["oof"] for n in names], [allm[n]["test"] for n in names], y, folds)
    return s
print(f"baseline stack (4)            {run(T2_MODELS):.4f}")
for extra in [["t2_lgb_factor"], ["t2_nb16"], ["t2_stumps_raw"], ["t2_stumps_FE"], ["t2_lgb_factor", "t2_nb16", "t2_stumps_FE"]]:
    print(f"+ {', '.join(extra):30s} {run(T2_MODELS + extra):.4f}")
print(f"swap lgb->lgb_factor          {run(['t2_lgb_factor', 't2_xgb', 't2_cat', 't2_lr']):.4f}")
# seed noise of the class-weight step: re-split the stacker folds
from src.modeling import make_folds
for sd in [1, 2, 3]:
    f2_ = make_folds(y, seed=sd)
    a = t2_score([allm[n]["oof"] for n in T2_MODELS], [allm[n]["test"] for n in T2_MODELS], y, f2_)[0]
    b = t2_score([allm[n]["oof"] for n in T2_MODELS + ["t2_lgb_factor", "t2_nb16", "t2_stumps_FE"]], [allm[n]["test"] for n in T2_MODELS + ["t2_lgb_factor", "t2_nb16", "t2_stumps_FE"]], y, f2_)[0]
    print(f"stacker seed {sd}: base {a:.4f}  +3 new {b:.4f}  diff {b - a:+.4f}")
