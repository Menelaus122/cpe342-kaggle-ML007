"""Task 3: non-linear level-2 stacker (LightGBM, L1 objective) on the v8 pool's OOF predictions + stage-1 p,
segment and log history. Nested: the meta-model for fold f is trained on the other 4 folds' OOF rows only."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import common  # noqa
from common import OUT, t3_blend_score
from src import task3
from src.metrics import nmae
from src.modeling import make_folds, fit_predict
from t3_fill import fill
from t3_segment import impute_segment
import pandas as pd

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, Xt, _ = impute_segment(task3.add_features(trf), task3.add_features(tef))
v8 = np.load(OUT / "t3_v8.npz"); qz = np.load(OUT / "t3_quantile_members.npz"); qr = np.load(OUT / "t3_quantile_readout.npz")
P, gate = np.column_stack([v8["P_oof"], qr["oof"]]), v8["gate_oof"]
s_lin, lin = t3_blend_score(y, folds, P, gate)
M = pd.DataFrame(np.log1p(P), columns=[f"m{i}" for i in range(P.shape[1])])
M["p"] = qz["p_oof"]; M["segment"] = X.segment.values; M["log_hist"] = np.log1p(trf.historical_spending.values)
M["lin_blend"] = np.log1p(lin)
res = {}
for tag, prm in [("shallow", {"objective": "regression_l1", "learning_rate": 0.02, "num_leaves": 7, "min_child_samples": 200}),
                 ("medium", {"objective": "regression_l1", "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 100})]:
    o = np.zeros(len(y))
    for a, v in folds:
        # inner split of the training folds for early stopping
        rng = np.random.default_rng(0); a = rng.permutation(a); cut = int(len(a) * 0.85)
        _, (pv,), _ = fit_predict("lgb", "regression", prm, M.iloc[a[:cut]], y[a[:cut]], M.iloc[a[cut:]], y[a[cut:]], [M.iloc[v]])
        o[v] = np.clip(pv, 0, None)
    res[tag] = o
    print(f"level-2 LightGBM-L1 ({tag}): {nmae(y, o):.5f}; 50/50 with the linear blend {nmae(y, 0.5 * o + 0.5 * lin):.5f} "
          f"(linear blend {s_lin:.5f})", flush=True)
