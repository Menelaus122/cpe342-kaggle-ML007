"""Task 3: is y == 0 a rule of the features (like segment) or random within a segment?"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from common import OUT
from src import task3
from t3_fill import fill
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

tr, _ = task3.load()
d = fill(tr)[0]
y = d[task3.TARGET].values
zero = (y == 0).astype(int)
h = np.load(OUT / "t3_identity_hurdle.npz")
seg = d.segment.values
print("stage-1 OOF AUC P(y>0): all", round(roc_auc_score(1 - zero, h["p_oof"]), 4))
for s in (0, 1, 2, 3):
    m = seg == s
    print(f"  segment {s}: zero share {zero[m].mean():.3f}, AUC within segment {roc_auc_score(1 - zero[m], h['p_oof'][m]):.4f}")
cols = [c for c in d.columns if c not in ("player_id", task3.TARGET)]
for s in (0, 3):
    m = seg == s
    t = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200, random_state=0).fit(d.loc[m, cols].fillna(-999), zero[m])
    print(f"\nsegment {s} depth-2 tree (zero vs positive):\n" + export_text(t, feature_names=cols, decimals=2, show_weights=True))
