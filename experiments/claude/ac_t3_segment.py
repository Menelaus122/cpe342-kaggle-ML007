"""Task 3: which columns determine `segment`? (train rows with a known segment only)"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from src import task3
from t3_fill import fill
from sklearn.tree import DecisionTreeClassifier, export_text

tr, te = task3.load()
d = fill(tr)[0]
k = d.segment.notna()
cols = [c for c in d.columns if c not in ("player_id", "segment", task3.TARGET)]
for depth in (1, 2, 3, 4):
    Xk = d.loc[k, cols].fillna(-999)
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xk, d.segment[k])
    print(f"depth {depth}: train accuracy {t.score(Xk, d.segment[k]):.4f}; top features",
          sorted(zip(t.feature_importances_, cols), reverse=True)[:3])
t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(d.loc[k, cols].fillna(-999), d.segment[k])
print(export_text(t, feature_names=cols, decimals=3))
for c in sorted(zip(t.feature_importances_, cols), reverse=True)[:3]:
    print(c[1], "missing share train", round(d[c[1]].isna().mean(), 4), "| among segment-missing rows",
          round(d.loc[~k, c[1]].isna().mean(), 4))
