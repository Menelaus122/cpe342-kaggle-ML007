"""Insight round 7: what Task 4 score does the TAs' own "ground baseline" imply?

Instruction PDF, slide 27: ground baseline = Logistic Regression / Random Forest / CNN / Isolation Forest, models
from class, Weighted Score (Public) = 0.68; challenge baseline = 0.75 (public). We rebuild plain default versions of
the tabular parts (single stratified 80/20 split, median imputation, sklearn defaults), score them with the official
metrics, and solve 0.68 = 0.2 * (S1 + S2 + S3 + S4 + S5) for S4. Task 5 is graded against the id block
(ids >= 23,819), which the v5 upload confirmed. Nothing here touches our submission.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402


def f_beta(y, p, b):
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    return (1 + b * b) * tp / ((1 + b * b) * tp + b * b * fn + fp)


def split(d, t, strat=True):
    df = pd.read_csv(f"{d}/train.csv").dropna(subset=[t])
    X, y = df.drop(columns=["player_id", t]), df[t].values
    txt = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X = pd.get_dummies(X, columns=txt, dtype=float)          # e.g. Task 2 region ('APAC', ...)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y if strat else None)


S = {}
# ---------------------------------------------------------------- Task 1 (positive-class F2)
Xa, Xb, ya, yb = split("01_binary_classification", "is_cheater")
ya, yb = ya.astype(int), yb.astype(int)
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, ya)
p = lr.predict_proba(Xb)[:, 1]
S["T1 LR, threshold 0.5"] = f_beta(yb, (p >= 0.5).astype(int), 2)
S["T1 LR, best threshold"] = max(f_beta(yb, (p >= t).astype(int), 2) for t in np.linspace(0.02, 0.6, 59))
rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_jobs=-1, random_state=0)).fit(Xa, ya)
p = rf.predict_proba(Xb)[:, 1]
S["T1 RF, threshold 0.5"] = f_beta(yb, (p >= 0.5).astype(int), 2)
S["T1 RF, best threshold"] = max(f_beta(yb, (p >= t).astype(int), 2) for t in np.linspace(0.02, 0.6, 59))
print({k: round(v, 4) for k, v in S.items()}, flush=True)

# ---------------------------------------------------------------- Task 2 (macro F1)
Xa, Xb, ya, yb = split("02_multiclass_classification", "segment")
rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_jobs=-1, random_state=0)).fit(Xa, ya)
S["T2 RF"] = f1_score(yb, rf.predict(Xb), average="macro")
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xa, ya)
S["T2 LR"] = f1_score(yb, lr.predict(Xb), average="macro")
print({k: round(v, 4) for k, v in S.items() if k.startswith("T2")}, flush=True)

# ---------------------------------------------------------------- Task 3 (1 / (1 + sum|e| / sum y))
Xa, Xb, ya, yb = split("03_regression", "spending_30d", strat=False)
sc3 = lambda y, p: 1 / (1 + np.abs(y - p).sum() / y.sum())
rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=100)).fit(Xa, ya)
S["T3 RF"] = sc3(yb, rf.predict(Xb))
lin = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(Xa, ya)
S["T3 linear"] = sc3(yb, lin.predict(Xb))
print({k: round(v, 4) for k, v in S.items() if k.startswith("T3")}, flush=True)

# ---------------------------------------------------------------- Task 5 (positive-class F3 vs the id block)
t5 = pd.read_csv("05_anomaly_detection/test.csv")
truth = (t5.player_id.str[1:].astype(int) >= 23819).astype(int).values
X5 = SimpleImputer(strategy="median").fit_transform(t5.drop(columns=["player_id"]))
for c in ("auto", 0.05, 0.08, 0.1):
    iso = IsolationForest(random_state=0, contamination=c, n_jobs=-1).fit(X5)
    S[f"T5 IsolationForest, contamination {c}"] = f_beta(truth, (iso.predict(X5) == -1).astype(int), 3)
print({k: round(v, 4) for k, v in S.items() if k.startswith("T5")}, flush=True)

# ---------------------------------------------------------------- implied Task 4 for the ground baseline
print("\nGround baseline 0.68 (public) = 0.2 * (S1 + S2 + S3 + S4 + S5)  ->  implied S4")
rows = []
for k1 in [k for k in S if k.startswith("T1")]:
    for k2 in ("T2 RF", "T2 LR"):
        for k3 in ("T3 RF",):
            for k5 in [k for k in S if k.startswith("T5")]:
                rest = S[k1] + S[k2] + S[k3] + S[k5]
                rows.append([k1, k2, k3, k5, rest, 5 * 0.68 - rest])
R = pd.DataFrame(rows, columns=["T1", "T2", "T3", "T5", "sum of 4 tabular", "implied S4"])
print(R.sort_values("implied S4").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nimplied S4 range: {R['implied S4'].min():.2f} .. {R['implied S4'].max():.2f}  (chance level about 0.19-0.21; "
      f"our own uploads imply about 0.21)")
best_tab = 0.8364 + 0.7549 + 0.79403 + 1.0
print(f"challenge baseline 0.75: even with OUR best leak-free tabular scores and a perfect Task 5 "
      f"(sum {best_tab:.3f}), S4 >= {5 * 0.75 - best_tab:.2f}")
