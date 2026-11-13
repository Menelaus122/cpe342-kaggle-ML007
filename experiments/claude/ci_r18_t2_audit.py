"""Round 18, Task 2 on the re-released data: where do the errors come from, and is 0.7568 a ceiling?

    python ci_r18_t2_audit.py

Uses the pipeline's cached member OOF predictions (t2_lgb / t2_xgb / t2_cat / t2_lr) and the pipeline's folds; training rows only.
  A. the notebook's LogReg stacker + class multipliers, rebuilt; multipliers also chosen NESTED (4 folds -> 5th) to see their real gain
  B. confusion matrix, per-class precision / recall / F1
  C. label-noise floor: error rate among confident predictions, calibration by confidence band
  D. are the old feature assumptions still true? (noise columns, 2x2 engagement x spending design, sign of every indicator)
  E. class-conditional independence (within-class rank correlations) and a naive-Bayes member: if features are independent given
     the class, NB on per-feature class histograms is the Bayes rule, so its score estimates the ceiling
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402

NAMES = ["Casual", "Grinder", "Social", "Whale"]
train, test = task2.load()
X, y, Xt = task2.build(train, test)
folds = make_folds(y)
M = {k: np.load(f"outputs/preds/t2_{k}.npz") for k in ("lgb", "xgb", "cat", "lr")}
mf1 = lambda p: f1_score(y, p, average="macro")

# ---------------------------------------------------------------- A. stacker, multipliers in-sample vs nested
S = np.column_stack([np.log(np.clip(M[k]["oof"], 1e-6, 1)) for k in M])
stack = np.zeros((len(y), 4))
for tr, va in folds:
    stack[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
W, sc = optimize_class_weights(y, stack)
nested = np.zeros(len(y), int)
for tr, va in folds:
    w, _ = optimize_class_weights(y[tr], stack[tr])
    nested[va] = apply_class_weights(stack[va], w)
print("== A. members and stack (OOF macro-F1)")
for k in M:
    print(f"  {k:4s} {mf1(M[k]['oof'].argmax(1)):.4f}")
print(f"  stack argmax {mf1(stack.argmax(1)):.4f} | + multipliers in-sample {sc:.4f} (w = {np.round(W, 3).tolist()}) | nested {mf1(nested):.4f}")
corr = pd.DataFrame({k: M[k]["oof"][np.arange(len(y)), y] for k in M}).corr().round(3)
print("  correlation of members' P(true class):\n" + corr.to_string())
pred = apply_class_weights(stack, W)

# ---------------------------------------------------------------- B. confusion
print("\n== B. confusion (rows = true, counts) and per-class scores")
print(pd.DataFrame(confusion_matrix(y, pred), index=NAMES, columns=NAMES).to_string())
p, r, f, n = precision_recall_fscore_support(y, pred)
print(pd.DataFrame({"precision": p, "recall": r, "F1": f, "n": n}, index=NAMES).round(4).to_string())
cm = confusion_matrix(y, pred)
off = [(cm[i, j] + cm[j, i], NAMES[i], NAMES[j]) for i in range(4) for j in range(i + 1, 4)]
print("  most confused pairs (both directions):", sorted(off, reverse=True)[:3])

# ---------------------------------------------------------------- C. label-noise floor
print("\n== C. errors by confidence (stack max-probability)")
mx = stack.max(1)
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .95), (.95, 1.01)]:
    m = (mx >= lo) & (mx < hi)
    print(f"  p in [{lo:.2f},{hi:.2f}): {m.mean():6.1%} of rows, accuracy {(stack[m].argmax(1) == y[m]).mean():.4f}, mean p {mx[m].mean():.3f}")

# ---------------------------------------------------------------- D. old assumptions on the new data
print("\n== D. feature assumptions on the new data")
raw = train.drop(columns=["player_id", "segment"])
num = raw.select_dtypes("number")
eng = np.isin(y, [1, 3]).astype(int)   # Grinder / Whale = high engagement
spd = np.isin(y, [2, 3]).astype(int)   # Social / Whale = high spending
rows = []
for c in num.columns:
    v = num[c]
    ok = v.notna()
    rows.append([c, spearmanr(v[ok], eng[ok])[0], spearmanr(v[ok], spd[ok])[0], v.isna().mean()])
D = pd.DataFrame(rows, columns=["col", "rho_engagement", "rho_spending", "nan"]).set_index("col")
D["group"] = ["ENG+" if c in task2.ENGAGEMENT_POS else "ENG-" if c in task2.ENGAGEMENT_NEG else "SPD+" if c in task2.SPENDING_POS
              else "noise" if c in task2.NOISE_COLS else "-" for c in D.index]
print(D.round(3).sort_values("group").to_string())
bad = D[((D.group == "ENG+") & (D.rho_engagement <= 0.02)) | ((D.group == "ENG-") & (D.rho_engagement >= -0.02))
        | ((D.group == "SPD+") & (D.rho_spending <= 0.02))]
print("  indicators whose sign no longer fits their group:", bad.index.tolist() or "none")
strong_other = D[(D.group == "-") & ((D.rho_engagement.abs() > 0.05) | (D.rho_spending.abs() > 0.05))]
print("  ungrouped columns with |rho| > 0.05:", strong_other.index.tolist() or "none")
print("  noise columns:", D.loc[D.group == "noise", ["rho_engagement", "rho_spending"]].abs().max().round(4).to_dict())
print("  class shares by engagement x spending:", pd.crosstab(eng, spd, normalize=True).round(3).values.tolist())

# ---------------------------------------------------------------- E. class-conditional independence and naive Bayes
print("\n== E. class-conditional dependence and naive Bayes")
Q = num.rank(pct=True)
for k in range(4):
    C = Q[y == k].corr(method="pearson").values
    iu = np.triu_indices_from(C, 1)
    a = np.abs(C[iu])
    print(f"  {NAMES[k]:7s}: within-class |rank corr| mean {np.nanmean(a):.4f}, max {np.nanmax(a):.3f}")
allc = Q.corr().values
print(f"  all classes pooled: |rank corr| mean {np.nanmean(np.abs(allc[np.triu_indices_from(allc, 1)])):.4f}")


def nb_oof(df, n_bins=24, alpha=1.0):
    """Naive Bayes on per-column class histograms (quantile bins fitted on the training fold, NaN = own bin)."""
    out = np.zeros((len(y), 4))
    cols = list(df.columns)
    for tr, va in folds:
        L = np.tile(np.log(np.bincount(y[tr], minlength=4) / len(tr)), (len(va), 1))
        for c in cols:
            v = df[c]
            if v.dtype == object or str(v.dtype) in ("category", "str", "string"):
                codes = v.astype(str).fillna("NA")
                cats = {u: i for i, u in enumerate(sorted(codes.iloc[tr].unique()))}
                b = codes.map(cats).fillna(len(cats)).astype(int).values
                nb = len(cats) + 1
            else:
                vv = v.values.astype(float)
                edges = np.unique(np.nanquantile(vv[tr], np.linspace(0, 1, n_bins + 1)[1:-1]))
                b = np.where(np.isnan(vv), len(edges) + 1, np.searchsorted(edges, vv))
                nb = len(edges) + 2
            H = np.zeros((4, nb))
            np.add.at(H, (y[tr], b[tr]), 1)
            H = np.log((H + alpha) / (H.sum(1, keepdims=True) + alpha * nb))
            L += H[:, b[va]].T
        P = np.exp(L - L.max(1, keepdims=True))
        out[va] = P / P.sum(1, keepdims=True)
    return out


nb = nb_oof(raw)
print(f"  naive Bayes on all raw columns: OOF macro-F1 {mf1(nb.argmax(1)):.4f}; with multipliers {optimize_class_weights(y, nb)[1]:.4f}")
nb2 = nb_oof(raw.drop(columns=task2.NOISE_COLS))
print(f"  naive Bayes without the 3 noise columns: {mf1(nb2.argmax(1)):.4f}")
S2 = np.column_stack([S, np.log(np.clip(nb, 1e-6, 1))])
st2 = np.zeros((len(y), 4))
for tr, va in folds:
    st2[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S2[tr], y[tr]).predict_proba(S2[va])
nest2 = np.zeros(len(y), int)
for tr, va in folds:
    w, _ = optimize_class_weights(y[tr], st2[tr])
    nest2[va] = apply_class_weights(st2[va], w)
print(f"  stack + NB member: argmax {mf1(st2.argmax(1)):.4f}, nested multipliers {mf1(nest2):.4f} (stack alone nested {mf1(nested):.4f})")
np.savez("experiments/claude/outputs/r18_t2_nb.npz", nb_oof=nb, stack_oof=stack)
