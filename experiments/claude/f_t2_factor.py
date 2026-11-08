"""Task 2: supervised latent-factor scores (in-fold logistic regressions) as extra GBDT features."""
import time
from common import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from src.modeling import run_cv

tr, te, X, y, Xt, folds = load_t2()
cat2 = [c for c in X.columns if str(X[c].dtype) == "category"]
raw_num = [c for c in X.columns if c in tr.columns and c not in cat2] + ["engagement_level", "loyalty_tier", "skill_tier"]
raw_num = list(dict.fromkeys(raw_num))
ll = lambda p: -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def _lin(Za, ya, Zbs):
    out = [[] for _ in Zbs]
    for tgt in (np.isin(ya, [1, 3]).astype(int), np.isin(ya, [2, 3]).astype(int)):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Za, tgt)
        for i, Zb in enumerate(Zbs):
            out[i].append(m.decision_function(Zb))
    m = LogisticRegression(C=1.0, max_iter=3000).fit(Za, ya)
    for i, Zb in enumerate(Zbs):
        lp = m.predict_log_proba(Zb)
        out[i] += [lp[:, k] for k in range(4)]
    return [np.column_stack(o) for o in out]


NAMES = ["lin_E", "lin_S", "lin_mn0", "lin_mn1", "lin_mn2", "lin_mn3"]


def factor_fold_fn(X_tr, y_tr, X_others):
    qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", subsample=10**6, random_state=SEED)
    Za = np.nan_to_num(qt.fit_transform(X_tr[raw_num]))
    Zos = [np.nan_to_num(qt.transform(x[raw_num])) for x in X_others]
    oof = np.zeros((len(Za), len(NAMES)))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Za, y_tr):
        oof[b] = _lin(Za[a], y_tr[a], [Za[b]])[0]
    applied = _lin(Za, y_tr, Zos)
    X_tr = X_tr.copy(); X_others = [x.copy() for x in X_others]
    for j, n in enumerate(NAMES):
        X_tr[n] = oof[:, j]
        for x, a in zip(X_others, applied):
            x[n] = a[:, j]
    X_tr["lin_ExS"] = X_tr.lin_E * X_tr.lin_S
    for x in X_others:
        x["lin_ExS"] = x.lin_E * x.lin_S
    return X_tr, X_others


if __name__ == "__main__":
    base = load_preds("t2_lgb")
    wb, sb = optimize_class_weights(y, base["oof"])
    print(f"t2_lgb baseline: argmax {macro_f1(y, base['oof'].argmax(1)):.4f} +w {sb:.4f} logloss {ll(base['oof']):.4f}")
    t0 = time.time()
    r = run_cv("lgb", "multiclass", P2_lgb, X, y, Xt, folds, n_classes=4, name="lgb+factor", fold_fn=factor_fold_fn, verbose=False)
    w, s = optimize_class_weights(y, r["oof"])
    print(f"lgb + factor scores: argmax {macro_f1(y, r['oof'].argmax(1)):.4f} +w {s:.4f} logloss {ll(r['oof']):.4f} [{time.time()-t0:.0f}s]")
    print(r["importance"].head(12).round(4).to_string())
    np.savez_compressed(OUT / "t2_lgb_factor.npz", oof=r["oof"], test=r["test"])
