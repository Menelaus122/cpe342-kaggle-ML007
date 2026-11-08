"""Task 2: histogram Naive Bayes (Bayes-optimal if features are class-conditionally independent)."""
import time
from common import *

tr, te, X, y, Xt, folds = load_t2()
raw = [c for c in tr.columns if c not in ("player_id", "segment")]
num = [c for c in raw if pd.api.types.is_numeric_dtype(tr[c])]
cat = [c for c in raw if not pd.api.types.is_numeric_dtype(tr[c])]

# missingness vs class
mr = pd.DataFrame({c: [tr.loc[y == k, c].isna().mean() for k in range(4)] for c in raw}).T
print("max spread of missing rate across classes:", (mr.max(1) - mr.min(1)).max().round(4))
# categorical vs numeric dependence within class (ordinal codes)
ordc = {"engagement_level": task2.ORDINAL_MAPS["engagement_level"], "loyalty_tier": task2.ORDINAL_MAPS["loyalty_tier"]}
for c, m in ordc.items():
    v = tr[c].map(m)
    for k in range(4):
        cc = pd.concat([v[y == k], tr.loc[y == k, num]], axis=1).corr(method="spearman")[c].drop(c).abs()
        print(f"  {c} class {k}: max within-class |rho| with numeric {cc.max():.3f} ({cc.idxmax()})")


def encode(df_fit, df_apply, col, nb):
    """bin index per row; last bin = missing."""
    if col in cat:
        cats = sorted(df_fit[col].dropna().unique())
        mp = {v: i for i, v in enumerate(cats)}
        f = lambda s: s.map(mp).fillna(len(cats)).astype(int).values
        return f(df_fit[col]), f(df_apply[col]), len(cats) + 1
    x = df_fit[col].values
    u = np.unique(x[~np.isnan(x)])
    if len(u) <= nb:
        edges = (u[:-1] + u[1:]) / 2
    else:
        edges = np.unique(np.nanquantile(x, np.linspace(0, 1, nb + 1)[1:-1]))
    def f(s):
        v = s.values.astype(float)
        b = np.searchsorted(edges, v)
        b[np.isnan(v)] = len(edges) + 1
        return b
    return f(df_fit[col]), f(df_apply[col]), len(edges) + 2


def nb_fit_predict(df_fit, y_fit, applies, cols, nb=64, alpha=1.0):
    K = 4
    prior = np.log(np.bincount(y_fit, minlength=K) / len(y_fit))
    outs = [np.tile(prior, (len(a), 1)) for a in applies]
    for c in cols:
        bf = None
        for i, a in enumerate(applies):
            b_fit, b_ap, nbin = encode(df_fit, a, c, nb)
            cnt = np.zeros((nbin, K))
            np.add.at(cnt, (b_fit, y_fit), 1)
            logp = np.log((cnt + alpha) / (cnt.sum(0, keepdims=True) + alpha * nbin))
            outs[i] += logp[b_ap]
    return [np.exp(o - o.max(1, keepdims=True)) / np.exp(o - o.max(1, keepdims=True)).sum(1, keepdims=True) for o in outs]


def nb_cv(cols, nb=64, alpha=1.0):
    oof = np.zeros((len(y), 4)); test = np.zeros((len(te), 4))
    for a, b in folds:
        po, pt = nb_fit_predict(tr.iloc[a], y[a], [tr.iloc[b], te], cols, nb, alpha)
        oof[b] = po; test += pt / len(folds)
    return oof, test

drop = ["random_metric_1", "random_metric_2", "random_metric_3"]
cols = [c for c in raw if c not in drop]
for nb in [16, 32, 64, 128]:
    t0 = time.time()
    oof, test = nb_cv(cols, nb)
    w, s = optimize_class_weights(y, oof)
    print(f"NB bins={nb:4d}: argmax {macro_f1(y, oof.argmax(1)):.4f}  +weights {s:.4f}  logloss {-np.mean(np.log(oof[np.arange(len(y)), y] + 1e-12)):.4f} [{time.time()-t0:.0f}s]")
    np.savez_compressed(OUT / f"t2_nb{nb}.npz", oof=oof, test=test)
for n in T2_MODELS:
    d = load_preds(n); print(n, "logloss", round(-np.mean(np.log(d["oof"][np.arange(len(y)), y] + 1e-12)), 4))
