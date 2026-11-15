"""Data-quality audit used by the notebooks (section 3.6): distribution shift, missingness, impossible values, duplicates.

Uses train and test FEATURES only; nothing here is fitted for a model and no test label exists.
* Adversarial validation: a LightGBM separating train (0) from test (1). Three views tell WHERE a shift is:
  all features / only the missing-value indicators / values with missing cells filled by the train median.
* Missingness vs target: chi-square (classes) or Mann-Whitney (spending) of target vs "cell is missing". p < 0.001 = informative.
* Physical checks: negatives, percentages outside [0, 100], coordinates outside the globe, sentinel codes, value spikes at a
  column's min/max (clipping), duplicate rows, constant columns, collinear pairs.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SENTINELS = [-1, -99, -999, -9999, 999, 9999, 99999, 999999]


def _same_precision(train, test, cols):
    """Cast numeric columns of both frames to float32: the notebooks shrink only the TRAIN frames (reduce_mem_usage), and a
    train/test classifier would otherwise separate them by number precision alone."""
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c]):
            tr[c], te[c] = tr[c].astype("float32"), te[c].astype("float32")
        else:
            tr[c], te[c] = tr[c].astype("object"), te[c].astype("object")
    return tr, te


def adversarial_auc(train, test, cols, n_splits=3, n_estimators=200, seed=0):
    train, test = _same_precision(train, test, cols)
    A = pd.concat([train[cols], test[cols]], ignore_index=True)
    for c in A.columns:
        if not pd.api.types.is_numeric_dtype(A[c]):
            A[c] = A[c].astype("category")
    lab = np.r_[np.zeros(len(train)), np.ones(len(test))]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(A[c])]
    views = {"all features": A,
             "missing indicators only": A[cols].isna().astype(int),
             "values (missing filled by train median)": A[num].fillna(train[num].median())}
    out = {}
    for name, V in views.items():
        oof = np.zeros(len(lab))
        for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(V, lab):
            m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
            oof[b] = m.fit(V.iloc[a], lab[a]).predict_proba(V.iloc[b])[:, 1]
        out[name] = roc_auc_score(lab, oof)
    ks = max(ks_2samp(train[c].dropna(), test[c].dropna()).statistic for c in num)
    return out, ks


def missing_vs_target(train, target, regression=False):
    rows = []
    y = train[target]
    for c in train.columns:
        if c in ("player_id", target):
            continue
        m = train[c].isna()
        if m.sum() < 30:
            continue
        p = mannwhitneyu(y[m], y[~m]).pvalue if regression else chi2_contingency(pd.crosstab(m, y))[1]
        rows.append((c, m.mean(), p))
    return pd.DataFrame(rows, columns=["column", "missing rate", "p"])


def physical_checks(df):
    issues = []
    for c in df.columns:
        if c == "player_id" or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if v.empty:
            continue
        lc = c.lower()
        if any(k in lc for k in ("percentage", "percent", "win_rate", "headshot")) and ((v > 100) | (v < 0)).any():
            issues.append(f"{c}: {int(((v > 100) | (v < 0)).sum())} values outside [0, 100]")
        if "lat" in lc and ((v < -90) | (v > 90)).any():
            issues.append(f"{c}: {int(((v < -90) | (v > 90)).sum())} latitudes outside [-90, 90]")
        if "lon" in lc and ((v < -180) | (v > 180)).any():
            issues.append(f"{c}: {int(((v < -180) | (v > 180)).sum())} longitudes outside [-180, 180]")
        if "count" in lc and (v < 0).any():
            issues.append(f"{c}: {int((v < 0).sum())} negative counts (min {v.min():g})")
        if v.nunique() > 50:
            vc = v.value_counts()
            if vc.iloc[0] / len(v) > 0.01 and vc.index[0] in (v.min(), v.max()):
                issues.append(f"{c}: {vc.iloc[0] / len(v):.1%} of values exactly at the {'min' if vc.index[0] == v.min() else 'max'} "
                              f"({vc.index[0]:.4g}), i.e. clipped")
    return issues


def duplicates_collinear(train, test):
    cols = [c for c in test.columns if c != "player_id"]
    if train is not None:
        train, test = _same_precision(train, test, cols)
    h = lambda d: pd.util.hash_pandas_object(d[cols], index=False)
    out = {"duplicate test rows": int(h(test).duplicated().sum())}
    if train is not None:
        htr = h(train)
        out["duplicate train rows"] = int(htr.duplicated().sum())
        out["test rows equal to a train row"] = int(h(test).isin(set(htr)).sum())
    df = test[cols] if train is None else pd.concat([train[cols], test[cols]], ignore_index=True)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out["constant columns"] = int(sum(df[c].nunique(dropna=True) <= 1 for c in cols))
    cs = df[num].rank().corr().abs().values
    iu = np.triu_indices(len(num), 1)
    out["collinear pairs (|Spearman| > 0.98)"] = [f"{num[i]} ~ {num[j]}" for i, j in zip(*iu) if cs[i, j] > 0.98]
    return out


def task_audit(train, test, target=None, regression=False):
    """One-line summary dict per task (train may be None for the unlabeled Task 5)."""
    cols = [c for c in test.columns if c != "player_id"]
    s = {}
    if train is not None:
        aucs, ks = adversarial_auc(train, test, cols)
        s.update({f"train-vs-test AUC: {k}": round(v, 3) for k, v in aucs.items()})
        s["largest value shift (KS)"] = round(ks, 3)
        s["missing rate train / test"] = f"{train[cols].isna().mean().mean():.3f} / {test[cols].isna().mean().mean():.3f}"
        mt = missing_vs_target(train, target, regression)
        s["informative missing columns (p < 0.001)"] = f"{int((mt.p < 0.001).sum())} of {len(mt)}"
    phys = physical_checks(test if train is None else pd.concat([train[cols], test[cols]], ignore_index=True))
    s["physical issues"] = "; ".join(phys) if phys else "none"
    for k, v in duplicates_collinear(train, test).items():
        s[k] = v if not isinstance(v, list) else (", ".join(v) if v else "none")
    return s


def image_audit(data_dir, frames, n_threads=12):
    """Readability, size, near-blank frames (pixel std < 5) and byte-identical duplicates of the Task 4 images."""
    from PIL import Image

    def scan(path):
        try:
            b = open(path, "rb").read()
            im = Image.open(path)
            im.load()
            a = np.asarray(im.convert("L"), dtype=np.float32)
            return hashlib.md5(b).hexdigest(), im.size, a.std(), a.mean(), False
        except Exception:  # noqa: BLE001
            return None, None, np.nan, np.nan, True

    rows, md5 = {}, {}
    for split, df in frames.items():
        with ThreadPoolExecutor(n_threads) as ex:
            res = list(ex.map(scan, [str(data_dir / split / f"{i}.jpg") for i in df.id]))
        h, size, sd, mean, bad = zip(*res)
        md5[split] = pd.Series(h)
        rows[split] = {"images": len(df), "unreadable": int(sum(bad)), "sizes": dict(pd.Series(size).value_counts()),
                       "near-blank (pixel std < 5)": int(np.nansum(np.array(sd) < 5)), "mean brightness": round(float(np.nanmean(mean)), 1)}
    rows["duplicates"] = {"within train": int(md5["train"].duplicated().sum()), "within test": int(md5["test"].duplicated().sum()),
                          "test equal to a train image": int(md5["test"].isin(set(md5["train"])).sum())}
    return rows
