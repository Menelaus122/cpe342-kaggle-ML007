"""Phase 1 forensic audit (2026-11-10): distribution shift, impossible values, missingness, duplicates/collinearity, images.

Uses train and test FEATURES only (no test labels exist). Nothing here is used for fitting a model.
1. Adversarial validation (Tasks 1-3): LightGBM separating train (0) from test (1), 5-fold AUC, three views:
   all features as given / only the missing-value indicators / values with missing cells filled by the train median.
   If only the indicator view separates the sets, the shift is in WHICH cells are missing, not in the values.
2. Physical / domain checks: negatives, percentages outside [0, 100], lat/lon outside the globe, sentinel codes,
   and "spikes" (one value holding > 1% of a continuous column at its min or max = clipping or a fill code).
3. Missingness vs target (Tasks 1-3): is a missing cell informative (MNAR) or random (MCAR)?
4. Duplicate rows, constant columns, collinear pairs (|r| > 0.98).
5. Task 4 images: unreadable files, wrong sizes, near-blank frames, byte-identical duplicates (within / across splits).
"""
import hashlib
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402

pd.set_option("display.width", 220)
TASKS = {1: ("01_binary_classification", "is_cheater"), 2: ("02_multiclass_classification", "segment"),
         3: ("03_regression", "spending_30d"), 5: ("05_anomaly_detection", None)}
SENTINELS = [-1, -99, -999, -9999, 999, 9999, 99999, 999999]


def load(t):
    d, tgt = TASKS[t]
    te = pd.read_csv(f"{d}/test.csv")
    if tgt is None:
        return None, te, None
    tr = pd.read_csv(f"{d}/train.csv")
    if t == 1:
        tr = tr[tr[tgt].notna()].reset_index(drop=True)
    return tr, te, tgt


def adversarial(tr, te, cols):
    A = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in A.columns:
        if not pd.api.types.is_numeric_dtype(A[c]):
            A[c] = A[c].astype("category")
    lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(A[c])]
    views = {"all features": A,
             "missing indicators only": A[cols].isna().astype(int).add_prefix("na_"),
             "values, missing filled by train median": A[num].fillna(tr[num].median())}
    out = {}
    for name, V in views.items():
        oof, imp = np.zeros(len(lab)), np.zeros(V.shape[1])
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(V, lab):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=-1)
            m.fit(V.iloc[a], lab[a])
            oof[b] = m.predict_proba(V.iloc[b])[:, 1]
            imp += m.booster_.feature_importance("gain")
        top = pd.Series(imp, index=V.columns).sort_values(ascending=False).head(4)
        out[name] = (roc_auc_score(lab, oof), [(k, round(v / imp.sum(), 2)) for k, v in top.items()])
    return out


def physical(df, name):
    issues = []
    for c in df.columns:
        if c == "player_id" or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if v.empty:
            continue
        lc = c.lower()
        if (v < 0).any() and not any(k in lc for k in ("lon", "lat", "change", "delta", "diff", "trend", "slope", "z")):
            issues.append(f"{c}: {int((v < 0).sum())} negative values (min {v.min():.3g})")
        if any(k in lc for k in ("percentage", "percent", "_pct", "win_rate", "headshot")) and ((v > 100) | (v < 0)).any():
            issues.append(f"{c}: {int(((v > 100) | (v < 0)).sum())} values outside [0, 100] (min {v.min():.3g}, max {v.max():.3g})")
        if "lat" in lc and ((v < -90) | (v > 90)).any():
            issues.append(f"{c}: {int(((v < -90) | (v > 90)).sum())} latitudes outside [-90, 90]")
        if "lon" in lc and ((v < -180) | (v > 180)).any():
            issues.append(f"{c}: {int(((v < -180) | (v > 180)).sum())} longitudes outside [-180, 180]")
        sent = {s: int((v == s).sum()) for s in SENTINELS if (v == s).sum() > 0}
        if sent and v.nunique() > 20:
            issues.append(f"{c}: sentinel-like values {sent}")
        if v.nunique() > 50:                                  # spikes at the edges of a continuous column
            vc = v.value_counts()
            top_val, top_n = vc.index[0], vc.iloc[0]
            if top_n / len(v) > 0.01 and top_val in (v.min(), v.max()):
                issues.append(f"{c}: {top_n / len(v):.1%} of values sit exactly at the {'min' if top_val == v.min() else 'max'} "
                              f"({top_val:.4g}) - clipping / floor")
    return issues


def missing_vs_target(tr, tgt, t):
    rows = []
    y = tr[tgt]
    for c in tr.columns:
        if c in ("player_id", tgt):
            continue
        m = tr[c].isna()
        if m.sum() < 30:
            continue
        if t == 3:
            p = mannwhitneyu(y[m], y[~m]).pvalue
            eff = f"zero share {np.mean(y[m] == 0):.3f} vs {np.mean(y[~m] == 0):.3f}; median {y[m].median():.0f} vs {y[~m].median():.0f}"
        else:
            ct = pd.crosstab(m, y)
            p = chi2_contingency(ct)[1]
            share = ct.div(ct.sum(1), axis=0)
            eff = "class shares missing vs present: " + ", ".join(f"{k}: {share.loc[True, k]:.3f}/{share.loc[False, k]:.3f}" for k in share.columns[:4])
        rows.append((c, m.mean(), p, eff))
    return rows


def dup_const_collinear(tr, te, tgt):
    cols = [c for c in te.columns if c != "player_id"]
    h = lambda df: pd.util.hash_pandas_object(df[cols], index=False)
    res = {}
    if tr is not None:
        htr, hte = h(tr), h(te)
        res["duplicate rows in train"] = int(htr.duplicated().sum())
        res["test rows identical to a train row"] = int(hte.isin(set(htr)).sum())
    res["duplicate rows in test"] = int(h(te).duplicated().sum())
    df = te if tr is None else pd.concat([tr[cols], te[cols]], ignore_index=True)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    res["constant columns"] = [c for c in cols if df[c].nunique(dropna=True) <= 1]
    res["near-constant (top value > 99%)"] = [c for c in cols if df[c].value_counts(normalize=True, dropna=True).iloc[0] > 0.99]
    cp = df[num].corr().abs()
    cs = df[num].rank().corr().abs()
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if cp.loc[a, b] > 0.98 or cs.loc[a, b] > 0.98:
                pairs.append((a, b, round(cp.loc[a, b], 3), round(cs.loc[a, b], 3)))
    res["collinear pairs (|pearson| or |spearman| > 0.98)"] = pairs
    return res


for t in (1, 2, 3, 5):
    tr, te, tgt = load(t)
    print(f"\n{'=' * 100}\nTASK {t}: train {0 if tr is None else len(tr):,} rows, test {len(te):,} rows")
    cols = [c for c in te.columns if c != "player_id"]
    if tr is not None:
        print("-- 1. adversarial validation (train vs test), 5-fold AUC")
        for k, (auc, top) in adversarial(tr, te, cols).items():
            print(f"   {k:42s} AUC {auc:.3f}   top: {top}")
        num = [c for c in cols if pd.api.types.is_numeric_dtype(te[c])]
        ks = sorted(((ks_2samp(tr[c].dropna(), te[c].dropna()).statistic, c) for c in num), reverse=True)[:4]
        print("   largest per-column value shifts (KS on observed values):", [(c, round(s, 3)) for s, c in ks])
        mr = pd.DataFrame({"train": tr[cols].isna().mean(), "test": te[cols].isna().mean()})
        mr["diff"] = mr.test - mr.train
        big = mr[mr["diff"].abs() > 0.03].sort_values("diff")
        print(f"   missing rate train vs test: mean {mr.train.mean():.3f} vs {mr.test.mean():.3f}; columns differing by > 3 points: "
              + ("none" if big.empty else ", ".join(f"{c} {r.train:.3f}->{r.test:.3f}" for c, r in big.iterrows())))
    print("-- 2. physical / domain checks (train + test)")
    iss = physical(te if tr is None else pd.concat([tr[cols], te[cols]], ignore_index=True), t)
    print("   none" if not iss else "\n".join("   " + s for s in iss))
    if tr is not None:
        print("-- 3. missingness vs target")
        mt = missing_vs_target(tr, tgt, t)
        sig = [r for r in mt if r[2] < 0.001]
        print(f"   {len(mt)} columns with missing cells; informative at p < 0.001: {len(sig)}")
        for c, rate, p, eff in sorted(mt, key=lambda r: r[2])[:3]:
            print(f"   smallest p: {c} ({rate:.1%} missing) p = {p:.2g}; {eff}")
    print("-- 4. duplicates / constant / collinear")
    for k, v in dup_const_collinear(tr, te, tgt).items():
        print(f"   {k}: {v}")

# ------------------------------------------------------------------ Task 4 images
print(f"\n{'=' * 100}\nTASK 4: image audit")
D = "04_image_classification/_download/task4_image_classification"
tr4, te4 = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")


def scan(path):
    try:
        b = open(path, "rb").read()
        im = Image.open(path)
        im.load()
        a = np.asarray(im.convert("L"), dtype=np.float32)
        return hashlib.md5(b).hexdigest(), im.size, float(a.mean()), float(a.std()), None
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, repr(e)


for split, df in (("train", tr4), ("test", te4)):
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(scan, [f"{D}/{split}/{i}.jpg" for i in df.id]))
    df["md5"], df["size"], df["mean"], df["std"], df["err"] = zip(*res)
    print(f"   {split}: {len(df):,} images | unreadable {int(df.err.notna().sum())} | sizes {dict(pd.Series(df['size']).value_counts().head(3))} "
          f"| near-blank (pixel std < 5) {int((df['std'] < 5).sum())} | mean brightness {np.nanmean(df['mean']):.1f}")
print("   train class counts:", tr4["class"].value_counts().to_dict())
g = tr4.groupby("md5")["class"].agg(["size", "nunique"])
print(f"   byte-identical duplicates: within train {int((g['size'] > 1).sum())} groups ({int((g['nunique'] > 1).sum())} with conflicting labels); "
      f"within test {int(te4.md5.duplicated().sum())}; test images identical to a train image {int(te4.md5.isin(set(tr4.md5)).sum())}")
print("   tabular metadata in train.csv / test.csv:", list(tr4.columns), "/", list(te4.columns))
