"""Data-quality audit used by the notebooks (section 3.6): missingness, impossible values, duplicates.

TRAINING data only (TA ruling, 2026-11-15: test statistics must not shape validation, features or augmentation). Task 5 has no
training file, so its checks run on the one file the task provides. Nothing here is fitted for a model.
* Missingness vs target: chi-square (classes) or Mann-Whitney (spending) of target vs "cell is missing". p < 0.001 = informative.
* Physical checks: negatives, percentages outside [0, 100], coordinates outside the globe, value spikes at a column's min/max
  (clipping), duplicate rows, constant columns, collinear pairs.
* Task 4: readability, size, near-blank frames and byte-identical duplicates of the training images.
(Until 2026-11-15 this module also ran a train-vs-test adversarial validation and scanned the test images; both were removed.)
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


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


def duplicates_collinear(df, target=None):
    cols = [c for c in df.columns if c not in ("player_id", target)]
    out = {"duplicate rows": int(pd.util.hash_pandas_object(df[cols], index=False).duplicated().sum())}
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out["constant columns"] = int(sum(df[c].nunique(dropna=True) <= 1 for c in cols))
    cs = df[num].rank().corr().abs().values
    iu = np.triu_indices(len(num), 1)
    out["collinear pairs (|Spearman| > 0.98)"] = [f"{num[i]} ~ {num[j]}" for i, j in zip(*iu) if cs[i, j] > 0.98]
    return out


def task_audit(df, target=None, regression=False):
    """One-line summary dict per task: `df` is the training frame (Tasks 1-3) or Task 5's only file (no target)."""
    cols = [c for c in df.columns if c not in ("player_id", target)]
    s = {"rows": len(df), "missing rate": round(float(df[cols].isna().mean().mean()), 3)}
    if target is not None:
        mt = missing_vs_target(df, target, regression)
        s["informative missing columns (p < 0.001)"] = f"{int((mt.p < 0.001).sum())} of {len(mt)}"
    phys = physical_checks(df[cols])
    s["physical issues"] = "; ".join(phys) if phys else "none"
    for k, v in duplicates_collinear(df, target).items():
        s[k] = v if not isinstance(v, list) else (", ".join(v) if v else "none")
    return s


def image_audit(data_dir, df, split="train", n_threads=12):
    """Readability, size, near-blank frames (pixel std < 5) and byte-identical duplicates of the Task 4 training images."""
    from PIL import Image

    def scan(path):
        try:
            b = open(path, "rb").read()
            im = Image.open(path)
            im.load()
            a = np.asarray(im.convert("L"), dtype=np.float32)
            return hashlib.md5(b).hexdigest(), im.size, a.std(), False
        except Exception:  # noqa: BLE001
            return None, None, np.nan, True

    with ThreadPoolExecutor(n_threads) as ex:
        res = list(ex.map(scan, [str(data_dir / split / f) for f in df.file_name]))
    h, size, sd, bad = zip(*res)
    return {"images": len(df), "unreadable": int(sum(bad)), "sizes": dict(pd.Series(size).value_counts()),
            "near-blank (pixel std < 5)": int(np.nansum(np.array(sd) < 5)), "duplicates": int(pd.Series(h).duplicated().sum())}
