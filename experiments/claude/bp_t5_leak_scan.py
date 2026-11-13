"""Task 5: is there a second leaking column besides player_id? (user: "my friends said there is one more feature that leaks")

Reference = the id block (ids >= 23,819, 2,070 accounts), which our feature-only families already match 1,811/1,811 with 0 outside.
For every raw column, compare block vs the rest on value AND on data-generation artefacts:
  AUC of the value, NaN rate, share of integer values, decimal digits as written in the csv, share of the most common value,
  values outside the normal accounts' range. A leak = a column that separates the block (or the 259 unflagged block accounts)
  almost perfectly through an artefact, or a flag column that is ~1 only inside the block.
Read-only analysis of the given test file; nothing is merged or uploaded.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
path = sys.argv[1] if len(sys.argv) > 1 else "05_anomaly_detection/test.csv"
num = pd.read_csv(path)
txt = pd.read_csv(path, dtype=str, keep_default_na=False)
idn = num.player_id.str[1:].astype(int).values
blk = idn >= 23819
print(f"rows {len(num)}, block {blk.sum()} ({blk.mean():.3%})")

fam = pd.read_csv("experiments/claude/outputs/t5_id_families.csv") if False else None


def decimals(s):
    s = s[s != ""]
    d = s.str.split(".").str[1].fillna("").str.len()
    return d


rows = []
for c in num.columns[1:]:
    v = num[c].values.astype(float)
    ok = ~np.isnan(v)
    try:
        auc = roc_auc_score(blk[ok], v[ok])
    except ValueError:
        auc = np.nan
    dec = decimals(txt[c])
    dec_b, dec_n = dec[blk[txt[c].values != ""]], dec[~blk[txt[c].values != ""]]
    intshare = lambda m: np.mean(np.isclose(v[m & ok], np.round(v[m & ok])))
    lo, hi = np.nanpercentile(v[~blk], [0.0, 100.0])
    out_b = np.mean((v[blk & ok] < lo) | (v[blk & ok] > hi)) if (blk & ok).any() else np.nan
    mode = pd.Series(v[ok]).mode().iloc[0]
    rows.append(dict(col=c, auc=max(auc, 1 - auc) if auc == auc else np.nan, nan_blk=np.isnan(v[blk]).mean(), nan_rest=np.isnan(v[~blk]).mean(),
                     int_blk=intshare(blk), int_rest=intshare(~blk), dec_blk=dec_b.mean(), dec_rest=dec_n.mean(),
                     nuniq=len(np.unique(v[ok])), mode=mode, mode_blk=np.mean(v[blk & ok] == mode), mode_rest=np.mean(v[~blk & ok] == mode),
                     mean_blk=np.nanmean(v[blk]), mean_rest=np.nanmean(v[~blk])))
R = pd.DataFrame(rows)
R["nan_gap"] = (R.nan_blk - R.nan_rest).abs()
R["int_gap"] = (R.int_blk - R.int_rest).abs()
R["dec_gap"] = (R.dec_blk - R.dec_rest).abs()
R["mode_gap"] = (R.mode_blk - R.mode_rest).abs()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n== low-cardinality / flag columns (<= 10 distinct values) ==")
print(R[R.nuniq <= 10][["col", "nuniq", "auc", "nan_blk", "nan_rest", "mode", "mode_blk", "mode_rest", "mean_blk", "mean_rest"]].to_string(index=False))
for c in R[R.nuniq <= 10].col:
    print(f"\n{c}: crosstab vs block (rows = value, NaN shown as -1)")
    print(pd.crosstab(num[c].fillna(-1), np.where(blk, "block", "rest")).to_string())
print("\n== top 15 by AUC (value separates block) ==")
print(R.sort_values("auc", ascending=False).head(15)[["col", "auc", "mean_blk", "mean_rest", "nuniq"]].to_string(index=False))
print("\n== top 10 by NaN-rate gap ==")
print(R.sort_values("nan_gap", ascending=False).head(10)[["col", "nan_blk", "nan_rest"]].to_string(index=False))
print("\n== top 10 by integer-share gap ==")
print(R.sort_values("int_gap", ascending=False).head(10)[["col", "int_blk", "int_rest"]].to_string(index=False))
print("\n== top 10 by decimal-digits gap (as written in the csv) ==")
print(R.sort_values("dec_gap", ascending=False).head(10)[["col", "dec_blk", "dec_rest"]].to_string(index=False))
print("\n== top 10 by most-common-value share gap ==")
print(R.sort_values("mode_gap", ascending=False).head(10)[["col", "mode", "mode_blk", "mode_rest"]].to_string(index=False))
nan_cnt = num.iloc[:, 1:].isna().sum(1).values
print(f"\nNaN cells per row: block mean {nan_cnt[blk].mean():.2f}, rest {nan_cnt[~blk].mean():.2f}; "
      f"AUC {max(roc_auc_score(blk, nan_cnt), 1 - roc_auc_score(blk, nan_cnt)):.3f}")
pos = np.arange(len(num))
print(f"row position in the file vs block: AUC {max(roc_auc_score(blk, pos), 1 - roc_auc_score(blk, pos)):.3f}")
R.to_csv("experiments/claude/outputs/bp_t5_leak_scan.csv", index=False)
