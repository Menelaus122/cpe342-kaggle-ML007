"""Brief Phase 2.1: lag/lead autocorrelation of neighbouring rows (file order and player_id order), Tasks 1-3 train/test and Task 5.
Earlier rounds checked trends along the file order (Spearman with position, decile means); this checks neighbour-to-neighbour
dependence, which a trend test can miss. Read-only."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

def lag1(v):
    v = np.asarray(v, float); ok = ~np.isnan(v[:-1]) & ~np.isnan(v[1:])
    return np.corrcoef(v[:-1][ok], v[1:][ok])[0, 1] if ok.sum() > 10 else np.nan

files = {"T1": ("01_binary_classification", "is_cheater"), "T2": ("02_multiclass_classification", "segment"),
         "T3": ("03_regression", "spending_30d")}
for t, (d, tgt) in files.items():
    for part in ("train", "test"):
        f = next(__import__("pathlib").Path(d).rglob(f"{part}.csv"))
        df = pd.read_csv(f)
        num = df.select_dtypes("number").drop(columns=[c for c in ("id", tgt) if c in df.columns], errors="ignore")
        n = len(df); band = 4 / np.sqrt(n)
        for order in ("file", "id"):
            o = np.arange(n) if order == "file" else np.argsort(df.player_id.str.extract(r"(\d+)")[0].astype(int).values, kind="stable")
            ac = np.array([lag1(num[c].values[o]) for c in num.columns])
            msg = f"{t} {part:5s} {order:4s} order: features with |lag-1 r| > {band:.3f}: {(np.abs(ac) > band).sum()}/{len(ac)} (max {np.nanmax(np.abs(ac)):.3f})"
            if tgt in df.columns:
                y = df[tgt].values[o]
                if t == "T2":
                    yy = pd.Series(y).dropna().values; p = pd.Series(yy).value_counts(normalize=True).values
                    msg += f"; P(same segment as next row) {np.mean(yy[:-1] == yy[1:]):.4f} vs chance {np.sum(p ** 2):.4f}"
                elif t == "T3":
                    msg += f"; target lag-1 Spearman {spearmanr(y[:-1], y[1:], nan_policy='omit')[0]:+.4f}"
                else:
                    msg += f"; target lag-1 r {lag1(y):+.4f}"
            print(msg, flush=True)
t5 = pd.read_csv("05_anomaly_detection/test.csv"); num = t5.drop(columns="player_id")
ac = np.array([lag1(num[c].values) for c in num.columns]); band = 4 / np.sqrt(len(t5))
print(f"T5 test  file order: features with |lag-1 r| > {band:.3f}: {(np.abs(ac) > band).sum()}/{len(ac)} (max {np.nanmax(np.abs(ac)):.3f})")
