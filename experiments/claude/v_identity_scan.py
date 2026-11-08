"""Look for deterministic relations between columns (a = b*c, a = b/c, a = b+c, a = b-c, a = k*b).

If a column is a formula of others, its missing values can be filled exactly from them.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402

TASKS = {1: ("01_binary_classification", "is_cheater"), 3: ("03_regression", "spending_30d")}


def r2(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 500:
        return np.nan, 0
    a, b = a[ok], b[ok]
    # fit a = k*b (through origin); score = 1 - SSE/SST
    k = (a @ b) / (b @ b) if (b @ b) > 0 else 0
    sst = ((a - a.mean()) ** 2).sum()
    return 1 - ((a - k * b) ** 2).sum() / sst if sst > 0 else np.nan, ok.sum()


for t, (d, tgt) in TASKS.items():
    df = pd.concat([pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")]).drop(columns=["player_id", tgt])
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 20]
    V = {c: df[c].to_numpy(float) for c in num}
    L = {c: np.where(V[c] > 0, np.log(np.where(V[c] > 0, V[c], 1)), np.nan) for c in num}
    hits = []
    for a in num:
        # a = k * b  (pairwise, raw scale)
        for b in num:
            if b != a:
                s, n = r2(V[a], V[b])
                if s > 0.98:
                    hits.append((s, f"{a} = k*{b}", n))
        for b, c in itertools.combinations([x for x in num if x != a], 2):
            for form, x in [(f"{b} * {c}", L[b] + L[c]), (f"{b} / {c}", L[b] - L[c]), (f"{c} / {b}", L[c] - L[b])]:
                ok = np.isfinite(L[a]) & np.isfinite(x)
                if ok.sum() < 500:
                    continue
                # log a = log k + log(form): residual spread after removing the mean
                res = L[a][ok] - x[ok]
                s = 1 - res.var() / L[a][ok].var()
                if s > 0.98:
                    hits.append((s, f"log {a} = const + log({form})", int(ok.sum())))
            for form, x in [(f"{b} + {c}", V[b] + V[c]), (f"{b} - {c}", V[b] - V[c]), (f"{c} - {b}", V[c] - V[b])]:
                s, n = r2(V[a], x)
                if s > 0.98:
                    hits.append((s, f"{a} = k*({form})", n))
    hits.sort(reverse=True)
    print(f"== Task {t}: {len(num)} continuous columns; relations with R^2 > 0.98:")
    for s, f, n in hits[:15]:
        print(f"  R2={s:.5f}  {f}  (n={n:,})")
    if not hits:
        print("  none")
    # strongest pairwise rank correlations, for reference
    c = df[num].rank().corr().abs()
    pairs = c.where(np.triu(np.ones(c.shape, bool), 1)).stack().sort_values(ascending=False).head(6)
    print("  strongest pairwise |Spearman|:", [(f"{i}~{j}", round(v, 3)) for (i, j), v in pairs.items()])
