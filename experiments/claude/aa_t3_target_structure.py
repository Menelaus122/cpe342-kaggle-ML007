"""Task 3: which base quantity does spending_30d scale with? For y > 0, compare the spread (MAD) of
log(y / base) within each segment for candidate bases. The smallest spread = the generator's driver."""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from src import task3
from t3_fill import fill

tr, _ = task3.load()
d = fill(tr)[0]
y = d[task3.TARGET].values
seg = d.segment.values
pos = y > 0
mad = lambda v: np.nanmedian(np.abs(v - np.nanmedian(v)))
cands = {
    "prev_month_spending": d.prev_month_spending,
    "historical_spending": d.historical_spending,
    "hist / account_age * 30": d.historical_spending / d.account_age_days * 30,
    "avg_txn_value * purchase_freq": d.avg_transaction_value * d.purchase_frequency,
    "avg_txn_value": d.avg_transaction_value,
    "hist / total_txn": d.historical_spending / d.total_transactions,
    "purchase_frequency": d.purchase_frequency,
}
print("segment sizes:", pd.Series(seg).value_counts(dropna=False).sort_index().to_dict())
rows = []
for name, b in cands.items():
    b = b.values.astype(float)
    r = np.log(y) - np.log(b)
    ok = pos & np.isfinite(r)
    row = {"base": name, "n": int(ok.sum())}
    for s in (0, 1, 2, 3):
        m = ok & (seg == s)
        row[f"seg{s} med"] = round(float(np.median(r[m])), 3)
        row[f"seg{s} MAD"] = round(float(mad(r[m])), 3)
    row["overall MAD (seg-centred)"] = round(float(np.nanmedian(np.abs(np.concatenate(
        [r[ok & (seg == s)] - np.median(r[ok & (seg == s)]) for s in (0, 1, 2, 3)])))), 3)
    rows.append(row)
print(pd.DataFrame(rows).set_index("base").to_string())
print("\nMAD of log(y) itself by segment:", {s: round(float(mad(np.log(y[pos & (seg == s)]))), 3) for s in (0, 1, 2, 3)})
