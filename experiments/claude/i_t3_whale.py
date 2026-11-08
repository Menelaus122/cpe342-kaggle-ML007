from common import *
from sklearn.linear_model import QuantileRegressor, LinearRegression, HuberRegressor
X3, y3, X3t, folds3, m3, g_oof, g_te = t3_pool()
tr, te = task3.load()
P = np.column_stack([v[0] for v in m3.values()])
s3, pred = t3_blend_score(y3, folds3, P, g_oof)
seg = tr.segment.fillna(-1).values
pw = (seg == 3) & (y3 > 0)
d = tr.loc[pw].copy()
ly = np.log(d.spending_30d.values)
lh, lp, la = np.log1p(d.historical_spending), np.log1p(d.prev_month_spending), np.log1p(d.avg_transaction_value)
print("whale spenders:", pw.sum(), " missing hist/prev/both:", lh.isna().mean().round(3), lp.isna().mean().round(3), (lh.isna() & lp.isna()).mean().round(4))
ok = lh.notna() & lp.notna()
print("corr(log y, log hist, log prev) on complete rows:\n", np.corrcoef(np.vstack([ly[ok], lh[ok], lp[ok]])).round(3))
# simple ratios
print("median log(y/prev)", np.median(ly[ok] - lp[ok]).round(3), "MAD", np.median(np.abs((ly[ok] - lp[ok]) - np.median(ly[ok] - lp[ok]))).round(3))
print("median log(y/hist)", np.median(ly[ok] - lh[ok]).round(3), "MAD", np.median(np.abs((ly[ok] - lh[ok]) - np.median(ly[ok] - lh[ok]))).round(3))
print("median log(prev/hist)", np.median(lp[ok] - lh[ok]).round(3), "MAD", np.median(np.abs((lp[ok] - lh[ok]) - np.median(lp[ok] - lh[ok]))).round(3))
# OLS of log y on log hist + log prev (complete rows)
A = np.column_stack([lh[ok], lp[ok]])
lr = LinearRegression().fit(A, ly[ok]); r = ly[ok] - lr.predict(A)
print("OLS log y ~ log hist + log prev: coefs", lr.coef_.round(3), "icpt", round(lr.intercept_, 3), "resid MAD", np.median(np.abs(r - np.median(r))).round(4))
cols = ["historical_spending", "prev_month_spending", "avg_transaction_value", "vip_status", "is_premium_member",
        "days_since_last_purchase", "purchase_frequency", "games_played", "cross_game_activity", "event_participation_rate"]
Z = d[cols].copy()
for c in ["historical_spending", "prev_month_spending", "avg_transaction_value"]:
    Z[c] = np.log1p(Z[c])
okk = Z.notna().all(1).values
lr2 = LinearRegression().fit(Z[okk], ly[okk]); r2 = ly[okk] - lr2.predict(Z[okk])
print("OLS with 10 feats (complete rows):", dict(zip(cols, lr2.coef_.round(3))), "resid MAD", np.median(np.abs(r2 - np.median(r2))).round(4))
rg = ly - np.log(np.maximum(pred[pw], 1))
print("current blend on the same complete rows: resid MAD", np.median(np.abs(rg[okk] - np.median(rg[okk]))).round(4))
# residual vs features (does anything explain the blend's residual?)
dd = d.assign(res=rg)
print(dd.corr(method="spearman", numeric_only=True)["res"].drop(["res"]).sort_values(key=abs, ascending=False).head(8).round(3).to_string())
