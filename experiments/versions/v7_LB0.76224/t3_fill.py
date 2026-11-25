"""Fill missing Task 3 spending-history values from the exact row identities (see x_t3_identity.py)."""
import numpy as np  # noqa: F401

RATIO = 0.3498  # median prev / hist


def fill(df):
    d = df.copy()
    H, T, A, P = d.historical_spending, d.total_transactions, d.avg_transaction_value, d.prev_month_spending
    POD, DR = d.purchases_on_discount, d.discount_rate_used
    exact_h = T * A
    # exact first, then the approximate prev/hist ratio as a last resort
    H2 = H.fillna(exact_h.where(T > 0)).fillna(P / RATIO)
    T2 = T.fillna((H / A).where(A > 0).round()).fillna((POD / DR).where(DR > 0).round())
    A2 = A.fillna((H2 / T2).where(T2 > 0))
    P2 = P.fillna(H2 * RATIO)
    POD2 = POD.fillna(T2 * DR)
    DR2 = DR.fillna((POD / T2).where(T2 > 0))
    stats = {c: int(new.notna().sum() - old.notna().sum()) for c, old, new in
             [("historical_spending", H, H2), ("total_transactions", T, T2), ("avg_transaction_value", A, A2),
              ("prev_month_spending", P, P2), ("purchases_on_discount", POD, POD2), ("discount_rate_used", DR, DR2)]}
    d["historical_spending"], d["total_transactions"], d["avg_transaction_value"] = H2, T2, A2
    d["prev_month_spending"], d["purchases_on_discount"], d["discount_rate_used"] = P2, POD2, DR2
    return d, stats
