from common import *
tr, te, X, y, Xt, folds = load_t2()
E = [c for c in task2.ENGAGEMENT_POS + task2.ENGAGEMENT_NEG if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
S = [c for c in task2.SPENDING_POS if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
N = ["random_metric_1", "random_metric_2", "random_metric_3", "account_age_days", "avg_match_length"]
for k in range(4):
    C = tr.loc[y == k].corr(method="spearman", numeric_only=True)
    def m(a, b):
        v = [C.loc[i, j] * (-1 if (i == "days_since_last_login") ^ (j == "days_since_last_login") else 1) for i in a for j in b if i != j]
        return np.mean(v)
    print(f"class {k}: within-class mean rho  E-E {m(E, E):+.4f}  S-S {m(S, S):+.4f}  E-S {m(E, S):+.4f}  noise-all {m(N, E + S):+.4f}")
