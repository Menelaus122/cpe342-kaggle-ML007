"""Round 13 follow-up to bv_t3_scale_whale.py.
(1) W1 left the blend unchanged to six decimals: print the weight the exact L1 solver gives the Huber member, to tell a true zero from a no-op.
(2) The only positive cross-task signal (LGB-L1 raw + Task 2 segment probabilities, +0.00014 on the member, 4/5 folds) is tested where it counts: as a member of the blend.
    The auxiliary segment model is fitted on Task 2 TRAINING rows only and applied to Task 3 rows.
(3) A prediction cap at the largest training value (nested), because the largest OOF prediction exceeds the largest y.
Same baseline and acceptance rule as bv_t3_scale_whale.py.
"""
from pathlib import Path

_src = open(Path(__file__).with_name("bv_t3_scale_whale.py"), encoding="utf-8").read()
exec(_src.split("bo, inner = fold_blend(P)")[0].split('"""', 2)[2])

from src.modeling import run_cv  # noqa: E402


def blend_w(P_):
    bo_, ws = np.zeros(len(y)), []
    for a, b in folds:
        w = task3.l1_blend_weights_exact(y[a], [P_[a, j] for j in range(P_.shape[1])])
        bo_[b] = P_[b] @ w
        ws.append(w)
    return bo_, np.array(ws)


bo, W0 = blend_w(P)
pred0 = bo * gate
s0 = nmae(y, pred0)
pf = lambda v: np.array([nmae(y[b], v[b]) for _, b in folds])
g0 = pf(pred0)
log(f"B0: NMAE {s0:.5f}, folds {np.round(g0, 4)}")


def boot(pn, po, B=300):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        T = y[i].sum()
        d.append(1 / (1 + np.abs(y[i] - pn[i]).sum() / T) - 1 / (1 + np.abs(y[i] - po[i]).sum() / T))
    return np.percentile(d, [2.5, 97.5])


rows = []


def judge(name, v):
    s, g = nmae(y, v), pf(v)
    lo, hi = boot(v, pred0)
    better = int((g > g0).sum())
    keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
    rows.append([name, s, s - s0, f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", g.std(), "KEEP" if keep else "discard"])
    log(f"{name}: {s:.5f} ({s - s0:+.5f}) CI {rows[-1][3]}, {better}/5, std {g.std():.5f} -> {rows[-1][-1]}")


# (1) Huber weight
hub_o = np.zeros(len(y))
for k, (a, b) in enumerate(folds):
    delta = float(np.quantile(np.abs(y[a] - np.median(y[a])), 0.9))
    _, (pv,), _ = fit_predict("lgb", "regression", {**P3_lgb, "objective": "huber", "alpha": delta, "metric": "l1"}, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
    hub_o[b] = np.clip(pv, 0, None)
print(f"Huber member alone {nmae(y, hub_o):.5f}; correlation with the baseline blend {np.corrcoef(hub_o, bo)[0, 1]:.4f}", flush=True)
b1, W1 = blend_w(np.column_stack([P, hub_o]))
print("Huber weight per fold:", np.round(W1[:, -1], 5).tolist(), flush=True)
print("largest weight change on the other members per fold:", np.round(np.abs(W1[:, :-1] - W0).max(1), 5).tolist(), flush=True)
judge("W1 + raw-scale Huber member", b1 * gate)

# (2) cross-task member in the blend
num2 = pd.read_csv("02_multiclass_classification/train.csv")
G = {"friends": ("friend_count", "friend_count"), "session_length": ("avg_session_duration", "avg_session_length"), "account_age": ("account_age_days", "account_age_days"),
     "playtime": ("total_playtime_hours", "total_playtime_hours"), "last_login": ("days_since_last_login", "days_since_last_login"),
     "achievement": ("achievement_completion_rate", "achievement_completion_rate"), "rank": ("competitive_rank", "competitive_rank"), "spending": ("total_spending_thb", "historical_spending"),
     "streak": ("login_streak", "daily_login_streak"), "frequency": ("play_frequency", "sessions_per_week")}


def rank_map(train_v, v):
    s = np.sort(train_v[~np.isnan(train_v)])
    return np.where(np.isnan(v), np.nan, np.searchsorted(s, v, side="right") / len(s))


col = lambda df, c: pd.to_numeric(df[c], errors="coerce").values.astype(float)
src_x = pd.DataFrame({g: rank_map(col(num2, a), col(num2, a)) for g, (a, _) in G.items()})
tgt = lambda df: pd.DataFrame({g: rank_map(col(tr, b), col(df, b)) for g, (_, b) in G.items()})
ys = num2.segment.values.astype(int)
idx = np.random.default_rng(0).permutation(len(ys))
va, fi = idx[:len(ys) // 5], idx[len(ys) // 5:]
prm = {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 2000}
_, (a3, at3), _ = fit_predict("lgb", "multiclass", prm, src_x.iloc[fi], ys[fi], src_x.iloc[va], ys[va], [tgt(tr), tgt(te)], early_stopping=100)
add = lambda X_, A_: pd.concat([X_.reset_index(drop=True), pd.DataFrame(A_, columns=[f"aux_seg{i}" for i in range(4)])], axis=1)
r = run_cv("lgb", "regression", P3_lgb, add(X, a3), y, add(Xt, at3), folds, name="T3 aux", verbose=False)
aux_o = np.clip(r["oof"], 0, None)
print(f"member: cached {nmae(y, plain['LGB-L1 raw THB']):.5f}, with Task 2 segment probabilities {nmae(y, aux_o):.5f}", flush=True)
b2, W2 = blend_w(np.column_stack([P, aux_o]))
print("cross-task member weight per fold:", np.round(W2[:, -1], 5).tolist(), flush=True)
judge("E1 + LGB-L1 raw with Task 2 segment probabilities, in the blend", b2 * gate)

# (3) cap at the largest training value
print(f"max y {y.max():,.0f}; rows with y at that value {(y == y.max()).sum()}; share of y >= 299,000: {(y >= 299000).mean():.4%}; OOF predictions above max y: {(pred0 > y.max()).sum()}", flush=True)
vc = pred0.copy()
for a, b in folds:
    vc[b] = np.minimum(pred0[b], y[a].max())
judge("C1 cap predictions at the training maximum", vc)

pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "fold std", "Decision"]).to_string(index=False))
log("done")
