"""Round 13, item 1: are the five tasks linked (same players), and does cross-task feature transfer help?

A. schema: shared and semantically matching columns, their ranges and written precision in every file.
B. entity match: rows of different tasks with identical values on all shared columns, against a column-shuffled baseline.
C. alignment: Spearman of the matching columns when rows are paired by ANS position and by player_id.
D. cross-tabulation of the v11 and clean submissions across tasks for the same ANS id.
E. transfer: auxiliary models fitted on one task's TRAINING rows (shared columns, rank-mapped using each task's own training rows),
   applied to another task's rows as extra features. Only training labels are used, and the members are compared with the cached
   pipeline members on the same folds.
Rules: test rows are only predicted. No pseudo-labels, no AutoML, no pre-trained models.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import load_t1, load_t2, tuned, P2_lgb, P3_lgb, _logit  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import macro_f1, nmae, optimize_class_weights, apply_class_weights, optimize_threshold, f2  # noqa: E402
from src.modeling import fit_predict, run_cv, make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
D = {"t1": "01_binary_classification", "t2": "02_multiclass_classification", "t3": "03_regression", "t5": "05_anomaly_detection"}
num, txt = {}, {}
for t, d in D.items():
    for s in ("train", "test"):
        if t == "t5" and s == "train":
            continue
        num[t, s] = pd.read_csv(f"{d}/{s}.csv")
        txt[t, s] = pd.read_csv(f"{d}/{s}.csv", dtype=str, keep_default_na=False)
log("files loaded", {k: v.shape for k, v in num.items()})

GROUPS = {
    "friends": {"t1": "friend_network_size", "t2": "friend_count", "t3": "friend_count"},
    "session_length": {"t1": "avg_session_length_min", "t2": "avg_session_duration", "t3": "avg_session_length"},
    "account_age": {"t1": "account_age_days", "t2": "account_age_days", "t3": "account_age_days", "t5": "account_age_days"},
    "playtime": {"t2": "total_playtime_hours", "t3": "total_playtime_hours"},
    "last_login": {"t2": "days_since_last_login", "t3": "days_since_last_login"},
    "achievement": {"t2": "achievement_completion_rate", "t3": "achievement_completion_rate"},
    "rank": {"t2": "competitive_rank", "t3": "competitive_rank"},
    "spending": {"t2": "total_spending_thb", "t3": "historical_spending"},
    "streak": {"t2": "login_streak", "t3": "daily_login_streak"},
    "win_rate": {"t1": "win_rate", "t2": "win_rate_ranked"},
    "frequency": {"t1": "sessions_per_day", "t2": "play_frequency", "t3": "sessions_per_week"},
}

print("\n=== A. matching columns: distribution and precision ===")
rows = []
for g, m in GROUPS.items():
    for (t, s), df in num.items():
        if t in m:
            v = pd.to_numeric(df[m[t]], errors="coerce")
            tx = txt[t, s][m[t]]
            dec = tx[tx != ""].str.split(".").str[1].fillna("").str.len()
            rows.append([g, t, s, m[t], v.isna().mean(), v.mean(), v.median(), v.quantile(.01), v.quantile(.99), v.nunique(), int(dec.mode().iloc[0]), int(dec.max())])
print(pd.DataFrame(rows, columns=["group", "task", "split", "column", "NaN", "mean", "median", "p1", "p99", "distinct", "modal decimals", "max decimals"]).round(3).to_string(index=False))
print("\nexact shared column names per pair:")
for a, b in [("t1", "t2"), ("t1", "t3"), ("t1", "t5"), ("t2", "t3"), ("t2", "t5"), ("t3", "t5")]:
    print(f"  {a}-{b}:", sorted((set(num[a, 'test'].columns) & set(num[b, 'test'].columns)) - {"player_id"}))

print("\n=== B. entity match on all matching columns (exact after rounding to 2 decimals) ===")


def key(df, cols, rng=None):
    z = df[cols].apply(pd.to_numeric, errors="coerce").dropna().round(2)
    if rng is not None:
        z = pd.DataFrame({c: rng.permutation(z[c].values) for c in cols})
    return pd.util.hash_pandas_object(z, index=False).values


rng = np.random.default_rng(0)
rows = []
keys_t = ["t1", "t2", "t3", "t5"]
for i, a in enumerate(keys_t):
    for b in keys_t[i + 1:]:
        gs = [g for g, m in GROUPS.items() if a in m and b in m]
        if not gs:
            continue
        for sa in ("train", "test"):
            for sb in ("train", "test"):
                if (a, sa) not in num or (b, sb) not in num:
                    continue
                ca, cb = [GROUPS[g][a] for g in gs], [GROUPS[g][b] for g in gs]
                ka, kb = key(num[a, sa], ca), key(num[b, sb].rename(columns=dict(zip(cb, ca))), ca)
                kb_null = key(num[b, sb].rename(columns=dict(zip(cb, ca))), ca, rng)
                rows.append([f"{a} {sa}", f"{b} {sb}", len(gs), len(ka), int(np.isin(ka, kb).sum()), int(np.isin(ka, kb_null).sum())])
print(pd.DataFrame(rows, columns=["A", "B", "columns matched on", "complete rows in A", "rows with an identical row in B", "same, shuffled columns (chance)"]).to_string(index=False))

print("\n=== C. same-position and same-player_id alignment (Spearman on matching columns) ===")
rows = []
for i, a in enumerate(keys_t):
    for b in keys_t[i + 1:]:
        for g, m in GROUPS.items():
            if a not in m or b not in m:
                continue
            for how, sa, sb in [("ANS position, test", "test", "test"), ("player_id, test", "test", "test"), ("position, train", "train", "train"), ("player_id, train", "train", "train")]:
                if (a, sa) not in num or (b, sb) not in num:
                    continue
                A, B = num[a, sa], num[b, sb]
                if how.startswith("player_id"):
                    J = A[["player_id", m[a]]].rename(columns={m[a]: "x"}).merge(B[["player_id", m[b]]].rename(columns={m[b]: "y"}), on="player_id")
                else:
                    k = min(len(A), len(B))
                    J = pd.DataFrame({"x": A[m[a]].values[:k], "y": B[m[b]].values[:k]})
                J = J.apply(pd.to_numeric, errors="coerce").dropna(subset=["x", "y"])
                rows.append([f"{a}-{b}", g, how, len(J), spearmanr(J.x, J.y).statistic if len(J) > 10 else np.nan])
C_ = pd.DataFrame(rows, columns=["pair", "group", "alignment", "n", "spearman"])
print(C_.round(4).to_string(index=False))
print(f"largest |Spearman| of {len(C_)} pairings: {C_.spearman.abs().max():.4f}")
seg_t2, seg_t3 = num["t2", "train"].segment.values, num["t3", "train"].segment.values
ok = ~(pd.isna(seg_t2) | pd.isna(seg_t3))
print(f"train by position: Task 2 segment = Task 3 segment on {np.mean(seg_t2[ok] == seg_t3[ok]):.3%} of rows")
print("mean total_spending_thb by Task 2 segment:", num["t2", "train"].groupby("segment").total_spending_thb.mean().round(0).to_dict())
print("mean historical_spending by Task 3 segment:", num["t3", "train"].groupby("segment").historical_spending.mean().round(0).to_dict())
print("mean friend_count by Task 2 segment:", num["t2", "train"].groupby("segment").friend_count.mean().round(1).to_dict())
print("mean friend_count by Task 3 segment:", num["t3", "train"].groupby("segment").friend_count.mean().round(1).to_dict())

print("\n=== D. cross-tabulation of submissions for the same ANS id ===")
for nm, f in [("v11 (Task 2 and 5 from player_id blocks)", "submission_v11_try.csv"), ("clean pipeline (no player_id)", "submission_clean_try.csv")]:
    S = pd.read_csv(f)
    print(f"\n--- {nm}: test rows {len(S)}")
    print("Task 3 predicted spending by Task 2 class:\n", S.groupby("task2").task3.agg(["count", "mean", "median", lambda v: (v > 0).mean()]).rename(columns={"<lambda_0>": "share > 0"}).round(3).to_string())
    print("Task 1 flag rate by Task 2 class:", S.groupby("task2").task1.mean().round(4).to_dict())
    print("Task 5 flag rate by Task 2 class:", S.groupby("task2").task5.mean().round(4).to_dict())
    print("Task 1 flag rate by Task 5:", S.groupby("task5").task1.mean().round(4).to_dict())
    print("Task 3 > 0 rate by Task 1:", S.groupby("task1").task3.apply(lambda v: (v > 0).mean()).round(4).to_dict(), " by Task 5:", S.groupby("task5").task3.apply(lambda v: (v > 0).mean()).round(4).to_dict())
    print("Task 4 class shares by Task 2 class:\n", pd.crosstab(S.task2, S.task4, normalize="index").round(3).to_string())
    for a, b in [("task1", "task2"), ("task2", "task5"), ("task1", "task5"), ("task2", "task4"), ("task1", "task4"), ("task4", "task5")]:
        ct = pd.crosstab(S[a], S[b])
        chi, p, _, _ = chi2_contingency(ct)
        print(f"  {a} x {b}: chi2 p = {p:.3f}, Cramer's V = {np.sqrt(chi / (len(S) * (min(ct.shape) - 1))):.4f}")
    print(f"  task3 x task1 Spearman {spearmanr(S.task3, S.task1).statistic:+.4f}, task3 x task5 {spearmanr(S.task3, S.task5).statistic:+.4f}, task3 x task2 {spearmanr(S.task3, S.task2).statistic:+.4f}")

print("\n=== E. cross-task feature transfer (training rows only) ===")


def rank_map(train_v, v):
    """Empirical CDF of the source's own training rows: makes the different column scales comparable. NaN stays NaN."""
    tr = np.sort(train_v[~np.isnan(train_v)])
    return np.where(np.isnan(v), np.nan, np.searchsorted(tr, v, side="right") / len(tr))


def mapped(t, df_all_train, df, groups):
    return pd.DataFrame({g: rank_map(pd.to_numeric(df_all_train[GROUPS[g][t]], errors="coerce").values.astype(float),
                                     pd.to_numeric(df[GROUPS[g][t]], errors="coerce").values.astype(float)) for g in groups}, index=df.index)


def aux_fit(src, y_src, groups, task, targets, n_classes=None):
    """Fit on source-task training rows and return predictions on each (task, frame); hold-out check on a 20% slice."""
    tr = num[src, "train"]
    Xs = mapped(src, tr, tr, groups)
    keep = ~pd.isna(y_src)
    Xs, ys = Xs[keep], np.asarray(y_src)[keep]
    if task != "regression":
        ys = ys.astype(int)
    idx = np.random.default_rng(0).permutation(len(ys))
    va, fi = idx[:len(ys) // 5], idx[len(ys) // 5:]
    params = {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 2000}
    _, (pva,), it = fit_predict("lgb", task, params, Xs.iloc[fi], ys[fi], Xs.iloc[va], ys[va], [Xs.iloc[va]], early_stopping=100)
    quality = macro_f1(ys[va], pva.argmax(1)) if task == "multiclass" else nmae(np.expm1(ys[va]), np.expm1(pva))
    _, preds, _ = fit_predict("lgb", task, params, Xs.iloc[fi], ys[fi], Xs.iloc[va], ys[va],
                              [mapped(t, num[t, "train"], f, groups) for t, f in targets], early_stopping=100)
    return quality, preds


T3_G = ["friends", "session_length", "account_age", "playtime", "last_login", "achievement", "rank", "spending", "streak", "frequency"]
T1_G = ["friends", "session_length", "account_age", "win_rate", "frequency"]

# ------- Task 2 segment model -> Task 3 rows and Task 1 rows
tr3, te3 = task3.load()
X3, y3, Xt3, _ = task3.build_repaired(tr3, te3)
folds3 = make_folds(y3, kind="regression")
assert len(X3) == len(tr3), (len(X3), len(tr3))
q, (a3, at3) = aux_fit("t2", num["t2", "train"].segment.values, T3_G, "multiclass", [("t3", tr3), ("t3", te3)])
log(f"aux Task 2 segment model on {len(T3_G)} shared columns: hold-out macro-F1 {q:.4f} (full Task 2 feature model 0.7536)")
add = lambda X_, A_, pre: pd.concat([X_.reset_index(drop=True), pd.DataFrame(A_, columns=[f"{pre}{i}" for i in range(A_.shape[1])])], axis=1)
base3 = load_preds("t3r_lgb_raw")
r = run_cv("lgb", "regression", P3_lgb, add(X3, a3, "aux_seg"), y3, add(Xt3, at3, "aux_seg"), folds3, name="T3 aux", verbose=False)
o0, o1 = np.clip(base3["oof"], 0, None), np.clip(r["oof"], 0, None)
pf = lambda p: np.array([nmae(y3[b], p[b]) for _, b in folds3])
print(f"Task 3 LGB-L1 raw member: cached {nmae(y3, o0):.5f}; with Task 2 segment probabilities {nmae(y3, o1):.5f}; delta {nmae(y3, o1) - nmae(y3, o0):+.5f}; folds better {(pf(o1) > pf(o0)).sum()}/5", flush=True)

# ------- Task 3 models (segment and spending) -> Task 2 rows
tr2, te2, X2, y2, Xt2, folds2 = load_t2()
assert len(X2) == len(num["t2", "train"]), (len(X2), len(num["t2", "train"]))
q1, (b2, bt2) = aux_fit("t3", num["t3", "train"].segment.values, T3_G, "multiclass", [("t2", num["t2", "train"]), ("t2", num["t2", "test"])])
q2, (c2, ct2) = aux_fit("t3", np.log1p(num["t3", "train"].spending_30d.values), T3_G, "regression", [("t2", num["t2", "train"]), ("t2", num["t2", "test"])])
log(f"aux Task 3 models on shared columns: segment hold-out macro-F1 {q1:.4f}, spending hold-out NMAE {q2:.4f}")
A2, At2 = np.column_stack([b2, c2]), np.column_stack([bt2, ct2])
base2 = load_preds("t2_lgb")["oof"]
r = run_cv("lgb", "multiclass", P2_lgb, add(X2, A2, "aux"), y2, add(Xt2, At2, "aux"), folds2, n_classes=4, name="T2 aux", verbose=False)


def nested(P):
    pred = np.zeros(len(y2), int)
    for a, b in folds2:
        w, _ = optimize_class_weights(y2[a], P[a])
        pred[b] = apply_class_weights(P[b], w)
    return pred


n0, n1 = nested(base2), nested(r["oof"])
pf2 = lambda p: np.array([macro_f1(y2[b], p[b]) for _, b in folds2])
print(f"Task 2 LightGBM member (no player_id): cached {macro_f1(y2, n0):.4f}; with Task 3 aux features {macro_f1(y2, n1):.4f}; delta {macro_f1(y2, n1) - macro_f1(y2, n0):+.4f}; folds better {(pf2(n1) > pf2(n0)).sum()}/5", flush=True)

# ------- Task 2 segment model -> Task 1 rows
tr1, te1, X1, y1, Xt1, folds1, _ = load_t1()
tr1_all = num["t1", "train"]
lab = tr1_all[tr1_all.is_cheater.notna()]
assert len(lab) == len(X1), (len(lab), len(X1))
q3, (d1, dt1) = aux_fit("t2", num["t2", "train"].segment.values, T1_G, "multiclass", [("t1", lab), ("t1", num["t1", "test"])])
log(f"aux Task 2 segment model on {len(T1_G)} columns matching Task 1: hold-out macro-F1 {q3:.4f}")


spw =(y1 == 0).sum() / (y1 == 1).sum()
P1 = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base1 = load_preds("t1_lgb")["oof"]
r = run_cv("lgb", "binary", P1, add(X1, d1, "aux"), y1, add(Xt1, dt1, "aux"), folds1, name="T1 aux", verbose=False)


def nested_f2(p):
    out = np.zeros(len(y1), int)
    for a, b in folds1:
        t = optimize_threshold(y1[a], p[a], 2.0, average="binary")[0]
        out[b] = p[b] >= t
    return out


e0, e1 = nested_f2(base1), nested_f2(r["oof"])
pf1 = lambda v: np.array([f2(y1[b], v[b], average="binary") for _, b in folds1])
print(f"Task 1 LightGBM member: cached AUC {roc_auc_score(y1, base1):.4f}, nested F2 {f2(y1, e0, average='binary'):.4f}; with Task 2 aux features AUC {roc_auc_score(y1, r['oof']):.4f}, "
      f"nested F2 {f2(y1, e1, average='binary'):.4f}; delta F2 {f2(y1, e1, average='binary') - f2(y1, e0, average='binary'):+.4f}; folds better {(pf1(e1) > pf1(e0)).sum()}/5", flush=True)
log("done")
