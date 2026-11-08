"""Task 2: is there any signal in player_id (or in the row order) about the segment?

Usage: python p_t2_id.py <path-to-original-train.csv>
The original file (as committed / as downloaded) is passed in because the working copy may have been re-saved.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "02_multiclass_classification"
TARGET = "segment"

orig = pd.read_csv(sys.argv[1])
cur = pd.read_csv(D / "train.csv")
te = pd.read_csv(D / "test.csv")

# ------------------------------------------------------------------ 0. is the working copy the same data?
print("=== 0. working copy vs original file ===")
print(f"rows: original {len(orig):,}, working copy {len(cur):,}; same columns: {list(orig.columns) == list(cur.columns)}")
o = orig.sort_values("player_id").reset_index(drop=True)
c = cur.sort_values("player_id").reset_index(drop=True)
print(f"same player_ids: {o.player_id.equals(c.player_id)}; same labels: {o[TARGET].equals(c[TARGET])}; "
      f"working copy already sorted by id: {cur.player_id.is_monotonic_increasing}")
num = [k for k in o.columns if pd.api.types.is_numeric_dtype(o[k]) and k != TARGET]
cat = [k for k in o.columns if k not in num and k not in ("player_id", TARGET)]
nan_same = all((o[k].isna() == c[k].isna()).all() for k in o.columns)
rel = max(float(np.nanmax(np.abs(o[k] - c[k]) / np.maximum(np.abs(o[k]), 1e-12))) for k in num)
cat_same = all(o[k].fillna("<NA>").astype(str).equals(c[k].fillna("<NA>").astype(str)) for k in cat)
print(f"same NaN pattern: {nan_same}; categorical columns identical: {cat_same}; "
      f"largest relative change in a numeric value: {rel:.1e}")

# ------------------------------------------------------------------ 1. id layout
df = o.copy()
df["id"] = df.player_id.str[1:].astype(int)
te_id = te.player_id.str[1:].astype(int)
allid = np.sort(np.r_[df.id.values, te_id.values])
print("\n=== 1. id layout ===")
print(f"train ids {df.id.min()}..{df.id.max()} ({len(df):,}), test ids {te_id.min()}..{te_id.max()} ({len(te):,}); "
      f"overlap {len(set(df.id) & set(te_id))}; train+test = 0..{allid.max()} without gaps: "
      f"{np.array_equal(allid, np.arange(allid.max() + 1))}")
is_test = np.zeros(allid.max() + 1, bool); is_test[te_id] = True
dec = pd.qcut(np.arange(len(is_test)), 10, labels=False)
print("test share per id decile:", np.round(pd.Series(is_test).groupby(dec).mean().values, 3))
y = df[TARGET].values
K = int(y.max()) + 1
prior = np.bincount(y, minlength=K) / len(y)
print("class prior:", np.round(prior, 4))

# ------------------------------------------------------------------ 2. global relation id <-> class
print("\n=== 2. global relation between id and segment ===")
rho, p = stats.spearmanr(df.id, y)
print(f"Spearman(id, segment) = {rho:+.4f} (p={p:.2f})")
H, p = stats.kruskal(*[df.id[y == k] for k in range(K)])
print(f"Kruskal-Wallis id across segments: H={H:.2f}, p={p:.3f}")
for k in range(K):
    print(f"  segment {k}: mean id {df.id[y == k].mean():9.0f}, one-vs-rest AUC of id = {roc_auc_score(y == k, df.id):.4f}")
for nb in (10, 50, 200):
    tab = pd.crosstab(pd.qcut(df.id, nb, labels=False), y)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    print(f"chi-square class x {nb} id-bins: chi2={chi2:.1f}, dof={dof}, p={p:.3f}")
tab10 = pd.crosstab(pd.qcut(df.id, 10, labels=False), y, normalize="index")
print("class share per id decile (rows = decile):\n", tab10.round(3).to_string())

# ------------------------------------------------------------------ 3. local structure (neighbours, runs, periodicity)
print("\n=== 3. local structure in id order ===")
exp_same = float((prior ** 2).sum())
for lag in (1, 2, 3, 5, 10, 100):
    same = (y[lag:] == y[:-lag]).mean()
    print(f"lag {lag:3d}: P(same segment) = {same:.4f} vs {exp_same:.4f} expected if unrelated "
          f"(z={(same - exp_same) / np.sqrt(exp_same * (1 - exp_same) / (len(y) - lag)):+.2f})")
# only pairs whose ids are truly consecutive (no test row in between)
cons = np.diff(df.id.values) == 1
same_c = (y[1:][cons] == y[:-1][cons]).mean()
print(f"truly consecutive ids ({cons.sum():,} pairs): P(same) = {same_c:.4f}")
runs = 1 + int((y[1:] != y[:-1]).sum())
n = len(y)
exp_runs = 1 + (n - 1) * (1 - exp_same)
print(f"runs of equal segment: {runs:,} vs {exp_runs:,.0f} expected")
worst = []
for m in range(2, 201):
    tab = pd.crosstab(df.id % m, y)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    worst.append((p, m, chi2, dof))
worst.sort()
print("periodicity id mod m, m=2..200, smallest p-values:", [(m, round(p, 4)) for p, m, _, _ in worst[:5]],
      f"(with 199 tests, expect the smallest p near {1 / 199:.4f})")
digit = pd.crosstab(df.id % 10, y)
print(f"last digit x segment: p={stats.chi2_contingency(digit)[1]:.3f}")

# ------------------------------------------------------------------ 4. can the id predict the segment? (5-fold OOF)
print("\n=== 4. predictive test (5-fold OOF macro-F1; majority-class guess = "
      f"{f1_score(y, np.full_like(y, prior.argmax()), average='macro'):.4f}) ===")
from lightgbm import LGBMClassifier
skf = StratifiedKFold(5, shuffle=True, random_state=42)
ids = df.id.values.reshape(-1, 1)
oof = np.zeros((n, K)); oof_knn = np.zeros((n, K))
for a, b in skf.split(ids, y):
    m = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, verbose=-1)
    oof[b] = m.fit(ids[a], y[a]).predict_proba(ids[b])
    # neighbour vote: classes of the 10 nearest training ids
    sa = np.argsort(ids[a, 0]); ia, ya = ids[a, 0][sa], y[a][sa]
    pos = np.searchsorted(ia, ids[b, 0])
    for j, pp in enumerate(pos):
        lo, hi = max(pp - 5, 0), min(pp + 5, len(ia))
        oof_knn[b[j]] = np.bincount(ya[lo:hi], minlength=K) / (hi - lo)
from sklearn.metrics import log_loss
sys.path.insert(0, str(ROOT))
from src.metrics import optimize_class_weights  # noqa: E402
base_ll = log_loss(y, np.tile(prior, (n, 1)))
for name, P in [("LightGBM on id", oof), ("10 nearest training ids vote", oof_knn)]:
    Pc = np.clip(P, 1e-6, 1); Pc = Pc / Pc.sum(1, keepdims=True)
    _, f = optimize_class_weights(y, Pc)
    print(f"{name:30s}: log-loss {log_loss(y, Pc):.4f} (prior-only {base_ll:.4f}), "
          f"macro-F1 with tuned class weights {f:.4f}, AUC per class "
          f"{[round(roc_auc_score(y == k, Pc[:, k]), 3) for k in range(K)]}")
_, f_prior = optimize_class_weights(y, np.tile(prior, (n, 1)) + 1e-9 * np.random.default_rng(0).random((n, K)))
print(f"{'prior + tiny noise (reference)':30s}: macro-F1 with tuned class weights {f_prior:.4f}")

# ------------------------------------------------------------------ 5. id vs features, and file row order
print("\n=== 5. id vs features / row order ===")
feat_rho = {k: stats.spearmanr(df.id, df[k], nan_policy="omit")[0] for k in num}
top = sorted(feat_rho.items(), key=lambda t: -abs(t[1]))[:5]
print("largest |Spearman(id, feature)|:", [(k, round(v, 4)) for k, v in top], f"(noise level ~{2 / np.sqrt(n):.4f})")
pos_in_file = orig.reset_index().rename(columns={"index": "row"})
rho, p = stats.spearmanr(pos_in_file.row, pos_in_file[TARGET])
print(f"original file row position vs segment: Spearman {rho:+.4f} (p={p:.2f}); "
      f"lag-1 P(same) in file order {(pos_in_file[TARGET].values[1:] == pos_in_file[TARGET].values[:-1]).mean():.4f}")
rho, p = stats.spearmanr(pos_in_file.row, pos_in_file.player_id.str[1:].astype(int))
print(f"original file row position vs id: Spearman {rho:+.4f} (p={p:.2f})  -> file was shuffled: {abs(rho) < 0.01}")
rho_t, _ = stats.spearmanr(np.arange(len(te)), te_id)
print(f"test file row position vs id: Spearman {rho_t:+.4f}")

out = ROOT / "experiments" / "claude" / "outputs"
out.mkdir(parents=True, exist_ok=True)
df.drop(columns="id").to_csv(out / "t2_train_sorted_by_id.csv", index=False)
print(f"\nsorted copy (full precision) written to {out / 't2_train_sorted_by_id.csv'}")
