"""Round 19 forensics, tabular tasks (1, 2, 3, 5) on the re-released data. Diagnostics only: nothing here trains a model on test rows.

    python ck_r19_forensics_tab.py

  1. Task 2 / Task 5 old vs new: what did the TAs change (regenerated rows or perturbed old rows? same generator?)
  2. Task 2 generating process: per-class marginal families, discreteness, clipping, hidden sub-groups (within-class eigenvalues
     against the random-matrix bound), row-level signals (row order, NaN count, decimal places) against the label and the OOF errors
  3. Tasks 1, 3, 5: per-row precision artifacts (decimal places, integer-valued floats, zeros, NaNs) against the target
  4. Task 5: the anomaly archetypes our detector flags, and whether a geo-only "VPN regional exploiter" group is left unflagged
The old release is read from git (commit fe3d291), so the script is reproducible.
"""
import io
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

OLD_REV = "fe3d291"
rng = np.random.default_rng(0)
NAMES2 = ["Casual", "Grinder", "Social", "Whale"]


def git_csv(path, **kw):
    return pd.read_csv(io.StringIO(subprocess.run(["git", "show", f"{OLD_REV}:{path}"], capture_output=True, text=True, check=True).stdout), **kw)


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def decimals_frame(raw_str, cols):
    """Number of decimal places of every cell (NaN for empty cells), from the csv text itself."""
    out = {}
    for c in cols:
        s = raw_str[c]
        d = s.str.split(".").str[1].str.len()
        out[c] = np.where(s.isna(), np.nan, d.fillna(0))
    return pd.DataFrame(out)


# ======================================================================================== 1. old vs new
hdr("1. Task 2 / Task 5: old release vs new release")
o2, n2 = git_csv("02_multiclass_classification/train.csv"), pd.read_csv("02_multiclass_classification/train.csv")
o2t, n2t = git_csv("02_multiclass_classification/test.csv"), pd.read_csv("02_multiclass_classification/test.csv")
o5, n5 = git_csv("05_anomaly_detection/test.csv"), pd.read_csv("05_anomaly_detection/test.csv")
num2 = [c for c in n2.columns if c not in ("player_id", "segment") and pd.api.types.is_numeric_dtype(n2[c])]
cat2 = [c for c in n2.columns if c not in num2 + ["player_id", "segment"]]

ks_all = {c: stats.ks_2samp(o2[c].dropna(), n2[c].dropna()).statistic for c in num2}
ks_cls = {c: max(stats.ks_2samp(o2.loc[o2.segment == k, c].dropna(), n2.loc[n2.segment == k, c].dropna()).statistic for k in range(4)) for c in num2}
crit = 1.36 * np.sqrt(2 / 20000)   # 5% KS critical value for ~20k vs ~20k rows (smallest class)
print(f"T2 numeric columns: KS old vs new, all rows: max {max(ks_all.values()):.4f}; within class: max {max(ks_cls.values()):.4f} "
      f"(5% critical value at the smallest class size ~{crit:.4f})")
print("  largest within-class KS:", {c: round(v, 4) for c, v in sorted(ks_cls.items(), key=lambda kv: -kv[1])[:5]})
tv = {c: 0.5 * (o2[c].fillna("NA").value_counts(normalize=True).subtract(n2[c].fillna("NA").value_counts(normalize=True), fill_value=0).abs().sum()) for c in cat2}
print("  categorical columns: same value sets:", all(set(o2[c].dropna()) == set(n2[c].dropna()) for c in cat2),
      "| max total-variation distance:", round(max(tv.values()), 4))
dm = pd.DataFrame({k: (n2.loc[n2.segment == k, num2].mean() - o2.loc[o2.segment == k, num2].mean()) / o2[num2].std() for k in range(4)})
print(f"  class-conditional means, new - old in SD units: max |diff| {dm.abs().max().max():.4f} (sampling noise ~{2 / np.sqrt(16000):.4f})")


def std_matrix(df, ref):
    med, iqr = ref[num2].median(), (ref[num2].quantile(.75) - ref[num2].quantile(.25)).replace(0, 1)
    return ((df[num2] - med) / iqr).fillna(0).values.astype(np.float32)


def nn_dist(A, B, exclude_self=False):
    out = np.empty(len(A), np.float32)
    bb = (B ** 2).sum(1)
    for i in range(0, len(A), 500):
        a = A[i:i + 500]
        d = (a ** 2).sum(1)[:, None] + bb[None] - 2 * a @ B.T
        if exclude_self:
            d[d < 1e-6] = np.inf
        out[i:i + 500] = np.sqrt(np.maximum(d.min(1), 0))
    return out


Zn, Zo = std_matrix(n2, n2), std_matrix(o2, n2)
s = rng.choice(len(Zn), 3000, replace=False)
d_old, d_new = nn_dist(Zn[s], Zo), nn_dist(Zn[s], Zn, exclude_self=True)
print(f"  nearest-neighbour distance of 3,000 new rows: to the OLD rows median {np.median(d_old):.3f} | to OTHER NEW rows median {np.median(d_new):.3f}"
      f" | share closer to an old row than the typical new-new distance / 10: {np.mean(d_old < np.median(d_new) / 10):.4f}")
for nm, o, n in (("T2 train", o2, n2), ("T2 test", o2t, n2t), ("T5 test", o5, n5)):
    oi, ni = o.player_id.str[1:].astype(int), n.player_id.str[1:].astype(int)
    print(f"  {nm}: ids old {oi.min()}-{oi.max()} (n {len(oi)}), new {ni.min()}-{ni.max()} (n {len(ni)}); same id set: {set(oi) == set(ni)}; "
          f"rank corr(row, id) new {stats.spearmanr(np.arange(len(ni)), ni)[0]:+.4f}")
allid_o = set(o2.player_id) | set(o2t.player_id)
allid_n = set(n2.player_id) | set(n2t.player_id)
print(f"  T2 train+test ids: old {len(allid_o):,}, new {len(allid_n):,} of 130,000 possible; missing ids identical: "
      f"{set(f'P{i:06d}' for i in range(130000)) - allid_o == set(f'P{i:06d}' for i in range(130000)) - allid_n}")
num5 = [c for c in n5.columns if c != "player_id"]
ks5 = {c: stats.ks_2samp(o5[c].dropna(), n5[c].dropna()).statistic for c in num5}
print(f"T5 test: KS old vs new per column: max {max(ks5.values()):.4f} ({max(ks5, key=ks5.get)}); 5% critical ~{1.36 * np.sqrt(2 / 25889):.4f}; "
      f"NaN rate old {o5[num5].isna().mean().mean():.4f} new {n5[num5].isna().mean().mean():.4f}")
for nm, df in (("old", o5), ("new", n5)):
    print(f"  T5 {nm}: invalid coordinates (|lat|>90 or |lon|>180) in {int(((df.filter(like='login_lat').abs() > 90).any(axis=1) | (df.filter(like='login_lon').abs() > 180).any(axis=1)).sum())} rows; "
          f"binary flag means {df[task5.FLAG_COLS].mean().round(4).to_dict()}")

# ======================================================================================== 2. Task 2 generating process
hdr("2. Task 2 (new train): how the features were generated")
y2 = n2.segment.values
raw2 = pd.read_csv("02_multiclass_classification/train.csv", dtype=str)
dec2 = decimals_frame(raw2, num2)
rows = []
for c in num2:
    v = n2[c]
    vv = v.dropna()
    integer = np.mean(vv == np.round(vv))
    fams = {}
    for k in range(4):
        x = n2.loc[y2 == k, c].dropna().values
        cand = {"normal": stats.norm(*stats.norm.fit(x))}
        if x.min() > 0:
            cand["lognormal"] = stats.lognorm(*stats.lognorm.fit(x, floc=0))
            cand["gamma"] = stats.gamma(*stats.gamma.fit(x, floc=0))
        if x.min() >= 0:
            cand["exponential"] = stats.expon(*stats.expon.fit(x, floc=0))
        cand["uniform"] = stats.uniform(x.min(), x.max() - x.min())
        sub = rng.choice(x, min(len(x), 4000), replace=False)
        ksd = {f: stats.kstest(sub, d.cdf).statistic for f, d in cand.items()}
        fams[k] = min(ksd, key=ksd.get) + f"({ksd[min(ksd, key=ksd.get)]:.3f})"
    lo, hi = vv.min(), vv.max()
    rows.append([c, vv.nunique(), round(integer, 3), int(np.nanmedian(dec2[c])), round(np.nanmax(dec2[c])), lo, hi,
                 round(np.mean(vv == lo), 4), round(np.mean(vv == hi), 4), round(v.isna().mean(), 4),
                 round(max(n2.loc[y2 == k, c].isna().mean() for k in range(4)) - min(n2.loc[y2 == k, c].isna().mean() for k in range(4)), 4),
                 " ".join(fams[k] for k in range(4))])
G = pd.DataFrame(rows, columns=["col", "n_unique", "int_share", "dec_med", "dec_max", "min", "max", "at_min", "at_max", "nan",
                                "nan_range_by_class", "best family by class (KS) Casual/Grinder/Social/Whale"]).set_index("col")
with pd.option_context("display.width", 250, "display.max_colwidth", 120):
    print(G.to_string())
print("  columns with >= 1% of values exactly at the min or max (clipping):",
      G.index[(G.at_min >= 0.01) | (G.at_max >= 0.01)].tolist())
print("  columns whose NaN rate differs by class by > 0.005:", G.index[G.nan_range_by_class > 0.005].tolist() or "none")

# hidden sub-groups: within-class correlation eigenvalues vs the Marchenko-Pastur edge of pure noise
from sklearn.preprocessing import QuantileTransformer
Q = pd.DataFrame(QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(n2[num2].fillna(n2[num2].median())), columns=num2)
print("  within-class top eigenvalue of the correlation matrix (no sub-groups -> at the random bound (1+sqrt(p/n))^2):")
for k in range(4):
    Xk = Q[y2 == k].values
    ev = np.linalg.eigvalsh(np.corrcoef(Xk, rowvar=False))[::-1]
    mp = (1 + np.sqrt(Xk.shape[1] / len(Xk))) ** 2
    print(f"    {NAMES2[k]:7s} n={len(Xk):6d}: top 3 eigenvalues {np.round(ev[:3], 3).tolist()} vs bound {mp:.3f}")
from sklearn.mixture import GaussianMixture
top =list(pd.Series({c: abs(stats.spearmanr(n2[c], np.isin(y2, [1, 3]), nan_policy="omit")[0]) + abs(stats.spearmanr(n2[c], np.isin(y2, [2, 3]), nan_policy="omit")[0]) for c in num2}).nlargest(12).index)
print(f"  Gaussian mixtures per class on the 12 most informative columns (quantile-normal), BIC(k) - BIC(1), lower = better:")
for k in range(4):
    Xk = Q.loc[y2 == k, top].values
    b = [GaussianMixture(m, covariance_type="full", random_state=0, n_init=2).fit(Xk).bic(Xk) for m in (1, 2, 3)]
    print(f"    {NAMES2[k]:7s}: 2 comps {b[1] - b[0]:+.0f}, 3 comps {b[2] - b[0]:+.0f}")

# row-level signals against the label and against the OOF errors
oof = np.load("experiments/claude/outputs/r18_t2_nb.npz")["stack_oof"]
err = oof.argmax(1) != y2
rowinfo = pd.DataFrame({"row_decile": np.arange(len(y2)) * 10 // len(y2), "na_count": n2[num2 + cat2].isna().sum(1).clip(upper=6),
                        "dec_sum": dec2.sum(1, skipna=True), "y": y2, "err": err})
rowinfo["dec_bin"] = pd.qcut(rowinfo.dec_sum, 5, duplicates="drop")
for col in ("row_decile", "na_count", "dec_bin"):
    t = rowinfo.groupby(col, observed=True).agg(n=("y", "size"), casual=("y", lambda v: np.mean(v == 0)), whale=("y", lambda v: np.mean(v == 3)),
                                                err_rate=("err", "mean"))
    chi = stats.chi2_contingency(pd.crosstab(rowinfo[col], rowinfo.y))[1]
    print(f"  {col}: chi2 p-value vs label {chi:.3g}\n" + t.round(4).to_string())
print(f"  AUC of 'decimal places in the row' for an OOF error: {roc_auc_score(err, rowinfo.dec_sum):.4f}; "
      f"of NaN count: {roc_auc_score(err, rowinfo.na_count):.4f}")
# 2b. are decimal places just the value's magnitude? (shortest float repr = ~15-17 significant digits)
for c in ("total_spending_thb", "total_playtime_hours", "avg_monthly_spending", "play_frequency"):
    v, d = n2[c], dec2[c]
    ok = v.notna() & (v != 0)
    intd = np.floor(np.log10(v[ok].abs())) + 1
    print(f"  {c:22s}: rank corr(decimals, integer digits) {stats.spearmanr(d[ok], intd)[0]:+.3f}; significant digits (int + dec) "
          f"quantiles 5/50/95% {np.quantile(d[ok] + np.maximum(intd, 0), [.05, .5, .95]).tolist()}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score
L = np.log(np.clip(oof, 1e-6, 1))
sig = dec2.add(np.floor(np.log10(n2[num2].abs().replace(0, np.nan))) + 1).mean(1)   # mean significant digits per row
extra = np.column_stack([(rowinfo.dec_sum - rowinfo.dec_sum.mean()) / rowinfo.dec_sum.std(), (sig - sig.mean()) / sig.std()])
for nm, M_ in (("stack probabilities", L), ("+ decimal places & significant digits", np.column_stack([L, extra]))):
    pr = np.zeros_like(oof)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(M_, y2):
        pr[b] = LogisticRegression(C=1.0, max_iter=2000).fit(M_[a], y2[a]).predict_proba(M_[b])
    print(f"  re-stack on {nm:40s}: log-loss {log_loss(y2, pr):.5f}, macro-F1 {f1_score(y2, pr.argmax(1), average='macro'):.4f}")
# 2c. Whale's extra factor
Xw = Q[y2 == 3]
ev, V = np.linalg.eigh(np.corrcoef(Xw.values, rowvar=False))
v1 = pd.Series(V[:, -1], index=num2)
print("  Whale top eigenvector, largest loadings:", v1.abs().sort_values(ascending=False).head(8).round(3).to_dict())
score = Q[num2].values @ V[:, -1]
print("  that factor's mean score by class (Casual/Grinder/Social/Whale):", [round(float(score[y2 == k].mean()), 3) for k in range(4)],
      "| SD in Whale", round(float(score[y2 == 3].std()), 3), "vs Social", round(float(score[y2 == 2].std()), 3))

# ======================================================================================== 3. precision artifacts, Tasks 1 / 3 / 5
hdr("3. Per-row precision artifacts vs the target (Tasks 1, 3; Task 5 vs our flags)")
for nm, path, target in (("T1", "01_binary_classification/train.csv", "is_cheater"), ("T3", "03_regression/train.csv", "spending_30d"),
                         ("T5", "05_anomaly_detection/test.csv", None)):
    raw = pd.read_csv(path, dtype=str)
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("player_id", target) and pd.api.types.is_numeric_dtype(df[c])]
    dec = decimals_frame(raw, cols)
    feat = pd.DataFrame({"decimals_sum": dec.sum(1), "decimals_max": dec.max(1), "n_zero": (df[cols] == 0).sum(1),
                         "n_nan": df[cols].isna().sum(1), "n_int_valued": ((df[cols] == np.round(df[cols])) & df[cols].notna()).sum(1)})
    if nm == "T5":
        t = pd.read_csv("submission_v14.csv").task5.values
        lab, desc = t, "our v14 flags"
    elif nm == "T1":
        ok = df[target].notna()
        feat, lab, desc = feat[ok], df.loc[ok, target].values, "is_cheater"
    else:
        lab, desc = (df[target] > 0).astype(int).values, "spending > 0"
    aucs = {c: round(roc_auc_score(lab, feat[c]), 4) for c in feat}
    print(f"  {nm} ({desc}): AUC of each row artifact {aucs}")
    if nm == "T3":
        pos = df[target] > 0
        print(f"     T3 Spearman with log spending on spenders: {({c: round(stats.spearmanr(feat.loc[pos, c], np.log(df.loc[pos, target]))[0], 4) for c in feat})}")
        zc = (df[cols] == 0).mean().sort_values(ascending=False)
        print(f"     T3 columns with the most exact zeros: {zc.head(6).round(3).to_dict()}")
        h = np.load("outputs/preds/t3r_hurdle.npz")
        pred = h["oof"] if "oof" in h.files else h["p_oof"] * h["mu_oof"]
        res = np.log1p(df[target].values) - np.log1p(np.clip(pred, 0, None))
        print(f"     hurdle-member OOF residual (log1p) vs n_zero: Spearman {stats.spearmanr(res, feat.n_zero)[0]:+.4f}; mean residual by n_zero: "
              f"{pd.Series(res).groupby(feat.n_zero.clip(upper=8).values).mean().round(3).to_dict()}")
        print(f"     ... vs decimals_sum: Spearman {stats.spearmanr(res, feat.decimals_sum)[0]:+.4f}; vs n_int_valued {stats.spearmanr(res, feat.n_int_valued)[0]:+.4f}")
    if nm == "T5":
        T5DEC = (dec, cols)

# ======================================================================================== 4. Task 5 archetypes
hdr("4. Task 5 (new test): which archetypes we flag, and what is left")
X5, _ = task5.build_features(n5)
gc_table, gc_masks = task5.gap_clusters(n5, X5)
fams = task5.merge_families(gc_masks)
flags = pd.read_csv("submission_v14.csv").task5.values.astype(bool)
fam_any = np.zeros(len(n5), bool)
Z = X5.apply(lambda s: (s - s.median()) / (1.4826 * (s - s.median()).abs().median() + 1e-9))
for i, f in enumerate(sorted(fams, key=lambda f: -gc_masks[f].any(axis=1).sum())):
    m = gc_masks[f].any(axis=1).values
    fam_any |= m
    drivers = Z[m].median().abs().sort_values(ascending=False).head(6)
    print(f"  family {i + 1}: {m.sum():5d} accounts, columns {list(f)[:6]}{' ...' if len(f) > 6 else ''}\n      median |robust z| of top drivers: {drivers.round(1).to_dict()}")
pocket = flags & ~fam_any
print(f"  dependence pocket: {pocket.sum()} accounts; median |robust z| top drivers: {Z[pocket].median().abs().sort_values(ascending=False).head(4).round(1).to_dict()}")
print(f"  flags total {flags.sum()} = families {fam_any.sum()} + pocket {pocket.sum()} (+ {int((flags & ~fam_any & ~pocket).sum())} other)")
ids5 = n5.player_id.str[1:].astype(int).values
dec_share = pd.Series(flags).groupby(pd.qcut(ids5, 10, labels=False)).mean().round(4).tolist()
print(f"  share flagged by player_id decile (old data: all anomalies in the top 8% of ids): {dec_share}")
# geo-only exploiters: big jumps, normal login counts, one device, valid coordinates, not flagged
valid = X5.geo_invalid_coord == 0
dev = n5.filter(like="device_count").max(axis=1)
logins = n5.filter(like="login_count").mean(axis=1)
hop = X5.geo_hop_max
print(f"  unflagged accounts with valid coordinates: {int((~flags & valid).sum())}")
for q in (0.99, 0.995, 0.999):
    thr = hop[valid].quantile(q)
    m = (~flags) & valid & (hop > thr)
    print(f"    max hop > {thr:8.0f} km (top {1 - q:.1%} of valid accounts): unflagged {int(m.sum())}, of which device_count <= 1: {int((m & (dev <= 1)).sum())}, "
          f"login count within the 10-90% band: {int((m & logins.between(logins.quantile(.1), logins.quantile(.9))).sum())}")
h = np.sort(hop[valid & ~flags].dropna().values)[::-1][:400]
gaps = -np.diff(h)
j = int(np.argmax(gaps[:300]))
print(f"    largest gap in the top-400 unflagged max-hop values: {gaps[j]:.0f} km between rank {j + 1} ({h[j]:.0f} km) and {j + 2} ({h[j + 1]:.0f} km); "
      f"median gap there {np.median(gaps[:300]):.1f} km")
for c in ("geo_hop_max", "geo_path_km", "geo_p4_from_home_km", "geo_hop34_over_hist"):
    a = X5.loc[valid, c]
    print(f"    {c:22s}: flagged median {a[flags[valid]].median():9.1f} | unflagged median {a[~flags[valid]].median():9.1f} | "
          f"unflagged p99.9 {a[~flags[valid]].quantile(.999):9.1f} | flagged p50 of the VPN-like family (vpn_usage=1): "
          f"{a[(flags & (n5.vpn_usage == 1)).values[valid.values]].median():9.1f}")
print(f"  vpn_usage = 1: {int((n5.vpn_usage == 1).sum())} accounts, flagged {int(((n5.vpn_usage == 1) & flags).sum())}; "
      f"suspicious_login_time = 1: {int((n5.suspicious_login_time == 1).sum())}, flagged {int(((n5.suspicious_login_time == 1) & flags).sum())}")

dec5, cols5 = T5DEC
grp5 = np.where(pocket, "pocket", np.where(fam_any, "family", "unflagged"))
short = (dec5 <= 6)
by = short.groupby(grp5).mean().T
by["family - unflagged"] = by["family"] - by["unflagged"]
by["pocket - unflagged"] = by["pocket"] - by["unflagged"]
print("  share of cells with <= 6 decimal places, by group (largest family/pocket differences):")
print(by.reindex(by["family - unflagged"].abs().sort_values(ascending=False).index).head(8).round(3).to_string())
print(by.reindex(by["pocket - unflagged"].abs().sort_values(ascending=False).index).head(5).round(3).to_string())
mx5 = dec5.max(1)
print("  max decimals per row, by group (quantiles 10/50/90):", {gname: np.quantile(mx5[grp5 == gname], [.1, .5, .9]).tolist() for gname in ("family", "pocket", "unflagged")})
