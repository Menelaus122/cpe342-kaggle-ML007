"""Task 5 round 4: label-free error analysis + three experiments. Task 5 has NO labels, so there is no OOF score.

Decision metric (label-free): expected F3 under a permutation null.
  * Shuffling each feature column independently keeps every marginal and destroys any group of accounts that is
    unusual in several features at once. For a score s and its top-k accounts, the expected number of NORMAL
    accounts is the mean count of shuffled rows scoring >= the k-th score; est_TP(k) = k - that count.
  * The six gap families are separated by empty gaps (the notebook's detector) and are taken as true.
  * K_hat = families + estimated size of the new group (the largest est_TP over k), shared by all candidates.
  * Expected F3(k) = 10 (n_fam + est_TP(k)) / (9 K_hat + n_fam + k)   (positive-class F3, slide 25).
The player_id is NOT used for any decision. One reference column ("F3 if the id block were the truth") is printed
at the very end for transparency only.

Error analysis : blind 20x20 quantile "dependence pocket" scan of all feature pairs outside the families, against
                 the maximum z of the same scan on column-shuffled data (the noise level); repeated after flagging.
E1 feature eng.: volatility score = Fisher combination of the features that form the pockets above the noise
                 level (selected by the scan itself; here kda_std_level and accuracy_std_level).
E2 model div.  : alternative scorers on the same features: min-rank ("all three high") and a cross-fitted
                 LightGBM density-ratio classifier (real accounts vs column-shuffled accounts).
E3 post-proc.  : cut-off rules: expected-F3 optimum (null) vs estimated-FDR 50% vs "flag the estimated group size".
"""
import itertools
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task5  # noqa: E402
from src.modeling import fit_predict  # noqa: E402

G, N_PERM = 20, 300
rng = np.random.default_rng(0)
t5 = task5.load()
X5, _ = task5.build_features(t5)
_, gc_masks = task5.gap_clusters(t5, X5)
fam = gc_masks.any(axis=1).values
n_fam = int(fam.sum())
print(f"six gap families: {n_fam:,} accounts ({n_fam / len(fam):.2%}); per family "
      f"{[int(gc_masks[c].sum()) for c in gc_masks.columns]}")


# ------------------------------------------------------------------ error analysis: residual structure
def base_name(c):
    c = re.sub(r"_(level|cv|burst|maxjump|dev4|selfz4)$", "", c)
    return "geo" if c.startswith("geo") else re.sub(r"_(mean|std)$", "", c)


cols = [c for c in X5.columns if pd.api.types.is_numeric_dtype(X5[c]) and X5[c].nunique() > 50]
# mechanical pairs: same base feature, or one is a per-login / per-account ratio built from the other
MECH = {("geo", "login_count")}


def quantile_codes(R):
    Q = np.full(R.shape, -1, dtype=np.int16)
    for j in range(R.shape[1]):
        v = R[:, j]
        ok = ~np.isnan(v)
        r = pd.Series(v[ok]).rank(pct=True, method="first").values
        Q[ok, j] = np.minimum((r * G).astype(int), G - 1)
    return Q


def pocket_scan(R):
    Q = quantile_codes(R)
    out = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        bi, bj = base_name(cols[i]), base_name(cols[j])
        if bi == bj or (bi, bj) in MECH or (bj, bi) in MECH or bi.startswith("login_count") and bj == "geo" \
                or bj.startswith("login_count") and bi == "geo":
            continue
        ok = (Q[:, i] >= 0) & (Q[:, j] >= 0)
        cnt = np.bincount(Q[ok, i] * G + Q[ok, j], minlength=G * G).reshape(G, G).astype(float)
        exp = np.outer(cnt.sum(1), cnt.sum(0)) / cnt.sum()
        z = (cnt - exp) / np.sqrt(np.maximum(exp, 1))
        k = int(np.argmax(z))
        out.append((z.flat[k], cols[i], cols[j], k // G, k % G, cnt.flat[k], exp.flat[k]))
    return pd.DataFrame(out, columns=["z", "a", "b", "bin_a", "bin_b", "obs", "exp"]).sort_values("z", ascending=False)


def shuffled(R):
    return np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])


R = X5.loc[~fam, cols].to_numpy(float)
scan = pocket_scan(R)
null_max = max(pocket_scan(shuffled(R)).z.max() for _ in range(2))
sig = scan[scan.z > null_max]
print(f"\nERROR ANALYSIS - residual structure outside the families: {len(scan):,} feature pairs scanned; "
      f"noise level (max z on shuffled data, 2 runs) = {null_max:.1f}; pairs above it: {len(sig)}")
print(sig.head(12).round(1).to_string(index=False))
feat_counts = pd.concat([sig.a, sig.b]).value_counts()
print("features in the significant pockets:", feat_counts.head(8).to_dict())

FEATS = sorted(feat_counts.index)        # chosen by the blind scan only: features of the pockets above noise
print("detector features (from the scan):", FEATS)
rest = np.where(~fam)[0]
U = X5.loc[~fam, FEATS].rank(pct=True).to_numpy()


def null_curve(score_fn, T, n_perm=N_PERM):
    S = score_fn(T)
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(len(Ss))
    for _ in range(n_perm):
        Sp = np.sort(score_fn(shuffled(T)))
        cnt += len(Sp) - np.searchsorted(Sp, Ss, side="left")
    cnt /= n_perm
    k = np.arange(1, len(Ss) + 1)
    return order, np.maximum.accumulate(k - cnt)


def fisher(T):
    L = -np.log(np.clip(1 - T + 0.5 / len(T), 1e-12, 1))
    return np.nansum(L, axis=1) * T.shape[1] / np.maximum(np.isfinite(L).sum(1), 1)


def minrank(T):
    return np.nanmin(np.where(np.isnan(T), np.nan, T), axis=1) + 1e-9 * np.nansum(T, axis=1)


curves = {}
curves["E1 Fisher volatility score"] = null_curve(fisher, U)
curves["E2a min-rank (all features high)"] = null_curve(minrank, U)

# E2b: cross-fitted density-ratio classifier, real vs column-shuffled (2 shuffled copies)
Rv = X5.loc[~fam, FEATS].to_numpy(float)
Pv = np.vstack([shuffled(Rv), shuffled(Rv)])
Xall = pd.DataFrame(np.vstack([Rv, Pv]), columns=FEATS)
lab = np.r_[np.ones(len(Rv)), np.zeros(len(Pv))].astype(int)
fold = rng.integers(0, 5, len(lab))
sc = np.zeros(len(lab))
prm = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 50}
for f in range(5):
    a, b = np.where(fold != f)[0], np.where(fold == f)[0]
    _, (pb,), _ = fit_predict("lgb", "binary", prm, Xall.iloc[a], lab[a], Xall.iloc[b], lab[b], [Xall.iloc[b]])
    sc[b] = pb
s_real, s_null = sc[:len(Rv)], np.sort(sc[len(Rv):])
order_dr = np.argsort(-s_real)
k = np.arange(1, len(Rv) + 1)
null_dr = (len(s_null) - np.searchsorted(s_null, s_real[order_dr], side="left")) * len(Rv) / len(s_null)
curves["E2b density-ratio classifier"] = (order_dr, np.maximum.accumulate(k - null_dr))

G_hat = max(c[1][:3000].max() for c in curves.values())
K_hat = n_fam + G_hat
kk = np.arange(1, len(rest) + 1)
print(f"\nestimated size of the new group (best scorer): {G_hat:.0f} accounts -> K_hat = {K_hat:.0f}")
exp_f3 = lambda est, k: 10 * (n_fam + est) / (9 * K_hat + n_fam + k)
f3_fam_only = 10 * n_fam / (9 * K_hat + n_fam)

idn = t5.player_id.str[1:].astype(int).values                # reference only, printed at the end
blk = idn >= 23819
rows = [["T5-B0", "six gap families (current pipeline)", n_fam, np.nan, f3_fam_only, 0.0, "baseline"]]
flags = {}
for name, (order, est) in curves.items():
    f3c = exp_f3(est, kk)
    k_star = int(np.argmax(f3c[:3000])) + 1
    m = np.zeros(len(fam), bool)
    m[rest[order[:k_star]]] = True
    flags[name] = (m, order, est)
    rows.append([name.split()[0], name + f", top k* = {k_star} by expected F3", n_fam + k_star,
                 est[k_star - 1] / k_star, f3c[k_star - 1], f3c[k_star - 1] - f3_fam_only, ""])
best = max(flags, key=lambda n: exp_f3(flags[n][2], kk)[:3000].max())
order, est = flags[best][1], flags[best][2]
f3c = exp_f3(est, kk)
k_opt = int(np.argmax(f3c[:3000])) + 1
prec_cum = est / kk
k_fdr = int(np.where(prec_cum[:3000] >= 0.5)[0].max()) + 1
k_size = int(round(est[:3000].max()))
for tag, kx in [("E3a cut-off = expected-F3 optimum", k_opt), ("E3b cut-off = estimated FDR 50%", k_fdr),
                ("E3c cut-off = estimated group size", k_size)]:
    rows.append([tag.split()[0], f"{best.split()[0]} scorer, {tag[4:]} (k = {kx})", n_fam + kx, est[kx - 1] / kx,
                 f3c[kx - 1], f3c[kx - 1] - f3_fam_only, ""])
res = pd.DataFrame(rows, columns=["Experiment", "Changes", "flags", "est. precision of new flags",
                                  "expected F3 (label-free)", "Delta vs B0", "Decision"])
top = res["expected F3 (label-free)"].iloc[1:].max()
res.loc[1:, "Decision"] = np.where(res["expected F3 (label-free)"].iloc[1:] >= top - 1e-9, "KEEP", "discard")
# reference column, never used above
ref = []
for _, r in res.iterrows():
    if r.Experiment == "T5-B0":
        m = fam
    elif r.Experiment.startswith("E3"):
        kx = int(r.Changes.split("k = ")[1].rstrip(")"))
        m = fam.copy(); m[rest[order[:kx]]] = True
    else:
        m = fam | flags[[n for n in flags if n.startswith(r.Experiment)][0]][0]
    tp = int((m & blk).sum())
    ref.append(10 * tp / (9 * blk.sum() + m.sum()))
res["reference only: F3 if id block = truth"] = ref
print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
res.to_csv(OUT / "t5_round4_results.csv", index=False)

# residual check after flagging with the kept configuration
keep = fam.copy(); keep[rest[order[:k_opt]]] = True
R2 = X5.loc[~keep, cols].to_numpy(float)
scan2 = pocket_scan(R2)
print(f"\nafter flagging (kept configuration): pairs above the noise level {null_max:.1f}: "
      f"{int((scan2.z > null_max).sum())}; top pocket z = {scan2.z.iloc[0]:.1f} "
      f"({scan2.a.iloc[0]} x {scan2.b.iloc[0]})")
pd.DataFrame({"player_id": t5.player_id, "family": fam.astype(int), "pocket": (keep & ~fam).astype(int)}).to_csv(
    OUT / "t5_round4_flags.csv", index=False)
