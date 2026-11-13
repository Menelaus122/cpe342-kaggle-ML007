"""Round 25, Task 5: Bayes boundary of the six families under F3 (flag an account if P(anomaly | x) > F3 / 10 ~ 0.10).

    python de_r25_t5_gmm_boundary.py

Label-free (test features only). For each family (merged gap groups), in the subspace of its gap columns (robust-scaled):
  A. 2-component full-covariance Gaussian mixture (normal vs family), EM on all accounts, started from the current split.
     Posterior of the family component for every account NOT flagged in the current file.
  B. The same posterior without a Gaussian bulk (a Gaussian bulk has too-thin tails, so its tail accounts leak into the family
     component): per gap column, family density = Gaussian fitted to the members (generous tail towards the bulk), bulk density =
     kernel density of the unflagged accounts; posterior = pi_f f_f / (pi_f f_f + pi_n f_n); an account's posterior is the max over
     the family's columns.
  C. Independent check of the candidates (posterior > 0.10 in A or B): the four yes/no columns are not in any subspace, and the
     families have their own rates there (e.g. password_changed), so share = (rate in candidates - normal rate) / (family rate -
     normal rate) estimates how many candidates really belong to the family.
Expected F3 change of adding a candidate set S (sum of posteriors = expected members m, |S| - m expected normals), counting the
families as true, the pocket at its estimated 194 members of 339 and estimated size 201:
    F3 = 10 TP / (10 TP + 9 FN + FP);  before: FN includes the m members inside S;  after: they are TP.
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from sklearn.mixture import GaussianMixture

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t5 = task5.load()
n = len(t5)
X5, _ = task5.build_features(t5)
gc_table, gc_masks = task5.gap_clusters(t5, X5)
fams = task5.merge_families(gc_masks)
final = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
gap5 = gc_masks.any(axis=1).values
print(f"{n} accounts; flags {final.sum()} (families {gap5.sum()}, pocket {(final & ~gap5).sum()}); {len(fams)} families")
gt = gc_table.set_index("group")
col = lambda c: (X5[c] if c in X5 else t5[c]).values.astype(float)
TP0, FP0, FN0 = gap5.sum() + 194.0, (final & ~gap5).sum() - 194.0, 201.0 - 194.0
f3 = lambda tp, fp, fn: 10 * tp / (10 * tp + 9 * fn + fp)


def gain(p):
    m, k = float(np.sum(p)), len(p)
    return f3(TP0 + m, FP0 + k - m, FN0) - f3(TP0, FP0, FN0 + m)


post_a, post_b = np.zeros(n), np.zeros(n)
for fi, fam in enumerate(fams):
    member = gc_masks[fam].any(axis=1).values
    cols = list(dict.fromkeys(gt.loc[g, "column"] for g in fam))
    Z = np.column_stack([col(c) for c in cols])
    med = np.nanmedian(Z[~final], 0)
    mad = 1.4826 * np.nanmedian(np.abs(Z[~final] - med), 0) + 1e-9
    Z = (Z - med) / mad
    Z = np.where(np.isnan(Z), 0.0, Z)                      # missing -> bulk median
    # A. Gaussian mixture, EM from the current split
    gm = GaussianMixture(2, covariance_type="full", reg_covar=1e-4, max_iter=500, random_state=0,
                         weights_init=[1 - member.mean(), member.mean()],
                         means_init=[Z[~member].mean(0), Z[member].mean(0)],
                         precisions_init=[np.linalg.pinv(np.atleast_2d(np.cov(Z[~member].T)) + 1e-4 * np.eye(len(cols))),
                                          np.linalg.pinv(np.atleast_2d(np.cov(Z[member].T)) + 1e-4 * np.eye(len(cols)))]).fit(Z)
    k_f = int(np.argmin(np.linalg.norm(gm.means_ - Z[member].mean(0), axis=1)))
    pa = gm.predict_proba(Z)[:, k_f]
    # B. per column: Gaussian family vs kernel-density bulk
    pb = np.zeros(n)
    for g in fam:
        j = cols.index(gt.loc[g, "column"])
        mg = gc_masks[g].values
        z = Z[:, j]
        mu, sd = z[mg].mean(), z[mg].std() + 1e-9
        bulk = z[~final]
        kde = gaussian_kde(bulk[np.isfinite(bulk)])
        pi_f = mg.mean()
        ff, fn_ = pi_f * norm.pdf(z, mu, sd), (1 - pi_f) * kde(z)
        pb = np.maximum(pb, ff / (ff + fn_ + 1e-300))
    post_a, post_b = np.maximum(post_a, pa), np.maximum(post_b, pb)
    ua, ub = pa[~final], pb[~final]
    print(f"\nfamily {fi + 1}: {member.sum()} members, {len(cols)} columns {cols}")
    print(f"  A mixture : members with posterior > 0.5: {np.mean(pa[member] > 0.5):.3f} | component weight {gm.weights_[k_f]:.4f} (members {member.mean():.4f})"
          f" | unflagged > 0.10: {(ua > 0.10).sum()}, > 0.12: {(ua > 0.12).sum()}, > 0.5: {(ua > 0.5).sum()}")
    print(f"  B kde bulk: members with posterior > 0.5: {np.mean(pb[member] > 0.5):.3f} | unflagged > 0.10: {(ub > 0.10).sum()}, > 0.12: {(ub > 0.12).sum()}, "
          f"> 0.5: {(ub > 0.5).sum()}; largest unflagged posterior {ub.max():.2e}")

U = ~final
print("\n== all families together (max posterior over families), unflagged accounts")
base = {c: np.nanmean(t5.loc[U, c]) for c in task5.FLAG_COLS}
famr = {c: np.nanmean(t5.loc[gap5, c]) for c in task5.FLAG_COLS}
for nm, post in [("A mixture", post_a), ("B kde bulk", post_b)]:
    for thr in (0.10, 0.12, 0.5):
        S = U & (post > thr)
        if S.sum() == 0:
            print(f"  {nm:10s} posterior > {thr}: 0 accounts")
            continue
        p = post[S]
        chk = []
        for c in task5.FLAG_COLS:
            if abs(famr[c] - base[c]) > 0.02:
                r = np.nanmean(t5.loc[S, c])
                chk.append(f"{c[:12]} {r:.3f} (normal {base[c]:.3f}, families {famr[c]:.3f}, share {(r - base[c]) / (famr[c] - base[c]):+.2f})")
        print(f"  {nm:10s} posterior > {thr}: {S.sum():5d} accounts, model's expected members {p.sum():7.1f} (mean posterior {p.mean():.2f}); "
              f"expected F3 change if the model is right {gain(p):+.4f}")
        print(f"             C check: " + "; ".join(chk))
        for s in (0.0, 0.02, 0.05):
            print(f"             if the true share of members among them is {s:.2f}: F3 change {gain(np.full(int(S.sum()), s)):+.4f}")
np.savez("experiments/claude/outputs/de_r25_t5_post.npz", post_a=post_a, post_b=post_b)
