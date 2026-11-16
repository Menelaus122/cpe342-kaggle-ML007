"""Round 31, Task 5 item 3: the shape and the cut-off of the dependence pocket (label-free; Task 5 has only test.csv, by design).

    python ff_r31_t5_pocket_density.py [path to the first-release test.csv for the reference run]

The pipeline flags the accounts outside the gap families by min(rank of accuracy_std_level, rank of kda_std_level), a square
(L-infinity) corner in rank space, cut at k* = 339 where the expected F3 under a column-shuffle null peaks. By the Neyman-Pearson
lemma the most powerful region for "real joint density p(x1, x2)" against "independent null p0(x1) p0(x2)" is a level set of the
density ratio r = p / (p0 p0); in rank space p0 p0 is uniform, so r is the copula density, and 1 - 1/r is the local posterior
P(anomaly | x) (Efron's local false discovery rate, lfdr = pi0 f0 / f, with every non-independent account counted as anomalous).
Scores compared on the accounts outside the gap families (ranks among them):
  min-rank (pipeline), Fisher (-log(1-u1) - log(1-u2)), Stouffer (probit sum), and the density ratio estimated as a smoothed
  histogram ratio in normal-score space (bandwidth h in {0.15, 0.25, 0.40} SD), CROSS-FITTED: two random halves, each account scored
  by the ratio fitted on the other half (5 repeats averaged), so the region is not tuned to the noise of the accounts it scores;
  the same with performance_stability_level as a third column.
For every score: est_TP(k) = k - mean count of null scores above the k-th score (column shuffles, 200 draws; the density ratio is
evaluated on the shuffled points with the other half's fit), expected F3 with the gap families counted as true and ONE common group
size (the largest est_TP of any score, so a weak score is not rewarded for under-estimating the group), its k*, and the posterior
rule (flag where 1 - 1/r > F3 / 10). Null uncertainty: 30 bootstrap resamples of the accounts, curves recomputed (30 null draws); for the density ratio this
bootstrap is biased (a duplicated account sits in both cross-fitting halves), see ff2_r31 for 80% subsamples instead.
Diagnostics of the null: corner counts of a 10 x 10 rank grid (if normal accounts were dependent the low-low corner would be
over-full too), and performance_stability's rank among the top-ranked accounts.
Reference run (optional argument): the same on the first release (2026-11-03) of test.csv, kept in git history (commit 99ed4b9),
whose anomalies were the id block P023819-P025888 (2,070 accounts; FINDINGS "Task 5: the anomalies are the top 8% of ids"). There
the F3 of gap families + pocket against the id block is printed, for reference only: the cut-off rule is never tuned on it.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.stats import norm  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
B2 = 9.0
F2 = ["accuracy_std_level", "kda_std_level"]
F3C = F2 + ["performance_stability_level"]


def log(m):
    print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)


def fbeta(flags, truth):
    tp = np.sum(flags & truth)
    return (1 + B2) * tp / ((1 + B2) * tp + B2 * np.sum(~flags & truth) + np.sum(flags & ~truth))


class Ratio:
    """Cross-fitted smoothed-histogram density ratio in normal-score space (d = 2 or 3)."""

    def __init__(self, Z, h, rng, reps=5):
        self.d, self.h = Z.shape[1], h
        self.w = 0.05 if self.d == 2 else 0.1
        self.edges = np.arange(-4.2, 4.2 + 1e-9, self.w)
        p = np.diff(norm.cdf(np.r_[-np.inf, self.edges[1:-1], np.inf]))
        self.p = p
        self.fits = []
        for _ in range(reps):
            half = rng.random(len(Z)) < 0.5
            self.fits.append((half, self._fit(Z[half]), self._fit(Z[~half])))

    def _fit(self, Z):
        H, _ = np.histogramdd(np.clip(Z, -4.199, 4.199), bins=[self.edges] * self.d)
        E = len(Z) * (np.einsum("i,j->ij", self.p, self.p) if self.d == 2 else np.einsum("i,j,k->ijk", self.p, self.p, self.p))
        s = self.h / self.w
        eps = 0.5 / ((2 * np.pi) ** (self.d / 2) * s ** self.d)
        return np.log((gaussian_filter(H, s, mode="constant") + eps) / (gaussian_filter(E, s, mode="constant") + eps))

    def _look(self, L, Z):
        idx = np.clip(((np.clip(Z, -4.199, 4.199) + 4.2) / self.w).astype(int), 0, len(self.edges) - 2)
        return L[tuple(idx.T)]

    def score(self, Z, halves=None):
        """log-ratio of each row from the fit on the half it is NOT in (for null points a random half each repeat)."""
        out = np.zeros(len(Z))
        for k, (half, LA, LB) in enumerate(self.fits):
            hh = half if halves is None else halves[k]
            out += np.where(hh, self._look(LB, Z), self._look(LA, Z))
        return out / len(self.fits)


def curve(S, null_fn, n_perm, rng):
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(len(Ss))
    for _ in range(n_perm):
        Sp = np.sort(null_fn(rng))
        cnt += len(Sp) - np.searchsorted(Sp, Ss, side="left")
    cnt /= n_perm
    k = np.arange(1, len(Ss) + 1)
    return order, np.maximum.accumulate(k - cnt)


def scores_for(U, rng, with3=True):
    """U: rank percentiles (rows x 3 columns, NaN allowed). Returns {name: (score, null_fn)}."""
    n = len(U)
    Ui = np.where(np.isnan(U), 0.5, U)
    uz = (pd.DataFrame(Ui).rank(method="first").to_numpy() - 0.5) / n          # mid-ranks in (0, 1) for the probit scores
    Z = norm.ppf(uz)
    shuf = lambda A, r: np.column_stack([r.permutation(A[:, j]) for j in range(A.shape[1])])
    out = {}

    def minrank(T):
        with np.errstate(all="ignore"):
            s = np.nanmin(T, axis=1) + 1e-9 * np.nansum(T, axis=1)
        return np.where(np.isnan(s), -np.inf, s)

    out["min-rank (pipeline)"] = (minrank(U[:, :2]), lambda r: minrank(shuf(U[:, :2], r)))
    fis = lambda T: -np.log(1 - T).sum(1)
    out["Fisher"] = (fis(uz[:, :2]), lambda r: fis(shuf(uz[:, :2], r)))
    out["Stouffer"] = (Z[:, :2].sum(1), lambda r: shuf(Z[:, :2], r).sum(1))
    for h in (0.15, 0.25, 0.40):
        R = Ratio(Z[:, :2], h, rng)
        out[f"density ratio h={h:.2f}"] = (R.score(Z[:, :2]),
                                          lambda r, R=R: R.score(shuf(Z[:, :2], r), [r.random(n) < 0.5 for _ in R.fits]))
    if with3:
        out["min-rank, 3 columns"] = (minrank(U), lambda r: minrank(shuf(U, r)))
        out["Fisher, 3 columns"] = (fis(uz), lambda r: fis(shuf(uz, r)))
        R3 = Ratio(Z, 0.40, rng)
        out["density ratio 3 columns h=0.40"] = (R3.score(Z), lambda r, R=R3: R.score(shuf(Z, r), [r.random(n) < 0.5 for _ in R.fits]))
    return out


def analyse(df, label, truth=None, n_perm=200, n_boot=30, max_k=3000):
    rng = np.random.default_rng(0)
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    rest = np.where(~gap)[0]
    nk = int(gap.sum())
    U = X.loc[~gap, F3C].rank(pct=True).to_numpy()
    log(f"\n===== {label}: {len(df)} accounts, gap families {nk}, outside them {len(rest)}; NaN in "
        + ", ".join(f"{c} {int(np.isnan(U[:, j]).sum())}" for j, c in enumerate(F3C)))
    # null diagnostics: 10 x 10 grid of the two pocket columns
    q = np.minimum((np.nan_to_num(U[:, :2], nan=0.5) * 10).astype(int), 9)
    cnt = np.bincount(q[:, 0] * 10 + q[:, 1], minlength=100).reshape(10, 10)
    e = len(rest) / 100
    log("10 x 10 rank grid (accuracy_std x kda_std): corner z = (obs - exp) / sqrt(exp): "
        f"high-high {(cnt[9, 9] - e) / np.sqrt(e):+.1f} ({cnt[9, 9]}), low-low {(cnt[0, 0] - e) / np.sqrt(e):+.1f}, high-low "
        f"{(cnt[9, 0] - e) / np.sqrt(e):+.1f}, low-high {(cnt[0, 9] - e) / np.sqrt(e):+.1f}; next to the corner (8,9)/(9,8) "
        f"{(cnt[8, 9] - e) / np.sqrt(e):+.1f}/{(cnt[9, 8] - e) / np.sqrt(e):+.1f}; Spearman of the two outside the top 20% corner "
        f"{pd.DataFrame(U[(U[:, 0] < 0.8) | (U[:, 1] < 0.8), :2]).corr('spearman').iloc[0, 1]:+.3f}")
    if truth is not None:
        tr = truth[rest]
        nm = ~tr
        sp = pd.DataFrame(U[nm, :2]).corr("spearman").iloc[0, 1]
        log(f"reference: anomalies outside the gap families {int(tr.sum())} (of {int(truth.sum())}; gap families vs truth: "
            f"precision {np.mean(truth[gap]):.3f}, recall {np.sum(truth & gap) / truth.sum():.3f}); among the TRUE normals outside the "
            f"families Spearman(accuracy_std, kda_std) {sp:+.3f}; mean rank of performance_stability among those anomalies "
            f"{np.nanmean(U[tr, 2]):.3f}")
    S = scores_for(U, rng)
    curves = {}
    for name, (s, nf) in S.items():
        curves[name] = curve(s, nf, n_perm, rng)
        log(f"  null curve done: {name}")
    G = max(est[:max_k].max() for _, est in curves.values())
    k = np.arange(1, max_k + 1)
    rows = []
    for name, (order, est) in curves.items():
        f = (1 + B2) * (nk + est[:max_k]) / (B2 * (nk + G) + nk + k)
        ks = int(np.argmax(f)) + 1
        row = dict(score=name, group_hat=float(est[:max_k].max()), k_star=ks, est_tp=float(est[ks - 1]), prec=float(est[ks - 1] / ks),
                   exp_F3=float(f[ks - 1]), F3_k339=float(f[338]))
        if name.startswith("density ratio"):
            post = 1 - np.exp(-S[name][0])
            kp = int(np.sum(post > row["exp_F3"] / 10))
            row.update(k_post=kp, exp_F3_post=float(f[min(max(kp, 1), max_k) - 1]))
        if truth is not None:
            for tag, kk in (("ref_F3_kstar", ks), ("ref_F3_k339", 339)):
                fl = gap.copy()
                fl[rest[order[:kk]]] = True
                row[tag] = fbeta(fl, truth)
            best = (0, 0)
            tr_sorted = truth[rest[order[:max_k]]]
            tp_c = np.cumsum(tr_sorted)
            tp_g = np.sum(truth & gap)
            fr = (1 + B2) * (tp_g + tp_c) / ((1 + B2) * (tp_g + tp_c) + B2 * (truth.sum() - tp_g - tp_c) + (nk - tp_g) + (k - tp_c))
            best = (int(np.argmax(fr)) + 1, float(fr.max()))
            row.update(ref_prec_kstar=float(tr_sorted[:ks].mean()), ref_oracle_k=best[0], ref_oracle_F3=best[1],
                       ref_true_in_top_kstar=int(tp_c[ks - 1]))
            if "k_post" in row:
                fl = gap.copy()
                fl[rest[order[:max(row["k_post"], 1)]]] = True
                row["ref_F3_post"] = fbeta(fl, truth)
        rows.append(row)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    log(f"common group size G = {G:.0f}; expected F3 with the gap families as true:\n{tab.round(4).to_string(index=False)}")
    # bootstrap: uncertainty of k* and of the expected F3 for the pipeline score and the best density ratio
    best_ratio = max([r for r in rows if r["score"].startswith("density ratio") and "3 col" not in r["score"]], key=lambda r: r["exp_F3"])["score"]
    for name in ("min-rank (pipeline)", best_ratio):
        kstars, fcurves = [], []
        for b in range(n_boot):
            rb = np.random.default_rng(100 + b)
            idx = rb.integers(0, len(rest), len(rest))
            Ub = pd.DataFrame(U[idx]).rank(pct=True).to_numpy()
            Sb = scores_for(Ub, rb, with3=False)[name]
            _, est = curve(Sb[0], Sb[1], 30, rb)
            Gb = est[:max_k].max()
            f = (1 + B2) * (nk + est[:max_k]) / (B2 * (nk + Gb) + nk + k)
            kstars.append(int(np.argmax(f)) + 1)
            fcurves.append(f)
        mf = np.mean(fcurves, 0)
        log(f"bootstrap ({n_boot}) {name}: k* mean {np.mean(kstars):.0f}, SD {np.std(kstars):.0f}, 10-90% [{np.quantile(kstars, 0.1):.0f}, "
            f"{np.quantile(kstars, 0.9):.0f}]; k maximising the mean expected F3 {int(np.argmax(mf)) + 1} (F3 {mf.max():.4f}); mean F3 at "
            + ", ".join(f"k={kk}: {mf[kk - 1]:.4f}" for kk in (200, 250, 300, 339, 400, 500)))
    return tab


if __name__ == "__main__":
    df_new = task5.load()
    analyse(df_new, "re-released test.csv (2026-11-12, the competition data)")
    if len(sys.argv) > 1:
        df_old = pd.read_csv(sys.argv[1])
        truth = df_old.player_id.str[1:].astype(int).values >= 23819
        analyse(df_old, "first release (2026-11-03), reference only", truth=truth)
    log("done")
