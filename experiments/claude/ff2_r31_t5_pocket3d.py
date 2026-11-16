"""Round 31, Task 5 item 3, follow-up: the 3-column density-ratio pocket (accuracy_std, kda_std, performance_stability levels).

    python ff2_r31_t5_pocket3d.py [first-release test.csv for the reference run]

ff_r31 found the cross-fitted 3-column density ratio (h = 0.40) ahead of the pipeline's 2-column min-rank, label-free on the
competition data (expected F3 0.983 vs 0.964 at a common group size) and on the first release's id block (F3 0.9888 vs 0.9830,
reference only). Checks before it can be proposed:
  1. which columns are shifted in the pocket: mean rank of every *_level column among the top accounts (and, on the first release,
     among the 259 true anomalies outside the gap families) -> is a 3-column pattern the right space?
  2. null validity: Spearman correlations of the three columns outside their joint upper corner (independence is the null)
     and, on the first release, among the true normals;
  3. bandwidth h in {0.30, 0.40, 0.50, 0.60} and 20 cross-fitting repeats: expected F3 (common group size) and k*;
  4. stability: 5 cross-fitting seeds (Jaccard of the flagged sets, k*), and 20 subsamples of 80% of the accounts without replacement
     (bootstrap duplicates would leak across the cross-fitting halves) -> k* / 0.8;
  5. overlap with the pipeline's 339 pocket flags.
Expected F3 counts the gap families as true; the group size G is the largest est_TP of the compared scores.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm  # noqa: E402

import ff_r31_t5_pocket_density as FF  # noqa: E402  (module-level runs are guarded below)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
B2 = 9.0
F3C = ["accuracy_std_level", "kda_std_level", "performance_stability_level"]
MAXK = 3000


def zscores(U):
    n = len(U)
    return norm.ppf((pd.DataFrame(U).rank(method="first").to_numpy() - 0.5) / n)


def ratio_curve(Z, h, seed, reps=20, n_perm=200):
    rng = np.random.default_rng(seed)
    R = FF.Ratio(Z, h, rng, reps=reps)
    s = R.score(Z)
    shuf = lambda r: np.column_stack([r.permutation(Z[:, j]) for j in range(Z.shape[1])])
    order, est = FF.curve(s, lambda r: R.score(shuf(r), [r.random(len(Z)) < 0.5 for _ in R.fits]), n_perm, rng)
    return order, est, s


def minrank_curve(U, seed, n_perm=200):
    rng = np.random.default_rng(seed)
    mr = lambda T: np.min(T, axis=1) + 1e-9 * T.sum(1)
    shuf = lambda r: np.column_stack([r.permutation(U[:, j]) for j in range(U.shape[1])])
    order, est = FF.curve(mr(U[:, :2]), lambda r: mr(shuf(r)[:, :2]), n_perm, rng)
    return order, est


def f3curve(est, nk, G):
    k = np.arange(1, MAXK + 1)
    return (1 + B2) * (nk + est[:MAXK]) / (B2 * (nk + G) + nk + k)


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    rest = np.where(~gap)[0]
    nk = int(gap.sum())
    lv = [c for c in X.columns if c.endswith("_level") and X[c].nunique() > 50]
    UL = X.loc[~gap, lv].rank(pct=True)
    U = UL[F3C].to_numpy()
    Z = zscores(U)
    log(f"\n===== {label}: gap families {nk}, outside {len(rest)}")
    o2, e2 = minrank_curve(U, 0)
    res = {h: ratio_curve(Z, h, 0) for h in (0.30, 0.40, 0.50, 0.60)}
    G = max([e2[:MAXK].max()] + [r[1][:MAXK].max() for r in res.values()])
    f2 = f3curve(e2, nk, G)
    k2 = int(np.argmax(f2)) + 1
    line = f"common G {G:.0f}; pipeline 2-column min-rank: k* {k2}, expected F3 {f2[k2 - 1]:.4f} (at 339: {f2[338]:.4f})"
    ref = {}
    if truth is not None:
        def ref_f3(order, k):
            fl = gap.copy()
            fl[rest[order[:k]]] = True
            tp = np.sum(fl & truth)
            return (1 + B2) * tp / ((1 + B2) * tp + B2 * np.sum(~fl & truth) + np.sum(fl & ~truth))
        line += f", reference F3 {ref_f3(o2, k2):.4f}"
    log(line)
    for h, (order, est, s) in res.items():
        f = f3curve(est, nk, G)
        ks = int(np.argmax(f)) + 1
        line = (f"  3-column ratio h={h:.2f} (20 repeats): group est {est[:MAXK].max():.0f}, k* {ks}, est. precision {est[ks - 1] / ks:.3f}, "
                f"expected F3 {f[ks - 1]:.4f}; at k = 250 / 300 / 339 / 400: " + " / ".join(f"{f[k - 1]:.4f}" for k in (250, 300, 339, 400)))
        if truth is not None:
            line += f" | reference F3 at k* {ref_f3(order, ks):.4f}, at 300 {ref_f3(order, 300):.4f}, at 339 {ref_f3(order, 339):.4f}"
            ref[h] = ref_f3(order, ks)
        log(line)
    order, est, s = res[0.40]
    f = f3curve(est, nk, G)
    ks = int(np.argmax(f)) + 1
    top = order[:ks]
    # 1. which columns are shifted
    sh = UL.iloc[top].mean().sort_values(ascending=False)
    log(f"1. mean rank among the top {ks} (h 0.40) of the *_level columns, highest: " + ", ".join(f"{c} {v:.2f}" for c, v in sh.head(6).items())
        + "; lowest: " + ", ".join(f"{c} {v:.2f}" for c, v in sh.tail(3).items()))
    if truth is not None:
        tr = truth[rest]
        sht = UL[tr].mean().sort_values(ascending=False)
        log("   first release, the true anomalies outside the gap families: " + ", ".join(f"{c} {v:.2f}" for c, v in sht.head(6).items())
            + "; lowest: " + ", ".join(f"{c} {v:.2f}" for c, v in sht.tail(3).items()))
    # 2. null validity
    out = ~((U[:, 0] > 0.8) & (U[:, 1] > 0.8)) & ~((U[:, 0] > 0.8) & (U[:, 2] > 0.8)) & ~((U[:, 1] > 0.8) & (U[:, 2] > 0.8))
    cm = pd.DataFrame(U[out], columns=["acc_std", "kda_std", "perf_stab"]).corr("spearman")
    sim = np.random.default_rng(1).random((len(U), 3))
    so = ~((sim[:, 0] > 0.8) & (sim[:, 1] > 0.8)) & ~((sim[:, 0] > 0.8) & (sim[:, 2] > 0.8)) & ~((sim[:, 1] > 0.8) & (sim[:, 2] > 0.8))
    cs = pd.DataFrame(sim[so]).corr("spearman")
    log(f"2. Spearman outside the pairwise upper corners: acc-kda {cm.iloc[0, 1]:+.3f}, acc-perf {cm.iloc[0, 2]:+.3f}, kda-perf {cm.iloc[1, 2]:+.3f} "
        f"(independent uniforms with the same cut: {cs.iloc[0, 1]:+.3f})")
    if truth is not None:
        cn = pd.DataFrame(U[~truth[rest]], columns=["acc_std", "kda_std", "perf_stab"]).corr("spearman")
        log(f"   first release, true normals: acc-kda {cn.iloc[0, 1]:+.3f}, acc-perf {cn.iloc[0, 2]:+.3f}, kda-perf {cn.iloc[1, 2]:+.3f}")
    # 4. stability
    sets, kss = [], []
    for seed in range(5):
        o_, e_, _ = ratio_curve(Z, 0.40, 10 + seed, n_perm=100)
        f_ = f3curve(e_, nk, G)
        k_ = int(np.argmax(f_)) + 1
        kss.append(k_)
        sets.append(set(rest[o_[:k_]]))
    jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    log(f"4. cross-fitting seeds: k* {kss}; Jaccard of the flagged sets {np.min(jac):.3f}-{np.max(jac):.3f}")
    kk_sub, kk_sub2 = [], []
    for b in range(20):
        rb = np.random.default_rng(200 + b)
        idx = rb.choice(len(rest), int(0.8 * len(rest)), replace=False)
        Ub = pd.DataFrame(U[idx]).rank(pct=True).to_numpy()
        _, eb, _ = ratio_curve(zscores(Ub), 0.40, 300 + b, reps=10, n_perm=50)
        _, eb2 = minrank_curve(Ub, 400 + b, n_perm=50)
        Gb = max(eb[:MAXK].max(), eb2[:MAXK].max())
        kk_sub.append((int(np.argmax(f3curve(eb, int(0.8 * nk), Gb))) + 1) / 0.8)
        kk_sub2.append((int(np.argmax(f3curve(eb2, int(0.8 * nk), Gb))) + 1) / 0.8)
    log(f"   80% subsamples (k* / 0.8): 3-column ratio mean {np.mean(kk_sub):.0f}, SD {np.std(kk_sub):.0f}, 10-90% "
        f"[{np.quantile(kk_sub, 0.1):.0f}, {np.quantile(kk_sub, 0.9):.0f}]; 2-column min-rank mean {np.mean(kk_sub2):.0f}, SD {np.std(kk_sub2):.0f}")
    # 5. overlap with the pipeline's pocket
    pocket = set(rest[o2[:339]])
    new = set(rest[top])
    line = f"5. overlap with the pipeline's 339: kept {len(pocket & new)}, dropped {len(pocket - new)}, added {len(new - pocket)}"
    if truth is not None:
        tset = set(np.where(truth)[0])
        line += (f" | first release: of the kept {len(pocket & new & tset)} true, dropped {len((pocket - new) & tset)} true of {len(pocket - new)}, "
                 f"added {len((new - pocket) & tset)} true of {len(new - pocket)}")
    log(line)


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1:
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
