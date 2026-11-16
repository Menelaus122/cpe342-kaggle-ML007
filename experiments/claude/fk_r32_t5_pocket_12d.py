"""Round 32, Task 5 item 4: 12-D classifier density ratio for the pocket (3 behaviours x 4 periods, block-shuffled null) (label-free;
Task 5 has only test.csv).

    python fk_r32_t5_pocket_12d.py [first-release test.csv for the reference run]
    env: REPS (halvings, default 20), NP (null draws, 200), M (shuffles per real row in training, 3), SEEDS ("42,1,2"), ONLY_CURRENT

Item 1 found that the pocket's true members are high in every period (3.8 of 4 periods in the top 20% against 2.7 for normal accounts
at the same level, and 2.8 for the false members on the first release), but no single per-account aggregate (median, mean of normal
scores, second-lowest period) beat the period mean in the 3-D density ratio. Here the ratio is estimated on the whole per-period
profile instead:
  data: the accounts outside the gap families; for each of the pocket's three behaviours b a block of features computed from b's four
  periods only: the normal score of each period's rank, of the period mean (the pipeline's _level), and the min / max / SD of the four
  period scores (functions of the block, so they travel with it when the block is shuffled), plus across-behaviour summaries of
  each row (see rows());
  null: block shuffle = each behaviour's block permuted over the accounts independently (keeps every within-behaviour dependence,
  persistence over periods included; removes only the dependence across behaviours, which is what the pocket is). It is the same
  hypothesis as the 3-D null (independent _level columns), so the two est_TP curves estimate the same group;
  ratio: a gradient-boosted classifier (sklearn HistGradientBoosting, 300 small trees, leaves of at least 200 rows) of real rows against M block shuffles of its training half; log-odds = log ratio + constant;
  cross-fitting: each of REPS random halvings fits one classifier per half; a real row is scored by the classifier of the half it is
  not in, scores averaged over halvings; null: NP block shuffles of all accounts fixed in advance, every null row scored by a random
  half's classifier in every halving (as in task5.density_ratio_curve); est_TP(k) = k - mean null count above the k-th score.
Compared at ONE common group size (the largest estimate) with the pipeline's pocket (3-D smoothed-histogram ratio of the three _level
columns, 200 halvings / 1000 null draws) and the min-rank score: expected F3, cut-off, estimated precision, overlap of the flagged
sets, and the 12-D cut over seeds.
Reference run (optional argument): the first release with truth = id block P023819-P025888, reference only (never used to choose).
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
FEATS = ["accuracy_std_level", "kda_std_level"]
REPS = int(os.environ.get("REPS", 20))
NP = int(os.environ.get("NP", 200))
M = int(os.environ.get("M", 3))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,1,2").split(",")]
B2, MAXK = 9.0, 3000
# Setting picked from four in a 6-halving benchmark by the label-free estimate (early stopping; 100 iterations at 0.05; 150 iterations
# with 4 leaves; this one): expected F3 at a group of 260 was 0.9828 / 0.9864 / 0.9864 / 0.9876. No early stopping: its validation part
# only breaks the exact 1 : M balance of the block values by chance.
CLF = dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False)


def nscore(v):
    """Normal score of the rank among the non-missing rows; missing stays NaN (the classifier routes it natively)."""
    r = pd.Series(v).rank(method="first").to_numpy()
    return norm.ppf((r - 0.5) / np.sum(~np.isnan(v)))


def block_features(P):
    """Features of one behaviour from its four periods only: period scores, score of the period mean, min / max / SD of the
    observed period scores (about 18% of the accounts miss at least one period of a behaviour)."""
    Zp = np.column_stack([nscore(P[:, j]) for j in range(P.shape[1])])
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        lev = nscore(np.nanmean(P, 1))
        return np.column_stack([Zp, lev, np.nanmin(Zp, 1), np.nanmax(Zp, 1), np.nanstd(Zp, 1)]).astype(np.float32)


def rows(blocks):
    """One row per account from its (possibly shuffled) blocks: the block features plus across-behaviour summaries. A split on a
    single block feature has exactly zero gain (every block value sits in 1 real and M shuffled training rows), so greedy trees
    need the across-behaviour columns to start: min / mean / max of the level scores, min and mean of the per-behaviour lowest
    period scores, and the number of periods in the top 20% over all behaviours."""
    Xb = np.hstack(blocks)
    w = blocks[0].shape[1]
    L = Xb[:, 4::w]
    Mn = Xb[:, 5::w]
    P = np.hstack([b[:, :4] for b in blocks])
    with np.errstate(all="ignore"):
        cross = np.column_stack([L.min(1), L.mean(1), L.max(1), Mn.min(1), Mn.mean(1), np.nansum(P > 0.8416, axis=1)])
    return np.hstack([Xb, cross.astype(np.float32)])


def classifier_curve(Fb, seed):
    """Cross-fitted classifier log-ratio of the real rows and the block-shuffle null est_TP curve."""
    rng = np.random.default_rng(seed)
    n = len(Fb[0])
    perms = np.stack([np.stack([rng.permutation(n) for _ in Fb]) for _ in range(NP)]).astype(np.int32)   # NP x blocks x n
    real = rows(Fb)
    S = np.zeros(n)
    SN = np.zeros((NP, n), np.float32)
    iters = []
    for _ in range(REPS):
        half = rng.random(n) < 0.5
        models = []
        for m in (half, ~half):
            idx = np.where(m)[0]
            Xs = [real[idx]] + [rows([f[idx[rng.permutation(len(idx))]] for f in Fb]) for _ in range(M)]
            y = np.r_[np.ones(len(idx)), np.zeros(M * len(idx))]
            clf = HistGradientBoostingClassifier(**CLF, random_state=int(rng.integers(2**31 - 1)))
            clf.fit(np.vstack(Xs), y)
            models.append(clf)
            iters.append(clf.n_iter_)
        mA, mB = models
        S[half] += mB.decision_function(real[half])
        S[~half] += mA.decision_function(real[~half])
        for j in range(NP):
            Xn = rows([f[perms[j, b]] for b, f in enumerate(Fb)])
            pick = rng.random(n) < 0.5
            SN[j, pick] += mB.decision_function(Xn[pick])
            SN[j, ~pick] += mA.decision_function(Xn[~pick])
    S /= REPS
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(n)
    for j in range(NP):
        Sp = np.sort(SN[j].astype(float) / REPS)
        cnt += n - np.searchsorted(Sp, Ss, side="left")
    cnt /= NP
    est = np.maximum.accumulate(np.arange(1, n + 1) - cnt)
    return order, est[:MAXK], S, iters


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    rest = np.where(~gap)[0]
    nk = int(gap.sum())
    pocket0, pinfo0 = task5.pocket_detector(X, gap, FEATS)
    extra, _, _ = task5.pocket_extra_column(X, gap, pocket0, FEATS)
    cols3 = FEATS + ([extra] if extra else [])
    beh = [c[:-len("_level")] for c in cols3]
    log(f"\n===== {label}: gap families {nk}, outside {len(rest)}; pocket columns {cols3}; REPS {REPS}, NP {NP}, M {M}")
    curves = {"min-rank (2-D)": (pinfo0["order"], pinfo0["curve"].est_tp.to_numpy())}
    o3, e3, _ = task5.pocket_density_ratio(X, gap, cols3)
    curves["density ratio 3-D (pipeline)"] = (o3, e3)
    log("   3-D density ratio done")
    sub = df.iloc[rest].reset_index(drop=True)
    Fb = [block_features(sub[[f"{b}_{p}" for p in range(1, 5)]].to_numpy(float)) for b in beh]
    for s in SEEDS:
        order, est, S, iters = classifier_curve(Fb, s)
        curves[f"classifier 12-D, seed {s}"] = (rest[order], est)
        log(f"   12-D classifier seed {s}: boosting iterations per fit {min(iters)}-{max(iters)} (median {int(np.median(iters))}); "
            f"estimated group {est.max():.0f}")
    G = max(float(e.max()) for _, e in curves.values())
    k = np.arange(1, MAXK + 1)
    rows, flags = [], {}
    for name, (order, est) in curves.items():
        f = (1 + B2) * (nk + est) / (B2 * (nk + G) + nk + k)
        ks = int(np.argmax(f)) + 1
        fl = np.zeros(len(df), bool)
        fl[order[:ks]] = True
        flags[name] = fl
        row = dict(score=name, group=float(est.max()), k=ks, est_tp=float(est[ks - 1]), est_prec=float(est[ks - 1] / ks),
                   exp_F3=float(f[ks - 1]))
        if truth is not None:
            a = gap | fl
            tp = np.sum(a & truth)
            row["ref_F3"] = (1 + B2) * tp / ((1 + B2) * tp + B2 * np.sum(~a & truth) + np.sum(a & ~truth))
            row["ref_prec_pocket"] = float(truth[fl].mean())
        rows.append(row)
    pd.set_option("display.width", 250)
    log(f"common group size {G:.0f}:\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    names = list(flags)
    base = "density ratio 3-D (pipeline)"
    for nm in names:
        if nm.startswith("classifier"):
            a, b = flags[nm], flags[base]
            line = (f"   {nm} vs 3-D pipeline: both {np.sum(a & b)}, only 12-D {np.sum(a & ~b)}, only 3-D {np.sum(~a & b)}, "
                    f"Jaccard {np.sum(a & b) / np.sum(a | b):.3f}")
            if truth is not None:
                line += (f" | reference: true among only-12-D {int(truth[a & ~b].sum())}/{np.sum(a & ~b)}, among only-3-D "
                         f"{int(truth[~a & b].sum())}/{np.sum(~a & b)}")
            log(line)
    cl = [nm for nm in names if nm.startswith("classifier")]
    if len(cl) > 1:
        jac = [np.sum(flags[x] & flags[y]) / np.sum(flags[x] | flags[y]) for i, x in enumerate(cl) for y in cl[i + 1:]]
        log(f"   12-D seed stability: Jaccard {', '.join(f'{j:.3f}' for j in jac)}")


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1 and not os.environ.get("ONLY_CURRENT"):
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
