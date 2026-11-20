"""Round 36, Task 5: rank fusion of the period-profile classifier ratio with a multi-scale k-NN distance ratio (label-free; Task 5
has only test.csv).

    python gb_r36_t5_knn_fusion.py [first-release test.csv for the reference run]
    env: REPS (halvings of the classifier, default 20), NP (null draws, 200), NPK (null draws scored by the k-NN ratio, 100),
         SEEDS ("42,1"), ONLY_CURRENT

v25's pocket score is the cross-fitted classifier estimate of the density ratio "real accounts" / "behaviour blocks shuffled
independently" on the 3 x 4 period profile (task5.profile_ratio_curve). A second estimator of the SAME ratio with a different
error structure, distance-based instead of tree-based:
    s_k(x) = log r_k^null(x) - log r_k^real(x),
r_k^real = distance to the k-th nearest real account (itself excluded), r_k^null = distance to the k-th nearest row of a
block-shuffled copy (mean of the log over 2 independent copies), in the 12 per-period normal scores (missing -> 0); k = 15 and 50.
In d dimensions the density ratio is (r^null / r^real)^d, so s_k is monotone in it.
Fusion: every score is turned into its percentile among the real accounts' scores (a null row gets the percentile it would have
among the real ones), and the percentiles are averaged. The est_TP curve of a fused score comes from the same block-shuffle null
draws scored by both estimators (the first NPK of the classifier's NP draws), so the cut-off rule is unchanged:
    est_TP(k) = k - mean number of null rows above the k-th real score,   F3 expected at one common group size.
Scores compared: classifier alone (on the NPK draws); k-NN alone (mean of the k = 15 and k = 50 percentiles); classifier + k-NN
at 1/2 : 1/2 and at 2/3 : 1/3. Per seed: cut-off, estimated precision, expected F3, overlap with the classifier's flags; over
seeds: Jaccard of the flagged sets (does the fusion reduce the seed variance?).
Decision (label-free): the fusion replaces the classifier only if its expected F3 at the common group size is higher in every seed
and its seed Jaccard is not lower. The first-release run (truth = id block P023819-P025888) is a reference check, never used to
choose; the user's bar there is F3 above 0.9911.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
REPS = int(os.environ.get("REPS", 20))
NP = int(os.environ.get("NP", 200))
NPK = int(os.environ.get("NPK", 100))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,1").split(",")]
NT = int(os.environ["OMP_NUM_THREADS"])
B2, MAXK, KS = 9.0, 3000, (15, 50)
COLS = ["accuracy_std_level", "kda_std_level", "performance_stability_level"]


def classifier_scores(blocks, seed):
    """task5.profile_ratio_curve with the null scores kept: (S of the real rows, SN of the NP null draws, the permutations)."""
    rng = np.random.default_rng(seed)
    n = len(blocks[0])
    perms = np.stack([np.stack([rng.permutation(n) for _ in blocks]) for _ in range(NP)]).astype(np.int32)
    real = task5._profile_rows(blocks)
    S = np.zeros(n)
    SN = np.zeros((NP, n), np.float32)
    for _ in range(REPS):
        half = rng.random(n) < 0.5
        models = []
        for hm in (half, ~half):
            idx = np.where(hm)[0]
            Xs = [real[idx]] + [task5._profile_rows([f[idx[rng.permutation(len(idx))]] for f in blocks]) for _ in range(3)]
            yy = np.r_[np.ones(len(idx)), np.zeros(3 * len(idx))]
            models.append(HistGradientBoostingClassifier(**task5.PROFILE_CLF, random_state=int(rng.integers(2**31 - 1))).fit(np.vstack(Xs), yy))
        mA, mB = models
        S[half] += mB.decision_function(real[half])
        S[~half] += mA.decision_function(real[~half])
        for j in range(NP):
            Xn = task5._profile_rows([f[perms[j, b]] for b, f in enumerate(blocks)])
            pick = rng.random(n) < 0.5
            SN[j, pick] += mB.decision_function(Xn[pick])
            SN[j, ~pick] += mA.decision_function(Xn[~pick])
    return S / REPS, SN / REPS, perms


def knn_scores(blocks, perms, seed):
    """k-NN log distance ratio of the real rows and of the first NPK null draws, for each k in KS: (dict k -> S, dict k -> SN)."""
    rng = np.random.default_rng(10_000 + seed)
    Z = [np.nan_to_num(b[:, :4]).astype(np.float32) for b in blocks]
    R = np.hstack(Z)
    n = len(R)
    shuffled = lambda pp: np.hstack([z[pp[b]] for b, z in enumerate(Z)])
    refs = [shuffled(np.stack([rng.permutation(n) for _ in Z])) for _ in range(2)]            # independent null copies
    nn_real = NearestNeighbors(n_neighbors=max(KS) + 1, algorithm="brute", n_jobs=NT).fit(R)
    nn_refs = [NearestNeighbors(n_neighbors=max(KS), algorithm="brute", n_jobs=NT).fit(r) for r in refs]

    def score(Q, is_real):
        d_real = nn_real.kneighbors(Q)[0]
        if is_real:
            d_real = d_real[:, 1:]                                                                # drop the row itself
        d_null = [m.kneighbors(Q)[0] for m in nn_refs]
        return {k: np.mean([np.log(d[:, k - 1] + 1e-9) for d in d_null], 0) - np.log(d_real[:, k - 1] + 1e-9) for k in KS}

    S = score(R, True)
    SN = {k: np.zeros((NPK, n), np.float32) for k in KS}
    for j in range(NPK):
        sj = score(shuffled(perms[j]), False)
        for k in KS:
            SN[k][j] = sj[k]
    return S, SN


def pct(S, SN):
    """Percentile of every score among the real rows' scores (real rows: own rank; null rows: where they would fall)."""
    srt = np.sort(S)
    n = len(S)
    return np.searchsorted(srt, S, side="right") / n, np.searchsorted(srt, SN, side="right") / n


def curve(U, UN):
    """(row order, est_TP curve) of a fused score U (real) with null scores UN (draws x rows)."""
    n = len(U)
    order = np.argsort(-U)
    Us = U[order]
    cnt = np.zeros(n)
    for j in range(len(UN)):
        cnt += n - np.searchsorted(np.sort(UN[j]), Us, side="left")
    est = np.maximum.accumulate(np.arange(1, n + 1) - cnt / len(UN))
    return order, est[:MAXK]


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    rest = np.where(~gap)[0]
    nk = int(gap.sum())
    sub = df.iloc[rest]
    blocks = [task5.profile_block(sub[[f"{c[:-6]}_{i}" for i in range(1, 5)]].to_numpy(float)) for c in COLS]
    log(f"\n===== {label}: gap families {nk}, outside {len(rest)}; REPS {REPS}, NP {NP}, NPK {NPK}, k {KS}")
    flags_by = {}
    for seed in SEEDS:
        S, SN, perms = classifier_scores(blocks, seed)
        log(f"   seed {seed}: classifier scores done")
        KSc, KSN = knn_scores(blocks, perms, seed)
        log(f"   seed {seed}: k-NN scores done; Spearman of the classifier score with k-NN 15 / 50 on the real rows: "
            + " / ".join(f"{pd.Series(S).corr(pd.Series(KSc[k]), method='spearman'):.3f}" for k in KS)
            + f"; among the classifier's top 300: " + " / ".join(
                f"{pd.Series(S[np.argsort(-S)[:300]]).corr(pd.Series(KSc[k][np.argsort(-S)[:300]]), method='spearman'):.3f}" for k in KS))
        pc, pcn = pct(S, SN[:NPK])
        pk = [pct(KSc[k], KSN[k]) for k in KS]
        pkr, pkn = np.mean([p[0] for p in pk], 0), np.mean([p[1] for p in pk], 0)
        cands = {"classifier 12-D alone": (pc, pcn), "k-NN ratio alone (k 15 + 50)": (pkr, pkn),
                 "classifier + k-NN, 1/2 : 1/2": (0.5 * pc + 0.5 * pkr, 0.5 * pcn + 0.5 * pkn),
                 "classifier + k-NN, 2/3 : 1/3": (2 / 3 * pc + 1 / 3 * pkr, 2 / 3 * pcn + 1 / 3 * pkn)}
        curves = {nm: curve(U, UN) for nm, (U, UN) in cands.items()}
        G = max(float(e.max()) for _, e in curves.values())
        k = np.arange(1, MAXK + 1)
        rows = []
        for nm, (order, est) in curves.items():
            f = (1 + B2) * (nk + est) / (B2 * (nk + G) + nk + k)
            ks = int(np.argmax(f)) + 1
            fl = np.zeros(len(df), bool)
            fl[rest[order[:ks]]] = True
            flags_by.setdefault(nm, {})[seed] = fl
            row = dict(score=nm, group=float(est.max()), k=ks, est_prec=float(est[ks - 1] / ks), exp_F3=float(f[ks - 1]))
            if truth is not None:
                a = gap | fl
                tp = np.sum(a & truth)
                row["ref_F3"] = (1 + B2) * tp / ((1 + B2) * tp + B2 * np.sum(~a & truth) + np.sum(a & ~truth))
                row["ref_prec_pocket"] = float(truth[fl].mean())
            rows.append(row)
        pd.set_option("display.width", 250)
        log(f"   seed {seed}, common group size {G:.0f}:\n" + pd.DataFrame(rows).round(4).to_string(index=False))
        base = flags_by["classifier 12-D alone"][seed]
        for nm in list(cands)[1:]:
            a = flags_by[nm][seed]
            line = f"   {nm} vs the classifier: both {np.sum(a & base)}, only this {np.sum(a & ~base)}, only the classifier {np.sum(~a & base)}"
            if truth is not None:
                line += f" | reference: true among only-this {int(truth[a & ~base].sum())}/{np.sum(a & ~base)}, among only-classifier {int(truth[~a & base].sum())}/{np.sum(~a & base)}"
            log(line)
    if len(SEEDS) > 1:
        for nm, fs in flags_by.items():
            v = list(fs.values())
            jac = [np.sum(v[i] & v[j]) / np.sum(v[i] | v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
            log(f"   seed stability, {nm}: Jaccard {', '.join(f'{j:.3f}' for j in jac)}")


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1 and not os.environ.get("ONLY_CURRENT"):
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
