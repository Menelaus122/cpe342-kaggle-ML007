"""Round 32, Task 5 item 1: every column and every period for the 7th group (the pocket) (label-free; Task 5 has only test.csv).

    python fi_r32_t5_pocket_columns.py [first-release test.csv for the reference run]

v24 tested only the 20 other all-period *_level columns and found performance_stability_level (z 16.4). Here:
  A. the same regression-adjusted rank-shift test (pocket mean rank against what its three columns predict, least squares on the
     ranks fitted on the other accounts outside the families) on EVERY column: all derived features of build_features (_cv, _burst,
     _maxjump, _dev4, _selfz4, _level, geo, statics, flags, counts) and every raw column (each period separately), two-sided
     Bonferroni at alpha 0.001 over all candidates; reported separately for the pocket's own behaviours (accuracy_std, kda_std,
     performance_stability: mechanically tied to the three columns, so a linear adjustment may not remove all of it) and for the
     other behaviours (a candidate 4th dimension);
  B. persistence over the four periods: is the group high in all periods, while the normal accounts that reach a high level do so
     through one period? Per-period profile of the pocket (number of periods in the top 20%, lowest period rank) against normal
     accounts at the same level (the accounts outside the pocket whose three level ranks are all above 0.8), and on the first
     release (reference) true members against false members of the v24 pocket;
  C. alternative per-account aggregates of the three behaviours, each scored by the same cross-fitted 3-D density ratio, shuffle
     null and expected-F3 cut-off, compared at ONE common group size: mean over periods (v24's _level), median over periods, mean of
     per-period normal scores, second-lowest per-period rank (high in at least 3 of 4 periods);
  D. if a column of another behaviour clears the bound in A: a 4-D density ratio with that column added (coarser bins, 0.2 SD).
The estimator is task5.density_ratio_curve with the stabilised settings of item 3 (reps, n_perm below).
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

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
FEATS = ["accuracy_std_level", "kda_std_level"]
B3 = ["accuracy_std", "kda_std", "performance_stability"]
B2 = 9.0
MAXK = 3000
REPS, NPERM = 200, 1000


def behaviour(c):
    import re
    return task5._base_feature(re.sub(r"_[1-4]$", "", c))


def shift_table(R, pk, adj_cols, cands):
    A = np.column_stack([np.ones(len(R)), R[adj_cols].to_numpy()])
    rows = []
    for c in cands:
        y = R[c].to_numpy()
        b, *_ = np.linalg.lstsq(A[~pk], y[~pk], rcond=None)
        res = y - A @ b
        sd = res[~pk].std()
        if sd < 1e-9:
            continue
        rows.append((c, behaviour(c), float(y[pk].mean()), float((A @ b)[pk].mean()), float(res[pk].mean() / (sd / np.sqrt(pk.sum())))))
    t = pd.DataFrame(rows, columns=["column", "behaviour", "pocket mean rank", "predicted", "z"])
    return t.reindex(t.z.abs().sort_values(ascending=False).index).reset_index(drop=True)


def f3curve(est, nk, G):
    k = np.arange(1, MAXK + 1)
    return (1 + B2) * (nk + est[:MAXK]) / (B2 * (nk + G) + nk + k)


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    _, gm = task5.gap_clusters(df, X)
    gap = gm.any(axis=1).values
    rest = np.where(~gap)[0]
    nk = int(gap.sum())
    pocket0, pinfo0 = task5.pocket_detector(X, gap, FEATS)
    pocket, pinfo = task5.pocket_refine(X, gap, pocket0, pinfo0, FEATS)
    cols3 = pinfo["cols"]
    pk = pocket[rest]
    log(f"\n===== {label}: gap families {nk}, outside {len(rest)}; pocket ({pinfo['method']}, {cols3}) k {pinfo['k']}")
    raw = df.drop(columns=["player_id"])
    raw = raw[[c for c in raw.columns if c not in X.columns]]
    A = pd.concat([X, raw], axis=1).iloc[rest].reset_index(drop=True)
    A = A[[c for c in A.columns if pd.api.types.is_numeric_dtype(A[c]) and A[c].nunique() > 1]]
    R = A.rank(pct=True).fillna(0.5)
    # A. every column
    cands = [c for c in A.columns if c not in cols3]
    t = shift_table(R, pk, cols3, cands)
    bound = norm.isf(0.0005 / len(t))
    own = t.behaviour.isin({behaviour(c) for c in cols3})
    log(f"A. {len(t)} candidate columns, Bonferroni bound {bound:.2f}; above it: own behaviours {int((own & (t.z.abs() > bound)).sum())}, "
        f"other behaviours {int((~own & (t.z.abs() > bound)).sum())}")
    pd.set_option("display.width", 250)
    log("   own behaviours, largest |z|:\n" + t[own].head(12).round(3).to_string(index=False))
    log("   other behaviours, largest |z|:\n" + t[~own].head(10).round(3).to_string(index=False))
    if truth is not None:
        tr = truth[rest]
        tt = shift_table(R, tr & pk, cols3, cands)
        ff = shift_table(R, ~tr & pk, cols3, cands)
        log("   first release: true members of the pocket (adjusted z), largest |z| of other behaviours: "
            + ", ".join(f"{r.column} {r.z:+.1f}" for r in tt[~tt.behaviour.isin({behaviour(c) for c in cols3})].head(6).itertuples())
            + f" | false members ({int((~tr & pk).sum())}): " + ", ".join(f"{r.column} {r.z:+.1f}" for r in ff.head(6).itertuples()))
    # B. persistence over periods
    P = {b: R[[f"{b}_{p}" for p in range(1, 5)]].to_numpy() for b in B3}
    nhigh = {b: (P[b] > 0.8).sum(1) for b in B3}
    lowp = {b: P[b].min(1) for b in B3}
    lv = R[cols3].to_numpy()
    same_level = (lv > 0.8).all(1) & ~pk
    line = f"B. per-period profile, pocket ({pk.sum()}) vs normal accounts at the same level (all three level ranks > 0.8, outside the pocket: {same_level.sum()}): "
    line += "; ".join(f"{b}: periods in top 20% {nhigh[b][pk].mean():.2f} vs {nhigh[b][same_level].mean():.2f}, lowest period rank "
                      f"{lowp[b][pk].mean():.2f} vs {lowp[b][same_level].mean():.2f}" for b in B3)
    log(line)
    if truth is not None:
        tp, fp = pk & tr, pk & ~tr
        log(f"   first release, v24 pocket true ({tp.sum()}) vs false ({fp.sum()}) members: "
            + "; ".join(f"{b}: periods in top 20% {nhigh[b][tp].mean():.2f} vs {nhigh[b][fp].mean():.2f}, lowest period rank "
                        f"{lowp[b][tp].mean():.2f} vs {lowp[b][fp].mean():.2f}, level rank {R[f'{b}_level'][tp].mean():.2f} vs "
                        f"{R[f'{b}_level'][fp].mean():.2f}" for b in B3))
    # C. aggregates
    rawp = {b: A[[f"{b}_{p}" for p in range(1, 5)]].to_numpy(float) for b in B3}
    nsp = {b: norm.ppf(np.clip(P[b], 0.5 / len(R), 1 - 0.5 / len(R))) for b in B3}
    aggs = {"mean over periods (v24 _level)": np.column_stack([np.nanmean(rawp[b], 1) for b in B3]),
            "median over periods": np.column_stack([np.nanmedian(rawp[b], 1) for b in B3]),
            "mean of per-period normal scores": np.column_stack([nsp[b].mean(1) for b in B3]),
            "second-lowest period rank": np.column_stack([np.sort(P[b], 1)[:, 1] for b in B3])}
    D = {}
    t_other = t[~own]
    if len(t_other) and abs(t_other.z.iloc[0]) > bound:
        c4 = t_other.column.iloc[0]
        aggs[f"levels + {c4} (4-D)"] = np.column_stack([aggs["mean over periods (v24 _level)"], A[c4].to_numpy(float)])
    for name, V in aggs.items():
        order, est, _ = task5.density_ratio_curve(V, reps=REPS, n_perm=NPERM, max_k=MAXK)
        D[name] = (order, est)
        log(f"   C. scored: {name}")
    G = max(e.max() for _, e in D.values())
    rows = []
    for name, (order, est) in D.items():
        f = f3curve(est, nk, G)
        ks = int(np.argmax(f)) + 1
        row = dict(score=name, group=float(est.max()), k=ks, est_prec=float(est[ks - 1] / ks), exp_F3=float(f[ks - 1]))
        if truth is not None:
            fl = gap.copy()
            fl[rest[order[:ks]]] = True
            tpv = np.sum(fl & truth)
            row["ref_F3"] = (1 + B2) * tpv / ((1 + B2) * tpv + B2 * np.sum(~fl & truth) + np.sum(fl & ~truth))
            row["ref_prec_pocket"] = float(truth[rest[order[:ks]]].mean())
        rows.append(row)
    log(f"C/D. common group size {G:.0f}:\n" + pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1:
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
