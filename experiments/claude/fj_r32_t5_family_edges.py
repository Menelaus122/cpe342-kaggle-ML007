"""Round 32, Task 5 item 2: multi-column signatures at the edges of the six gap families (label-free; Task 5 has only test.csv).

    python fj_r32_t5_family_edges.py [first-release test.csv for the reference run]

Each family is cut by 1-D density gaps (gap_clusters: column > cut or < cut, merged into families by overlap), but the planted
behaviour of a family moves several columns at once. Two questions:
  (1) missed members: are there unflagged accounts just inside the bulk side of a family's gap column that carry the family's
      multi-column signature?
  (2) false members: are there family members just beyond the cut that do not carry it?
Signature of family f (no labels): for every column outside the behaviours of f's own gap columns (derived features of build_features
and raw columns), z of the mean rank of f's members against the unflagged accounts U; the columns above the two-sided Bonferroni
bound (alpha 0.001), at most one per behaviour (the largest |z|), at most 8. Signature score = signed Stouffer sum of the normal
scores (ranks over all accounts) / sqrt(m).
Tests, for every gap column c of f (side and cut from gap_clusters):
  edge zone E = the 1% of U nearest to the cut on the bulk side of c (largest c for a high-side gap). Under the null (no missed
  members; the signature does not depend on c among normal accounts) E has the signature distribution of the rest of U:
  z of the mean signature of E against U \\ E, the count of E above U's 99th signature percentile (expected 1% of |E|), and the
  same for the next 1-2% band and for 0.5% as a control of the trend towards the cut;
  false members: f's members in the 10% nearest the cut against the rest of f (mean signature, share below U's median).
Also reported: the Spearman of signature and c within U \\ E (dependence the null assumes away), and the tail of each signature
inside U against 200 column shuffles of its signature columns (accounts above U's own 99.9% / 99% quantile; valid only where the
signature columns are not built from each other, e.g. not with n_flags next to the flags it sums).
Behaviours: a column belongs to the longest temporal group name it starts with; geo and login_count count as one behaviour
(km per login is computed from the login count).
Reference run (first release, truth = id block P023819-P025888, reference only): precision of each family and of its near-cut
members, true anomalies in each edge zone, and where the 24 true anomalies that the pipeline leaves unflagged sit (largest family
signature percentile, 3-column pocket score percentile, distance to the nearest cut in units of the bulk density).
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, {psutil.Process().memory_info().rss / 2**20:.0f} MB]", flush=True)
FEATS = ["accuracy_std_level", "kda_std_level"]


GROUPS = sorted(task5.temporal_groups(task5.load()), key=len, reverse=True)


def behaviour(c):
    """Behaviour of a raw or derived column: the longest temporal group name it starts with (so activity_level_level ->
    activity_level, not 'activity'); geo features -> geo; anything else is its own behaviour."""
    if c.startswith("geo") or c.startswith("login_lat") or c.startswith("login_lon"):
        return "geo"
    for g in GROUPS:
        if c == g or c.startswith(g + "_"):
            return g
    return c


def own_behaviours(cols):
    """Behaviours of a family's gap columns, plus their built-in partners (km per login is computed from the login count)."""
    own = {behaviour(c) for c in cols}
    if own & {"geo", "login_count"}:
        own |= {"geo", "login_count"}
    return own


def run(df, label, truth=None):
    X, _ = task5.build_features(df)
    table, masks = task5.gap_clusters(df, X)
    fams = task5.merge_families(masks)
    gap = masks.any(axis=1).values
    pocket0, pinfo0 = task5.pocket_detector(X, gap, FEATS)
    pocket, pinfo = task5.pocket_refine(X, gap, pocket0, pinfo0, FEATS)
    flagged = gap | pocket
    U = ~flagged
    raw = df.drop(columns=["player_id"])
    raw = raw[[c for c in raw.columns if c not in X.columns]]
    A = pd.concat([X, raw], axis=1)
    A = A[[c for c in A.columns if pd.api.types.is_numeric_dtype(A[c]) and A[c].nunique() > 1]]
    R = A.rank(pct=True).fillna(0.5)                                      # ranks over all accounts
    NS = pd.DataFrame(norm.ppf(np.clip(R.to_numpy(), 0.5 / len(A), 1 - 0.5 / len(A))), columns=A.columns)
    log(f"\n===== {label}: {len(df)} accounts; gap families {gap.sum()} in {len(fams)} families; pocket {pocket.sum()}; "
        f"unflagged {U.sum()}; columns {A.shape[1]}")
    if truth is not None:
        log(f"reference: true anomalies {truth.sum()}; flagged precision {truth[flagged].mean():.3f}, recall "
            f"{truth[flagged].sum() / truth.sum():.3f}; true anomalies left unflagged {int((truth & U).sum())}")
    sig_all = {}
    rows = []
    for fam in sorted(fams, key=lambda f: -masks[f].any(axis=1).sum()):
        fm = masks[fam].any(axis=1).values
        own = own_behaviours([table.set_index("group").column[g] for g in fam])
        cand = [c for c in A.columns if behaviour(c) not in own]
        nf = fm.sum()
        Rf, Ru = R.loc[fm, cand], R.loc[U, cand]
        z = (Rf.mean() - Ru.mean()) / np.sqrt(Ru.var() / U.sum() + Rf.var() / nf + 1e-12)
        bound = norm.isf(0.0005 / len(cand))
        zz = z[z.abs() > bound].reindex(z[z.abs() > bound].abs().sort_values(ascending=False).index)
        pick, seen = [], set()
        for c in zz.index:
            if behaviour(c) not in seen:
                pick.append(c)
                seen.add(behaviour(c))
            if len(pick) == 8:
                break
        name = " | ".join(fam)
        if not pick:
            log(f"family [{name}] ({nf}): no signature column outside its own behaviours")
            continue
        s = (NS[pick].to_numpy() * np.sign(zz[pick].to_numpy())).sum(1) / np.sqrt(len(pick))
        sig_all[name] = s
        u99, umed = np.quantile(s[U], 0.99), np.median(s[U])
        Wu = NS.loc[U, pick].to_numpy() * np.sign(zz[pick].to_numpy())
        su = Wu.sum(1) / np.sqrt(len(pick))
        rngn = np.random.default_rng(0)
        tails = {}
        for q in (0.999, 0.99):
            th = np.quantile(su, q)
            null = [np.sum(np.column_stack([rngn.permutation(Wu[:, j]) for j in range(Wu.shape[1])]).sum(1) / np.sqrt(len(pick)) > th)
                    for _ in range(200)]
            tails[q] = (int(np.sum(su > th)), float(np.mean(null)), float(np.std(null)))
        log(f"family [{name}] ({nf} accounts): signature {', '.join(f'{c} {zz[c]:+.1f}' for c in pick)} "
            f"(bound {bound:.2f}); family members above U's 99th signature percentile {np.mean(s[fm] > u99):.2f}; U above its own "
            + ", ".join(f"{q:.1%} signature quantile: {a} vs shuffle null {b:.1f} +- {c:.1f}" for q, (a, b, c) in tails.items())
            + (f"; true anomalies in U's top 0.1% / 1% of this signature: {int(truth[np.where(U)[0][su > np.quantile(su, 0.999)]].sum())} / "
               f"{int(truth[np.where(U)[0][su > np.quantile(su, 0.99)]].sum())}" if truth is not None else ""))
        for g in fam:
            row = table.set_index("group").loc[g]
            col, side, cut = row.column, row.side, row.cut
            x = (X[col] if col in X.columns else df[col]).to_numpy(float)
            xu = np.where(U & ~np.isnan(x), x, np.nan)
            ordr = np.argsort(-xu if side == "high" else xu)               # U sorted towards the cut, NaN last
            ordr = ordr[~np.isnan(xu[ordr])]
            nU = len(ordr)
            res = {}
            for tag, lo, hi in (("0-0.5%", 0, 0.005), ("0-1%", 0, 0.01), ("1-2%", 0.01, 0.02), ("2-3%", 0.02, 0.03)):
                E = ordr[int(lo * nU):int(hi * nU)]
                rest = np.setdiff1d(np.where(U)[0], E)
                zE = (s[E].mean() - s[rest].mean()) / (s[rest].std() / np.sqrt(len(E)))
                res[tag] = (len(E), zE, int(np.sum(s[E] > u99)), 0.01 * len(E),
                            int(truth[E].sum()) if truth is not None else None)
            E1 = ordr[:int(0.01 * nU)]
            rest1 = np.setdiff1d(np.where(U & ~np.isnan(x))[0], E1)
            rho = spearmanr(x[rest1], s[rest1]).correlation
            # false members: members nearest the cut
            xm = np.where(masks[g].values, x, np.nan)
            om = np.argsort(xm if side == "high" else -xm)
            om = om[~np.isnan(xm[om])]
            near, far = om[:max(int(0.1 * len(om)), 1)], om[max(int(0.1 * len(om)), 1):]
            line = (f"   gap column {col} {'>' if side == 'high' else '<'} {cut:.4g} ({masks[g].sum()} members): edge zones of U "
                    + "; ".join(f"{t}: n {v[0]}, z {v[1]:+.1f}, above U99 {v[2]} (exp {v[3]:.1f})" + (f", true {v[4]}" if v[4] is not None else "")
                                for t, v in res.items())
                    + f" | Spearman(signature, column) in U {rho:+.3f} | members nearest the cut (10%, n {len(near)}): mean signature "
                      f"{s[near].mean():+.2f} vs rest {s[far].mean():+.2f}, below U median {np.mean(s[near] < umed):.2f} vs "
                      f"{np.mean(s[far] < umed):.2f}")
            if truth is not None:
                line += f"; precision near the cut {truth[near].mean():.3f}, rest {truth[far].mean():.3f}"
            log(line)
            rows.append(dict(family=name, column=col, edge_z=res["0-1%"][1], edge_hi=res["0-1%"][2], edge_exp=res["0-1%"][3]))
    if truth is not None:
        miss = np.where(truth & U)[0]
        Rk = X.loc[~gap, pinfo["cols"]].rank(method="first").to_numpy()
        Zp = norm.ppf((Rk - 0.5) / (~gap).sum())
        D = task5.DensityRatio(Zp, 0.40, np.random.default_rng(0), 20)
        ps = pd.Series(D.score(Zp), index=np.where(~gap)[0]).rank(pct=True)
        log(f"reference: the {len(miss)} true anomalies left unflagged")
        out = []
        for i in miss:
            fs = {k: (v[U] < v[i]).mean() for k, v in sig_all.items()}
            kbest = max(fs, key=fs.get)
            out.append(dict(id=df.player_id[i], best_family=kbest.split(" | ")[0], family_sig_pct=fs[kbest],
                            pocket_score_pct=ps.get(i, np.nan), pocket_cols_rank=" / ".join(f"{R.loc[i, c]:.2f}" for c in pinfo["cols"])))
        tab = pd.DataFrame(out).sort_values("family_sig_pct", ascending=False)
        pd.set_option("display.width", 250)
        log(tab.round(3).to_string(index=False))
        log(f"   of them with a family signature above U's 99th percentile: {int((tab.family_sig_pct > 0.99).sum())}; pocket score above "
            f"U's 99th percentile: {int((tab.pocket_score_pct > 0.99).sum())}")


if __name__ == "__main__":
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    run(task5.load(), "competition data (2026-11-12 release)")
    if len(sys.argv) > 1:
        d = pd.read_csv(sys.argv[1])
        run(d, "first release, reference only", truth=d.player_id.str[1:].astype(int).values >= 23819)
    log("done")
