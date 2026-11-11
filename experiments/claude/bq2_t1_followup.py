"""Round 12 follow-up to bq_t1_paradigms.py:
(1) N2 (LightGBM + in-fold class-conditional Mahalanobis distances) with LightGBM seeds 7 and 2026, then the 3-seed average,
    because single-seed gains of this size moved by ~0.001 between seeds in round 11;
(2) S2 hill climbing fixed: the cut-off search (src.metrics.optimize_threshold, grid 0.05-0.95) needs probabilities, so the
    weighted logit average goes back through the sigmoid (the first run fed it logits, which broke both the search and the result).
Same baseline, members, test-like check and acceptance rule as bq_t1_paradigms.py.
"""
from pathlib import Path

_src = open(Path(__file__).with_name("bq_t1_paradigms.py"), encoding="utf-8").read()
exec(_src.split("# ------------------------------------------------ N1 spline logistic regression")[0].split('"""', 2)[2])

sig = lambda v: 1 / (1 + np.exp(-v))
Zq = np.load(OUT / "bq_t1_members.npz")
n2s = [(Zq["N2 LightGBM + Mahalanobis_oof"], Zq["N2 LightGBM + Mahalanobis_tl"])]
for seed_ in (7, 2026):
    o, otl = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        za, (zb, ztl) = prep(Xv[a], [Xv[b], Xtlv[b]])
        stats = []
        for c in (0, 1):
            lw = LedoitWolf().fit(za[y[a] == c])
            stats.append((lw.location_, lw.precision_))
        dist = lambda Zm: np.column_stack([np.sqrt(np.einsum("ij,jk,ik->i", Zm - mu, P_, Zm - mu)) for mu, P_ in stats])

        def feat(Zm, Xm):
            d = dist(Zm)
            return pd.DataFrame(np.hstack([Xm, d, d[:, :1] - d[:, 1:]]), columns=list(X.columns) + ["maha0", "maha1", "maha_diff"])

        _, (pv, pt), _ = fit_predict("lgb", "binary", {**P["lgb"], "random_state": seed_}, feat(za, Xv[a]), y[a], feat(zb, Xv[b]), y[b],
                                     [feat(zb, Xv[b]), feat(ztl, Xtlv[b])])
        o[b], otl[b] = pv, pt
    n2s.append((o, otl))
    add_member(f"N2 Mahalanobis, seed {seed_}", o, otl)
add_member("N2 Mahalanobis, 3-seed average", np.mean([v[0] for v in n2s], 0), np.mean([v[1] for v in n2s], 0))


def hill(Lo, a, iters=40):
    """Caruana forward selection with replacement on the training-fold rows a; objective = F2 at the best cut-off."""
    score = lambda s: optimize_threshold(y[a], sig(s[a]), 2.0, average="binary")[1]
    counts = np.zeros(Lo.shape[1])
    j0 = int(np.argmax([score(Lo[:, j]) for j in range(Lo.shape[1])]))
    counts[j0] = 1
    cur = score(Lo[:, j0])
    for _ in range(iters):
        cand = [score((Lo @ (counts + np.eye(len(counts))[j])) / (counts.sum() + 1)) for j in range(Lo.shape[1])]
        j = int(np.argmax(cand))
        if cand[j] <= cur:
            break
        counts[j] += 1
        cur = cand[j]
    return counts / counts.sum()


Lo = np.column_stack([_logit(M[k][0]) for k in BASE])
Ltl = np.column_stack([_logit(M[k][1]) for k in BASE])
p, p_tl = np.zeros(len(y), int), np.zeros(len(y), int)
for a, b in folds:
    w = hill(Lo, a)
    s, s_tl = sig(Lo @ w), sig(Ltl @ w)
    t = optimize_threshold(y[a], s[a], 2.0, average="binary")[0]
    p[b], p_tl[b] = s[b] >= t, s_tl[b] >= t
    log(f"  hill-climb weights: {dict(zip(BASE, np.round(w, 2).tolist()))}")
judge("S2 hill climbing (Caruana), nested, fixed", p, p_tl)

print("\n" + pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like delta", "test-like CI",
                                         "Decision"]).to_string(index=False), flush=True)
np.savez(OUT / "bq2_t1_n2_seeds.npz", **{f"seed{i}_oof": v[0] for i, v in enumerate(n2s)}, **{f"seed{i}_tl": v[1] for i, v in enumerate(n2s)})
log("done")
