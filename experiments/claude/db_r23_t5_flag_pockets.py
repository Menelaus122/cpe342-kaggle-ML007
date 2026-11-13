"""Round 23, Task 5: pockets that mix a yes/no flag with a continuous feature, the blind spot of the pipeline's pocket scan.

    python db_r23_t5_flag_pockets.py

task5.dependence_pockets scans pairs of continuous columns only (> 50 distinct values), so a planted group such as
"password changed AND a moderate location jump" (each alone common) would not be seen. Among the accounts NOT flagged in v15, for every
flag f in FLAG_COLS and every continuous column c of the engineered matrix, and each tail of c (top / bottom 1, 2, 5%):
    observed = #(f = 1 in the tail), expected = rate(f) * tail size, z = (obs - exp) / sqrt(exp (1 - rate)).
Noise level: the largest z of the same scan after shuffling f among these accounts (20 shuffles; flags are independent of each other
outside the families, FINDINGS 7.5). For each cell above the noise, the excess = obs - exp estimates planted members, and the
candidate set {f = 1 in the tail} has an estimated anomaly share excess / obs: worth adding under F3 only if above ~0.1.
Also: v15-flag edge accounts (closest to each family's gap) per family, with the same flag-excess estimate. Test features only.
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t5 = task5.load()
X5, _ = task5.build_features(t5)
final = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
U = ~final
cont = [c for c in X5.columns if pd.api.types.is_numeric_dtype(X5[c]) and X5[c].nunique() > 50]
TAILS = [("top", 0.01), ("top", 0.02), ("top", 0.05), ("bottom", 0.01), ("bottom", 0.02), ("bottom", 0.05)]
Xu = X5.loc[U, cont].to_numpy(float)
masks = []
for j, c in enumerate(cont):
    v = Xu[:, j]
    ok = ~np.isnan(v)
    for side, q in TAILS:
        thr = np.nanquantile(v, 1 - q if side == "top" else q)
        m = ok & ((v >= thr) if side == "top" else (v <= thr))
        masks.append((c, side, q, m))
M = np.column_stack([m for *_, m in masks]).astype(np.float32)          # accounts x tail cells
sizes = M.sum(0)


def scan(f):
    ok = ~np.isnan(f)
    r = f[ok].mean()
    obs = (M[ok] * f[ok, None]).sum(0)
    exp = r * M[ok].sum(0)
    return obs, exp, (obs - exp) / np.sqrt(np.maximum(exp * (1 - r), 1e-9))


rng = np.random.default_rng(0)
rows = []
for fc in task5.FLAG_COLS:
    f = t5.loc[U, fc].to_numpy(float)
    obs, exp, z = scan(f)
    noise = []
    for _ in range(20):
        g = f.copy()
        ok = ~np.isnan(g)
        g[ok] = rng.permutation(g[ok])
        noise.append(scan(g)[2].max())
    nz = max(noise)
    print(f"{fc:22s} rate among unflagged {np.nanmean(f):.4f}; max z {z.max():.1f} (cell {masks[int(z.argmax())][:3]}); noise level {nz:.1f}")
    for i in np.where(z > nz)[0]:
        c, side, q, _ = masks[i]
        rows.append(dict(flag=fc, column=c, tail=f"{side} {q:.0%}", z=z[i], noise=nz, observed=int(obs[i]), expected=exp[i],
                         excess=obs[i] - exp[i], est_share=(obs[i] - exp[i]) / max(obs[i], 1)))
res = pd.DataFrame(rows).sort_values("z", ascending=False) if rows else pd.DataFrame()
print("\ncells above the noise level (candidate set = flag 1 inside the tail):")
print(res.round(3).to_string(index=False) if len(res) else "  none")

print("\nfamily-edge accounts (closest unflagged accounts to each gap), flag excess per family:")
gc_table, gc_masks = task5.gap_clusters(t5, X5)
for _, g in gc_table.iterrows():
    x = (X5[g.column] if g.column in X5 else t5[g.column]).values.astype(float)
    d = (g.cut - x) if g.side == "high" else (x - g.cut)
    d = np.where(np.isnan(d) | (d < 0) | final, np.inf, d)
    fam = gc_masks[g.group].values
    for k in (50, 200):
        e = np.argsort(d)[:k]
        parts = []
        for fc in task5.FLAG_COLS:
            base = np.nanmean(t5.loc[U, fc])
            in_fam = np.nanmean(t5.loc[fam, fc])
            ob = np.nanmean(t5[fc].values[e])
            share = (ob - base) / (in_fam - base) if abs(in_fam - base) > 0.02 else np.nan   # mixture estimate, only if the family differs
            parts.append(f"{fc[:8]} {ob:.3f} (normal {base:.3f}, family {in_fam:.3f}{'' if np.isnan(share) else f', share {share:+.2f}'})")
        print(f"  {g.group:38s} nearest {k:3d}: " + "; ".join(parts))
