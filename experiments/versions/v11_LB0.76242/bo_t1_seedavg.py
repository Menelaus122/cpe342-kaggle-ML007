"""Round 11, Task 1: the gain from adding members (bn_t1_combined.py) depends on the seed (V3: +0.0015 with seed 42, +0.0005 with 7).
Seed-averaged members (mean of the seed-42 and seed-7 predictions, saved by bn) remove most of that luck; same nested stack + rule."""
import numpy as np
import pandas as pd
import common  # noqa: F401
from common import OUT
from pathlib import Path
exec(open(Path(__file__).with_name("bn_t1_combined.py"), encoding="utf-8").read().split("member_f2 = lambda")[0].split("# ------------------------------------------------ ablation")[0])
Z = np.load(OUT / "bn_t1_members.npz")
get = lambda k: (Z[f"{k}_oof"], Z[f"{k}_tl"])
avg = lambda k: tuple((Z[f"{k}_42_{s}"] + Z[f"{k}_7_{s}"]) / 2 for s in ("oof", "tl"))
M = {k: get(k) for k in ("lgb", "xgb", "cat", "et", "lr")}
M["rf"] = avg("rf")
for k in ("rf", "lgb", "xgb", "cat"):
    M[f"their_{k}"] = avg(f"their_{k}")
ours = ["lgb", "xgb", "cat", "et", "lr"]
ref = stack_pred([M[k][0] for k in ours]); ref_tl = stack_pred([M[k][0] for k in ours], [M[k][1] for k in ours]); fr = per_fold(ref)
th = ["their_rf", "their_lgb", "their_xgb", "their_cat"]
rows = []
for name, keys in (("V1 +RF", ours + ["rf"]), ("V2 +their 4", ours + th), ("V3 +RF +their 4", ours + ["rf"] + th),
                   ("V6 +their CatBoost only", ours + ["their_cat"]), ("V7 +RF +their CatBoost", ours + ["rf", "their_cat"])):
    p = stack_pred([M[k][0] for k in keys]); p_tl = stack_pred([M[k][0] for k in keys], [M[k][1] for k in keys]); fo = per_fold(p)
    lo, hi = boot(p, ref); tlo, thi = boot(p_tl, ref_tl, w_tl); dtl = wf2(p_tl, w_tl) - wf2(ref_tl, w_tl); better = int((fo > fr).sum())
    keep = lo > 0 and better >= 4 and fo.std() <= fr.std() * 1.1 and dtl >= 0
    rows.append([name + " (2-seed avg)", f2(y, p), f2(y, p) - f2(y, ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", fo.std(), f"{dtl:+.4f}",
                 f"[{tlo:+.4f}, {thi:+.4f}]", "KEEP" if keep else "discard"])
print(f"B0 (refitted members): F2 {f2(y, ref):.4f}, folds {np.round(fr, 4)}, std {fr.std():.4f}; test-like {wf2(ref_tl, w_tl):.4f}")
print(pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like delta", "test-like CI",
                                  "Decision"]).to_string(index=False))
