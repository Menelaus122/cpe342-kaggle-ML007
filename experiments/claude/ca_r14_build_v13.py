"""Round 14: build submission_v13.csv = v12 with the Task 3 column from the 10-seed bag (the only step-4 variant that passed, bz_r14_eval.py t3).

    python ca_r14_build_v13.py

Follows the pipeline's Task 3 path (scripts/nb_part2.py, sections 7.3 and 7.6) exactly:
  pool of 26 members -> out-of-fold readouts (sigma, gate shift, tau, smearing) -> exact L1 blend weights -> median zero gate -> scale -> cap at 300,000.
  Final test input = 0.5 x 5-fold models + 0.5 x Train+Val refit, per seed; the bag averages seeds (probabilities for stage 1, log scale for stage 2,
  as in bz_r14_eval.py). Readouts, weights, gate and scale are re-learned on the bagged OOF; nothing looks at the test set.
Checks before anything is written:
  1. the same code fed the cached v12 members reproduces the uploaded v12 Task 3 column text exactly,
  2. the bagged OOF score equals the evaluation run,
  3. the seed-42 Train+Val refits equal the pipeline cache.
Writes submission_v13.csv (v12 + new Task 3) and submission_clean_v13.csv (clean fallback + new Task 3); every other column is copied byte for byte.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from by_r14_bag import BAG3, DIR, SEEDS  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

ROOT = Path.cwd()
V12, CLEAN = ROOT / "submission_v12_try.csv", ROOT / "submission_clean_try.csv"
OUT_V13, OUT_CLEAN = ROOT / "submission_v13.csv", ROOT / "submission_clean_v13.csv"
SHA_V12 = "de380b025cdfe095154ffdfe5ef8d57fd7ed0e48160b90d3ceb94241ca29c384"
SHA_CLEAN = "061cb9866cab7de5d203799a890834c6a4186790e38ec107bf90bd9f33620b28"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(V12) == SHA_V12, "v12 file changed"
assert sha(CLEAN) == SHA_CLEAN, "clean fallback changed"

FULL_W = 0.5
blend_full = lambda fold_test, full_test: (1 - FULL_W) * np.asarray(fold_test) + FULL_W * np.asarray(full_test)
clip0 = lambda v: np.clip(v, 0, None)
slug3 = lambda t: "t3r_" + t.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
TAGS = ["Hurdle", "ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
BCOL = {t: ("prev_month_spending" if "[prev]" in t else "historical_spending") for t in TAGS[1:]}
READ = ("mixture-median", "hard gate", "E[y] + smearing")
BAGGED = {"ratio[hist] LGB tuned": "ratio_hist_lgb_tuned", "ratio[prev] LGB tuned": "ratio_prev_lgb_tuned",
          "ratio[hist] XGB": "ratio_hist_xgb", "ratio[hist] Cat": "ratio_hist_cat"}
assert set(BAGGED.values()) == set(BAG3)

tr, te = task3.load()
reduce_mem_usage(tr, verbose=False)  # as the notebook does (training frame only)
X, y, Xt, _ = task3.build_repaired(tr, te)
folds = make_folds(y, kind="regression")
bt = {t: np.log1p(Xt[c].values) for t, c in BCOL.items()}

# plain members (not bagged): 5-fold test + Train+Val refit, as in the pipeline
plain_oof, plain_fin = {}, {}
for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                        ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r, f = load_preds(key), load_preds(key + "_full")
    tf = (lambda v: clip0(np.expm1(v))) if logt else clip0
    plain_oof[name], plain_fin[name] = tf(r["oof"]), blend_full(tf(r["test"]), tf(f["test"]))


def ratio_mu_fin(tag, hurdle_mu_fin, ratio_test_fin):
    mu = hurdle_mu_fin.copy()
    ok = np.isfinite(bt[tag])
    mu[ok] = bt[tag][ok] + ratio_test_fin[ok]
    return mu


def build(p_oof, mu_oof, p_fin, mu_fin):
    """Pipeline sections 7.3 + 7.6 for given stage inputs. Returns OOF final, test final, info."""
    names, Po, Pt = list(plain_oof), list(plain_oof.values()), list(plain_fin.values())
    RO = {}
    for tag in TAGS:
        RO[tag] = task3.readouts(y, p_oof, mu_oof[tag])
        for rk in READ:
            names.append(f"{tag} {rk}")
            Po.append(task3.apply_readout(rk, p_oof, mu_oof[tag], RO[tag]))
            Pt.append(task3.apply_readout(rk, p_fin, mu_fin[tag], RO[tag]))
    Po, Pt = np.column_stack(Po), np.column_stack(Pt)
    bo, cap = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        bo[b] = Po[b] @ task3.l1_blend_weights_exact(y[a], [Po[a, j] for j in range(Po.shape[1])])
        cap[b] = y[a].max()
    W = task3.l1_blend_weights_exact(y, [Po[:, j] for j in range(Po.shape[1])])
    shift = RO["Hurdle"]["shift"]
    gate_o = (np.clip(p_oof + shift, 0, 1) > 0.5).astype(float)
    gate_t = (np.clip(p_fin + shift, 0, 1) > 0.5).astype(float)
    scales = np.linspace(0.85, 1.15, 61)
    scale = float(scales[int(np.argmax([nmae(y, bo * s * gate_o) for s in scales]))])
    oof = np.minimum(bo * scale * gate_o, cap)
    test = np.minimum(np.clip(Pt @ W * scale, 0, None) * gate_t, float(y.max()))
    return oof, test, dict(W={k: round(float(w), 3) for k, w in zip(names, W) if w > 1e-3}, scale=scale, shift=shift, sigma=RO["Hurdle"]["sigma"])


# ---- v12 inputs (cached seed-42 members)
h, hf = load_preds("t3r_hurdle"), load_preds("t3r_hurdle_full")
mu0_oof = {"Hurdle": h["mu_oof"], **{t: load_preds(slug3(t))["mu_oof"] for t in TAGS[1:]}}
p0_fin = blend_full(h["p_test"], hf["p_test"])
mu0_fin = {"Hurdle": blend_full(h["mu_test"], hf["mu_test"])}
for t in TAGS[1:]:
    mu0_fin[t] = ratio_mu_fin(t, mu0_fin["Hurdle"], blend_full(load_preds(slug3(t))["test"], load_preds(slug3(t) + "_full")["test"]))
oof0, test0, info0 = build(h["p_oof"], mu0_oof, p0_fin, mu0_fin)

v12 = pd.read_csv(V12, dtype=str, keep_default_na=False)
clean = pd.read_csv(CLEAN, dtype=str, keep_default_na=False)
fmt = lambda v: [str(float(x)) for x in np.round(v, 2)]
same_text = int((pd.Series(fmt(test0)) == v12.task3).sum())
print(f"check 1, v12 rebuilt from cache: OOF NMAE {nmae(y, oof0):.6f} (pipeline 0.79434); Task 3 text equal to the uploaded v12 on {same_text:,} of {len(v12):,} rows; "
      f"max |diff| {np.abs(np.round(test0, 2) - v12.task3.astype(float).values).max():.4f}")
print(f"  v12 scale {info0['scale']:.3f}, gate shift {info0['shift']:+.4f}, sigma {info0['sigma']:.4f}; weights {info0['W']}")
assert same_text == len(v12), "rebuild does not reproduce v12, stop"
assert (clean.task3 == v12.task3).all(), "clean fallback and v12 Task 3 columns differ"

# ---- check 3: seed-42 Train+Val refits equal the cache
z = np.load(DIR / "t3full_hurdle_42.npz")
print(f"check 3, seed-42 refits vs cache: hurdle p {np.abs(z['p_test'] - hf['p_test']).max():.3g}, mu {np.abs(z['mu_test'] - hf['mu_test']).max():.3g}; "
      + ", ".join(f"{nm} {np.abs(np.load(DIR / f't3full_{nm}_42.npz')['test'] - load_preds(BAG3[nm][3] + '_full')['test']).max():.3g}" for nm in BAG3))

# ---- 10-seed bag
H = [np.load(DIR / f"t3_hurdle_{s}.npz") for s in SEEDS]
HF = [np.load(DIR / f"t3full_hurdle_{s}.npz") for s in SEEDS]
p_oof = np.mean([z["p_oof"] for z in H], axis=0)
p_fin = np.mean([blend_full(a["p_test"], b["p_test"]) for a, b in zip(H, HF)], axis=0)
mu_oof, mu_fin = dict(mu0_oof), dict(mu0_fin)  # zero-weight ratio members stay as in v12
mu_oof["Hurdle"] = np.mean([z["mu_oof"] for z in H], axis=0)
mu_fin["Hurdle"] = np.mean([blend_full(a["mu_test"], b["mu_test"]) for a, b in zip(H, HF)], axis=0)
for t, nm in BAGGED.items():
    R = [np.load(DIR / f"t3_{nm}_{s}.npz") for s in SEEDS]
    RF = [np.load(DIR / f"t3full_{nm}_{s}.npz") for s in SEEDS]
    mu_oof[t] = np.mean([r["mu_oof"] for r in R], axis=0)
    mu_fin[t] = ratio_mu_fin(t, mu_fin["Hurdle"], np.mean([blend_full(r["test"], f["test"]) for r, f in zip(R, RF)], axis=0))
oof1, test1, info1 = build(p_oof, mu_oof, p_fin, mu_fin)
print(f"check 2, 10-seed bag: OOF NMAE {nmae(y, oof1):.6f} (evaluation 0.794578), delta {nmae(y, oof1) - nmae(y, oof0):+.6f}")
print(f"  bag scale {info1['scale']:.3f}, gate shift {info1['shift']:+.4f}, sigma {info1['sigma']:.4f}; weights {info1['W']}")

# ---- write: replace only the task3 field, copy every other byte
def write(src, dst, values):
    raw = src.read_bytes()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.split(eol)
    col = lines[0].decode().split(",").index("task3")
    txt = fmt(values)
    for i in range(1, len(v12) + 1):
        parts = lines[i].decode().split(",")
        parts[col] = txt[i - 1]
        lines[i] = ",".join(parts).encode()
    dst.write_bytes(eol.join(lines))


write(V12, OUT_V13, test1)
write(CLEAN, OUT_CLEAN, test1)

# ---- row-by-row diff against v12, every task
a, b = pd.read_csv(V12), pd.read_csv(OUT_V13)
assert a.id.equals(b.id) and list(a.columns) == list(b.columns) and b.shape == (25_889, 6)
print("\n=== submission_v13.csv vs v12, rows that differ per task ===")
for c in ["task1", "task2", "task3", "task4", "task5"]:
    print(f"  {c}: {int((a[c] != b[c]).sum()):,}")
d = b.task3 - a.task3
nz = d != 0
print(f"Task 3 detail: rows changed {int(nz.sum()):,} of {len(d):,}; |change| mean {d.abs().mean():.2f} THB, median over changed rows {d[nz].abs().median():.2f}, "
      f"p99 {d.abs().quantile(0.99):.2f}, max {d.abs().max():.2f}")
print(f"  zero <-> positive switches: 0 -> positive {int(((a.task3 == 0) & (b.task3 > 0)).sum())}, positive -> 0 {int(((a.task3 > 0) & (b.task3 == 0)).sum())}; "
      f"zero share {np.mean(a.task3 == 0):.4f} -> {np.mean(b.task3 == 0):.4f}")
print(f"  sum {a.task3.sum():,.2f} -> {b.task3.sum():,.2f} (ratio {b.task3.sum() / a.task3.sum():.5f}); mean {a.task3.mean():,.2f} -> {b.task3.mean():,.2f}; "
      f"max {a.task3.max():,.2f} -> {b.task3.max():,.2f}; corr log1p {np.corrcoef(np.log1p(a.task3), np.log1p(b.task3))[0, 1]:.6f}")
pos = (a.task3 > 0) & (b.task3 > 0)
print(f"  relative change on rows positive in both: median {np.median(np.abs(d[pos]) / a.task3[pos]):.4f}, p95 {np.quantile(np.abs(d[pos]) / a.task3[pos], 0.95):.4f}")
c0, c1 = pd.read_csv(CLEAN), pd.read_csv(OUT_CLEAN)
print("submission_clean_v13.csv vs submission_clean_try.csv, rows that differ per task:", {c: int((c0[c] != c1[c]).sum()) for c in c0.columns[1:]})
print("submission_clean_v13.csv vs submission_v13.csv, rows that differ per task:", {c: int((c1[c] != b[c]).sum()) for c in c0.columns[1:]})
raw_a, raw_b = V12.read_bytes(), OUT_V13.read_bytes()
crlf = b"\r\n"
print(f"line endings kept: {(crlf in raw_a) == (crlf in raw_b)}; bytes {len(raw_a):,} -> {len(raw_b):,}")
print(f"sha256 submission_v13.csv       {sha(OUT_V13)}")
print(f"sha256 submission_clean_v13.csv {sha(OUT_CLEAN)}")
print(f"sha256 v12 (unchanged)          {sha(V12)}")
