"""Round 43, Task 4: Fold-0 Gate judge for DINOv3-7B Top-12 LoRA (`dinov3_7b_lora12`) vs v41 (CPU; training OOF only).

    python experiments/claude/ik_r43_t4_7b_lora_fold0.py
    python experiments/claude/ik_r43_t4_7b_lora_fold0.py --cand dinov3Hp_384  # stand-in sanity test

Reference v41 on Fold 0:
  14 FT members (v38 weights: H384 40, H144 40, D384 20, D144 10, RAD 10/2, L2 8/2, B2 8, HUD x5 0.75, OLD x2 0.25)
  mixed at 80% with P_7b (`ii_probe_v3_views.npz` p7b_*) at PW = 0.20, Q10 brightness/contrast gate from folds 1-4, v17 rule.

Pre-declared Fold-0 variants (add CAND as 15th FT member):
  A20     CAND added at weight x20 (PW = 0.20)
  A40     CAND added at weight x40 (PW = 0.20)
  A60     CAND added at weight x60 (PW = 0.20)
  A40_p15 CAND added at weight x40 (PW = 0.15)
Gate: at least one variant meets (zoom70 > 0, centre_grey > 0, dark > 0, full >= -0.0005) on Fold 0 -> run Folds 1-4 + full.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="dinov3_7b_lora12")
    args = ap.parse_args()
    CAND = args.cand

    L2, D3, D384, RAD, HP, HP384 = (
        "dinov2L_hud70",
        "dinov3L_lr8e5",
        "dinov3L_384",
        "cradio3L_384",
        "dinov3Hp_144",
        "dinov3Hp_384",
    )
    PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
    V20 = PLAIN + HUDM + ["dinov2_v2"]
    VIEWS = ["full", "zoom70", "centre_grey"]
    V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")

    train, _ = task4.load_frames()
    y = train.label.values.astype(np.int64)
    folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
    fo = np.zeros(len(y), int)
    for f, va_i in enumerate(folds):
        fo[va_i] = f
    va = folds[0]

    cand_pred = PRED_DIR / f"t4_{CAND}_fold0.npz"
    cand_view = V30 / f"{CAND}_fold0.npz"
    assert cand_pred.exists(), f"Missing {cand_pred}: pull fold-0 output from Drive first"
    assert cand_view.exists(), f"Missing {cand_view}: pull fold-0 r30_views from Drive first"
    assert np.array_equal(np.load(cand_pred)["val_idx"], va), f"{cand_pred.name} val_idx mismatch vs fold 0"

    members = sorted(set(V20 + [L2, D3, D384, RAD, HP, HP384, CAND]))
    P = {}
    for t in members:
        z30 = np.load(V30 / f"{t}_fold0.npz")
        z19 = np.load(V19 / f"{t}_fold0.npz") if t in V20 else z30
        P[t] = {v: (z30 if v in z30.files else z19)[v].astype(np.float64) for v in VIEWS}

    z_p7b = np.load("experiments/claude/outputs/r42/ii_probe_v3_views.npz")
    P7B = {v: z_p7b[f"p7b_{v}"][va].astype(np.float64) for v in VIEWS}

    ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    tr = fo != 0
    tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
    GATE = {v: (ST[f"brightness_{v}"][va] <= tb) | (ST[f"contrast_{v}"][va] <= tc) for v in VIEWS}

    st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
    dark = ((st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1)))[va]
    yv = y[va]
    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[yv]
    bal = lambda pred, m: f1_score(yv[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(len(va), bool)
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k: dark if k == "dark/low-contrast" else allm

    BASE_FT = {
        **{t: 0.25 for t in PLAIN},
        **{t: 0.75 for t in HUDM},
        "dinov2_v2": 8.0,
        D3: 10.0,
        D384: 20.0,
        HP: 40.0,
        HP384: 40.0,
    }
    GATED_BASE = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}

    def eval_config(extra_w=0.0, pw=0.20):
        out = {}
        for v in VIEWS:
            num = sum(w * P[t][v] for t, w in BASE_FT.items())
            den = np.full(len(va), sum(BASE_FT.values()), dtype=np.float64)
            for t, (wn, wd) in GATED_BASE.items():
                ww = np.where(GATE[v], wd, wn)
                num, den = num + ww[:, None] * P[t][v], den + ww
            if extra_w > 0:
                num = num + extra_w * P[CAND][v]
                den = den + extra_w
            out[v] = rule((1.0 - pw) * (num / den[:, None]) + pw * P7B[v])
        return out

    ref = eval_config(extra_w=0.0, pw=0.20)
    print("v41 reference on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
    print("alone on fold 0 (balanced mix, v17 rule):")
    for nm, pmap in [(HP384, P[HP384]), ("P_7b", P7B), (CAND, P[CAND])]:
        print(f"  {nm:18s}: " + ", ".join(f"{k} {bal(rule(pmap[VIEW_OF[k]]), msk(k)):.4f}({(rule(pmap[VIEW_OF[k]])[msk(k)] != yv[msk(k)]).sum()})" for k in SC))

    any_ok = False
    for vname, ew, pw in (
        (f"A20     {CAND} added at x20 (PW=0.20)", 20.0, 0.20),
        (f"A40     {CAND} added at x40 (PW=0.20)", 40.0, 0.20),
        (f"A60     {CAND} added at x60 (PW=0.20)", 60.0, 0.20),
        (f"A40_p15 {CAND} added at x40 (PW=0.15)", 40.0, 0.15),
    ):
        lab = eval_config(extra_w=ew, pw=pw)
        d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
        ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
        any_ok |= ok
        err_f, err_d = int((lab["full"] != yv).sum()), int(((lab["full"] != yv) & dark).sum())
        ref_ef, ref_ed = int((ref["full"] != yv).sum()), int(((ref["full"] != yv) & dark).sum())
        print(
            f"  {vname}\n    "
            + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
            + f" | full err {ref_ef}->{err_f}, dark err {ref_ed}->{err_d}, flips(full) {(lab['full'] != ref['full']).sum()}\n"
            + f"    -> fold-0 rule {'PASS' if ok else 'fail'}"
        )

    print(f"\n-> {'at least one variant passes vs v41: folds 1-4 + full worth running' if any_ok else 'no variant passes vs v41: stop'}")


if __name__ == "__main__":
    main()
