"""Round 43, Task 4: Strict nested re-weighting round around v41 (14 FT members + P_7b @ 0.20; CPU, training OOF only).

    python experiments/claude/ij_r43_t4_reweight_v41.py

Grid fixed before the run (2 x 2 x 2 x 3 x 2 = 48 settings; v41 = H384 40, H144 40, D384 20, PW 0.20, OLD 1):
  H384   ViT-H+ 192x384 (dinov3Hp_384) weight on every frame        {40, 60}
  H144   ViT-H+ 144x272 (dinov3Hp_144) weight on every frame        {20, 40}
  D384   DINOv3-L 192x384 (dinov3L_384) weight on every frame       {10, 20}
  PW     P_7b probe weight in the final mix                         {0.15, 0.20, 0.25}
  OLD    older seven relative to v41 (cnxt/deit3 0.25, HUD x5 0.75) {x1, x0.5}
  fixed at v41: DINOv2-B x8, DINOv2-L x8/x2 gated, DINOv3-L-144 x10, C-RADIOv3-L x10/x2 gated; Q10 brightness/contrast gate.
Acceptance rule vs v41 (nested N): zoom70 > 0, centre_grey > 0, dark > 0, full >= -0.0005, and zoom/HUD mean > 0 in >= 3/5 folds.
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

L2, D3, D384, RAD, HP, HP384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144", "dinov3Hp_384"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
LOG_PATH = Path("experiments/claude/outputs/r42/ij_reweight_v41.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    train, _ = task4.load_frames()
    y = train.label.values.astype(np.int64)
    folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
    fo = np.zeros(len(y), int)
    for f, va in enumerate(folds):
        fo[va] = f

    P = {t: {v: np.zeros((len(y), 5), dtype=np.float64) for v in VIEWS} for t in V20 + [L2, D3, D384, RAD, HP, HP384]}
    for t in P:
        for f, va in enumerate(folds):
            z30 = np.load(V30 / f"{t}_fold{f}.npz")
            z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
            for v in VIEWS:
                P[t][v][va] = (z30 if v in z30.files else z19)[v]

    z3 = np.load("experiments/claude/outputs/r42/ii_probe_v3_views.npz")
    PR = {v: z3[f"p7b_{v}"].astype(np.float64) for v in VIEWS}
    ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    GATE = {v: np.zeros(len(y), bool) for v in VIEWS}
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        for v in VIEWS:
            GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)

    st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
    dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(len(y), bool)
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m

    def labels(h384, h144, d384, pw, old):
        w = {**{t: 0.25 * old for t in PLAIN}, **{t: 0.75 * old for t in HUDM}, "dinov2_v2": 8.0, D3: 10.0, D384: d384, HP: h144, HP384: h384}
        gated = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}
        out = {}
        for v in VIEWS:
            num = sum(wt * P[t][v] for t, wt in w.items())
            den = np.full(len(y), sum(w.values()))
            for t, (wn, wd) in gated.items():
                ww = np.where(GATE[v], wd, wn)
                num, den = num + ww[:, None] * P[t][v], den + ww
            out[v] = rule((1 - pw) * num / den[:, None] + pw * PR[v])
        return out

    V41 = (40.0, 40.0, 20.0, 0.20, 1.0)
    GRID = list(itertools.product((40.0, 60.0), (20.0, 40.0), (10.0, 20.0), (0.15, 0.20, 0.25), (1.0, 0.5)))
    assert V41 in GRID and len(GRID) == 48
    ref = labels(*V41)
    ref_sc = {k: bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    log("v41 reference: " + ", ".join(f"{k} {ref_sc[k]:.4f}" for k in SC))
    for k, exp_v in {"full": 0.9959, "zoom70": 0.9897, "centre_grey": 0.9826, "dark/low-contrast": 0.9901}.items():
        assert abs(ref_sc[k] - exp_v) <= 1.5e-4, f"v41 mismatch on {k}: got {ref_sc[k]:.5f}, expected {exp_v:.4f}"

    name = lambda s: f"H384 {s[0]:g} H144 {s[1]:g} D384 {s[2]:g} PW {s[3]:.2f} OLD x{s[4]:g}"
    LAB = {s: labels(*s) for s in GRID}
    log(f"{len(GRID)} settings computed")

    deltas = lambda lab, m=allm: {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}
    meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    crit = lambda d: float(np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]))

    def report(nm, lab):
        d = deltas(lab)
        pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
        pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
        ok = meets(d) and sum(p > 0 for p in pf) >= 3
        log(
            f"  {nm}\n    "
            + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
            + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
            + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}"
        )
        return ok

    outN, picks = {v: ref[v].copy() for v in VIEWS}, []
    for f in range(5):
        okk = {s: crit(d) for s, d in ((s, deltas(LAB[s], fo != f)) for s in GRID if s != V41) if meets(d)}
        s = max(okk, key=okk.get) if okk else None
        picks.append(s)
        if s is not None:
            for v in VIEWS:
                outN[v][fo == f] = LAB[s][v][fo == f]

    log("\nNested choice per held-out fold:")
    for f, s in enumerate(picks):
        log(f"  fold {f}: {name(s) if s else 'v41 (no setting meets the rule)'}")

    log("\nCandidate (decision):")
    okN = report("N nested over the 48-setting grid", outN)

    ins = sorted(((crit(deltas(LAB[s])), s) for s in GRID if s != V41 and meets(deltas(LAB[s]))), reverse=True)
    log("\nInformation: top 5 in-sample settings meeting the rule")
    for c, s in ins[:5]:
        report(f"{name(s)} (criterion {c:+.4f})", LAB[s])

    log("\nInformation: one factor at a time from v41")
    for i, alt in enumerate([(60.0,), (20.0,), (10.0,), (0.15, 0.25), (0.5,)]):
        for a in alt:
            s = list(V41)
            s[i] = a
            report(name(tuple(s)), LAB[tuple(s)])

    best = ins[0][1] if ins else None
    EDGE = {"H384": (0, 60.0), "H144": (1, 20.0), "D384": (2, 10.0), "PW": (3, 0.25), "OLD": (4, 0.5)}
    edge = None if best is None else [nm for nm, (i, far) in EDGE.items() if best[i] == far]

    hp384_lab = {v: rule(P[HP384][v]) for v in VIEWS}
    log("\nVote-out summary:")
    rows_diag = [("v41", ref), ("N nested", outN)] + ([("top in-sample", LAB[best])] if best else [])
    for label_nm, lab in rows_diag:
        vo_f = int(((hp384_lab["full"] == y) & (lab["full"] != y)).sum())
        res_f = int(((hp384_lab["full"] != y) & (lab["full"] == y)).sum())
        vo_d = int(((hp384_lab["full"] == y) & (lab["full"] != y) & dark).sum())
        res_d = int(((hp384_lab["full"] != y) & (lab["full"] == y) & dark).sum())
        err_f = int((lab["full"] != y).sum())
        err_d = int(((lab["full"] != y) & dark).sum())
        log(f"  {label_nm:18s}: full err={err_f:3d} (Hp384 vote-out={vo_f:2d}, rescued={res_f:2d}) | dark err={err_d:2d} (vote-out={vo_d:2d}, rescued={res_d:2d})")

    log(f"\n-> N {'PASSES' if okN else 'fails'}; pipeline setting if N passes: {name(best) if best else None}"
        + (f" (at the far grid edge in: {', '.join(edge)})" if edge else ""))
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
