"""Round 42 part A, Task 4: v33's MLP probe with frozen C-RADIOv3-L features added (laptop GPU, 0 Colab CU; training images for the
decision).

    python ib_r42_t4_probe_radio.py

User brief of 2026-11-19 (after ia_r42_t4_radio_frozen.py: frozen C-RADIOv3-L summary @ 192 x 384 alone 0.9543 full / 0.9027 real
dark against DINOv3-L's 0.9400 / 0.8831 at the same geometry). Written before the run; nothing below is tuned after it.
Features: nvidia/C-RADIOv3-L (revision 9d041346, transformers AutoModel + trust_remote_code), frozen, `summary` (3072-d: class tokens
of the CLIP, SigLIP 2 and DINOv2 teachers), input [0, 1] RGB (no ImageNet normalisation: the model normalises inside with the CLIP
mean / std), bicubic resize of the 230 x 120 frame to 192 x 384 after the view transform (as the probe's other backbones), bf16,
mean of the image and its h-flip; training images under full (ia_r42's cache, checked against a recomputation), zoom70 and HUD hidden;
test images full frame (cached for a later pipeline run; not used below). Caches in the existing layout:
outputs/preds/t4_feat[_views]_cradio3L_192x384.npz.
Probes (scripts/train_task4_probe.fit_mlp, seeds 0, 1, 2, settings unchanged), fitted on the full / zoom70 / HUD-hidden rows of the
training folds, out of fold on the members' folds: P7 = v33's seven backbones (7,040 features); P8 = P7 + C-RADIOv3-L (10,112).
Reference v33 = members as v33 (older seven x0.5: ConvNeXt-T / DeiT-III 0.5, HUD members 1.5; DINOv2-B 8; DINOv2-L 8, 2 on gated
frames with thresholds from the other four folds; both DINOv3-L members x20 on every frame) mixed 0.65 / 0.35 with P7 recomputed in
this run (gs_r39's cached P7 compared as a reproduction check), v17 rule, balanced class mix.
Candidates (fixed now; the user asked for v33's weight and slightly higher ones): P8 in place of P7 at probe weight 0.35, 0.40, 0.45.
  N = per held-out fold, among the candidates meeting the rule on the other four folds, the best mean of the three hard deltas; none: v33.
Acceptance (user's rule against v33, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline setting is the candidate meeting the rule in-sample with
the best criterion on all five folds. Information only (not candidates): P7 at 0.40 / 0.45 (the weight without the new feature);
the C-RADIOv3-L probe alone.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402

REPO, REV, HW, TAG = "nvidia/C-RADIOv3-L", "9d0413465e8a91e67bbf2c1ad342815478d1b906", (192, 384), "cradio3L_192x384"
FIT = ["full", "zoom70", "centre_grey"]
OUT = Path("experiments/claude/outputs/r42")
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)
train, test = task4.load_frames()
y = train.label.values.astype(np.int64)


def embed(model, paths, views, bs=64):
    feats = {v: [] for v in views}
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            arr = task4.load_uint8(list(paths[i:i + 4096]))
            for j in range(0, len(arr), bs):
                x0 = torch.from_numpy(arr[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255)
                for v in views:
                    x = F.interpolate(P0.VIEWS[v](x0), size=HW, mode="bicubic", align_corners=False).clamp_(0, 1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f = model(x)[0].float() + model(torch.flip(x, dims=[3]))[0].float()
                    feats[v].append((f / 2).cpu().numpy().astype(np.float16))
    return {v: np.concatenate(f) for v, f in feats.items()}


f_full, f_views = PRED_DIR / f"t4_feat_{TAG}.npz", PRED_DIR / f"t4_feat_views_{TAG}.npz"
if not (f_full.exists() and f_views.exists()):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, revision=REV).cuda().eval()
    full = np.load(OUT / "ia_feat_C-RADIOv3-L_summary_192x384.npy")
    chk = embed(model, train.path.values[:256], ["full"])["full"]
    d = np.abs(chk.astype(np.float32) - full[:256].astype(np.float32)).max()
    log(f"full-frame cache of ia_r42 vs recomputation on 256 images: max |diff| {d:.2e}")
    assert d < 0.05, d
    if not f_views.exists():
        zv = embed(model, train.path.values, ["zoom70", "centre_grey"])
        np.savez(f_views, full=full, **zv)
        log("training features, zoom70 + HUD hidden")
    if not f_full.exists():
        np.savez(f_full, train=full, test=embed(model, test.path.values, ["full"])["full"],
                 test_file_name=np.array(test.file_name.tolist(), dtype="U"))
        log("test features")
    del model
    torch.cuda.empty_cache()

B = {}
for tag in [b[2] for b in P0.BACKBONES] + [TAG]:
    z, zv = np.load(PRED_DIR / f"t4_feat_{tag}.npz"), np.load(PRED_DIR / f"t4_feat_views_{tag}.npz")
    B[tag] = {"full": z["train"], "zoom70": zv["zoom70"], "centre_grey": zv["centre_grey"]}
    assert len(z["train"]) == len(y)
SEVEN = [b[2] for b in P0.BACKBONES]
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f


def oof(tags):
    # features stay float16 as cached (half the RAM of float32; the first run was stopped for low system memory); the
    # standardisation below works in float32 exactly as before
    X = {v: np.hstack([B[t][v] for t in tags]) for v in FIT}
    out = {v: np.zeros((len(y), 5)) for v in FIT}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        xf = X["full"][tr].astype(np.float32)
        mu, sd = xf.mean(0), xf.std(0) + 1e-6
        del xf
        p = lambda Q: ((Q - mu) / sd).astype(np.float32)
        A, Y = np.empty((len(tr) * len(FIT), X["full"].shape[1]), np.float32), np.tile(y[tr], len(FIT))
        for i, v in enumerate(FIT):          # = np.vstack([p(X[v][tr]) for v in FIT]) without the list of copies
            A[i * len(tr):(i + 1) * len(tr)] = p(X[v][tr])
        preds = [P0.fit_mlp(A, Y, s) for s in P0.MLP_SEEDS]
        del A
        torch.cuda.empty_cache()
        for v in FIT:
            out[v][va] = np.mean([q(p(X[v][va])) for q in preds], 0)
        del preds
    return out


# ---- v33's members on the three views (as hi_r41_t4_dinov3_384_eval.py)
L2, D3, D384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 0.5 for t in PLAIN}, **{t: 1.5 for t in HUDM}, "dinov2_v2": 8.0, D3: 20.0, D384: 20.0}
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
P = {t: {v: np.zeros((len(y), 5)) for v in FIT} for t in V20 + [L2, D3, D384]}
for t in P:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in FIT:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
MEM = {}
for v in FIT:
    gate = np.zeros(len(y), bool)
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        gate[ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)
    wn = np.where(gate, 2.0, 8.0)
    MEM[v] = (sum(w * P[t][v] for t, w in WM.items()) + wn[:, None] * P[L2][v]) / (sum(WM.values()) + wn)[:, None]

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
mix = lambda PR, w: {v: rule((1 - w) * MEM[v] + w * PR[v]) for v in FIT}
alone = lambda PR: ", ".join(f"{k} {bal(rule(PR[VIEW_OF[k]]), msk(k)):.4f}" for k in SC)

PR = {}
for name, tags in (("P7", SEVEN), ("P8", SEVEN + [TAG]), ("C-RADIOv3-L alone", [TAG])):
    PR[name] = oof(tags)
    log(f"probe alone, {name} ({sum(B[t]['full'].shape[1] for t in tags)} features): {alone(PR[name])}")
cached = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
print("P7 recomputed vs gs_r39's cache: max |diff| " + ", ".join(f"{v} {np.abs(PR['P7'][v] - cached[v]).max():.2e}" for v in FIT)
      + f"; v33 labels equal on full frames: {(mix(PR['P7'], 0.35)['full'] == mix(cached, 0.35)['full']).mean():.5f}", flush=True)
ref = mix(PR["P7"], 0.35)
print("v33 reference (members + 35% P7): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | "
          f"dark by fold {' '.join(f'{p:+.4f}' for p in pdk)} | labels changed (full) {(lab['full'] != ref['full']).sum()}\n"
          f"    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {f"P8 @ {w:.2f}": mix(PR["P8"], w) for w in (0.35, 0.40, 0.45)}
print("\nCandidates against v33 (in-sample):")
for k, lab in C.items():
    report(k, lab)
meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
crit = lambda d: np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])
outN, picks = {v: ref[v].copy() for v in FIT}, []
for f in range(5):
    okk = {k: crit(d) for k, d in ((k, deltas(lab, fo != f)) for k, lab in C.items()) if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in FIT:
            outN[v][fo == f] = C[s][v][fo == f]
print("Candidate:")
ok = report(f"N nested over {list(C)}; picks {picks}", outN)
print("Information (not candidates):")
for w in (0.40, 0.45):
    report(f"P7 @ {w:.2f} (probe weight only, no new feature)", mix(PR["P7"], w))
ins = {k: crit(deltas(lab)) for k, lab in C.items() if meets(deltas(lab))}
np.savez(OUT / "ib_probe_views.npz", **{f"{k}|{v}": PR[k][v] for k in PR for v in FIT})
print(f"\n-> N {'PASSES' if ok else 'fails'}; pipeline setting if N passes: {max(ins, key=ins.get) if ins else None}")
log("done")
