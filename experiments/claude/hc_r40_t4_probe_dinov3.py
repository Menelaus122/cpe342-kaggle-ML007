"""Round 40, Task 4: v30's MLP probe with frozen DINOv3-L features added (laptop GPU; training images for the decision).

    python hc_r40_t4_probe_dinov3.py

User brief of 2026-11-18 (after hb_r40_t4_dinov3_check.py: frozen DINOv3-L @ 144 x 272 alone 0.9363 full / 0.8716 real dark against
DINOv2-L's 0.9050 / 0.8318 at its fine-tune geometry): add DINOv3-L to v30's MLP probe (35% of the final mix) and test against v30.
Features: vit_large_patch16_dinov3.lvd1689m, frozen, pooled embedding (its own 'avg' pooling), img_size 144 x 272, bicubic resize
of the 230 x 120 frame after the view transform (as the existing probe's embed), mean of the image and its h-flip; training images
under full (from hb_r40's cache, checked against a recomputation), zoom70 and HUD hidden; test images full frame (cached for a later
pipeline run; not used below). Caches in the existing layout: outputs/preds/t4_feat[_views]_dinov3L_144x272.npz.
Probes (scripts/train_task4_probe.fit_mlp, seeds 0, 1, 2, settings unchanged), fitted on the full / zoom70 / HUD-hidden rows of the
training folds, out of fold on the members' folds:  P7 = v30's seven backbones (7,040 features);  P8 = P7 + DINOv3-L (8,064).
Reference v30 = members as v30 (v23 weights; DINOv2-L x8, x2 on gated frames, gate thresholds per held-out fold as gy_r40) mixed
0.65 / 0.35 with P7 recomputed in this run (the cached P7 of gs_r39 is compared as a reproduction check), v17 rule.
Candidate (one, fixed before the run, weight not tuned): P8 in place of P7 at 0.35.
Acceptance (user's rule against v30): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. One pre-declared candidate, no nested choice.
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

V3, HW, TAG = "vit_large_patch16_dinov3.lvd1689m", (144, 272), "dinov3L_144x272"
FIT = ["full", "zoom70", "centre_grey"]
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)
train, test = task4.load_frames()
y = train.label.values.astype(np.int64)


def embed(model, mean, std, paths, views):
    feats = {v: [] for v in views}
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            arr = task4.load_uint8(list(paths[i:i + 4096]))
            for j in range(0, len(arr), 128):
                x0 = torch.from_numpy(arr[j:j + 128]).cuda().permute(0, 3, 1, 2).float().div_(255)
                for v in views:
                    x = (F.interpolate(P0.VIEWS[v](x0), size=HW, mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f = model(x).float() + model(torch.flip(x, dims=[3])).float()
                    feats[v].append((f / 2).cpu().numpy().astype(np.float16))
    return {v: np.concatenate(f) for v, f in feats.items()}


f_full, f_views = PRED_DIR / f"t4_feat_{TAG}.npz", PRED_DIR / f"t4_feat_views_{TAG}.npz"
if not (f_full.exists() and f_views.exists()):
    import timm
    model = timm.create_model(V3, pretrained=True, num_classes=0, img_size=HW).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    full = np.load("experiments/claude/outputs/r40/hb_feat_DINOv3-L_144x272.npy")
    chk = embed(model, mean, std, train.path.values[:256], ["full"])["full"]
    d = np.abs(chk.astype(np.float32) - full[:256].astype(np.float32)).max()
    log(f"full-frame cache of hb_r40 vs recomputation on 256 images: max |diff| {d:.2e}")
    assert d < 0.05, d
    if not f_views.exists():
        zv = embed(model, mean, std, train.path.values, ["zoom70", "centre_grey"])
        np.savez(f_views, full=full, **zv)
        log("training features, zoom70 + HUD hidden")
    if not f_full.exists():
        np.savez(f_full, train=full, test=embed(model, mean, std, test.path.values, ["full"])["full"],
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
    X = {v: np.hstack([B[t][v].astype(np.float32) for t in tags]) for v in FIT}
    out = {v: np.zeros((len(y), 5)) for v in FIT}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        p = lambda Q: ((Q - mu) / sd).astype(np.float32)
        A, Y = np.vstack([p(X[v][tr]) for v in FIT]), np.tile(y[tr], len(FIT))
        preds = [P0.fit_mlp(A, Y, s) for s in P0.MLP_SEEDS]
        del A
        torch.cuda.empty_cache()
        for v in FIT:
            out[v][va] = np.mean([q(p(X[v][va])) for q in preds], 0)
        del preds
    return out


# ---- v30's members on the three views (as gy_r40_t4_temp_alpha.py)
NEW = "dinov2L_hud70"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 1.0 for t in PLAIN}, **{t: 3.0 for t in HUDM}, "dinov2_v2": 8.0}
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
P = {t: {v: np.zeros((len(y), 5)) for v in FIT} for t in V20 + [NEW]}
for t in V20 + [NEW]:
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
    MEM[v] = (sum(w * P[t][v] for t, w in WM.items()) + wn[:, None] * P[NEW][v]) / (sum(WM.values()) + wn)[:, None]

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
for name, tags in (("P7", SEVEN), ("P8", SEVEN + [TAG]), ("DINOv3-L alone", [TAG])):
    PR[name] = oof(tags)
    log(f"probe alone, {name} ({sum(B[t]['full'].shape[1] for t in tags)} features): {alone(PR[name])}")
cached = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
print("P7 recomputed vs gs_r39's cache: max |diff| " + ", ".join(f"{v} {np.abs(PR['P7'][v] - cached[v]).max():.2e}" for v in FIT)
      + f"; v30 labels equal on full frames: {(mix(PR['P7'], 0.35)['full'] == mix(cached, 0.35)['full']).mean():.5f}", flush=True)
ref = mix(PR["P7"], 0.35)
print("v30 reference (members + 35% P7): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def report(name, lab, base):
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(base[VIEW_OF[k]], msk(k)) for k in SC}
    pf = [np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(base["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | "
          f"dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


print("\nCandidate against v30:")
ok = report("P8 (P7 + DINOv3-L) in place of P7, 35%", mix(PR["P8"], 0.35), ref)
print("Information: against v30 with gs_r39's cached P7")
report("P8 in place of the cached P7, 35%", mix(PR["P8"], 0.35), mix(cached, 0.35))
np.savez("experiments/claude/outputs/r40/hc_probe_views.npz", **{f"{k}|{v}": PR[k][v] for k in PR for v in FIT})
print(f"\n-> the candidate {'PASSES' if ok else 'fails'}")
log("done")
