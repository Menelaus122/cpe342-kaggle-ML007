"""Round 37, Task 4: WiSE-FT for the DINOv2 member (training images only; GPU inference, no training).

    python gh_r37_t4_wiseft.py

Scoping (FINDINGS, round 37): the Task 4 gap between out-of-fold (0.989) and test is a train / test shift, and the leaderboard
rewarded the self-supervised DINOv2 backbone far more than the training-image proxies predicted. Fine-tuning moves the pretrained
features towards what separates the training frames (e.g. the HUD); WiSE-FT (Wortsman et al., CVPR 2022) interpolates the weights
    theta(alpha) = alpha * theta_fine-tuned + (1 - alpha) * theta_pretrained
and, for models fine-tuned from a pretrained start, recovers much of the robustness lost in fine-tuning at no training cost.
Here: each fold's fine-tuned dinov2_v2 (EMA weights) with the pretrained vit_base_patch14_reg4_dinov2.lvd142m weights at the same
126x238 token grid (exactly the fine-tuning start); every backbone tensor interpolated, the classifier head kept from the
fine-tuned model (the pretrained model has none). alpha in {0.5, 0.7, 0.85, 0.95}; alpha = 1 is the v23 member.
Each fold's interpolated model predicts its own validation images under the five scoring views (full, zoom70, HUD hidden, gamma 3.5,
75% grid; h-flip averaged, as em_r30_t4_hardviews.py), written to experiments/claude/outputs/r37_views/.
Scoring as fc_r31_t4_v23_eval.py: v23 ensemble (weights DINOv2 x8, HUD members x3, others x1; v17 rule; uniform class mix) with
dinov2_v2 replaced by the WiSE-FT model at weight 8; also each alone, and alpha chosen nested (on four folds by the hard-view mean
among the alphas with full >= -0.0005 there; alpha = 1 allowed). Rules fixed before the run: the hard-view rule (hard-view mean >=
+0.0015, full >= -0.0005), the generic rule, and the v24 check (zoom70, HUD hidden and real dark / low-contrast frames > 0, full >=
-0.0005).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

NAME, SIZE = "vit_base_patch14_reg4_dinov2.lvd142m", (126, 238)
ALPHAS = (0.5, 0.7, 0.85, 0.95)
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ALL5[1:]
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
V19, V30, V37 = (Path(f"experiments/claude/outputs/{d}") for d in ("r19_views", "r30_views", "r37_views"))
V37.mkdir(parents=True, exist_ok=True)
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey,
      "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75}

# ------------------------------------------------------------------------------------------------------------------ GPU phase
todo = [(a, f) for a in ALPHAS for f in range(5) if not (V37 / f"dinov2_wise{a}_fold{f}.npz").exists()]
if todo:
    pre = {k: v.float() for k, v in task4.create_model(NAME, pretrained=True, img_size=SIZE).state_dict().items()}
    for f in range(5):
        alphas = [a for a, ff in todo if ff == f]
        if not alphas:
            continue
        ft = torch.load(MODEL_DIR / f"t4_dinov2_v2_fold{f}.pt", map_location="cpu")
        assert set(ft) == set(pre), "state-dict keys differ"
        arr = task4.load_uint8(list(train.path.values[folds[f]]))
        for a in alphas:
            sd = {k: (v if k.startswith("head.") else (a * v.float() + (1 - a) * pre[k]).to(v.dtype)) for k, v in ft.items()}
            m = task4.create_model(NAME, pretrained=False, img_size=SIZE).cuda().to(memory_format=torch.channels_last)
            m.load_state_dict(sd)
            m.eval()
            res = {v: [] for v in ALL5}
            with torch.no_grad():
                for i in range(0, len(arr), 128):
                    x0 = task4.to_gpu_batch(arr[i:i + 128], SIZE)
                    for v in ALL5:
                        x = FN[v](x0)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                                       + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                        res[v].append(p.cpu().numpy())
            np.savez(V37 / f"dinov2_wise{a}_fold{f}.npz", **{v: np.concatenate(r) for v, r in res.items()})
            acc = np.mean(np.concatenate(res["full"]).argmax(1) == y[folds[f]])
            print(f"fold {f}, alpha {a}: views written; full-view accuracy {acc:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        del arr, ft

# ------------------------------------------------------------------------------------------------------------------ CPU phase
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
W = {a: {v: np.zeros((len(y), 5)) for v in ALL5} for a in ALPHAS}
for a in ALPHAS:
    for f, va in enumerate(folds):
        z = np.load(V37 / f"dinov2_wise{a}_fold{f}.npz")
        for v in ALL5:
            W[a][v][va] = z[v]
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
SCORE = ALL5 + ["dark/low-contrast"]
VIEW_OF = {**{v: v for v in ALL5}, "dark/low-contrast": "full"}
allm = np.ones(len(y), bool)
rule = lambda p: (p * pi ** -0.75).argmax(1)
others = {t: w for t, w in V23_W.items() if t != "dinov2_v2"}


def ens_labels(dmap):
    return {v: rule((sum(w * P[t][v] for t, w in others.items()) + 8.0 * dmap[v]) / (sum(others.values()) + 8.0)) for v in ALL5}


base = ens_labels(P["dinov2_v2"])
LAB = {a: ens_labels(W[a]) for a in ALPHAS}
print("v23 (uniform mix, rule): " + ", ".join(f"{k} {bal(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE))
print("member alone (rule): alpha 1.00 (dinov2_v2) " + ", ".join(f"{k} {bal(rule(P['dinov2_v2'][VIEW_OF[k]]), dark if k == 'dark/low-contrast' else allm):.4f}"
                                                                 for k in SCORE))
for a in ALPHAS:
    print(f"                    alpha {a:.2f}            " + ", ".join(f"{k} {bal(rule(W[a][VIEW_OF[k]]), dark if k == 'dark/low-contrast' else allm):.4f}"
                                                                      for k in SCORE))


def report(name, lab):
    d = {k: (lambda m: bal(lab[VIEW_OF[k]], m) - bal(base[VIEW_OF[k]], m))(dark if k == "dark/low-contrast" else allm) for k in SCORE}
    hard = np.mean([d[v] for v in HARD])
    pf = [np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(base["full"], (fo == f) & dark) for f in range(5)]
    ok_user = hard >= 0.0015 and d["full"] >= -0.0005
    ok_gen = hard > 0 and sum(p > 0 for p in pf) >= 3 and min(pf) >= -0.0005
    ok_v24 = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    print(f"  {name}\n    hard-view mean {hard:+.4f}, folds +{sum(p > 0 for p in pf)}/5 ({' '.join(f'{p:+.4f}' for p in pf)}); hard mean without the grid "
          f"{np.mean([d[v] for v in HARD[:-1]]):+.4f}\n    per view: " + ", ".join(f"{k} {d[k]:+.4f}" for k in SCORE)
          + f" | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> hard-view rule {'PASS' if ok_user else 'fail'}, generic rule "
          f"{'PASS' if ok_gen else 'fail'}, v24 check (beyond the grid) {'PASS' if ok_v24 else 'fail'}", flush=True)


for a in ALPHAS:
    report(f"WiSE-FT alpha {a:.2f} in place of dinov2_v2 (weight 8)", LAB[a])
nested = {v: base[v].copy() for v in ALL5}
picks = []
for f in range(5):
    m = fo != f
    sc = {a: np.mean([bal(LAB[a][v], m) - bal(base[v], m) for v in HARD]) for a in ALPHAS
          if bal(LAB[a]["full"], m) - bal(base["full"], m) >= -0.0005}
    a = max(sc, key=sc.get) if sc and max(sc.values()) > 0 else 1.0
    picks.append(a)
    if a != 1.0:
        for v in ALL5:
            nested[v][fo == f] = LAB[a][v][fo == f]
report("alpha chosen nested", nested)
print(f"  nested picks: {picks}")
