"""Write colab/t4_colab_train.ipynb (run from the repo root):  python colab/build_t4_colab_nb.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# CPE342 Task 4: GPU training on Colab (round 39 step 3: DINOv2-L with HUD dropout 0.70; fold 0 + gate, then folds 1-4 + full refit)

Runs the same trainer as the laptop (`scripts/train_task4.py`, identical folds) on a bigger GPU. Training images only; no test
image or test statistic enters any decision (the full refit only *predicts* the test images, as on the laptop).

**Recipe (user brief of 2026-11-17 evening, round 39 step 3; `LP_FT = False`):** DINOv2 ViT-L/14 at 126×238, plain fine-tuning (the
head starts from scratch, as every laptop member), **lr 3e-5**, layer-wise lr decay 0.85, 2 warm-up + 13 cosine epochs, **HUD dropout
0.70** (round 38's LP-FT run lost on HUD-hidden frames; a stronger HUD dropout targets exactly that), batch 64, tag `dinov2L_hud70`.
No WiSE-FT (`ALPHAS = []`): without an LP head θ0 would carry a random head.

**Round 38 recipe (`LP_FT = True`, kept for reference):** DINOv2 ViT-L/14 at 126×238 with **LP-FT** (Kumar et al., ICLR 2022): for each fold a
logistic probe head is first fitted on the frozen pretrained features of the fold's training images (full / 70% zoom / HUD-hidden
views, C = 0.03 as the frozen probe), and fine-tuning starts from that head (lr 2e-5, layer decay 0.85, 2 warm-up + 13 cosine
epochs, HUD dropout 0.5). Then **full WiSE-FT** (Wortsman et al., CVPR 2022): every weight, head included, interpolated between the
fine-tuned model and θ0 = pretrained backbone + LP head, α ∈ {0.5, 0.7, 0.85} (α = 1 is the fine-tuned model, α = 0 the frozen
backbone with the LP head; `experiments/claude/gn_r38_t4_wiseft_full.py`).

**Frozen features (first cell after the packages):** the backbones in `EXTRACT` for the super-probe
(`experiments/claude/gm_r38_t4_feat_extract.py --only <tag>`, inference only: training images under the five scoring views, test
images full frame). Round 38: EVA-02-L and ConvNeXt-L CLIP; round 39 step 2: OpenCLIP ViT-H/14 and Swin-L. Set `EXTRACT = []` to skip
it. With `TRAIN = False` (round 39) nothing else runs: no benchmark, no training.

**Before running**
1. On the laptop: `python scripts/make_colab_bundle.py` → `colab/upload/cpe342_code.zip` (≈34 MB) and `colab/upload/t4_data.zip`
   (≈0.6 GB). Upload both to Google Drive, folder `MyDrive/cpe342-colab/` (replace an older `cpe342_code.zip`). Keep the folder
   private (competition data).
2. Colab: *Runtime → Change runtime type → A100* (or L4 / H100). A T4 has no bf16 and is slower than the laptop: do not use it.
3. *Runtime → Run all*. Keep the browser tab open (Colab Pro stops when the tab closes or after long idle).

**Resuming:** everything is written to `MyDrive/cpe342-colab/out/` as soon as it exists, and finished folds are skipped. If the
session ends, reconnect and *Run all* again; it continues with the next unfinished step.

**Gate (fixed before the run; round 39 step 3, user's rule):** after fold 0 the five views of the fine-tuned model are scored against
the v23 members on fold 0 (`fc_r31_t4_v23_eval.py`); folds 1-4 and the full refit run only if it passes, in one fixed-weight variant,
the **v24 check: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005** (`GATE_NEEDS`; the hard-view
rule is printed but no longer required). Otherwise the notebook stops. Round 38 (LP-FT) required the hard-view rule as well and also
scored the WiSE-FT points. After all folds the member is judged on the laptop against v29 (members + 35% probe) under the usual rule.

**When it is done:** copy from `MyDrive/cpe342-colab/out/` to the laptop repo (the last cell lists the files):
`preds/t4_<TAG>*.npz` → `outputs/preds/`; `r30_views/<TAG>*.npz` → `experiments/claude/outputs/r30_views/`;
`logs/*<TAG>*` → `outputs/logs/`; the feature caches `preds/t4_feat_views_<tag>.npz` and `preds/t4_feat_<tag>.npz` of the
`EXTRACT` backbones → `outputs/preds/` (needed by `go_r38_t4_superprobe.py`). The checkpoints (`models/`, 1.2 GB each) stay on Drive. Whether the member (and, with
LP-FT, which α) enters the ensemble is decided on the laptop from the five folds, under the rules above.
""")

code(r"""
# ---- settings -------------------------------------------------------------------------------------------------------------
EXTRACT = []                  # frozen-feature caches made here first ([] = skip); round 39 step 2 used ["vitH14_clip_laion2b_224", "swinL_in22k_224"]
TRAIN = True                  # False: only the feature cell runs; True: benchmark + training as below
TAG = "dinov2L_hud70"         # round 39 step 3 (round 38: "dinov2L_lpft")
LP_FT = False                 # round 38: True (LP head first, then fine-tuning from it, plus WiSE-FT points)
LR, P_HUD = "3e-5", "0.70"    # round 38: 2e-5, 0.5
GATE_NEEDS = ["v24 check (beyond the grid) PASS"]   # round 38 also needed "hard-view rule PASS"
FOLDS = [0, 1, 2, 3, 4]       # fold 0 first; folds 1-4 only if fold 0 passes the gate
GATE = True                   # False: train every fold without the fold-0 gate
DO_FULL = True                # full Train+Val refit -> test predictions (the member's test input)
DRIVE_DIR = "/content/drive/MyDrive/cpe342-colab"   # the team's Drive folder
MODELS_ON_DRIVE = True        # 1.2 GB per checkpoint (about 6 GB in all); False keeps them on the Colab disk (lost with the session)
GRAD_CKPT = "auto"            # gradient checkpointing changes memory only, not the result: auto = on below 60 GB of GPU memory
ACCUM = 1                     # 64 images per optimiser step in one pass (2 = 2 x 32, same effective batch, for a small GPU)
ALPHAS = [0.0, 0.5, 0.7, 0.85] if LP_FT else []    # WiSE-FT points (alpha = 1 is the fine-tuned model itself); LP-FT only
MEMBER_ALPHAS = [0.5, 0.7, 0.85] if LP_FT else []  # after all folds: member files (val + test probabilities) for these alphas
MODEL = "vit_large_patch14_reg4_dinov2.lvd142m"
RECIPE = ["--model", MODEL, "--h", "126", "--w", "238", "--bs", "64", "--accum", str(ACCUM), *(["--lp-ft"] if LP_FT else []),
          "--lr", LR, "--layer-decay", "0.85", "--warmup-epochs", "2", "--epochs", "15", "--p-hud", P_HUD]
print("recipe:", " ".join(RECIPE))
""")

code(r"""
# ---- GPU check ------------------------------------------------------------------------------------------------------------
import subprocess
import torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
cap = torch.cuda.get_device_capability(0)
print(torch.cuda.get_device_name(0), "compute capability", cap, "| torch", torch.__version__)
assert cap[0] >= 8, "this GPU has no native bf16 (T4?): change the runtime type to A100, L4 or H100"
mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
if GRAD_CKPT == "auto":
    GRAD_CKPT = mem_gb < 60
RECIPE = [a for a in RECIPE if a != "--grad-ckpt"] + (["--grad-ckpt"] if GRAD_CKPT else [])
print(f"{mem_gb:.0f} GiB GPU memory -> gradient checkpointing {'on' if GRAD_CKPT else 'off'}")
""")

code(r"""
# ---- Drive, code and data --------------------------------------------------------------------------------------------------
import os
import shutil
import zipfile
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
D, R = Path(DRIVE_DIR), Path("/content/cpe342")
for z in ("cpe342_code.zip", "t4_data.zip"):
    assert (D / z).exists(), f"upload {z} to {DRIVE_DIR}"
if not (R / "src" / "task4.py").exists():
    zipfile.ZipFile(D / "cpe342_code.zip").extractall(R)
DATA = R / "04_image_classification/_download/task4_image_classification"
if not (DATA / "test.csv").exists() or len(os.listdir(DATA / "test")) < 25889:
    shutil.copy(D / "t4_data.zip", "/content/t4_data.zip")          # local copy first: unzipping straight from Drive is slow
    zipfile.ZipFile("/content/t4_data.zip").extractall(R)
    os.remove("/content/t4_data.zip")
print(len(os.listdir(DATA / "train")), "training images,", len(os.listdir(DATA / "test")), "test images")

# outputs live on Drive, so they survive the session; the fold indices from the bundle are copied over first
persist = {"outputs/preds": "preds", "outputs/logs": "logs", "experiments/claude/outputs/r30_views": "r30_views"}
if MODELS_ON_DRIVE:
    persist["outputs/models"] = "models"
for local, remote in persist.items():
    src_dir, dst = R / local, D / "out" / remote
    dst.mkdir(parents=True, exist_ok=True)
    if src_dir.is_symlink():
        continue
    if src_dir.exists():
        for f in src_dir.iterdir():
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
        shutil.rmtree(src_dir)
    src_dir.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(dst, src_dir)
print({k: sorted(p.name for p in (D / "out" / v).iterdir())[:8] for k, v in persist.items()})
""")

code(r"""
# ---- packages (timm as on the laptop) --------------------------------------------------------------------------------------
!pip -q install timm==1.0.30
import numpy, pandas, sklearn, timm, torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__, "| sklearn", sklearn.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__)
""")

code(r"""
# ---- round 38 step 1: frozen features for the super-probe (inference only; written to Drive as soon as each is done) ----------
import subprocess
import time

def stream(args):
    print("$ python", " ".join(args), flush=True)
    p = subprocess.Popen(["python", "-u", *args], cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    for line in p.stdout:
        if "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    return p.wait()

for tag in EXTRACT:
    caches = [R / f"outputs/preds/t4_feat_views_{tag}.npz", R / f"outputs/preds/t4_feat_{tag}.npz"]
    if all(c.exists() for c in caches):
        print(f"{tag}: cached on Drive")
        continue
    t1 = time.time()
    assert stream(["experiments/claude/gm_r38_t4_feat_extract.py", "--only", tag]) == 0, f"feature extraction {tag} failed"
    print(f"{tag}: {(time.time() - t1) / 60:.1f} min; " + ", ".join(f"{c.name} {c.stat().st_size / 1e6:.0f} MB" for c in caches))
""")

code(r"""
# ---- 1-epoch benchmark on 1,408 training images (LP head step with LP_FT; also downloads the DINOv2-L weights, 1.2 GB) ------
import sys
import time
import numpy as np
os.chdir(R)
sys.path.insert(0, str(R))
from src import task4

tr_df, _ = task4.load_frames()
idx = np.arange(1408)
arr = task4.load_uint8(list(tr_df.path.values[idx]))
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
head = None
if LP_FT:
    t_lp = time.time()
    m0 = task4.create_model(MODEL, img_size=(126, 238)).cuda().to(memory_format=torch.channels_last)
    head = task4.fit_lp_head(m0, arr, tr_df.label.values, np.arange(1280), (126, 238))
    vp0 = task4.predict_array(task4.set_head(m0, head), arr[1280:], (126, 238))
    print(f"LP head on 1,280 images in {time.time() - t_lp:.0f}s; theta_0 accuracy on 128 others "
          f"{np.mean(vp0.argmax(1) == tr_df.label.values[1280:1408]):.3f}")
    del m0
m, h = task4.train_fold(MODEL, arr, tr_df.label.values[idx], np.arange(1280), np.arange(1280, 1408),
                        img_size=(126, 238), epochs=1, bs=64, lr=float(LR), layer_decay=0.85, aug_kw={"p_hud": float(P_HUD)}, grad_ckpt=GRAD_CKPT,
                        warmup_epochs=1, accum=ACCUM, init_head=head)
per_img = h[-1]["secs"] / 1280
print(f"peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB; {per_img * 1000:.1f} ms per training image (first epoch, "
      f"includes warm-up) -> about {per_img * 25237 / 60:.0f} min per fold epoch, {per_img * 25237 * 15 / 3600:.1f} h per fold "
      f"(laptop: 656 s per epoch, 2.7 h per fold), {per_img * 25237 * 15 * (len(FOLDS) + 1.25 * DO_FULL) / 3600:.1f} h for this run")
del m, arr
torch.cuda.empty_cache()
""")

code(r"""
# ---- training: folds, their views, then the full refit -----------------------------------------------------------------------
import subprocess

def run(args, extra_env=None):
    print("$ python", " ".join(args), flush=True)
    p = subprocess.Popen(["python", "-u", *args], cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8", **(extra_env or {})})
    for line in p.stdout:
        if "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    rc = p.wait()
    print(f"exit {rc} [{time.strftime('%H:%M')}]", flush=True)
    return rc

WISE = [f"{TAG}_wise{round(a * 100):02d}" for a in ALPHAS]           # e.g. dinov2L_lpft_wise70

def score(cands, name):
    # v23-based scores of the folds with views (fc_r31_t4_v23_eval.py), saved to Drive
    out = subprocess.run(["python", "-u", "experiments/claude/fc_r31_t4_v23_eval.py", *cands], cwd=R, capture_output=True, text=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    (D / "out" / "logs" / f"fc_{TAG}_{name}_colab.log").write_text(out, encoding="utf-8")
    print("\n".join(l for l in out.splitlines() if "Warning" not in l))
    return out

def gate_passed():
    # PASS if the fine-tuned model (or, with LP-FT, a WiSE-FT point with alpha > 0) passes every rule in GATE_NEEDS in one variant
    out = score([TAG] + [w for w, a in zip(WISE, ALPHAS) if a > 0], "fold0_gate")
    if 0.0 in ALPHAS:
        score([w for w, a in zip(WISE, ALPHAS) if a == 0], "fold0_theta0_info")
    return any(all(g in l for g in GATE_NEEDS) for l in out.splitlines())

t_start = time.time()
for f in FOLDS:
    if not (R / f"outputs/preds/t4_{TAG}_fold{f}.npz").exists():
        assert run(["scripts/train_task4.py", "--tag", TAG, *RECIPE, "--folds", str(f)]) == 0, f"fold {f} failed"
    if not (R / f"experiments/claude/outputs/r30_views/{TAG}_fold{f}.npz").exists():
        # the five scoring views of this fold's validation images (full, zoom70, HUD hidden, gamma 3.5, 75% grid)
        assert run(["experiments/claude/em_r30_t4_hardviews.py", TAG], {"VIEWS_ONLY": "1"}) == 0, f"views of fold {f} failed"
    if WISE and not all((R / f"experiments/claude/outputs/r30_views/{w}_fold{f}.npz").exists() for w in WISE):
        assert run(["experiments/claude/gn_r38_t4_wiseft_full.py", TAG, "--model", MODEL, "--h", "126", "--w", "238",
                    "--alphas", *map(str, ALPHAS), "--folds", str(f)]) == 0, f"WiSE-FT views of fold {f} failed"
    if f == 0 and GATE:
        ok = gate_passed()
        print(f"\nfold-0 gate: {'PASS -> folds 1-4 and the full refit' if ok else 'FAIL -> stop (no further compute units spent)'}")
        if not ok:
            break
else:
    score([TAG] + WISE, "all_folds")
    if DO_FULL and not (R / f"outputs/preds/t4_{TAG}_full.npz").exists():
        assert run(["scripts/train_task4.py", "--tag", TAG, *RECIPE, "--full"]) == 0, "full refit failed"
    if MEMBER_ALPHAS:   # member files of the WiSE-FT points (inference only: validation folds + test images)
        assert run(["experiments/claude/gn_r38_t4_wiseft_full.py", TAG, "--model", MODEL, "--h", "126", "--w", "238",
                    "--alphas", *map(str, MEMBER_ALPHAS), "--member"]) == 0, "WiSE-FT member files failed"
print(f"done in {(time.time() - t_start) / 3600:.2f} h")
""")

code(r"""
# ---- what to copy back to the laptop -------------------------------------------------------------------------------------------
for sub in ("preds", "r30_views", "logs"):
    for f in sorted((D / "out" / sub).glob(f"*{TAG}*")):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")
for tag in EXTRACT:
    for f in sorted((D / "out" / "preds").glob(f"t4_feat*_{tag}.npz")):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")
""")

# the benchmark and training cells run only with TRAIN = True
for c in nb.cells:
    if c.cell_type == "code" and c.source.startswith(("# ---- 1-epoch benchmark", "# ---- training:")):
        head, body = c.source.split("\n", 1)
        c.source = head + "\nif not TRAIN:\n    print('TRAIN = False: skipped')\nelse:\n" + "\n".join(("    " + l) if l.strip() else l for l in body.split("\n"))
nbf.write(nb, "colab/t4_colab_train.ipynb")
print("written", len(nb.cells), "cells")
