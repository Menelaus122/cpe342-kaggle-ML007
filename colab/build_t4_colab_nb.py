"""Write colab/t4_colab_train.ipynb (run from the repo root):  python colab/build_t4_colab_nb.py"""
import sys

import nbformat as nbf

PROFILE = sys.argv[1] if len(sys.argv) > 1 else "seed2"     # seed2 -> colab/t4_colab_train.ipynb; dinov3 -> colab/t4_colab_dinov3.ipynb;
assert PROFILE in ("seed2", "dinov3", "dinov3b", "dinov3c", "radio3", "dinov3h", "dinov2g"), PROFILE   # dinov2g ->
# colab/t4_colab_dinov2g.ipynb (round 42: DINOv2-g/14 at 126 x 238, DINOv2-L recipe with lr 2e-5 / layer decay 0.90, fold 0 only)   # dinov3b -> colab/t4_colab_dinov3b.ipynb (DINOv3 at lr 8e-5,
# fold 0 only); dinov3c -> colab/t4_colab_dinov3c.ipynb (round 41: the same DINOv3 recipe at 192 x 384, fold 0 only); radio3 ->
# colab/t4_colab_radio3.ipynb (round 42: NVIDIA C-RADIOv3-L at 192 x 384 with the dinov3c recipe, fold 0 only); dinov3h ->
# colab/t4_colab_dinov3h.ipynb (round 42: DINOv3 ViT-H+/16 at 144 x 272 with the dinov3b recipe, fold 0 only)

nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# CPE342 Task 4: GPU training on Colab (round 40 step 3: a second seed of the v30 DINOv2-L member; all folds + full refit)

Runs the same trainer as the laptop (`scripts/train_task4.py`, identical folds) on a bigger GPU. Training images only; no test
image or test statistic enters any decision (the full refit only *predicts* the test images, as on the laptop).

**Round 40 step 3 (user brief of 2026-11-18):** the same recipe as v30's member `dinov2L_hud70` with other training seeds
(`--train-seed-offset 1000`; the folds are unchanged), tag `dinov2L_hud70_s2`. It is a replicate of a recipe that already passed, so
`GATE = False`: all five folds and the full refit run, and the decision is made on the laptop against v30
(`experiments/claude/gx_r40_t4_dinov2L_seed2_eval.py`, written before the run). About 3 h on an A100.

**Round 39 step 3 recipe (user brief of 2026-11-17 evening; `LP_FT = False`):** DINOv2 ViT-L/14 at 126×238, plain fine-tuning (the
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

**Auto-disconnect (`AUTO_DISCONNECT = True`):** the last cell ends the runtime 60 s after the run, and an error in any cell does the
same after printing it, so no compute units are spent on an idle runtime overnight; all outputs and logs are already on Drive.

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
TAG = "dinov2L_hud70_s2"      # round 40 step 3 (round 39: "dinov2L_hud70"; round 38: "dinov2L_lpft")
LP_FT = False                 # round 38: True (LP head first, then fine-tuning from it, plus WiSE-FT points)
LR, P_HUD = "3e-5", "0.70"    # round 38: 2e-5, 0.5
LD, EPOCHS = "0.85", "15"     # layer-wise lr decay and epochs (2 warm-up + the rest cosine)
MIN_ALONE = None              # fold-0 gate also needs the member's own fold-0 macro-F1 (TTA) >= this (None = no floor)
GATE_NEEDS = ["v24 check (beyond the grid) PASS"]   # round 38 also needed "hard-view rule PASS"
FOLDS = [0, 1, 2, 3, 4]       # fold 0 first; folds 1-4 only if fold 0 passes the gate
GATE = False                  # False: train every fold without the fold-0 gate (round 40: a seed replicate, judged on the laptop)
SEED_OFFSET = 1000            # added to the training seeds only (round 39's member: 0)
AUTO_DISCONNECT = True        # end the runtime when the run is done or a cell fails: no compute units spent idle (outputs are on Drive)
DO_FULL = True                # full Train+Val refit -> test predictions (the member's test input)
DRIVE_DIR = "/content/drive/MyDrive/cpe342-colab"   # the team's Drive folder
MODELS_ON_DRIVE = True        # 1.2 GB per checkpoint (about 6 GB in all); False keeps them on the Colab disk (lost with the session)
GRAD_CKPT = "auto"            # gradient checkpointing changes memory only, not the result: auto = on below 60 GB of GPU memory
ACCUM = 1                     # 64 images per optimiser step in one pass (2 = 2 x 32, same effective batch, for a small GPU)
ALPHAS = [0.0, 0.5, 0.7, 0.85] if LP_FT else []    # WiSE-FT points (alpha = 1 is the fine-tuned model itself); LP-FT only
MEMBER_ALPHAS = [0.5, 0.7, 0.85] if LP_FT else []  # after all folds: member files (val + test probabilities) for these alphas
MODEL = "vit_large_patch14_reg4_dinov2.lvd142m"
H, W = 126, 238               # the frame's own aspect ratio: 9 x 17 patches of 14 px
RECIPE = ["--model", MODEL, "--h", str(H), "--w", str(W), "--bs", "64", "--accum", str(ACCUM), *(["--lp-ft"] if LP_FT else []),
          "--lr", LR, "--layer-decay", LD, "--warmup-epochs", "2", "--epochs", EPOCHS, "--p-hud", P_HUD,
          "--train-seed-offset", str(SEED_OFFSET)]
print("recipe:", " ".join(RECIPE))

# on an error in any later cell: show it, wait 60 s, then end the runtime (Ctrl+M I / "Interrupt" is not an error and keeps it)
try:
    _ip = get_ipython()
except NameError:
    _ip = None
if AUTO_DISCONNECT and _ip is not None:
    def _end_runtime_on_error(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        import time as _t
        print(f"\nerror -> ending the runtime in 60 s [{_t.strftime('%H:%M')} UTC]; the logs are in {DRIVE_DIR}/out/logs", flush=True)
        _t.sleep(60)
        from google.colab import runtime
        runtime.unassign()
    _ip.set_custom_exc((Exception,), _end_runtime_on_error)
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
    m0 = task4.create_model(MODEL, img_size=(H, W)).cuda().to(memory_format=torch.channels_last)
    head = task4.fit_lp_head(m0, arr, tr_df.label.values, np.arange(1280), (H, W))
    vp0 = task4.predict_array(task4.set_head(m0, head), arr[1280:], (H, W))
    print(f"LP head on 1,280 images in {time.time() - t_lp:.0f}s; theta_0 accuracy on 128 others "
          f"{np.mean(vp0.argmax(1) == tr_df.label.values[1280:1408]):.3f}")
    del m0
m, h = task4.train_fold(MODEL, arr, tr_df.label.values[idx], np.arange(1280), np.arange(1280, 1408),
                        img_size=(H, W), epochs=1, bs=64, lr=float(LR), layer_decay=float(LD), aug_kw={"p_hud": float(P_HUD)}, grad_ckpt=GRAD_CKPT,
                        warmup_epochs=1, accum=ACCUM, init_head=head)
per_img = h[-1]["secs"] / 1280
print(f"peak GPU memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB; {per_img * 1000:.1f} ms per training image (first epoch, "
      f"includes warm-up) -> about {per_img * 25237 / 60:.0f} min per fold epoch, {per_img * 25237 * int(EPOCHS) / 3600:.1f} h per fold "
      f"(laptop: 656 s per epoch, 2.7 h per fold), {per_img * 25237 * int(EPOCHS) * (len(FOLDS) + 1.25 * DO_FULL) / 3600:.1f} h for this run")
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
    ok = any(all(g in l for g in GATE_NEEDS) for l in out.splitlines())
    if MIN_ALONE is not None:
        import numpy as np
        from sklearn.metrics import f1_score
        from src import task4
        z = np.load(R / f"outputs/preds/t4_{TAG}_fold0.npz")
        f1 = f1_score(task4.load_frames()[0].label.values[z["val_idx"]], z["val_probs"].argmax(1), average="macro")
        print(f"{TAG} alone, fold 0 macro-F1 (TTA) {f1:.4f} -> {'>=' if f1 >= MIN_ALONE else 'below'} the floor {MIN_ALONE}; "
              f"{', '.join(GATE_NEEDS)}: {'yes' if ok else 'no'}")
        ok = ok and f1 >= MIN_ALONE
    return ok

t_start = time.time()
for f in FOLDS:
    if not (R / f"outputs/preds/t4_{TAG}_fold{f}.npz").exists():
        assert run(["scripts/train_task4.py", "--tag", TAG, *RECIPE, "--folds", str(f)]) == 0, f"fold {f} failed"
    if not (R / f"experiments/claude/outputs/r30_views/{TAG}_fold{f}.npz").exists():
        # the five scoring views of this fold's validation images (full, zoom70, HUD hidden, gamma 3.5, 75% grid)
        assert run(["experiments/claude/em_r30_t4_hardviews.py", TAG], {"VIEWS_ONLY": "1"}) == 0, f"views of fold {f} failed"
    if WISE and not all((R / f"experiments/claude/outputs/r30_views/{w}_fold{f}.npz").exists() for w in WISE):
        assert run(["experiments/claude/gn_r38_t4_wiseft_full.py", TAG, "--model", MODEL, "--h", str(H), "--w", str(W),
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
        assert run(["experiments/claude/gn_r38_t4_wiseft_full.py", TAG, "--model", MODEL, "--h", str(H), "--w", str(W),
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

code(r"""
# ---- end the runtime (stops compute-unit use; every output is already on Drive) ------------------------------------------------
if AUTO_DISCONNECT:
    import time
    print(f"done -> ending the runtime in 60 s [{time.strftime('%H:%M')} UTC] (set AUTO_DISCONNECT = False to keep it)", flush=True)
    time.sleep(60)
    from google.colab import runtime
    runtime.unassign()
""")

# the benchmark and training cells run only with TRAIN = True
for c in nb.cells:
    if c.cell_type == "code" and c.source.startswith(("# ---- 1-epoch benchmark", "# ---- training:")):
        head, body = c.source.split("\n", 1)
        c.source = head + "\nif not TRAIN:\n    print('TRAIN = False: skipped')\nelse:\n" + "\n".join(("    " + l) if l.strip() else l for l in body.split("\n"))
out_path = "colab/t4_colab_train.ipynb"
if PROFILE == "seed2":
    soup_cell = nbf.v4.new_code_cell(r"""
# ---- weight soup of the two DINOv2-L seeds (round 40; inference only, about 10-15 min on an A100) -------------------------------
# uniform average of the EMA weights of dinov2L_hud70 (v30's member) and this run, per fold and for the full refit; needs both
# seeds' checkpoints in out/models (do not delete t4_dinov2L_hud70_*.pt); compared on the laptop by gx_r40_t4_dinov2L_seed2_eval.py
SOUP_OUT = "dinov2L_hud70_soup"
if TRAIN and TAG == "dinov2L_hud70_s2":
    need = [R / f"outputs/models/t4_{t}_{k}.pt" for t in ("dinov2L_hud70", TAG) for k in [f"fold{f}" for f in FOLDS] + ["full"]]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print("soup skipped, missing checkpoints:", missing)
    else:
        assert run(["experiments/claude/gz_r40_t4_soup.py", "dinov2L_hud70", TAG, SOUP_OUT, "--model", MODEL, "--h", str(H), "--w", str(W),
                    "--folds", *map(str, FOLDS)]) == 0, "weight soup failed"
        for sub in ("preds", "r30_views"):
            for f in sorted((D / "out" / sub).glob(f"*{SOUP_OUT}*")):
                print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")
""".strip("\n"))
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.lstrip().startswith("# ---- what to copy back"))
    nb.cells.insert(k, soup_cell)
if PROFILE in ("dinov3", "dinov3b", "dinov3c", "radio3", "dinov3h"):
    out_path = "colab/t4_colab_dinov3.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 40 step 3: a second seed of the v30 DINOv2-L member; all folds + full refit)",
        "# CPE342 Task 4: GPU training on Colab (round 40, revised: DINOv3 ViT-L/16 with v30's recipe; fold 0 + gate, then folds 1-4 + full refit)\n\n"
        "**This notebook (user brief of 2026-11-18, revised round 40):** `vit_large_patch16_dinov3.lvd1689m` (distilled from the 7B "
        "DINOv3 teacher on LVD-1689M; RoPE positions, 4 registers) at **144x272** = 9 x 17 patches of 16 px, the same token grid as "
        "DINOv2-L at 126x238; v30's recipe otherwise (lr 3e-5, layer decay 0.85, HUD dropout 0.70, 15 epochs, gradient checkpointing "
        "below 60 GB). Tag `dinov3L_hud70`. `GATE = True`: after fold 0 the v24 check against the v23 members decides whether "
        "folds 1-4 and the full refit run. A compatibility cell (timm version, the model's presence, one forward / backward step "
        "at 144x272 with the peak GPU memory) runs before the benchmark. The weights are not gated on the Hugging Face hub "
        "(timm mirror, DINOv3 licence). The sections below describe the earlier runs of this notebook.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov2L_hud70_s2"', 'TAG = "dinov3L_hud70"'),
                     ('GATE = False', 'GATE = True'),
                     ('SEED_OFFSET = 1000', 'SEED_OFFSET = 0'),
                     ('MODEL = "vit_large_patch14_reg4_dinov2.lvd142m"', 'MODEL = "vit_large_patch16_dinov3.lvd1689m"'),
                     ('H, W = 126, 238               # the frame\'s own aspect ratio: 9 x 17 patches of 14 px',
                      'H, W = 144, 272               # 9 x 17 patches of 16 px, the token grid of DINOv2-L at 126 x 238')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
    check = nbf.v4.new_code_cell(r"""
# ---- DINOv3 compatibility check: timm version, the model, one forward / backward step at H x W (downloads the weights, 1.2 GB) ----
import timm, torch
print("timm", timm.__version__)
if MODEL.split(".")[0] not in timm.list_models("*dinov3*"):
    raise RuntimeError(f"{MODEL} is not in timm {timm.__version__}: !pip -q install -U timm, then restart and run all")
print("dinov3 models:", [n for n in timm.list_models("*dinov3*") if "large" in n])
import sys
sys.path.insert(0, str(R))
from src import task4
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
m = task4.create_model(MODEL, img_size=(H, W)).cuda().train()
if GRAD_CKPT:
    m.set_grad_checkpointing(True)
x = torch.rand(64, 3, H, W, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(task4.normalize(x))
    tokens = m.forward_features(task4.normalize(x[:1]))
out.float().sum().backward()
print(f"logits {tuple(out.shape)}, tokens {tuple(tokens.shape)} (9 x 17 = 153 patches + class + registers), "
      f"grad of the patch embedding finite {torch.isfinite(m.patch_embed.proj.weight.grad).all().item()}, "
      f"peak GPU memory for a batch of 64 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
del m, out, x
torch.cuda.empty_cache()
""".strip("\n"))
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.lstrip().startswith("# ---- 1-epoch benchmark"))
    nb.cells.insert(k, check)
if PROFILE in ("dinov3b", "dinov3c", "radio3", "dinov3h"):
    out_path = "colab/t4_colab_dinov3b.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 40, revised: DINOv3 ViT-L/16 with v30's recipe; fold 0 + gate, then folds 1-4 + full refit)",
        "# CPE342 Task 4: GPU training on Colab (round 40: DINOv3 ViT-L/16, new recipe lr 8e-5 / layer decay 0.90 / 18 epochs; fold 0 only)\n\n"
        "**This notebook (user brief of 2026-11-18 afternoon):** the first DINOv3 run (`dinov3L_hud70`, v30's recipe) reached only 0.9794 "
        "on fold 0 and was still rising, while the frozen DINOv3-L @ 144x272 probe beats frozen DINOv2-L by +0.031 (full) and +0.040 "
        "(real dark frames) with the set-up checked (`hb_r40_t4_dinov3_check.py`): the gap is the fine-tuning recipe. One new recipe, "
        "fixed before the run: **lr 8e-5, layer decay 0.90, 2 warm-up + 16 cosine epochs (18)**, everything else as before (144x272, "
        "HUD dropout 0.70, batch 64, its own 'avg' pooling, gradient checkpointing below 60 GB, which changes memory only). Tag "
        "`dinov3L_lr8e5`. **Fold 0 only** (`FOLDS = [0]`, `DO_FULL = False`, about 45-50 min on an A100). Gate, fixed before the run: "
        "the member alone >= **0.985** macro-F1 on fold 0 (`MIN_ALONE`; DINOv2-L: 0.9885) **and** the v24 check against the v23 "
        "members; it is then compared with v30 on fold 0 on the laptop. If it fails, DINOv3 is closed. If it passes and the user goes on: "
        "set `FOLDS = [0, 1, 2, 3, 4]`, `DO_FULL = True` and *Run all* again (fold 0 is reused from Drive). The sections below describe "
        "the earlier runs of this notebook.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov3L_hud70"', 'TAG = "dinov3L_lr8e5"'),
                     ('LR, P_HUD = "3e-5", "0.70"', 'LR, P_HUD = "8e-5", "0.70"'),
                     ('LD, EPOCHS = "0.85", "15"', 'LD, EPOCHS = "0.90", "18"'),
                     ('MIN_ALONE = None ', 'MIN_ALONE = 0.985'),
                     ('FOLDS = [0, 1, 2, 3, 4]       # fold 0 first; folds 1-4 only if fold 0 passes the gate',
                      'FOLDS = [0]                   # fold 0 only (user brief); after a PASS: [0, 1, 2, 3, 4] and DO_FULL = True'),
                     ('DO_FULL = True ', 'DO_FULL = False')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
if PROFILE in ("dinov3c", "radio3"):
    out_path = "colab/t4_colab_dinov3c.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 40: DINOv3 ViT-L/16, new recipe lr 8e-5 / layer decay 0.90 / 18 epochs; fold 0 only)",
        "# CPE342 Task 4: GPU training on Colab (round 41: DINOv3 ViT-L/16 at 192x384 with v31's recipe; fold 0 only)\n\n"
        "**This notebook (user brief of 2026-11-18 evening):** v31's DINOv3-L member (`dinov3L_lr8e5`, 144x272 = 9 x 17 patches) at a "
        "higher input resolution, **192x384 = 12 x 24 patches of 16 px (288 tokens)**; RoPE positions let the model take the larger "
        "grid without interpolating a position table. Recipe unchanged (lr 8e-5, layer decay 0.90, 2 warm-up + 16 cosine epochs, HUD "
        "dropout 0.70, batch 64, 'avg' pooling); **gradient checkpointing always on** (`GRAD_CKPT = True`; memory only, not the "
        "result). Tag `dinov3L_384`. **Fold 0 only** (`FOLDS = [0]`, `DO_FULL = False`; about 1.5 h on an A100, the benchmark cell "
        "prints the estimate). Gate, fixed before the run: the member alone >= **0.9925** macro-F1 on fold 0 (`MIN_ALONE` = what "
        "`dinov3L_lr8e5` reached on fold 0: a higher resolution that is not better alone is an expensive replicate) **and** the v24 "
        "check against the v23 members; then, on the laptop, the fold-0 comparison with v31 (`hh_r41_t4_dinov3_384_fold0.py`, "
        "variants fixed before the run). If it fails: stop. If it passes and the user goes on: `FOLDS = [0, 1, 2, 3, 4]`, "
        "`DO_FULL = True`, *Run all* (fold 0 reused from Drive; about 8 h more). The sections below describe the earlier runs.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov3L_lr8e5"', 'TAG = "dinov3L_384"'),
                     ('MIN_ALONE = 0.985', 'MIN_ALONE = 0.9925'),
                     ('GRAD_CKPT = "auto"            # gradient checkpointing changes memory only, not the result: auto = on below 60 GB of GPU memory',
                      'GRAD_CKPT = True              # always on at 192 x 384 (288 tokens); changes memory only, not the result'),
                     ('H, W = 144, 272               # 9 x 17 patches of 16 px, the token grid of DINOv2-L at 126 x 238',
                      'H, W = 192, 384               # 12 x 24 patches of 16 px (round 40 member: 144 x 272 = 9 x 17)')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "DINOv3 compatibility check" in c.source)
    assert nb.cells[k].source.count("(9 x 17 = 153 patches + class + registers)") == 1
    nb.cells[k].source = nb.cells[k].source.replace("(9 x 17 = 153 patches + class + registers)", "(12 x 24 = 288 patches + class + registers)")
if PROFILE == "radio3":
    out_path = "colab/t4_colab_radio3.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 41: DINOv3 ViT-L/16 at 192x384 with v31's recipe; fold 0 only)",
        "# CPE342 Task 4: GPU training on Colab (round 42: NVIDIA C-RADIOv3-L at 192x384 with the DINOv3 recipe; fold 0 only)\n\n"
        "**This notebook (user brief of 2026-11-19, after v33):** NVIDIA **C-RADIOv3-L** (`nvidia/C-RADIOv3-L`, 320 M parameters, a "
        "ViT-L/16 distilled from CLIP ViT-H, SigLIP 2-SO400M, DINOv2-g and SAM-H; NVIDIA Open Model License), loaded from the Hugging "
        "Face hub with `trust_remote_code` at a pinned revision (the repo code was read on the laptop first). Frozen at 192x384 its "
        "summary embedding beat frozen DINOv3-L by +0.0143 (full) and +0.0195 (real dark frames) in a logistic probe "
        "(`experiments/claude/ia_r42_t4_radio_frozen.py`). Pipeline name `c-radio_v3-l` (`src/task4.RadioClassifier`): the model sees "
        "[0, 1] images and normalises them itself (CLIP mean / std; the wrapper undoes the pipeline's ImageNet normalisation exactly), "
        "head Linear(3072 -> 5) on the summary, stochastic depth 0-0.1, layer decay with the CPE patch generator as the stem, gradient "
        "checkpointing through RADIO's own block loop; smoke-tested on the laptop (`ic_r42_t4_radio_smoke.py`). **Recipe = dinov3c's, "
        "fixed before the run:** 192x384 (12 x 24 patches), lr 8e-5, layer decay 0.90, 2 warm-up + 16 cosine epochs, HUD dropout 0.70, "
        "batch 64, gradient checkpointing on. Tag `cradio3L_384`. **Fold 0 only** (`FOLDS = [0]`, `DO_FULL = False`; about 1.2-1.5 h on "
        "an A100, the benchmark cell prints the estimate). Gate, fixed before the run: the member alone >= **0.9925** macro-F1 on fold 0 "
        "(`MIN_ALONE`, the bar of the DINOv3 members) **and** the v24 check against the v23 members; then, on the laptop, the fold-0 "
        "comparison with v33 (`experiments/claude/hj_r42_t4_radio_fold0.py`, variants fixed before the run). If it fails: stop. If it "
        "passes and the user goes on: `FOLDS = [0, 1, 2, 3, 4]`, `DO_FULL = True`, *Run all* (fold 0 reused from Drive). Extra packages: "
        "transformers, einops, open_clip_torch (the transformers loader checks the repo's optional open_clip adaptor imports; it is not "
        "used). The sections below describe the earlier runs.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov3L_384"', 'TAG = "cradio3L_384"'),
                     ('MODEL = "vit_large_patch16_dinov3.lvd1689m"', 'MODEL = "c-radio_v3-l"            # src/task4.RADIO_MODELS -> nvidia/C-RADIOv3-L @ 9d041346'),
                     ('H, W = 192, 384               # 12 x 24 patches of 16 px (round 40 member: 144 x 272 = 9 x 17)',
                      'H, W = 192, 384               # 12 x 24 patches of 16 px (native: RADIO takes any multiple of 16)')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.lstrip().startswith("# ---- packages"))
    assert nb.cells[k].source.count("!pip -q install timm==1.0.30") == 1
    nb.cells[k].source = (nb.cells[k].source
                          .replace("!pip -q install timm==1.0.30", "!pip -q install timm==1.0.30 transformers==5.23.0 einops==0.8.2 open_clip_torch==4.2.2")
                          .replace("import numpy, pandas, sklearn, timm, torch, torchvision", "import numpy, pandas, sklearn, timm, torch, torchvision, transformers")
                          .replace('"| timm", timm.__version__,', '"| timm", timm.__version__, "| transformers", transformers.__version__,'))
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "DINOv3 compatibility check" in c.source)
    nb.cells[k].source = r"""
# ---- C-RADIOv3-L compatibility check: load (pinned revision, downloads 1.3 GB), one forward / backward step at H x W -------------
import sys
import torch
import torch.utils.checkpoint as tuc
sys.path.insert(0, str(R))
from src import task4
BS_CHECK = 64
print("pipeline name", MODEL, "->", task4.RADIO_MODELS[MODEL])
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
m = task4.create_model(MODEL, img_size=(H, W)).cuda().eval()
x = torch.rand(BS_CHECK, 3, H, W, device="cuda")
with torch.no_grad():   # eval mode: no stochastic depth, so both passes are deterministic (train mode drops blocks at random)
    s_wrap, (s_own, _) = m.forward_summary(task4.normalize(x[:2])), m.radio.radio_model(x[:2])
diff = (s_wrap - s_own).abs().max().item()
_orig_ckpt, n_ckpt = tuc.checkpoint, [0]
def _counting_ckpt(fn, *a, **k):     # RADIO's CPE forward -> timm.checkpoint_seq -> one checkpoint call per block
    n_ckpt[0] += 1
    return _orig_ckpt(fn, *a, **k)
tuc.checkpoint = _counting_ckpt
m.train()
if GRAD_CKPT:
    m.set_grad_checkpointing(True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(task4.normalize(x))
out.float().sum().backward()
tuc.checkpoint = _orig_ckpt
pg = next(m.radio.radio_model.model.patch_generator.parameters())
n_blocks = len(m.radio.radio_model.model.blocks)
print(f"logits {tuple(out.shape)}, summary {tuple(s_wrap.shape)}; wrapper vs RADIO on [0, 1] (eval mode) max |diff| {diff:.1e}; "
      f"checkpointed segments per step {n_ckpt[0]} of {n_blocks} blocks; patch-generator grad finite {torch.isfinite(pg.grad).all().item()}; "
      f"peak GPU memory for a batch of {BS_CHECK} {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
assert diff < 1e-3, f"wrapper and RADIO disagree on the same image: {diff}"
assert n_ckpt[0] == (n_blocks if GRAD_CKPT else 0), f"gradient checkpointing: {n_ckpt[0]} segments"
del m, out, x
torch.cuda.empty_cache()
""".strip("\n")
if PROFILE == "dinov3h":
    out_path = "colab/t4_colab_dinov3h.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 40: DINOv3 ViT-L/16, new recipe lr 8e-5 / layer decay 0.90 / 18 epochs; fold 0 only)",
        "# CPE342 Task 4: GPU training on Colab (round 42: DINOv3 ViT-H+/16 at 144x272 with the dinov3b recipe; fold 0 only)\n\n"
        "**This notebook (user brief of 2026-11-19):** `vit_huge_plus_patch16_dinov3.lvd1689m` (DINOv3 ViT-H+/16, 841 M parameters, 32 "
        "blocks, 1280-d, RoPE, 4 registers; timm 1.0.30, ImageNet mean / std, DINOv3 licence) at **144x272** = 9 x 17 patches, the "
        "geometry of v31's `dinov3L_lr8e5`. Chosen by a frozen-probe screen on the laptop aimed at the real dark / low-contrast frames "
        "(`experiments/claude/id_r42_t4_dark_screen.py`, rule fixed before the run): frozen ViT-H+ beat frozen DINOv3-L at the same "
        "geometry by +0.0123 (full) and +0.0211 (real dark frames); frozen ConvNeXt-L DINOv3 and ConvNeXt V2-L were far behind. **Recipe "
        "= dinov3b's, unchanged (one change at a time):** lr 8e-5, layer decay 0.90 (over 32 blocks: block 0 at 0.034 of the head's lr), "
        "2 warm-up + 16 cosine epochs, HUD dropout 0.70, batch 64, 'avg' pooling, **gradient checkpointing always on** (about 16 GiB of "
        "training state; needs an A100). Tag `dinov3Hp_144`. **Fold 0 only** (`FOLDS = [0]`, `DO_FULL = False`; the benchmark cell prints "
        "the estimate). Gate, fixed before the run: alone >= **0.9925** macro-F1 on fold 0 (`MIN_ALONE`, `dinov3L_lr8e5`'s fold 0) **and** "
        "the v24 check against the v23 members; then, on the laptop, the fold-0 comparison with v33 "
        "(`experiments/claude/hl_r42_t4_dinov3h_fold0.py`, variants fixed before the run). Caveat known before the run: C-RADIOv3-L led "
        "frozen on the dark frames too and was weaker there after fine-tuning. The sections below describe the earlier runs.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov3L_lr8e5"', 'TAG = "dinov3Hp_144"'),
                     ('MODEL = "vit_large_patch16_dinov3.lvd1689m"', 'MODEL = "vit_huge_plus_patch16_dinov3.lvd1689m"'),
                     ('MIN_ALONE = 0.985', 'MIN_ALONE = 0.9925'),
                     ('GRAD_CKPT = "auto"            # gradient checkpointing changes memory only, not the result: auto = on below 60 GB of GPU memory',
                      'GRAD_CKPT = True              # always on for ViT-H+ (841 M parameters); changes memory only, not the result'),
                     ('MODELS_ON_DRIVE = True        # 1.2 GB per checkpoint (about 6 GB in all)',
                      'MODELS_ON_DRIVE = True        # 3.4 GB per checkpoint for ViT-H+ (fold 0 only here)')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
if PROFILE == "dinov2g":
    out_path = "colab/t4_colab_dinov2g.ipynb"
    nb.cells[0].source = nb.cells[0].source.replace(
        "# CPE342 Task 4: GPU training on Colab (round 40 step 3: a second seed of the v30 DINOv2-L member; all folds + full refit)",
        "# CPE342 Task 4: GPU training on Colab (round 42: DINOv2-g/14 at 126x238; fold 0 only)\n\n"
        "**This notebook (user brief of 2026-11-20, after v36 = 0.86727):** `vit_giant_patch14_reg4_dinov2.lvd142m` (DINOv2-g/14 with 4 "
        "registers, 1,135 M parameters, 40 blocks, 1536-d class token; timm 1.0.30, ImageNet mean / std) at **126x238** = 9 x 17 patches, "
        "the geometry of v30's DINOv2-L. Chosen by the second frozen-probe screen (`experiments/claude/ie_r42_t4_screen2.py`, rule fixed "
        "before the run): frozen DINOv2-g beat frozen DINOv3-L at the same 9 x 17 grid by +0.0092 (full) and +0.0229 (real dark frames), "
        "about level with frozen ViT-H+. **Recipe (user's, fixed before the run):** v30's DINOv2-L recipe with **lr 2e-5** and **layer "
        "decay 0.90** for the 40 blocks (block 0 at 0.015 of the head's lr; DINOv2-L at 0.85 over 24 blocks: 0.020), 2 warm-up + 13 "
        "cosine epochs (15), HUD dropout 0.70, batch 64, **gradient checkpointing always on** (about 21 GiB of training state: use an A100 "
        "80 GB; a VRAM smoke test - one full training step with AdamW and the EMA copy - runs before the benchmark). Tag `dinov2g_126`. "
        "**Fold 0 only** (`FOLDS = [0]`, `DO_FULL = False`). Gate, fixed before the run: alone >= **0.9925** macro-F1 on fold 0 "
        "(`MIN_ALONE`) **and** the v24 check against the v23 members; then, on the laptop, the fold-0 comparison with v36 "
        "(`experiments/claude/ho_r42_t4_dinov2g_fold0.py`, variants fixed before the run). If it passes and the user goes on: "
        "`FOLDS = [0, 1, 2, 3, 4]`, `DO_FULL = True`, *Run all* (fold 0 reused from Drive). The sections below describe earlier runs.")
    src = nb.cells[1].source
    for old, new in (('TAG = "dinov2L_hud70_s2"', 'TAG = "dinov2g_126"'),
                     ('LR, P_HUD = "3e-5", "0.70"', 'LR, P_HUD = "2e-5", "0.70"'),
                     ('LD, EPOCHS = "0.85", "15"', 'LD, EPOCHS = "0.90", "15"'),
                     ('MIN_ALONE = None ', 'MIN_ALONE = 0.9925'),
                     ('FOLDS = [0, 1, 2, 3, 4]       # fold 0 first; folds 1-4 only if fold 0 passes the gate',
                      'FOLDS = [0]                   # fold 0 only (user brief); after a PASS: [0, 1, 2, 3, 4] and DO_FULL = True'),
                     ('GATE = False ', 'GATE = True '),
                     ('SEED_OFFSET = 1000', 'SEED_OFFSET = 0'),
                     ('DO_FULL = True ', 'DO_FULL = False '),
                     ('MODELS_ON_DRIVE = True        # 1.2 GB per checkpoint (about 6 GB in all)',
                      'MODELS_ON_DRIVE = True        # 4.5 GB per checkpoint for DINOv2-g (about 27 GB for five folds + full)'),
                     ('GRAD_CKPT = "auto"            # gradient checkpointing changes memory only, not the result: auto = on below 60 GB of GPU memory',
                      'GRAD_CKPT = True              # always on for DINOv2-g (1,135 M parameters); changes memory only, not the result'),
                     ('MODEL = "vit_large_patch14_reg4_dinov2.lvd142m"', 'MODEL = "vit_giant_patch14_reg4_dinov2.lvd142m"')):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    nb.cells[1].source = src
    smoke = nbf.v4.new_code_cell(r"""
# ---- DINOv2-g VRAM smoke test: one full training step (forward + backward + AdamW step + EMA copy) at H x W, batch 64 -----------
# (downloads the weights, 4.5 GB). Peak memory here ~ the training run's; stop if it exceeds the GPU (switch to an A100 80 GB).
import copy
import sys
import timm
import torch
from timm.optim import param_groups_layer_decay
sys.path.insert(0, str(R))
from src import task4
print("timm", timm.__version__, "| model in timm:", MODEL.split(".")[0] in timm.list_models("vit_giant*"))
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
m = task4.create_model(MODEL, img_size=(H, W)).cuda().to(memory_format=torch.channels_last).train()
ema = copy.deepcopy(m).eval()
if GRAD_CKPT:
    m.set_grad_checkpointing(True)
groups = param_groups_layer_decay(m, weight_decay=0.05, layer_decay=float(LD))
for g in groups:
    g["lr"] = float(LR) * g.pop("lr_scale", 1.0)
opt = torch.optim.AdamW(groups, lr=float(LR), weight_decay=0.05)
x = torch.rand(64, 3, H, W, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(task4.normalize(x).to(memory_format=torch.channels_last))
    tokens = m.forward_features(task4.normalize(x[:1]))
out.float().logsumexp(1).mean().backward()
torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
opt.step()
peak, total = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"{len(m.blocks)} blocks, {sum(p.numel() for p in m.parameters()) / 1e6:.0f} M parameters, tokens {tuple(tokens.shape)} (9 x 17 = 153 "
      f"patches + class + 4 registers); grad of the patch embedding finite {torch.isfinite(m.patch_embed.proj.weight.grad).all().item()}; "
      f"peak GPU memory of one training step (batch 64, AdamW, EMA copy) {peak:.1f} of {total:.0f} GiB")
assert peak < 0.92 * total, "too close to the GPU memory: use an A100 80 GB (or ACCUM = 2)"
del m, ema, opt, out, x, tokens
torch.cuda.empty_cache()
""".strip("\n"))
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.lstrip().startswith("# ---- 1-epoch benchmark"))
    nb.cells.insert(k, smoke)
nbf.write(nb, out_path)
print("profile", PROFILE, "->", out_path)
print("written", len(nb.cells), "cells")
