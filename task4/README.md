# Task 4: Game Title Detection (final submission v41)

`task4.ipynb` is the final Task 4 code. It loads the cached out-of-fold and test probabilities of every member from
`outputs/preds/`, builds the weighted ensemble, applies the decision rule and writes `outputs/submission_parts/task4.csv`.
It does **not** retrain the networks: the members took about 2–3 GPU hours each on a laptop RTX 4070 or a Colab A100.
This file explains how each cached member was trained, so any of them can be rebuilt.

## Running the notebook
1. Set up the environment and data as in the main [README](../README.md#how-to-run). The Task 4 images are not in git. Download
   them from the Drive link in [`04_image_classification/README.md`](../04_image_classification/README.md) and extract them to
   `04_image_classification/_download/task4_image_classification/{train,test}/`. The notebook reads the images for the EDA and
   for the per-image brightness / contrast gate.
2. Run `task4.ipynb` top to bottom. A CPU is enough.
3. Optional: the Grad-CAM figure (section 8.4) needs the fold-0 ConvNeXt checkpoint. Without it the cell prints "skipped" and
   the rest of the notebook runs normally. To get the figure, download
   [`t4_cnxt_fold0.pt`](https://github.com/Menelaus122/cpe342-kaggle-ML007/releases/download/t4-checkpoints/t4_cnxt_fold0.pt)
   into `outputs/models/`.

## Final ensemble
| Tag | Backbone (timm) | Input H×W | Weight (normal / dark frames) | Trained on |
|---|---|---|---|---|
| `cnxt` | `convnext_tiny.in12k_ft_in1k` | 160×320 | 0.25 | laptop |
| `deit3` | `deit3_small_patch16_384.fb_in22k_ft_in1k` | 192×384 | 0.25 | laptop |
| `cnxt_hud` | `convnext_tiny.in12k_ft_in1k` + HUD dropout | 160×320 | 0.75 | laptop |
| `cnxt22k_hud` | `convnext_tiny.fb_in22k` + HUD dropout | 160×320 | 0.75 | laptop |
| `effv2_hud` | `tf_efficientnetv2_s.in21k_ft_in1k` + HUD dropout | 160×320 | 0.75 | laptop |
| `deit3_hud` | `deit3_small_patch16_384.fb_in22k_ft_in1k` + HUD dropout | 192×384 | 0.75 | laptop |
| `cnxtS_hud` | `convnext_small.in12k_ft_in1k` + HUD dropout | 160×320 | 0.75 | laptop |
| `dinov2_v2` | `vit_base_patch14_reg4_dinov2.lvd142m` | 126×238 | 8 | laptop |
| `dinov2L_hud70` | `vit_large_patch14_reg4_dinov2.lvd142m` | 126×238 | 8 / 2 | Colab A100 |
| `dinov3L_lr8e5` | `vit_large_patch16_dinov3.lvd1689m` | 144×272 | 10 | Colab A100 |
| `dinov3L_384` | `vit_large_patch16_dinov3.lvd1689m` | 192×384 | 20 | Colab A100 |
| `cradio3L_384` | NVIDIA C-RADIOv3-L (Hugging Face `nvidia/C-RADIOv3-L`) | 192×384 | 10 / 2 | Colab A100 |
| `dinov3Hp_144` | `vit_huge_plus_patch16_dinov3.lvd1689m` | 144×272 | 40 | Colab A100 |
| `dinov3Hp_384` | `vit_huge_plus_patch16_dinov3.lvd1689m` | 192×384 | 40 | Colab A100 |
| `probe7b` (P_7b) | MLP probe on frozen `vit_7b_patch16_dinov3.lvd1689m` features | 144×272 | 20% of the final mix | Colab A100 (features) + laptop (probe) |

- **Dark frames:** a frame counts as dark / low-contrast when its mean grey level or grey-level SD is at or below the 10% quantile
  of the training images.
- **Cross-validation:** every member uses the same 5 stratified folds (`StratifiedKFold`, seed 42) and horizontal-flip TTA.
- **Test input:** the average of the 5 fold models and the model refitted on all 31,546 training images (Train+Val), 50 / 50.
- **Decision rule:** the final class is `argmax_c p_c · π_c^-0.75`, where π is the class share in the training set.
- **Where the choices are explained:** the notebook, and `experiments/claude/FINDINGS.md` for the round-by-round checks.

## Re-training a member

### Laptop members (`scripts/train_task4.py`, any CUDA GPU with 8 GB)
Each command trains the 5 fold models (writes `outputs/preds/t4_<tag>_fold{0..4}.npz`). Run it again with `--full` for the
Train+Val refit (`t4_<tag>_full.npz`). Checkpoints go to `outputs/models/`.
```bash
python -u scripts/train_task4.py --tag cnxt  --model convnext_tiny.in12k_ft_in1k --epochs 10
python -u scripts/train_task4.py --tag deit3 --model deit3_small_patch16_384.fb_in22k_ft_in1k --h 192 --w 384 --lr 1e-4 --epochs 10
# HUD-dropout members: the same recipe as the plain backbone plus --p-hud 0.5, e.g.
python -u scripts/train_task4.py --tag cnxt_hud --model convnext_tiny.in12k_ft_in1k --epochs 10 --p-hud 0.5
#   cnxt22k_hud: --model convnext_tiny.fb_in22k
#   effv2_hud:   --model tf_efficientnetv2_s.in21k_ft_in1k
#   deit3_hud:   the deit3 command above + --p-hud 0.5
#   cnxtS_hud:   --model convnext_small.in12k_ft_in1k --grad-ckpt
python -u scripts/train_task4.py --tag dinov2_v2 --model vit_base_patch14_reg4_dinov2.lvd142m --h 126 --w 238 \
    --lr 5e-5 --layer-decay 0.8 --warmup-epochs 2 --epochs 15 --p-hud 0.5 --grad-ckpt
```

### Colab members (`task4/colab/`, A100)
The Colab notebooks run the same `scripts/train_task4.py` with the same folds:
1. Run `python scripts/make_colab_bundle.py`. It writes `task4/colab/upload/cpe342_code.zip` (code) and `t4_data.zip` (images).
2. Upload both files to Google Drive, folder `MyDrive/cpe342-colab/`.
3. Open the notebook in Colab, choose an A100 runtime and run all cells. Finished folds are saved to `MyDrive/cpe342-colab/out/`,
   and a disconnected run resumes from there.
4. Copy the `t4_<tag>_*.npz` files from `out/` into `outputs/preds/`.

| Notebook | Member | Recipe (`train_task4.py` arguments) |
|---|---|---|
| `t4_colab_train.ipynb` | `dinov2L_hud70` | `--lr 3e-5 --layer-decay 0.85 --warmup-epochs 2 --epochs 15 --p-hud 0.70 --grad-ckpt`. The committed notebook is set up for a later second-seed test; for the final member set `TAG = "dinov2L_hud70"` and `SEED_OFFSET = 0` |
| `t4_colab_dinov3b.ipynb` | `dinov3L_lr8e5` | `--lr 8e-5 --layer-decay 0.90 --warmup-epochs 2 --epochs 18 --p-hud 0.70 --grad-ckpt` |
| `t4_colab_dinov3c.ipynb` | `dinov3L_384` | as `dinov3L_lr8e5`, at 192×384 |
| `t4_colab_radio3.ipynb` | `cradio3L_384` | as `dinov3L_lr8e5`, at 192×384 (needs `transformers`, `einops`, `open_clip_torch`) |
| `t4_colab_dinov3h.ipynb` | `dinov3Hp_144` | as `dinov3L_lr8e5` |
| `t4_colab_dinov3h384.ipynb` | `dinov3Hp_384` | as `dinov3L_lr8e5` with `--layer-decay 0.93`, at 192×384 |
| `t4_colab_dinov3_7b_feat.ipynb` | features for `probe7b` | frozen DINOv3-7B features of 4 training views and the test images |

The notebooks were generated by `build_t4_colab_nb.py` and `build_r43_colab_nbs.py` in the same folder. Some notebooks start with
a fold-0 run and a gate; their text describes the run that was made.

### The P_7b probe
1. Run `t4_colab_dinov3_7b_feat.ipynb` on an A100. It writes the frozen features: `train_{full,zoom70,centre_grey,dark_lc}.npy`,
   `test_full.npy` and `test_file_name.npy`, about 5 GB in total.
2. Put them in `outputs/preds/dinov3_7b_144x272/`. They are not in git; only `test_file_name.npy` is.
3. Train the probe on any CUDA GPU (`--smoke` runs a fast end-to-end check first):
   ```bash
   python scripts/train_task4_probe_7b.py
   ```
   It writes `outputs/preds/t4_probe7b_fold{0..4}.npz` and `t4_probe7b_full.npz`.
