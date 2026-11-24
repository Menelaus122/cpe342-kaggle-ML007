"""Build the files for running Task 4 GPU training on Google Colab (round 37).

    python scripts/make_colab_bundle.py

Writes to colab/upload/ (git-ignored):
  cpe342_code.zip  src/, scripts/train_task4.py, the views and v23-evaluation scripts, a minimal experiments/claude/common.py (the
                   repo's one imports the tabular stack, which Colab does not need), outputs/preds/t4_cnxt_fold*.npz (fold indices)
                   and the v23 members' out-of-fold views + dark-frame statistics + probe views (training images only), so the fold-0
                   gate can be scored on Colab; about 40 MB;
  t4_data.zip      the Task 4 training / test images and CSVs under their repo-relative path (stored, not compressed: JPEGs);
                   about 0.6 GB.
Upload both to Google Drive (MyDrive/cpe342_colab/) and open colab/t4_colab_train.ipynb in Colab. Keep the competition data private.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "upload"
OUT.mkdir(parents=True, exist_ok=True)
DATA_REL = Path("04_image_classification/_download/task4_image_classification")
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
COMMON = '''"""Colab version of experiments/claude/common.py: only the working directory and the import path."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
'''

with zipfile.ZipFile(OUT / "cpe342_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((ROOT / "src").glob("*.py")):
        z.write(f, f.relative_to(ROOT).as_posix())
    for rel in ("scripts/train_task4.py", "experiments/claude/em_r30_t4_hardviews.py", "experiments/claude/fc_r31_t4_v23_eval.py",
                "experiments/claude/gn_r38_t4_wiseft_full.py", "experiments/claude/gm_r38_t4_feat_extract.py",
                "experiments/claude/gz_r40_t4_soup.py",
                "scripts/train_task4_probe.py",
                "requirements.txt", "experiments/claude/outputs/r27/t4_probe_views.npz", "experiments/claude/outputs/r30/t4_errors.npz"):
        z.write(ROOT / rel, rel)
    # the v23 members' out-of-fold views (training images), so the fold-0 gate can be scored on Colab
    for sub in ("r19_views", "r30_views"):
        for t in V20:
            for k in range(5):
                f = ROOT / "experiments/claude/outputs" / sub / f"{t}_fold{k}.npz"
                if f.exists():
                    z.write(f, f.relative_to(ROOT).as_posix())
    z.writestr("experiments/claude/common.py", COMMON)
    for k in range(5):
        rel = f"outputs/preds/t4_cnxt_fold{k}.npz"
        z.write(ROOT / rel, rel)
n = 0
with zipfile.ZipFile(OUT / "t4_data.zip", "w", zipfile.ZIP_STORED) as z:
    for name in ("train.csv", "test.csv"):
        z.write(ROOT / DATA_REL / name, (DATA_REL / name).as_posix())
    for sub in ("train", "test"):
        for f in sorted((ROOT / DATA_REL / sub).iterdir()):
            z.write(f, (DATA_REL / sub / f.name).as_posix())
            n += 1
for f in sorted(OUT.iterdir()):
    print(f"{f}: {f.stat().st_size / 2**20:.1f} MB")
print(f"{n} images")
