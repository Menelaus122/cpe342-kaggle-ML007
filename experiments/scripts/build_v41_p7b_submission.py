"""Standalone CPU builder for Task 4 Round 43 (`v41`: `v39` base + `P_7b @ 0.20` replacing `P7 @ 0.30`).

    python scripts/build_v41_p7b_submission.py

Does NOT modify `scripts/nb_part2.py` or `submission.csv` (`v39` remains untouched as final backup).
Verifies:
  1. Exact 0-diff reproduction of `submission_v39.csv` Task 4 using `t4_probe` (`P7 @ 0.30`).
  2. Exact 0-diff across Tasks 1, 2, 3, 5 vs `submission_v39.csv`.
  3. Writes `submission_v41_p7b.csv` and prints OOF + test label diff + SHA-256 diagnostics.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import task4  # noqa: E402
from src.config import N_FOLDS, PRED_DIR, SAMPLE_SUB  # noqa: E402

T4_TAGS = [
    "cnxt",
    "deit3",
    "cnxt_hud",
    "cnxt22k_hud",
    "effv2_hud",
    "deit3_hud",
    "cnxtS_hud",
    "dinov2_v2",
    "dinov2L_hud70",
    "dinov3L_lr8e5",
    "dinov3L_384",
    "cradio3L_384",
    "dinov3Hp_144",
    "dinov3Hp_384",
]
T4_MEMBER_W = {
    "dinov2_v2": 8.0,
    "cnxt": 0.25,
    "deit3": 0.25,
    "cnxt_hud": 0.75,
    "cnxt22k_hud": 0.75,
    "effv2_hud": 0.75,
    "deit3_hud": 0.75,
    "cnxtS_hud": 0.75,
    "dinov2L_hud70": 8.0,
    "dinov3L_lr8e5": 10.0,
    "dinov3L_384": 20.0,
    "cradio3L_384": 10.0,
    "dinov3Hp_144": 40.0,
    "dinov3Hp_384": 40.0,
}
T4_MEMBER_W_DARK = {**T4_MEMBER_W, "dinov2L_hud70": 2.0, "cradio3L_384": 2.0}
GATE_Q4 = 0.10
FULL_W = 0.5
ALPHA4 = 0.75


def check_t4_order(z, names, label):
    assert "test_file_name" in z.files and (z["test_file_name"] == names).all(), f"{label}: test row order mismatch"


def wmean4(arrs, dark_mask):
    w4m = np.array([T4_MEMBER_W[t] for t in T4_TAGS], dtype=np.float64)
    w4d = np.array([T4_MEMBER_W_DARK[t] for t in T4_TAGS], dtype=np.float64)
    s = np.stack(arrs)
    return np.where(
        dark_mask[:, None],
        np.tensordot(w4d / w4d.sum(), s, axes=1),
        np.tensordot(w4m / w4m.sum(), s, axes=1),
    )


def load_probe(prefix, y4, test_names, val_idx_ref):
    oof = np.zeros((len(y4), 5), dtype=np.float64)
    test_fold = np.zeros((len(test_names), 5), dtype=np.float64)
    for f in range(N_FOLDS):
        zf = np.load(PRED_DIR / f"{prefix}_fold{f}.npz")
        check_t4_order(zf, test_names, f"{prefix}_fold{f}")
        assert np.array_equal(zf["val_idx"], val_idx_ref[f]), f"{prefix}_fold{f}: val_idx mismatch"
        oof[zf["val_idx"]] = zf["val_probs"]
        test_fold += zf["test_probs"] / N_FOLDS
    z_full = np.load(PRED_DIR / f"{prefix}_full.npz")
    check_t4_order(z_full, test_names, f"{prefix}_full")
    test_full = z_full["test_probs"].astype(np.float64)
    test_blend = (1.0 - FULL_W) * test_fold + FULL_W * test_full
    return oof, test_fold, test_full, test_blend


def main():
    t4_train, t4_test = task4.load_frames()
    y4 = t4_train.label.values.astype(np.int64)
    test_names = np.array(t4_test.file_name.tolist(), dtype="U")
    prior4 = np.bincount(y4, minlength=5) / len(y4)
    w_bal4 = (0.2 / prior4)[y4]
    w4 = prior4 ** (-ALPHA4)

    sub_v39_path = ROOT / "experiments" / "submissions" / "submission_v39.csv"
    assert sub_v39_path.exists(), f"Missing {sub_v39_path}"
    sub_v39 = pd.read_csv(sub_v39_path)
    sample_sub = pd.read_csv(SAMPLE_SUB)

    zg = np.load(PRED_DIR / "t4_gate_stats.npz")
    check_t4_order(zg, test_names, "t4_gate_stats")
    thr4 = np.quantile(zg["train"], GATE_Q4, axis=0)
    dark4_tr = (zg["train"] <= thr4).any(1)
    dark4_te = (zg["test"] <= thr4).any(1)

    oof_list, test_list, val_idx_ref = [], [], []
    for tag in T4_TAGS:
        oof_t = np.zeros((len(y4), 5), dtype=np.float64)
        test_f = np.zeros((len(test_names), 5), dtype=np.float64)
        for f in range(N_FOLDS):
            z = np.load(PRED_DIR / f"t4_{tag}_fold{f}.npz")
            check_t4_order(z, test_names, f"t4_{tag}_fold{f}")
            if tag == T4_TAGS[0]:
                val_idx_ref.append(z["val_idx"])
            oof_t[z["val_idx"]] = z["val_probs"]
            test_f += z["test_probs"] / N_FOLDS
        z_full = np.load(PRED_DIR / f"t4_{tag}_full.npz")
        check_t4_order(z_full, test_names, f"t4_{tag}_full")
        oof_list.append(oof_t)
        test_list.append((1.0 - FULL_W) * test_f + FULL_W * z_full["test_probs"].astype(np.float64))

    ft_oof = wmean4(oof_list, dark4_tr)
    ft_test = wmean4(test_list, dark4_te)

    # 1. Exact reproduction of submission_v39.csv (P7 @ 0.30)
    p7_oof, _, _, p7_test = load_probe("t4_probe", y4, test_names, val_idx_ref)
    v39_oof = 0.70 * ft_oof + 0.30 * p7_oof
    v39_test = 0.70 * ft_test + 0.30 * p7_test
    pred4_v39_repro = (v39_test * w4).argmax(1)
    repro_diffs = int((pred4_v39_repro != sub_v39["task4"].values).sum())
    assert repro_diffs == 0, f"v39 parity check failed: {repro_diffs} diffs!"
    print(f"Parity check: v39 Task 4 reproduced with 0 diffs across {len(test_names)} rows")

    # 2. Build v41 (P_7b @ 0.20)
    p7b_oof, p7b_tfold, p7b_tfull, p7b_test = load_probe("t4_probe7b", y4, test_names, val_idx_ref)
    fold_vs_full_diff = float(np.abs(p7b_tfold - p7b_tfull).mean())
    fold_vs_full_agree = float(np.mean(p7b_tfold.argmax(1) == p7b_tfull.argmax(1)))
    print(f"P_7b test 5-fold mean vs full-refit: mean|diff|={fold_vs_full_diff:.4f}, argmax agree={fold_vs_full_agree:.4f}")

    v41_oof = 0.80 * ft_oof + 0.20 * p7b_oof
    v41_test = 0.80 * ft_test + 0.20 * p7b_test
    pred4_v41 = (v41_test * w4).argmax(1).astype(int)

    # OOF summary
    for label_nm, oof_p in [("v39 (P7 @ 0.30)", v39_oof), ("v41 (P_7b @ 0.20)", v41_oof)]:
        lab_oof = (oof_p * w4).argmax(1)
        f1_tr = f1_score(y4, lab_oof, average="macro")
        f1_bal = f1_score(y4, lab_oof, average="macro", sample_weight=w_bal4)
        print(f"OOF {label_nm:18s}: train-mix macro-F1 {f1_tr:.4f} | balanced-mix macro-F1 {f1_bal:.4f}")

    # Test diffs & sanity checks
    out_df = sub_v39.copy()
    out_df["task4"] = pred4_v41
    expected_ids = [f"ANS{i:05d}" for i in range(1, len(test_names) + 1)]
    assert out_df.shape == (25889, 6)
    assert list(out_df.columns) == list(sample_sub.columns)
    assert out_df["id"].tolist() == expected_ids == sample_sub["id"].tolist()
    assert out_df.isna().sum().sum() == 0
    assert set(out_df["task4"].unique()) <= {0, 1, 2, 3, 4}
    for col in ("task1", "task2", "task3", "task5"):
        assert np.array_equal(out_df[col].values, sub_v39[col].values), f"Unexpected change in {col}!"

    out_path = ROOT / "experiments" / "submissions" / "submission_v41_p7b.csv"
    out_df.to_csv(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    diff_v39 = pred4_v41 != sub_v39["task4"].values
    n_diff_v39 = int(diff_v39.sum())
    n_diff_v39_dark = int((diff_v39 & dark4_te).sum())
    print(f"Task 4 test diffs vs v39: {n_diff_v39} rows total ({n_diff_v39 / len(test_names):.2%}), of which {n_diff_v39_dark} are gated dark/low-contrast")

    v38_candidates = [ROOT / "experiments" / "submissions" / "submission_v38.csv", ROOT / "experiments" / "versions" / "v38_LB0.86891" / "submission.csv"]
    v38_path = next((p for p in v38_candidates if p.exists()), None)
    if v38_path is not None:
        sub_v38 = pd.read_csv(v38_path)
        diff_v38 = pred4_v41 != sub_v38["task4"].values
        print(f"Task 4 test diffs vs v38 ({v38_path.name}): {int(diff_v38.sum())} rows total, {int((diff_v38 & dark4_te).sum())} gated")

    print("Test class shares [0..4]:")
    print("  train :", np.round(prior4, 4).tolist())
    print("  v39   :", np.round(np.bincount(sub_v39["task4"].values, minlength=5) / len(test_names), 4).tolist())
    print("  v41   :", np.round(np.bincount(pred4_v41, minlength=5) / len(test_names), 4).tolist())
    print(f"Saved {out_path.name} ({out_df.shape}) | sha256: {sha}")


if __name__ == "__main__":
    main()
