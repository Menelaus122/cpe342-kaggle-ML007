"""Global configuration for the CPE342 Karena Player Intelligence pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIRS = {
    1: ROOT / "01_binary_classification",
    2: ROOT / "02_multiclass_classification",
    3: ROOT / "03_regression",
    4: ROOT / "04_image_classification" / "_download" / "task4_image_classification",
    5: ROOT / "05_anomaly_detection",
}
SAMPLE_SUB = ROOT / "sample_submission.csv"

OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
PRED_DIR = OUT / "preds"
PARAM_DIR = OUT / "params"
MODEL_DIR = OUT / "models"
for _d in (OUT, FIG_DIR, PRED_DIR, PARAM_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_JOBS = -1

# How the leaderboard averages F-beta for the binary tasks (1 and 5).
# The official instruction PDF defines F2 = 5PR/(4P+R) and F3 = 10PR/(9P+R) for the
# positive class (cheater / anomaly) and marks only Task 4 as "(Macro)", so we optimise
# the positive-class score. "macro" is kept as a switch for the alternative reading.
FBETA_AVERAGE = "binary"

TARGETS = {1: "is_cheater", 2: "segment", 3: "spending_30d"}
ID_COL = "player_id"

# Task 4: the datacard's integer coding is the alphabetical order of the string labels.
T4_CLASSES = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
T4_CLASS_NAMES = ["7-Eleven Knights", "FiveN", "Free Fried", "PUBE Mobile", "Roblock"]

N_TEST = 25_889
