"""Task 1 - Anti-Cheat Pre-Filter (binary, metric: F2)."""
import numpy as np
import pandas as pd

from .config import DATA_DIRS, ID_COL, TARGETS

TARGET = TARGETS[1]

# Raw-skill signals an aimbot inflates vs. "human" game-sense signals it cannot fake.
MECH_COLS = ["headshot_percentage", "accuracy_score", "aiming_smoothness", "spray_control_score",
             "crosshair_placement", "kill_death_ratio", "damage_per_round", "first_blood_rate"]
SENSE_COLS = ["game_sense_score", "map_knowledge", "team_play_score", "buy_decision_score",
              "communication_rate", "utility_usage_rate", "clutch_success_rate"]


def load():
    d = DATA_DIRS[1]
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    n_unlabeled = train[TARGET].isna().sum()
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    train[TARGET] = train[TARGET].astype(int)
    return train, test, n_unlabeled


def add_features(df, ref=None):
    """Mechanical-anomaly ratios. `ref` (train features) supplies z-score statistics so that
    train and test are standardised identically (no test statistics are used)."""
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns]).copy()
    raw_cols = list(X.columns)
    eps = 1e-6
    X["hs_per_accuracy"] = X.headshot_percentage / (X.accuracy_score + eps)
    X["kd_per_reaction"] = X.kill_death_ratio / (X.reaction_time_ms + eps) * 100
    X["accuracy_per_reaction"] = X.accuracy_score / (X.reaction_time_ms + eps) * 100
    X["hs_x_smooth"] = X.headshot_percentage * X.aiming_smoothness
    X["dmg_per_survival"] = X.damage_per_round / (X.survival_time_avg + eps)
    X["kd_x_winrate"] = X.kill_death_ratio * X.win_rate / 100
    X["level_per_day"] = X.level / (X.account_age_days + 1)
    X["reports_per_day"] = X.reports_received / (X.account_age_days + 1) * 100
    X["reports_per_friend"] = X.reports_received / (X.friend_network_size + 1)
    X["device_changes_per_day"] = X.device_changes_count / (X.account_age_days + 1) * 100
    X["kd_per_level"] = X.kill_death_ratio / (X.level + eps) * 100
    X["daily_playtime"] = X.avg_session_length_min * X.sessions_per_day
    X["skill_per_playtime"] = X.kill_death_ratio / (np.log1p(X.daily_playtime) + eps)
    X["input_x_smooth"] = X.input_consistency_score * X.aiming_smoothness

    # Composite indices: aimbots show mechanical skill far above their game sense.
    stats = (ref if ref is not None else X)[MECH_COLS + SENSE_COLS + ["reaction_time_ms"]].agg(["mean", "std"])
    z = lambda c: (X[c] - stats.loc["mean", c]) / stats.loc["std", c]
    mech = pd.concat([z(c) for c in MECH_COLS] + [-z("reaction_time_ms")], axis=1).mean(axis=1)
    sense = pd.concat([z(c) for c in SENSE_COLS], axis=1).mean(axis=1)
    X["mech_index"] = mech
    X["sense_index"] = sense
    X["mech_minus_sense"] = mech - sense
    X["n_missing"] = df[raw_cols].isna().sum(axis=1)
    return X


def build(train, test):
    Xtr_raw = train.drop(columns=[ID_COL, TARGET])
    X = add_features(train, ref=Xtr_raw)
    X_test = add_features(test, ref=Xtr_raw)
    return X, train[TARGET].values, X_test


# Round 11 (2026-11-11): extra stack members built the way last year's 1st-place team built theirs (CPE342 1/2025, public repo
# github.com/Saltywan/CPE342-KAGGLE; idea re-implemented, no code copied). They use 24 of the 31 raw columns, leaving out
# device_changes_count, input_consistency_score, avg_session_length_min, sessions_per_day, movement_pattern_score, map_knowledge and
# clutch_success_rate, and fill missing cells with the training median. Different inputs make them diverse members for our stack.
C24 = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms", "account_age_days",
       "level", "level_progression_speed", "friend_network_size", "reports_received", "night_play_ratio", "weapon_switch_speed",
       "aiming_smoothness", "spray_control_score", "game_sense_score", "communication_rate", "team_play_score", "buy_decision_score",
       "first_blood_rate", "survival_time_avg", "damage_per_round", "utility_usage_rate", "crosshair_placement"]


def build_c24(train, test):
    """The 24 columns above with missing cells filled by the TRAINING median (no test statistics)."""
    med = train[C24].median()
    return train[C24].fillna(med).reset_index(drop=True), test[C24].fillna(med).reset_index(drop=True)
