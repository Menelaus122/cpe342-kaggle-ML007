"""Task 2 - Player Segment Classification (4-class, metric: macro-F1).

EDA shows the four segments form a 2x2 design over two latent factors:
    engagement (Casual/Social low  vs. Grinder/Whale high)
    spending   (Casual/Grinder low vs. Social/Whale high)
Each factor is measured by many weak, noisy, partly-missing indicators, so NaN-aware composite
z-score indices are the key engineered features (trees struggle to sum many weak signals).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import DATA_DIRS, ID_COL, SEED, TARGETS

TARGET = TARGETS[2]
NOISE_COLS = ["random_metric_1", "random_metric_2", "random_metric_3"]
ORDINAL_MAPS = {
    "engagement_level": {"Low": 0, "Mid": 1, "High": 2, "Veteran": 3},
    "loyalty_tier": {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3, "Diamond": 4},
    "skill_tier": {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3, "Diamond": 4},
}
NOMINAL_COLS = ["region", "platform", "device_type", "payment_method", "language",
                "account_status", "player_type_tag"]

ENGAGEMENT_POS = ["play_frequency", "avg_session_duration", "total_playtime_hours", "login_streak",
                  "friend_count", "chat_activity_score", "friend_invites_sent", "gifts_sent_received",
                  "tournament_entries", "competitive_rank", "win_rate_ranked", "speed_of_progression",
                  "peak_concurrent_hours", "team_play_percentage", "engagement_level"]
ENGAGEMENT_NEG = ["days_since_last_login"]
SPENDING_POS = ["total_spending_thb", "avg_monthly_spending", "spending_frequency", "rare_items_count",
                "collection_progress", "item_type_preference_cosmetic", "vip_tier", "loyalty_tier"]


def load():
    d = DATA_DIRS[2]
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")


def _base(df):
    X = df.drop(columns=[c for c in (ID_COL, TARGET, *NOISE_COLS) if c in df.columns]).copy()
    for c, m in ORDINAL_MAPS.items():
        X[c] = X[c].map(m).astype(float)
    return X


def _composite(X, stats, pos, neg=()):
    zs = [(X[c] - stats.loc["mean", c]) / stats.loc["std", c] for c in pos]
    zs += [-(X[c] - stats.loc["mean", c]) / stats.loc["std", c] for c in neg]
    Z = pd.concat(zs, axis=1)
    return Z.mean(axis=1, skipna=True), Z.notna().sum(axis=1)


def add_features(X, stats):
    eps = 1e-5
    X["engagement_index"], X["engagement_n_obs"] = _composite(X, stats, ENGAGEMENT_POS, ENGAGEMENT_NEG)
    X["spending_index"], X["spending_n_obs"] = _composite(X, stats, SPENDING_POS)
    X["eng_x_spend"] = X.engagement_index * X.spending_index
    X["eng_minus_spend"] = X.engagement_index - X.spending_index
    X["spend_per_session"] = X.total_spending_thb / (X.play_frequency * X.avg_session_duration + eps)
    X["spend_per_hour"] = X.total_spending_thb / (X.total_playtime_hours + eps)
    X["social_engagement"] = X.friend_count / (X.play_frequency + eps)
    X["monetization_velocity"] = X.avg_monthly_spending * X.spending_frequency
    X["spend_per_txn"] = X.avg_monthly_spending / (X.spending_frequency + eps)
    X["rare_per_spend"] = X.rare_items_count / (X.total_spending_thb + eps) * 1000
    X["chat_per_friend"] = X.chat_activity_score / (X.friend_count + 1)
    X["gifts_per_friend"] = X.gifts_sent_received / (X.friend_count + 1)
    X["n_missing"] = X.isna().sum(axis=1)
    return X


def _te_encode(keys_fit, Y_fit, keys_apply, prior, smoothing):
    df = pd.DataFrame(Y_fit)
    df["k"] = keys_fit
    g = df.groupby("k")
    enc = (g.sum() + smoothing * prior).div(g.size() + smoothing, axis=0)
    return enc.reindex(keys_apply).fillna(pd.Series(prior)).values


def target_encode_oof(train_cat, y, apply_cats, n_classes=4, n_folds=5, smoothing=20, seed=SEED):
    """K-fold target encoding (per-class rates, smoothed toward the prior).

    Rows of `train_cat` are encoded with statistics from the *other* inner folds only; every
    frame in `apply_cats` is encoded with the full `train_cat` statistics. NaN is a category."""
    tr = train_cat.astype(object).fillna("NA").astype(str).values
    Y = np.eye(n_classes)[np.asarray(y)]
    prior = Y.mean(0)
    oof = np.zeros((len(tr), n_classes))
    for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tr, y):
        oof[b] = _te_encode(tr[a], Y[a], tr[b], prior, smoothing)
    applied = [_te_encode(tr, Y, c.astype(object).fillna("NA").astype(str).values, prior, smoothing)
               for c in apply_cats]
    return oof, applied


def te_fold_fn(X_tr, y_tr, X_others):
    """Per-CV-fold hook: target-encode nominal columns using ONLY the fold's training rows, so
    neither validation labels nor test data ever influence the encodings (fully leakage-free)."""
    X_tr = X_tr.copy()
    X_others = [x.copy() for x in X_others]
    for c in NOMINAL_COLS:
        oof, applied = target_encode_oof(X_tr[c], y_tr, [x[c] for x in X_others])
        for k in range(4):
            X_tr[f"{c}_te{k}"] = oof[:, k]
            for x, a in zip(X_others, applied):
                x[f"{c}_te{k}"] = a[:, k]
    return X_tr, X_others


def build(train, test):
    y = train[TARGET].values
    Xtr, Xte = _base(train), _base(test)
    stats = Xtr[ENGAGEMENT_POS + ENGAGEMENT_NEG + SPENDING_POS].agg(["mean", "std"])
    Xtr, Xte = add_features(Xtr, stats), add_features(Xte, stats)
    for c in NOMINAL_COLS:
        freq = Xtr[c].fillna("NA").value_counts(normalize=True)
        Xtr[f"{c}_freq"] = Xtr[c].fillna("NA").map(freq).values
        Xte[f"{c}_freq"] = Xte[c].fillna("NA").map(freq).fillna(0).values
        cats = sorted(Xtr[c].dropna().unique())
        Xtr[c] = pd.Categorical(Xtr[c], categories=cats)
        Xte[c] = pd.Categorical(Xte[c], categories=cats)
    return Xtr, y, Xte
