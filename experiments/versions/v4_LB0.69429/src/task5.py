"""Task 5 - Account Security Monitoring (unsupervised anomaly detection, metric: F3).

Only test.csv exists, so every detector is fitted on the unlabeled test features (allowed:
there are no labels and nothing else to fit on). Pipeline:
  1. Domain features: impossible-travel geodesics, invalid-coordinate flags, per-account
     period-4-vs-own-history deviation z-scores (the dominant takeover signature found in
     EDA), burstiness (max/mean, variance, first differences).
  2. Heterogeneous detectors (Aggarwal & Sathe, 2015): Isolation Forest, ECOD, COPOD,
     LOF, robust Mahalanobis (MCD) and a deep Autoencoder.
  3. Rank-normalised score fusion + F3-aware contamination choice.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from .config import DATA_DIRS, ID_COL, SEED

FLAG_COLS = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
STATIC_COLS = ["account_age_days", "behavior_stability"] + FLAG_COLS
EARTH_R = 6371.0


def load():
    return pd.read_csv(DATA_DIRS[5] / "test.csv")


def temporal_groups(df):
    return sorted({c.rsplit("_", 1)[0] for c in df.columns if c[-2:] in ("_1", "_2", "_3", "_4")})


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.clip(a, -90, 90) if i % 2 == 0 else a)
                              for i, a in enumerate((lat1, lon1, lat2, lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_features(df):
    """Return (features DataFrame, list of 'deviation' columns)."""
    X = pd.DataFrame(index=df.index)
    groups = [g for g in temporal_groups(df) if g not in ("login_lat", "login_lon")]

    # --- geo: impossible travel & invalid coordinates
    lat = df[[f"login_lat_{i}" for i in range(1, 5)]].values
    lon = df[[f"login_lon_{i}" for i in range(1, 5)]].values
    hops = np.column_stack([haversine(lat[:, i], lon[:, i], lat[:, i + 1], lon[:, i + 1]) for i in range(3)])
    X["geo_hop_12"], X["geo_hop_23"], X["geo_hop_34"] = hops.T
    X["geo_hop_max"] = np.nanmax(hops, axis=1)
    X["geo_path_km"] = np.nansum(hops, axis=1)
    base_lat, base_lon = np.nanmean(lat[:, :3], axis=1), np.nanmean(lon[:, :3], axis=1)
    X["geo_p4_from_home_km"] = haversine(base_lat, base_lon, lat[:, 3], lon[:, 3])
    X["geo_hop34_over_hist"] = X.geo_hop_34 / (np.nanmean(hops[:, :2], axis=1) + 50)
    X["geo_invalid_coord"] = ((np.abs(lat) > 90).any(axis=1) | (np.abs(lon) > 180).any(axis=1)).astype(float)
    # Impossible travel: kilometres moved per login in period 4 (VPN hopping / hijack)
    X["geo_km_per_login_4"] = X.geo_hop_34 / (df.login_count_4 + 1)

    # --- per-group temporal dynamics
    dev_cols = []
    for g in groups:
        v = df[[f"{g}_{i}" for i in range(1, 5)]].values.astype(float)
        hist = v[:, :3]
        pop_sd = np.nanstd(hist) + 1e-9          # population scale of the feature
        h_mean = np.nanmean(hist, axis=1)
        h_sd = np.nanstd(hist, axis=1)
        dev = (v[:, 3] - h_mean) / pop_sd         # period-4 shift vs own history
        X[f"{g}_dev4"] = dev
        X[f"{g}_selfz4"] = (v[:, 3] - h_mean) / (h_sd + 0.25 * pop_sd)
        diffs = np.diff(v, axis=1) / pop_sd
        X[f"{g}_maxjump"] = np.nanmax(np.abs(diffs), axis=1)
        X[f"{g}_burst"] = np.nanmax(v, axis=1) / (np.abs(np.nanmean(v, axis=1)) + 0.1 * pop_sd)
        X[f"{g}_cv"] = np.nanstd(v, axis=1) / pop_sd
        # All-period level: catches persistent anomalies (e.g. 24/7 bots on one IP with
        # clockwork login times) that never *change* and are invisible to dev4.
        X[f"{g}_level"] = np.nanmean(v, axis=1)
        dev_cols.append(f"{g}_dev4")
    for c in STATIC_COLS:
        X[c] = df[c]
    X["n_flags"] = df[FLAG_COLS].sum(axis=1, min_count=1)
    X["n_missing"] = df.isna().sum(axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, dev_cols


ENVELOPE_UP = ["skill_rating", "skill_trajectory", "kda_mean", "accuracy_mean", "location_changes",
               "purchase_frequency", "purchase_amount_mean", "currency_transfers", "friend_removes",
               "friend_adds", "device_count", "guild_changes"]
ENVELOPE_DOWN = ["reaction_time_mean", "play_pattern_consistency"]


def evidence_rules(df):
    """Deterministic 'hard evidence' rules derived from unsupervised EDA (no labels):
    a period-4 value outside the envelope that ANY account reached in periods 1-3, physically
    impossible coordinates, and the two bimodal gaps (behaviour_stability, IP entropy).
    Used (a) as a diagnostic proxy for detector recall and (b) as an OR-override in the final rule,
    the way production security stacks combine hard rules with ML scores."""
    R = pd.DataFrame(index=df.index)
    for g in ENVELOPE_UP:
        hist = df[[f"{g}_{i}" for i in range(1, 4)]].values
        R[f"{g}_spike"] = df[f"{g}_4"] > np.nanmax(hist)
    for g in ENVELOPE_DOWN:
        hist = df[[f"{g}_{i}" for i in range(1, 4)]].values
        R[f"{g}_drop"] = df[f"{g}_4"] < np.nanmin(hist)
    lat = df[[f"login_lat_{i}" for i in range(1, 5)]].abs()
    lon = df[[f"login_lon_{i}" for i in range(1, 5)]].abs()
    R["impossible_coords"] = (lat > 90).any(axis=1) | (lon > 180).any(axis=1)
    R["low_behavior_stability"] = df.behavior_stability < 0.5
    R["single_ip_bot"] = df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1) < 0.15
    return R


def gap_clusters(df, X=None, tail=0.10, k=200, min_lambda=50.0, min_size=20, max_share=0.03,
                 min_distinct=10, n_gaps=3):
    """Isolated tail clusters: a 1-D density-gap detector (DBSCAN-like, no labels, no thresholds by hand).

    For every continuous column, the widest empty gaps inside each 10% tail are tested. With `dens`
    the density of the k bulk accounts next to a gap of width `w`, lambda = dens * w is the number of
    accounts expected inside the gap if the bulk simply continued, and P(gap empty) ~ exp(-lambda).
    A large lambda therefore means the accounts beyond the gap are a *separate population*, not a
    long tail. Groups smaller than `min_size` (single outliers, e.g. longitudes wrapping past +-180)
    or larger than `max_share` (zero-inflated counts) and point masses are ignored.
    Returns (table of accepted groups, boolean DataFrame rows x groups)."""
    cols = {c: df[c] for c in df.columns if c != ID_COL and c not in FLAG_COLS}
    if X is not None:  # wrap-aware geo features (raw lat/lon gaps are coordinate artefacts)
        cols.update({c: X[c] for c in ("geo_hop_max", "geo_p4_from_home_km") if c in X})
    n_all = len(df)
    rows, masks = [], {}
    for c, x in cols.items():
        v = x.dropna().values.astype(float)
        if np.unique(v).size < 50:
            continue
        s = np.sort(v)
        n = len(s)
        g = np.diff(s)
        for side in ("low", "high"):
            idx = np.arange(int((1 - tail) * n), n - 1) if side == "high" else np.arange(0, int(tail * n))
            best = None
            for j in idx[np.argsort(-g[idx])[:n_gaps]]:
                size = (n - 1 - j) if side == "high" else (j + 1)
                bulk = s[max(0, j - k):j + 1] if side == "high" else s[j + 1:j + 2 + k]
                lam = (len(bulk) - 1) / max(bulk[-1] - bulk[0], 1e-12) * g[j]
                group = s[j + 1:] if side == "high" else s[:j + 1]
                ok = (lam >= min_lambda and min_size <= size <= max_share * n_all
                      and np.unique(group).size >= min_distinct)
                if ok and (best is None or lam > best[1]):
                    best = (j, lam, size)
            if best is None:
                continue
            j, lam, size = best
            cut = (s[j] + s[j + 1]) / 2
            m = (x > cut) if side == "high" else (x < cut)
            name = f"{c} {'>' if side == 'high' else '<'} {cut:.4g}"
            masks[name] = m.fillna(False).values
            rows.append(dict(group=name, column=c, side=side, cut=cut, size=int(size), share=size / n_all,
                             gap_from=s[j], gap_to=s[j + 1], expected_in_gap=lam))
    table = pd.DataFrame(rows).sort_values("expected_in_gap", ascending=False).reset_index(drop=True)
    return table, pd.DataFrame(masks, index=df.index)[table.group.tolist()]


def merge_families(masks, min_overlap=0.5):
    """Union-find over groups that share >= min_overlap of the smaller group: one family = one
    planted behaviour seen through several columns (e.g. a bot has low IP entropy AND clockwork logins)."""
    names = list(masks.columns)
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    M = masks.values
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            inter = (M[:, a] & M[:, b]).sum()
            if inter >= min_overlap * min(M[:, a].sum(), M[:, b].sum()):
                parent[find(a)] = find(b)
    fam = {}
    for i, nm in enumerate(names):
        fam.setdefault(find(i), []).append(nm)
    return [sorted(v, key=lambda n: -masks[n].sum()) for v in fam.values()]


def detector_columns(X):
    """Canonical detector space: period-4 deviations + all-period levels + geo + static/flags."""
    dev = [c for c in X.columns if c.endswith("_dev4")]
    level = [c for c in X.columns if c.endswith("_level")]
    geo = [c for c in X.columns if c.startswith("geo_")]
    return dev + level + geo + STATIC_COLS + ["n_flags"]


def detector_matrix(X, cols, method="robust", clip=50.0):
    """Median-impute and scale.

    'robust'  : (x - median) / IQR, clipped at +-clip. Keeps tail magnitude, which is exactly
                what separates injected takeovers from noisy-but-normal accounts.
    'quantile': map to a normal via ranks; bounded tails, useful for density/neighbour methods
                that would otherwise be dominated by a single extreme coordinate.
    """
    M = X[cols].copy()
    M = M.fillna(M.median())
    if method == "quantile":
        return QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                   random_state=SEED).fit_transform(M)
    Z = RobustScaler(quantile_range=(10, 90)).fit_transform(M)
    return np.clip(Z, -clip, clip)


# ------------------------------------------------------------------------ detectors
def score_iforest(Z, seed=SEED):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=600, max_samples=512, random_state=seed, n_jobs=-1).fit(Z)
    return -m.score_samples(Z)


def score_ecod(Z):
    from pyod.models.ecod import ECOD
    return ECOD().fit(Z).decision_scores_


def score_copod(Z):
    from pyod.models.copod import COPOD
    return COPOD().fit(Z).decision_scores_


def score_lof(Z, k=50):
    from sklearn.neighbors import LocalOutlierFactor
    m = LocalOutlierFactor(n_neighbors=k, n_jobs=-1).fit(Z)
    return -m.negative_outlier_factor_


def score_mahalanobis(Z, seed=SEED):
    from sklearn.covariance import MinCovDet
    mcd = MinCovDet(random_state=seed, support_fraction=0.9).fit(Z)
    return mcd.mahalanobis(Z)


def score_autoencoder(Z, epochs=60, seed=SEED, device=None):
    """Under-complete MLP autoencoder; anomaly score = per-row reconstruction MSE."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = device or "cpu"
    Xt = torch.tensor(Z, dtype=torch.float32, device=device)
    d = Z.shape[1]
    model = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 16), nn.GELU(),
                          nn.Linear(16, 64), nn.GELU(), nn.Linear(64, d)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(256):
            xb = Xt[idx.to(device)]
            # Huber loss: the few anomalies cannot dominate what "normal" looks like
            loss = nn.functional.smooth_l1_loss(model(xb + 0.05 * torch.randn_like(xb)), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        err = ((model(Xt) - Xt) ** 2).mean(1)
    return err.cpu().numpy()


def score_extreme_value(Z, top=2):
    """Extreme-value detector: mean of each row's `top` largest |robust z| values. Catches
    accounts that are wildly abnormal in only one or two behaviours, which sum-based scores
    (ECOD/COPOD) dilute across ~150 dimensions."""
    A = np.sort(np.abs(Z), axis=1)[:, -top:]
    return A.mean(axis=1)


DETECTORS = {
    "ExtremeValue": score_extreme_value,
    "IsolationForest": score_iforest,
    "ECOD": score_ecod,
    "COPOD": score_copod,
    "LOF": score_lof,
    "Mahalanobis(MCD)": score_mahalanobis,
    "AutoEncoder": score_autoencoder,
}


def rank_normalize(s):
    return pd.Series(s).rank(pct=True).values


def robust_z(s):
    """Median/MAD standardisation of a detector's raw scores. Unlike ranks, it keeps the *size* of
    an outlier's lead, so one detector that is certain is not out-voted by indifferent ones."""
    s = np.asarray(s, dtype=float)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med)) + 1e-12
    return (s - med) / mad


def consensus_weights(Z, floor=0.6):
    """Label-free detector selection in the spirit of SELECT (Rayana & Akoglu, 2016).

    Each detector is scored by its Spearman agreement with the leave-one-out consensus (the mean of
    the other detectors' robust-z scores); weights are proportional to agreement above `floor`, so
    detectors that disagree with everyone else (i.e. mostly fire on noise) are down-weighted or
    dropped without ever looking at labels."""
    Z = np.asarray(Z, dtype=float)
    k = Z.shape[1]
    agree = np.zeros(k)
    for j in range(k):
        loo = np.delete(Z, j, axis=1).mean(axis=1)
        agree[j] = pd.Series(Z[:, j]).corr(pd.Series(loo), method="spearman")
    w = np.clip(agree - floor, 0, None)
    return w / w.sum(), agree


def fuse(rank_matrix, how="mean"):
    """Combine rank-normalised detector scores (rows x detectors).
    mean : consensus averaging (variance reduction, favours precision)
    max  : OR-fusion, any detector can raise an alarm (favours recall)
    top2 : mean of each row's two highest ranks, a compromise that needs a second opinion."""
    R = np.asarray(rank_matrix)
    if how == "mean":
        return R.mean(axis=1)
    if how == "max":
        return R.max(axis=1)
    if how == "top2":
        return np.sort(R, axis=1)[:, -2:].mean(axis=1)
    raise ValueError(how)


def knee_point(sorted_scores):
    """Kneedle-style knee: point of the descending score curve farthest from its chord."""
    y = np.asarray(sorted_scores, dtype=float)
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    dist = np.abs((yn[-1] - yn[0]) * x - (x[-1] - x[0]) * yn + x[-1] * yn[0] - yn[-1] * x[0])
    return int(np.argmax(dist))


def expected_fbeta(rate_flagged, true_rate, beta=3.0):
    """F-beta of the positive class for a *perfect ranker* that flags `rate_flagged` of rows when
    `true_rate` are anomalous; shows why over-flagging is far cheaper than under-flagging."""
    tp = np.minimum(rate_flagged, true_rate)
    p, r = tp / rate_flagged, tp / true_rate
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)
