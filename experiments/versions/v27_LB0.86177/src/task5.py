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
    """Under-complete MLP autoencoder; anomaly score = per-row reconstruction MSE.

    Always trained on the CPU: `device` is accepted for old callers and ignored, so the detector never takes GPU memory from a
    Task 4 run on the same machine (on Windows an empty CUDA_VISIBLE_DEVICES does not hide the GPU)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    device = "cpu"
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


# --------------------------------------------------------------------------- residual dependence pockets
# A group that is unusual in two features at once, but not extreme in either, has no 1-D gap and hides from
# gap_clusters. Normal accounts draw (almost) independent features, so on a quantile grid of a feature pair every
# cell should hold n / G^2 accounts; a hidden group leaves a pocket of excess accounts in one cell.
MECHANICAL_PAIRS = {frozenset(("geo", "login_count"))}      # km-per-login is computed from the login count


def _base_feature(c):
    import re
    c = re.sub(r"_(level|cv|burst|maxjump|dev4|selfz4)$", "", c)
    return "geo" if c.startswith("geo") else re.sub(r"_(mean|std)$", "", c)


def _quantile_codes(R, G):
    Q = np.full(R.shape, -1, dtype=np.int16)
    for j in range(R.shape[1]):
        v = R[:, j]
        ok = ~np.isnan(v)
        r = pd.Series(v[ok]).rank(pct=True, method="first").values
        Q[ok, j] = np.minimum((r * G).astype(int), G - 1)
    return Q


def _shuffle_columns(R, rng):
    return np.column_stack([rng.permutation(R[:, j]) for j in range(R.shape[1])])


def _max_cell_z(R, cols, G):
    Q = _quantile_codes(R, G)
    base = [_base_feature(c) for c in cols]
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if base[i] == base[j] or frozenset((base[i], base[j])) in MECHANICAL_PAIRS:
                continue                                     # same behaviour seen twice, or a built-in ratio
            ok = (Q[:, i] >= 0) & (Q[:, j] >= 0)
            cnt = np.bincount(Q[ok, i] * G + Q[ok, j], minlength=G * G).reshape(G, G).astype(float)
            exp = np.outer(cnt.sum(1), cnt.sum(0)) / cnt.sum()
            z = (cnt - exp) / np.sqrt(np.maximum(exp, 1))
            k = int(np.argmax(z))
            out.append((z.flat[k], cols[i], cols[j], k // G, k % G, cnt.flat[k], exp.flat[k]))
    return pd.DataFrame(out, columns=["z", "a", "b", "bin_a", "bin_b", "observed", "expected"]).sort_values(
        "z", ascending=False, ignore_index=True)


def dependence_pockets(X, mask, G=20, n_null=2, seed=SEED):
    """Blind scan of every feature pair among the accounts in `mask` (20x20 quantile grid). Returns the table of
    each pair's most over-full cell, z = (observed - expected) / sqrt(expected), and the noise level: the largest
    z of the same scan on column-shuffled data (every marginal kept, every group destroyed)."""
    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 50]
    R = X.loc[mask, cols].to_numpy(float)
    rng = np.random.default_rng(seed)
    noise = max(_max_cell_z(_shuffle_columns(R, rng), cols, G).z.max() for _ in range(n_null))
    return _max_cell_z(R, cols, G), float(noise)


def pocket_detector(X, known, feats, n_perm=300, max_k=3000, beta=3.0, seed=SEED):
    """Label-free detector for a dependence pocket, with a label-free cut-off.

    Score: min over `feats` of the rank percentile among accounts not in `known` (high = high in all features).
    Null: the same score after shuffling each column independently, repeated n_perm times. For the top-k accounts
    the expected number of NORMAL accounts is the mean shuffled count above the k-th score, so
    est_TP(k) = k - that count, and the group size is estimated as max_k est_TP(k).
    Cut-off: k maximising the expected positive-class F_beta, with the `known` accounts counted as true:
        F(k) = (1 + b^2) (n_known + est_TP(k)) / (b^2 (n_known + G_hat) + n_known + k).
    Returns (flags over all accounts, info dict)."""
    rng = np.random.default_rng(seed)
    known = np.asarray(known, bool)
    rest = np.where(~known)[0]
    U = X.loc[~known, feats].rank(pct=True).to_numpy()

    def score(T):
        with np.errstate(all="ignore"):
            return np.nanmin(T, axis=1) + 1e-9 * np.nansum(T, axis=1)

    S = score(U)
    S = np.where(np.isnan(S), -np.inf, S)
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(len(Ss))
    for _ in range(n_perm):
        Sp = score(_shuffle_columns(U, rng))
        Sp = np.sort(np.where(np.isnan(Sp), -np.inf, Sp))
        cnt += len(Sp) - np.searchsorted(Sp, Ss, side="left")
    cnt /= n_perm
    k = np.arange(1, len(Ss) + 1)
    est = np.maximum.accumulate(k - cnt)
    n_known = int(known.sum())
    g_hat = float(est[:max_k].max())
    b2 = beta ** 2
    f = (1 + b2) * (n_known + est) / (b2 * (n_known + g_hat) + n_known + k)
    k_star = int(np.argmax(f[:max_k])) + 1
    flags = np.zeros(len(known), bool)
    flags[rest[order[:k_star]]] = True
    info = dict(k=k_star, group_size=g_hat, est_tp=float(est[k_star - 1]), precision=float(est[k_star - 1] / k_star),
                expected_f=float(f[k_star - 1]), expected_f_known_only=float((1 + b2) * n_known / (b2 * (n_known + g_hat) + n_known)),
                curve=pd.DataFrame({"k": k[:max_k], "est_tp": est[:max_k], "expected_f": f[:max_k]}), order=rest[order])
    return flags, info


# --------------------------------------------------------------------------- pocket shape: one more column and the density ratio
# The min-rank score flags a square corner of the two pocket columns. Two refinements, both label-free:
#  (a) the provisional pocket may be shifted in a further column of the same kind (an all-period level) that the pair scan could not
#      pick, because a pair scan looks at two columns at a time;
#  (b) by the Neyman-Pearson lemma the most powerful region for "real joint density" against "independent columns" (the shuffle null)
#      is a level set of the density ratio p(x) / prod_j p_j(x_j). In rank space the denominator is uniform, so the ratio is the copula
#      density, and 1 - 1/ratio is the local posterior of not being a normal account (Efron's local false discovery rate).
def pocket_extra_column(X, known, pocket, feats, alpha=0.001):
    """Which further all-period level column is the provisional pocket shifted in?

    Candidates: the columns of the same kind as `feats` (suffix `_level`, more than 50 distinct values), from behaviours other than
    those of `feats`. Ranks are taken among the accounts not in `known`. For each candidate, the mean rank of the pocket accounts is
    compared with what the bulk dependence predicts (least squares of the candidate's rank on the ranks of `feats`, fitted on the
    accounts outside `known` and the pocket): z = mean residual of the pocket / (residual SD / sqrt(pocket size)). A column is added
    when |z| clears the two-sided Bonferroni bound for `alpha` over all candidates; at most one (the ratio below is estimated in at
    most three dimensions). Returns (column or None, table sorted by |z|, bound)."""
    known, pocket = np.asarray(known, bool), np.asarray(pocket, bool)
    from scipy.stats import norm
    base = {_base_feature(c) for c in feats}
    cands = [c for c in X.columns if c.endswith("_level") and c not in feats and _base_feature(c) not in base and X[c].nunique() > 50]
    R = X.loc[~known, list(feats) + cands].rank(pct=True)
    R = R.fillna(0.5)
    pk = pocket[~known]
    A = np.column_stack([np.ones(len(R)), R[list(feats)].to_numpy()])
    rows = []
    for c in cands:
        y = R[c].to_numpy()
        beta_, *_ = np.linalg.lstsq(A[~pk], y[~pk], rcond=None)
        res = y - A @ beta_
        z = res[pk].mean() / (res[~pk].std() / np.sqrt(pk.sum()))
        rows.append((c, float(y[pk].mean()), float((A @ beta_)[pk].mean()), float(z)))
    tab = pd.DataFrame(rows, columns=["column", "mean rank in pocket", "predicted from the pocket columns", "z"])
    tab = tab.reindex(tab.z.abs().sort_values(ascending=False).index).reset_index(drop=True)
    bound = float(norm.isf(alpha / (2 * len(cands))))
    extra = tab.column[0] if len(tab) and abs(tab.z[0]) > bound else None
    return extra, tab, bound


class DensityRatio:
    """Cross-fitted smoothed-histogram estimate of log p(z) / prod_j phi(z_j) in normal-score space (d = 2 or 3).

    Each repeat splits the accounts into two random halves; an account is scored with the ratio fitted on the half it is NOT in, and
    the repeats are averaged, so the region is not tuned to the noise of the accounts it scores. Histogram bins of w SD, Gaussian
    smoothing of h SD on both the observed counts and the expected (independent normal) counts."""

    def __init__(self, Z, h=0.40, rng=None, reps=20):
        from scipy.stats import norm
        rng = np.random.default_rng(SEED) if rng is None else rng
        self.d, self.h = Z.shape[1], h
        self.w = 0.05 if self.d == 2 else 0.1
        self.edges = np.arange(-4.2, 4.2 + 1e-9, self.w)
        self.p = np.diff(norm.cdf(np.r_[-np.inf, self.edges[1:-1], np.inf]))
        self.fits = []
        for _ in range(reps):
            half = rng.random(len(Z)) < 0.5
            self.fits.append((half, self._fit(Z[half]), self._fit(Z[~half])))

    def _fit(self, Z):
        from scipy.ndimage import gaussian_filter
        H, _ = np.histogramdd(np.clip(Z, -4.199, 4.199), bins=[self.edges] * self.d)
        E = len(Z) * (np.einsum("i,j->ij", self.p, self.p) if self.d == 2 else np.einsum("i,j,k->ijk", self.p, self.p, self.p))
        s = self.h / self.w
        eps = 0.5 / ((2 * np.pi) ** (self.d / 2) * s ** self.d)
        return np.log((gaussian_filter(H, s, mode="constant") + eps) / (gaussian_filter(E, s, mode="constant") + eps))

    def _look(self, L, Z):
        idx = np.clip(((np.clip(Z, -4.199, 4.199) + 4.2) / self.w).astype(int), 0, len(self.edges) - 2)
        return L[tuple(idx.T)]

    def score(self, Z, halves=None):
        """Log-ratio of each row from the fit on the half it is not in (for null rows: a random half per repeat)."""
        out = np.zeros(len(Z))
        for k, (half, LA, LB) in enumerate(self.fits):
            hh = half if halves is None else halves[k]
            out += np.where(hh, self._look(LB, Z), self._look(LA, Z))
        return out / len(self.fits)


def density_ratio_curve(V, h=0.40, reps=200, n_perm=1000, max_k=3000, seed=SEED, block=100):
    """Cross-fitted density-ratio score of the rows of V (rows x d, d = 2, 3 or 4; NaN sits at the median) with the shuffle-null
    estimate of the true members among the top k.

    Normal scores of the ranks; histogram bins of 0.05 / 0.1 / 0.2 SD for d = 2 / 3 / 4, Gaussian smoothing of h SD on the observed
    counts and on the expected counts of independent columns. Each of `reps` random halvings fits one map per half; a row is scored
    by the map of the half it is NOT in, and the maps are discarded after use (streaming), so hundreds of halvings fit in memory.
    Null: `n_perm` column shuffles, fixed in advance; every null row is scored by a random half's map in every halving.
    Defaults (round 32): 200 halvings and 1000 null draws; with v24's 20 / 300 the cut-off moved by up to 25 accounts between seeds,
    with 200 / 1000 k* is 263-265 over five seeds (Jaccard 0.9925-1.000, mean 0.997).
    Returns (row order by score, est_TP curve of length max_k, score)."""
    from functools import reduce
    from scipy.ndimage import gaussian_filter
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    V = np.asarray(V, float)
    n, d = V.shape
    Rk = pd.DataFrame(V).rank(method="first").to_numpy()
    Z = np.nan_to_num(norm.ppf((Rk - 0.5) / n))
    w = {2: 0.05, 3: 0.1, 4: 0.2}[d]
    edges = np.arange(-4.2, 4.2 + 1e-9, w)
    nb = len(edges) - 1
    p = np.diff(norm.cdf(np.r_[-np.inf, edges[1:-1], np.inf]))
    sig = h / w
    eps = 0.5 / ((2 * np.pi) ** (d / 2) * sig ** d)
    fE1 = gaussian_filter(reduce(np.multiply.outer, [p] * d), sig, mode="constant").ravel()      # smoothed expected counts per row

    def flat(Zx):
        idx = np.clip(((np.clip(Zx, -4.199, 4.199) + 4.2) / w).astype(int), 0, nb - 1)
        return np.ravel_multi_index(tuple(idx.T), (nb,) * d)

    fi = flat(Z)
    NI = np.stack([flat(_shuffle_columns(Z, rng)) for _ in range(n_perm)]).astype(np.int32)
    S = np.zeros(n)
    SN = np.zeros((n_perm, n), np.float32)
    for _ in range(reps):
        half = rng.random(n) < 0.5
        maps = []
        for m in (half, ~half):
            H = np.bincount(fi[m], minlength=nb ** d).reshape((nb,) * d).astype(float)
            maps.append(np.log((gaussian_filter(H, sig, mode="constant").ravel() + eps) / (m.sum() * fE1 + eps)))
        LA, LB = maps
        S += np.where(half, LB[fi], LA[fi])
        for j in range(0, n_perm, block):
            idx = NI[j:j + block]
            SN[j:j + block] += np.where(rng.random(idx.shape) < 0.5, LB[idx], LA[idx])
    S /= reps
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(n)
    for j in range(n_perm):
        Sp = np.sort(SN[j].astype(float) / reps)
        cnt += n - np.searchsorted(Sp, Ss, side="left")
    cnt /= n_perm
    est = np.maximum.accumulate(np.arange(1, n + 1) - cnt)
    return order, est[:max_k], S


def pocket_density_ratio(X, known, cols, h=0.40, reps=200, n_perm=1000, max_k=3000, seed=SEED):
    """Pocket score = density_ratio_curve of `cols` over the accounts not in `known`.
    Returns (account indices in score order, est_TP curve of length max_k, score of the accounts not in `known`)."""
    known = np.asarray(known, bool)
    rest = np.where(~known)[0]
    order, est, S = density_ratio_curve(X.loc[~known, cols].to_numpy(float), h, reps, n_perm, max_k, seed)
    return rest[order], est, S


# --------------------------------------------------------------------------- pocket shape (c): the whole period profile
# The pocket's members are high in EVERY period of its behaviours (round 32), which the all-period mean (_level) only partly uses.
# A level and a spread of the same behaviour are dependent for normal accounts too, so the column-shuffle null is not valid for
# per-period columns. Valid null: shuffle whole behaviour BLOCKS (a behaviour's four periods stay together), which keeps every
# within-behaviour dependence and removes only the dependence across behaviours, the same hypothesis as the 3-D null.
# Setting picked from four on the label-free expected F3 of a 6-halving run (round 32, fk_r32_t5_pocket_12d.py). No early stopping:
# a validation part only breaks the exact 1 : m balance of the block values by chance.
PROFILE_CLF = dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False)


def _nscore(v):
    """Normal score of the rank among the non-missing rows; missing stays NaN (the classifier routes it natively)."""
    from scipy.stats import norm
    r = pd.Series(v).rank(method="first").to_numpy()
    return norm.ppf((r - 0.5) / np.sum(~np.isnan(v)))


def profile_block(P):
    """Features of one behaviour from its periods only (rows x periods): the normal score of each period, of the period mean (the
    _level column), and the min / max / SD of the observed period scores."""
    import warnings
    Zp = np.column_stack([_nscore(P[:, j]) for j in range(P.shape[1])])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lev = _nscore(np.nanmean(P, 1))
        return np.column_stack([Zp, lev, np.nanmin(Zp, 1), np.nanmax(Zp, 1), np.nanstd(Zp, 1)]).astype(np.float32)


def _profile_rows(blocks, n_periods=4):
    """One row per account from its (possibly shuffled) blocks: the block features plus across-behaviour summaries. A split on a
    single block feature has exactly zero gain (every block value sits in 1 real and m shuffled training rows), so greedy trees need
    the across-behaviour columns to start: min / mean / max of the level scores, min and mean of the per-behaviour lowest period
    scores, and the number of periods in the top 20% over all behaviours."""
    Xb = np.hstack(blocks)
    w = blocks[0].shape[1]
    L, Mn = Xb[:, n_periods::w], Xb[:, n_periods + 1::w]
    P = np.hstack([b[:, :n_periods] for b in blocks])
    with np.errstate(all="ignore"):
        cross = np.column_stack([L.min(1), L.mean(1), L.max(1), Mn.min(1), Mn.mean(1), np.nansum(P > 0.8416, axis=1)])
    return np.hstack([Xb, cross.astype(np.float32)])


def profile_ratio_curve(blocks, reps=40, n_perm=200, m=3, max_k=3000, seed=SEED):
    """Cross-fitted classifier estimate of the log density ratio "real accounts" against "behaviour blocks shuffled independently",
    with the block-shuffle estimate of the true members among the top k.

    `blocks`: one profile_block array per behaviour. Each of `reps` random halvings fits one gradient-boosted classifier per half
    (real rows of the half against m block shuffles of the same half; log-odds = log ratio + constant); a real row is scored by the
    classifier of the half it is NOT in, scores averaged over the halvings. Null: `n_perm` block shuffles of all rows, fixed in
    advance; every null row is scored by a random half's classifier in every halving (as in density_ratio_curve).
    Returns (row order by score, est_TP curve of length max_k, score)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    n = len(blocks[0])
    perms = np.stack([np.stack([rng.permutation(n) for _ in blocks]) for _ in range(n_perm)]).astype(np.int32)
    real = _profile_rows(blocks)
    S = np.zeros(n)
    SN = np.zeros((n_perm, n), np.float32)
    for _ in range(reps):
        half = rng.random(n) < 0.5
        models = []
        for hm in (half, ~half):
            idx = np.where(hm)[0]
            Xs = [real[idx]] + [_profile_rows([f[idx[rng.permutation(len(idx))]] for f in blocks]) for _ in range(m)]
            yy = np.r_[np.ones(len(idx)), np.zeros(m * len(idx))]
            models.append(HistGradientBoostingClassifier(**PROFILE_CLF, random_state=int(rng.integers(2**31 - 1))).fit(np.vstack(Xs), yy))
        mA, mB = models
        S[half] += mB.decision_function(real[half])
        S[~half] += mA.decision_function(real[~half])
        for j in range(n_perm):
            Xn = _profile_rows([f[perms[j, b]] for b, f in enumerate(blocks)])
            pick = rng.random(n) < 0.5
            SN[j, pick] += mB.decision_function(Xn[pick])
            SN[j, ~pick] += mA.decision_function(Xn[~pick])
    S /= reps
    order = np.argsort(-S)
    Ss = S[order]
    cnt = np.zeros(n)
    for j in range(n_perm):
        Sp = np.sort(SN[j].astype(float) / reps)
        cnt += n - np.searchsorted(Sp, Ss, side="left")
    cnt /= n_perm
    est = np.maximum.accumulate(np.arange(1, n + 1) - cnt)
    return order, est[:max_k], S


def pocket_profile_ratio(df, known, cols, reps=40, n_perm=200, max_k=3000, seed=SEED):
    """Pocket score = profile_ratio_curve of the per-period columns behind the _level columns `cols`, over the accounts not in `known`.
    Returns (account indices in score order, est_TP curve, score of the accounts not in `known`), or None if a column of `cols` is
    not an all-period level with its four period columns in df."""
    known = np.asarray(known, bool)
    rest = np.where(~known)[0]
    beh = [c[:-len("_level")] for c in cols if c.endswith("_level")]
    if len(beh) != len(cols) or any(f"{b}_{i}" not in df.columns for b in beh for i in range(1, 5)):
        return None
    sub = df.iloc[rest]
    blocks = [profile_block(sub[[f"{b}_{i}" for i in range(1, 5)]].to_numpy(float)) for b in beh]
    order, est, S = profile_ratio_curve(blocks, reps, n_perm, 3, max_k, seed)
    return rest[order], est, S


def pocket_refine(X, known, pocket, pinfo, feats, max_k=3000, beta=3.0, seed=SEED, reps=200, n_perm=1000, df=None,
                  profile_reps=40, profile_perm=200):
    """Second pass of the pocket: (a) one more column if the provisional pocket is shifted in it (pocket_extra_column), then (b) the
    cross-fitted density ratio of the pocket columns against the min-rank score. The two est_TP curves are compared at ONE common
    group size (the larger estimate, so a weak score is not rewarded for under-estimating the group); the score with the higher
    expected F_beta is kept, cut at its own maximum. Without a further column the min-rank pocket is returned unchanged.
    With `df` (the raw table) a third score joins the comparison: (c) the cross-fitted classifier ratio on the whole period
    profile of the pocket's behaviours against the block-shuffle null (pocket_profile_ratio).
    Returns (flags over all accounts, info dict)."""
    known = np.asarray(known, bool)
    extra, tab, bound = pocket_extra_column(X, known, pocket, feats)
    info = dict(extra=extra, shift_table=tab, bound=bound, cols=list(feats), method="min-rank")
    if extra is None:
        return np.asarray(pocket, bool), {**pinfo, **info}
    cols = list(feats) + [extra]
    order_r, est_r, _ = pocket_density_ratio(X, known, cols, reps=reps, n_perm=n_perm, max_k=max_k, seed=seed)
    est_m = pinfo["curve"].est_tp.to_numpy()
    n_known, b2 = int(known.sum()), beta ** 2
    cands = [("min-rank", est_m, pinfo["order"]), ("density ratio", est_r, order_r)]
    prof = (pocket_profile_ratio(df, known, cols, reps=profile_reps, n_perm=profile_perm, max_k=max_k, seed=seed)
            if df is not None else None)
    if prof is not None:
        cands.append(("period-profile ratio", prof[1], prof[0]))
    G = max(float(est.max()) for _, est, _ in cands)
    k = np.arange(1, max_k + 1)
    curves = {}
    for name, est, order in cands:
        f = (1 + b2) * (n_known + est) / (b2 * (n_known + G) + n_known + k)
        ks = int(np.argmax(f)) + 1
        curves[name] = dict(k=ks, est_tp=float(est[ks - 1]), precision=float(est[ks - 1] / ks), expected_f=float(f[ks - 1]),
                            group_size=float(est.max()), order=order, f=f)
    best = max(curves, key=lambda m: curves[m]["expected_f"])
    c = curves[best]
    flags = np.zeros(len(known), bool)
    flags[c["order"][:c["k"]]] = True
    shown = {"min-rank": " x ".join(feats), "density ratio": " x ".join(cols),
             "period-profile ratio": "the 4 periods of " + ", ".join(c[:-len("_level")] for c in cols)}
    comp = pd.DataFrame({m: {"columns": shown[m], "estimated group size": v["group_size"],
                             "cut-off k": v["k"], "estimated members": v["est_tp"], "estimated precision": v["precision"],
                             f"expected F{beta:g} (common group size)": v["expected_f"]} for m, v in curves.items()}).T
    info.update(method=best, cols=list(feats) if best == "min-rank" else cols, k=c["k"], est_tp=c["est_tp"], precision=c["precision"],
                expected_f=c["expected_f"], group_size=G, comparison=comp,
                expected_f_known_only=float((1 + b2) * n_known / (b2 * (n_known + G) + n_known)))
    return flags, info
