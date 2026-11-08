"""Task 1: in-fold logistic score on quantile-normalised raw features as an extra LightGBM feature."""
import time
from common import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.modeling import run_cv

tr, te, X, y, Xt, folds, n_unl = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
SKILL = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms",
         "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement", "level"]
spw = (y == 0).sum() / (y == 1).sum()
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})


def _scores(Za, Zsa, ya, Zbs, Zsbs):
    m_all = LogisticRegression(C=1.0, max_iter=2000).fit(Za, ya)
    m_sk = LogisticRegression(C=1.0, max_iter=2000).fit(Zsa, ya)
    return [np.column_stack([m_all.decision_function(Zb), m_sk.decision_function(Zsb)]) for Zb, Zsb in zip(Zbs, Zsbs)]


def lin_fold_fn(X_tr, y_tr, X_others):
    qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal", subsample=10**6, random_state=SEED)
    Za = np.nan_to_num(qt.fit_transform(X_tr[raw]))
    Zo = [np.nan_to_num(qt.transform(x[raw])) for x in X_others]
    si = [raw.index(c) for c in SKILL]
    oof = np.zeros((len(Za), 2))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Za, y_tr):
        oof[b] = _scores(Za[a], Za[a][:, si], y_tr[a], [Za[b]], [Za[b][:, si]])[0]
    app = _scores(Za, Za[:, si], y_tr, Zo, [z[:, si] for z in Zo])
    X_tr = X_tr.copy(); X_others = [x.copy() for x in X_others]
    for j, n in enumerate(["lin_all", "lin_skill"]):
        X_tr[n] = oof[:, j]
        for x, a in zip(X_others, app):
            x[n] = a[:, j]
    return X_tr, X_others


if __name__ == "__main__":
    base = load_preds("t1_lgb")
    t, s = optimize_threshold(y, base["oof"], 2.0)
    print(f"t1_lgb baseline: AUC {roc_auc_score(y, base['oof']):.4f} F2 {s:.4f}")
    t0 = time.time()
    r = run_cv("lgb", "binary", P1_lgb, X, y, Xt, folds, name="lgb+lin", fold_fn=lin_fold_fn, verbose=False)
    t, s = optimize_threshold(y, r["oof"], 2.0)
    print(f"lgb + linear scores: AUC {roc_auc_score(y, r['oof']):.4f} F2 {s:.4f} [{time.time()-t0:.0f}s]")
    print(r["importance"].head(10).round(4).to_string())
    np.savez_compressed(OUT / "t1_lgb_lin.npz", oof=r["oof"], test=r["test"])
    ms = [load_preds(n) for n in T1_MODELS]
    for name, extra in [("baseline stack", []), ("+ lgb_lin", [r])]:
        s2, thr2, so, st = t1_score([d["oof"] for d in ms + extra], [d["test"] for d in ms + extra], y, folds)
        print(f"{name:16s} stack AUC {roc_auc_score(y, so):.4f} best F2 {s2:.4f} @ {thr2:.3f}")
