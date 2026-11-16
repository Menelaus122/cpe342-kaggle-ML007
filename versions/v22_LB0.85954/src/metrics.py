"""Competition metrics and metric-aware post-processing (threshold / class-weight search)."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score, fbeta_score

from .config import FBETA_AVERAGE


# --------------------------------------------------------------------------- metrics
def fbeta(y_true, y_pred, beta, average=FBETA_AVERAGE):
    return fbeta_score(y_true, y_pred, beta=beta, average=average, zero_division=0)


def f2(y_true, y_pred, average=FBETA_AVERAGE):
    return fbeta(y_true, y_pred, 2.0, average)


def f3(y_true, y_pred, average=FBETA_AVERAGE):
    return fbeta(y_true, y_pred, 3.0, average)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def nmae(y_true, y_pred):
    """Normalized MAE used by the leaderboard: 1 / (1 + sum|y - yhat| / sum y)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 1.0 / (1.0 + np.abs(y_true - y_pred).sum() / y_true.sum())


# ------------------------------------------------------------ binary threshold moving
def _fbeta_at(y_true, proba, t, beta, average):
    return fbeta(y_true, (proba >= t).astype(int), beta, average)


def optimize_threshold(y_true, proba, beta, average=FBETA_AVERAGE, lo=0.05, hi=0.95):
    """Post-hoc threshold moving (Provost, 2000).

    F-beta is a step function of the threshold, so gradient-based optimisers stall on
    its flat plateaus. We therefore (1) scan a dense grid to locate the best basin and
    (2) polish inside that basin with bounded Brent search (scipy.optimize).
    Returns (best_threshold, best_score).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    grid = np.linspace(lo, hi, 181)
    scores = np.array([_fbeta_at(y_true, proba, t, beta, average) for t in grid])
    i = int(scores.argmax())
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda t: -_fbeta_at(y_true, proba, t, beta, average),
                          bounds=(a, b), method="bounded", options={"xatol": 1e-4})
    if -res.fun > scores[i]:
        return float(res.x), float(-res.fun)
    return float(grid[i]), float(scores[i])


def threshold_curve(y_true, proba, beta, average=FBETA_AVERAGE, lo=0.02, hi=0.98, n=193):
    grid = np.linspace(lo, hi, n)
    return grid, np.array([_fbeta_at(y_true, proba, t, beta, average) for t in grid])


# ------------------------------------------------- multi-class decision-weight tuning
def apply_class_weights(proba, w):
    return np.argmax(proba * np.asarray(w)[None, :], axis=1)


def optimize_class_weights(y_true, proba, n_restarts=5, seed=0):
    """Find per-class multipliers w_c maximising macro-F1 of argmax_c(w_c * P_ic).

    Optimised in log-space (w = exp(v), v_0 fixed at 0 for identifiability) with Powell,
    which is derivative-free and copes with the piecewise-constant objective. A few
    random restarts guard against bad local plateaus. Returns (weights, score).
    """
    y_true = np.asarray(y_true)
    k = proba.shape[1]
    rng = np.random.default_rng(seed)

    def loss(v):
        w = np.exp(np.concatenate([[0.0], v]))
        return -macro_f1(y_true, apply_class_weights(proba, w))

    best_v, best = np.zeros(k - 1), loss(np.zeros(k - 1))
    for r in range(n_restarts):
        v0 = np.zeros(k - 1) if r == 0 else rng.normal(0, 0.3, k - 1)
        res = minimize(loss, v0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 3000})
        if res.fun < best:
            best, best_v = res.fun, res.x
    w = np.exp(np.concatenate([[0.0], best_v]))
    return w / w.sum() * k, -best
