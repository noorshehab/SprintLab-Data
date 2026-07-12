"""Question-difficulty formula: static (cold-start) -> empirical (shrunk) ->
blended, and the BKT prior-seeding formulas that consume it. Every function
is pure and bounded-in-[0,1] by construction (sigmoid / convex combination),
which is what the property tests in tests/test_difficulty.py assert.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def empirical_difficulty(wrong, attempted, mu: float = 0.20, m: float = 20.0) -> np.ndarray:
    """Beta-Binomial shrinkage of raw error rate toward the global mean `mu`,
    with pseudo-count `m` controlling how aggressively low-attempt items are
    pulled toward the mean. D_emp = (w + mu*m) / (a + m)."""
    wrong = np.asarray(wrong, dtype=float)
    attempted = np.asarray(attempted, dtype=float)
    return (wrong + mu * m) / (attempted + m)


def differentiation_correction(d_emp: np.ndarray, delta: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    """Optional correction for intrinsic hardness beyond topic mastery."""
    return np.clip(d_emp + gamma * np.asarray(delta, dtype=float), 0.0, 1.0)


def fit_static_difficulty(X: np.ndarray, d_emp_binary_target: np.ndarray) -> dict:
    """Fit D_stat = sigmoid(beta0 + sum(beta_i * x_i)) via logistic regression
    of available static features against a binarised empirical-difficulty
    target (above/below median). Returns weights + fit AUC. Any sklearn
    LogisticRegression output is inherently in [0,1] via predict_proba."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, d_emp_binary_target)
    proba = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(d_emp_binary_target, proba) if len(np.unique(d_emp_binary_target)) > 1 else float("nan")
    return {
        "beta0": float(clf.intercept_[0]),
        "betas": clf.coef_[0].tolist(),
        "auc": float(auc),
        "D_stat": proba,
        "model": clf,
    }


def blend_difficulty(d_emp: np.ndarray, d_stat: np.ndarray, attempted: np.ndarray, a0: float = 30.0) -> np.ndarray:
    """Convex combination D = lambda(a)*D_emp + (1-lambda(a))*D_stat,
    lambda(a) = a/(a+a0). At a=0 -> D_stat (cold start); as a grows -> D_emp.
    A convex combination of two [0,1] quantities is itself in [0,1]."""
    attempted = np.asarray(attempted, dtype=float)
    lam = attempted / (attempted + a0)
    return lam * np.asarray(d_emp, dtype=float) + (1 - lam) * np.asarray(d_stat, dtype=float)


def bkt_priors_from_difficulty(
    D: np.ndarray, cognitive_load: np.ndarray, n_options: np.ndarray,
    g0: float = 0.5, s0: float = 0.4, t0: float = 0.3,
) -> dict:
    """Seed per-item BKT parameters from difficulty D and static load/option
    count, clipped to identifiable ranges (P(S), P(G) < 0.5)."""
    D = np.asarray(D, dtype=float)
    cognitive_load = np.asarray(cognitive_load, dtype=float)
    n_options = np.clip(np.asarray(n_options, dtype=float), 1.0, None)
    p_guess = np.clip(g0 * (1 - D) / n_options, 0.0, 0.3)
    p_slip = np.clip(s0 * (0.5 + 0.5 * D) * (0.5 + 0.5 * cognitive_load), 0.0, 0.3)
    p_l0 = np.clip(0.5 * (1 - D), 0.0, 1.0)
    p_transit = np.clip(t0 * (1 - 0.5 * D), 0.0, 1.0)
    return {"P_G": p_guess, "P_S": p_slip, "P_L0": p_l0, "P_T": p_transit}
