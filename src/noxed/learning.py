"""Learning-dynamics modelling: parametric curve fits over the pre-aggregated
per-KC exposure table, inferential module/depth comparison (the existing EDA
stopped at descriptive boxplots), and a Performance-Factors-Analysis (PFA/AFM)
fit on the raw exploded logs that sidesteps the aggregated-cell no-N problem.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import curve_fit
from scipy.stats import kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def exponential_curve(k, a, b, c):
    return a * np.exp(-b * (k - 1)) + c


def power_law_curve(k, a, b, c):
    return a * np.power(k, -b) + c


def fit_kc_curve(exposures: np.ndarray, errors: np.ndarray, kind: str = "exponential") -> dict:
    """Bounded curve_fit of one KC's exposure->error series. Returns params,
    R^2, and a `reliable` flag (R^2>=0.5 and coverage>=6), mirroring the
    reliability gating already validated in the practice-effect EDA."""
    func = exponential_curve if kind == "exponential" else power_law_curve
    bounds = ([0, 0, 0], [1, 5, 1]) if kind == "exponential" else ([0, 0, 0], [1, 3, 1])
    p0 = [errors[0], 0.3, errors[-1]] if kind == "exponential" else [errors[0], 0.3, errors[-1]]
    try:
        popt, _ = curve_fit(func, exposures, errors, p0=p0, bounds=bounds, maxfev=5000)
        pred = func(exposures, *popt)
        ss_res = np.sum((errors - pred) ** 2)
        ss_tot = np.sum((errors - errors.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except (RuntimeError, ValueError):
        popt, r2 = (np.nan, np.nan, np.nan), np.nan
    coverage = len(exposures)
    reliable = bool(r2 is not np.nan and not np.isnan(r2) and r2 >= 0.5 and coverage >= 6)
    return {"a": popt[0], "b": popt[1], "c": popt[2], "r2": r2, "coverage": coverage, "reliable": reliable}


def module_depth_kruskal(df: pd.DataFrame, value_col: str, group_col: str) -> dict:
    """Kruskal-Wallis H-test for whether `value_col` differs across the
    groups in `group_col` (e.g. module or depth) -- the inferential test the
    prior EDA's boxplots lacked. Returns H, p, group medians, and n per group."""
    groups = [g[value_col].dropna().to_numpy() for _, g in df.groupby(group_col) if len(g) >= 3]
    if len(groups) < 2:
        return {"H": np.nan, "p": np.nan, "n_groups": len(groups)}
    h, p = kruskal(*groups)
    medians = df.groupby(group_col)[value_col].median().to_dict()
    counts = df.groupby(group_col)[value_col].count().to_dict()
    return {"H": float(h), "p": float(p), "n_groups": len(groups), "medians": medians, "counts": counts}


def pfa_features(canonical_events: pd.DataFrame) -> pd.DataFrame:
    """Explode canonical (student, question, concepts[], response) events
    into one row per (student, KC, exposure) with running prior
    success/failure counts on that KC -- the standard PFA/AFM feature set.
    """
    rows = canonical_events.explode("concepts").rename(columns={"concepts": "kc_id"})
    rows = rows.sort_values(["uid", "kc_id", "event_seq"], kind="stable")
    grp = rows.groupby(["uid", "kc_id"], sort=False)
    # Fully vectorised (no per-group Python callback): shift-then-cumsum for
    # prior successes, and prior_failure = attempts-so-far minus successes.
    prior_attempts = grp.cumcount()
    shifted_correct = grp["response"].shift(fill_value=0)
    rows["prior_success"] = shifted_correct.groupby([rows["uid"], rows["kc_id"]]).cumsum()
    rows["prior_failure"] = prior_attempts - rows["prior_success"]
    rows["exposure_n"] = prior_attempts + 1
    return rows[["uid", "kc_id", "exposure_n", "prior_success", "prior_failure", "response"]]


def fit_pfa(pfa_df: pd.DataFrame) -> dict:
    """Fit the AFM/PFA logistic model: logit(p_correct) = beta_kc
    + gamma*prior_success + rho*prior_failure, pooled across KCs (single
    gamma/rho, per-KC intercept via a sparse one-hot -- never densified, so
    this scales to the full ~5M-row / ~865-KC exploded log) -- gamma>0
    confirms a genuine per-attempt learning effect on the raw logs."""
    kc_onehot = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    X_kc = kc_onehot.fit_transform(pfa_df[["kc_id"]])
    X_num = sp.csr_matrix(pfa_df[["prior_success", "prior_failure"]].to_numpy(dtype=np.float32))
    X = sp.hstack([X_num, X_kc], format="csr")
    y = pfa_df["response"].to_numpy()
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(X, y)
    gamma, rho = clf.coef_[0][0], clf.coef_[0][1]
    return {"gamma_success": float(gamma), "rho_failure": float(rho), "n_obs": len(pfa_df), "n_kcs": X_kc.shape[1]}
