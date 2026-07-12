"""A small, self-contained Bayesian Knowledge Tracing implementation.

pyBKT's published wheel does not build in this environment (invalid wheel
metadata), and shipping a broken dependency into the Modal image is worse
than a from-scratch implementation of a well-understood, four-parameter HMM.
This module implements the standard BKT posterior update and a per-KC
maximum-likelihood parameter fit via bounded optimisation -- the same model
pyBKT fits, without the flaky native wheel.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def bkt_posterior_update(p_l: float, correct: bool, p_t: float, p_s: float, p_g: float) -> float:
    """One BKT step: given prior P(mastery)=p_l and an observed response,
    return the posterior P(mastery) after the evidence, then apply the
    learning-transition probability p_t. Always returns a value in [0,1]."""
    if correct:
        num = p_l * (1 - p_s)
        den = p_l * (1 - p_s) + (1 - p_l) * p_g
    else:
        num = p_l * p_s
        den = p_l * p_s + (1 - p_l) * (1 - p_g)
    posterior = num / den if den > 0 else p_l
    return posterior + (1 - posterior) * p_t


def predict_correct_prob(p_l: float, p_s: float, p_g: float) -> float:
    """P(correct) = P(L)*(1-P(S)) + (1-P(L))*P(G)."""
    return p_l * (1 - p_s) + (1 - p_l) * p_g


def _sequence_neg_log_likelihood(params, sequences: list[np.ndarray]) -> float:
    p_l0, p_t, p_s, p_g = params
    nll = 0.0
    for seq in sequences:
        p_l = p_l0
        for correct in seq:
            p_pred = predict_correct_prob(p_l, p_s, p_g)
            p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
            nll -= np.log(p_pred) if correct else np.log(1 - p_pred)
            p_l = bkt_posterior_update(p_l, bool(correct), p_t, p_s, p_g)
    return nll


def fit_bkt_kc(sequences: list[np.ndarray], init: tuple[float, float, float, float] = (0.3, 0.2, 0.1, 0.2)) -> dict:
    """Maximum-likelihood fit of (P_L0, P_T, P_S, P_G) for one KC's set of
    per-student response sequences, bounded to keep P_S, P_G identifiable
    (< 0.5) as in the RQ note's difficulty-formula design."""
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499), (0.001, 0.499)]
    res = minimize(_sequence_neg_log_likelihood, x0=list(init), args=(sequences,), bounds=bounds, method="L-BFGS-B")
    p_l0, p_t, p_s, p_g = res.x
    return {"P_L0": p_l0, "P_T": p_t, "P_S": p_s, "P_G": p_g, "neg_log_lik": float(res.fun), "converged": bool(res.success)}


def mastery_trajectory(sequence: np.ndarray, p_l0: float, p_t: float, p_s: float, p_g: float) -> np.ndarray:
    """Replay a student's response sequence through the fitted BKT parameters
    and return the P(mastery) trajectory (length = len(sequence)+1, starting
    at P_L0 before any evidence)."""
    traj = [p_l0]
    p_l = p_l0
    for correct in sequence:
        p_l = bkt_posterior_update(p_l, bool(correct), p_t, p_s, p_g)
        traj.append(p_l)
    return np.array(traj)


def detect_mastery_crossing(trajectory: np.ndarray, threshold: float = 0.95, sustain: int = 3) -> int | None:
    """Return the first index at which P(mastery) crosses `threshold` and
    stays >= threshold for the next `sustain` steps (mastery-advancement
    detection per RQ4); None if never reached."""
    for i in range(len(trajectory) - sustain):
        if np.all(trajectory[i : i + sustain] >= threshold):
            return i
    return None


def student_kc_sequences(pfa_like_df: pd.DataFrame) -> dict:
    """Group a long (uid, kc_id, response) table into {kc_id: [seq_per_student]}
    for BKT fitting, ordered by exposure_n if present."""
    sort_cols = ["uid", "exposure_n"] if "exposure_n" in pfa_like_df.columns else ["uid"]
    df = pfa_like_df.sort_values(sort_cols, kind="stable")
    out: dict = {}
    for kc_id, g in df.groupby("kc_id"):
        out[kc_id] = [s["response"].to_numpy() for _, s in g.groupby("uid")]
    return out
