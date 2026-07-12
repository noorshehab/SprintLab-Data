"""1PL/2PL Item Response Theory calibration via girth.

The full response matrix (~18k students x ~7.6k questions, ~3.7% dense) is
too large and too sparse for girth's dense-matrix MML estimators. We build
a tractable, documented sub-matrix from the most-attempted questions and
most-active students, filled with girth's INVALID_RESPONSE sentinel for
missing cells -- a standard, defensible subsampling strategy for IRT
calibration on sparse KT logs, not a silent truncation (the notebook logs
exactly how many students/items were dropped and why).
"""
import numpy as np
import pandas as pd
from girth import INVALID_RESPONSE, tag_missing_data, twopl_mml


def build_response_matrix(
    canonical_events: pd.DataFrame, n_items: int = 300, min_responses_per_student: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (item_ids, student_ids, matrix[items x students]) for the
    `n_items` most-attempted questions, restricted to students who answered
    at least `min_responses_per_student` of them."""
    top_items = canonical_events["questions"].value_counts().head(n_items).index.to_numpy()
    sub = canonical_events[canonical_events["questions"].isin(top_items)]
    counts = sub.groupby("uid")["questions"].nunique()
    keep_students = counts[counts >= min_responses_per_student].index.to_numpy()
    sub = sub[sub["uid"].isin(keep_students)]

    item_ids = np.sort(top_items)
    student_ids = np.sort(keep_students)
    item_pos = {q: i for i, q in enumerate(item_ids)}
    stu_pos = {u: i for i, u in enumerate(student_ids)}

    matrix = np.full((len(item_ids), len(student_ids)), np.nan)
    rows = sub["questions"].map(item_pos).to_numpy()
    cols = sub["uid"].map(stu_pos).to_numpy()
    matrix[rows, cols] = sub["response"].to_numpy()
    return item_ids, student_ids, matrix


def fit_2pl(matrix: np.ndarray) -> dict:
    """Fit a 2PL model on an [items x participants] matrix that may contain
    NaN for missing cells. Returns discrimination (a) and difficulty (b) per
    KEPT item (see `kept_mask`) -- b is in IRT logit-difficulty units, not
    [0,1]. Items with zero-variance valid responses (all-correct or
    all-wrong among the students who attempted them) are dropped before
    fitting: girth's MML estimator raises on such degenerate items (an
    empty-array comparison inside its polytomous solver), so this is an
    explicit, reported filter rather than an unexplained crash risk."""
    valid = ~np.isnan(matrix)
    variance = np.array([np.unique(matrix[i, valid[i]]).size if valid[i].any() else 0 for i in range(matrix.shape[0])])
    kept_mask = variance >= 2
    dropped = int((~kept_mask).sum())

    int_matrix = np.nan_to_num(matrix[kept_mask], nan=INVALID_RESPONSE).astype(int)
    tagged = tag_missing_data(int_matrix, [0, 1])
    results = twopl_mml(tagged)
    return {
        "discrimination": np.asarray(results["Discrimination"], dtype=float).squeeze(),
        "difficulty": np.asarray(results["Difficulty"], dtype=float).squeeze(),
        "kept_mask": kept_mask,
        "n_dropped_degenerate": dropped,
    }


def irt_difficulty_to_unit_interval(b: np.ndarray) -> np.ndarray:
    """Map IRT logit-difficulty b (typically in [-3, 3]) onto [0,1] via a
    sigmoid so it is directly comparable to D_emp / D_stat / D_blend."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(b, dtype=float), -35, 35)))
