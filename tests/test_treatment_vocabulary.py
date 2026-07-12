import numpy as np
from hypothesis import given, settings, strategies as st

from noxed.behavioural import (
    TREATMENT_VOCABULARY, CASE_TO_TREATMENTS, CASES, plan_smoothness,
    generate_synthetic_session, compute_metrics, score_to_plan,
)


def test_every_case_maps_only_to_vocabulary_treatments():
    for case in CASES:
        treatments = CASE_TO_TREATMENTS.get(case, [])
        assert treatments, f"case {case} has no treatments defined"
        for t in treatments:
            assert t in TREATMENT_VOCABULARY, f"{t} (case={case}) is outside the fixed vocabulary"


@given(
    a_vals=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=3, max_size=3),
    b_vals=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=3, max_size=3),
)
def test_plan_smoothness_distances_bounded(a_vals, b_vals):
    keys = ["m1", "m2", "m3"]
    scores_a = dict(zip(keys, a_vals))
    scores_b = dict(zip(keys, b_vals))
    result = plan_smoothness(scores_a, scores_b, ["t1", "t2"], ["t1", "t3"])
    assert result["score_l2_distance"] >= 0.0
    assert 0.0 <= result["plan_jaccard_distance"] <= 1.0


def test_plan_smoothness_identical_scores_zero_distance():
    scores = {"m1": 0.5, "m2": 0.3}
    result = plan_smoothness(scores, scores, ["t1"], ["t1"])
    assert result["score_l2_distance"] == 0.0
    assert result["plan_jaccard_distance"] == 0.0


def test_score_to_plan_end_to_end_stays_in_vocabulary():
    """Exercise the full score->plan pipeline (not just plan_smoothness in
    isolation) with a real thresholds dict shaped {metric: {case, cutoff}},
    the exact shape used in notebook 07 -- regression test for a bug where
    score_to_plan compared a float against the whole threshold dict instead
    of thresholds[metric]["cutoff"]."""
    thresholds = {
        "fast_error_rate": {"case": "impulsiveness", "cutoff": 0.4},
        "carelessness": {"case": "misconceptions", "cutoff": 0.3},
    }
    scores_over = {"fast_error_rate": 0.9, "carelessness": 0.9}
    plan = score_to_plan(scores_over, thresholds)
    assert plan, "expected treatments to be emitted when scores exceed cutoffs"
    for t in plan:
        assert t in TREATMENT_VOCABULARY

    scores_under = {"fast_error_rate": 0.1, "carelessness": 0.1}
    assert score_to_plan(scores_under, thresholds) == []


@given(
    fast_error_rate=st.floats(min_value=0, max_value=1, allow_nan=False),
    carelessness=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_score_to_plan_vocabulary_bounded_property(fast_error_rate, carelessness):
    thresholds = {
        "fast_error_rate": {"case": "impulsiveness", "cutoff": 0.4},
        "carelessness": {"case": "misconceptions", "cutoff": 0.3},
    }
    plan = score_to_plan({"fast_error_rate": fast_error_rate, "carelessness": carelessness}, thresholds)
    for t in plan:
        assert t in TREATMENT_VOCABULARY


@settings(deadline=None, max_examples=15)
@given(seed=st.integers(min_value=0, max_value=1000), case=st.sampled_from(CASES))
def test_synthetic_metrics_are_finite_and_in_expected_ranges(seed, case):
    rng = np.random.default_rng(seed)
    D_sample = rng.uniform(0.05, 0.6, size=200)
    session = generate_synthetic_session(rng, case, D_sample, n_items=40)
    metrics = compute_metrics(session)
    for key in ["fast_error_rate", "slow_correct_rate", "timeout_rate", "carelessness", "error_rate"]:
        val = metrics[key]
        assert not np.isnan(val)
        assert -1e-9 <= val <= 1.0 + 1e-9, f"{key}={val} out of [0,1] for case={case}"
