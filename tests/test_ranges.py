import numpy as np
from hypothesis import given, strategies as st

from noxed.bandit import success_probability, desirable_difficulty_match, reward_desirable
from noxed.tracing import bkt_posterior_update, predict_correct_prob

floats01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
sub05 = st.floats(min_value=0.0, max_value=0.499, allow_nan=False)


@given(p_l=floats01, p_s=sub05, p_g=sub05)
def test_success_probability_in_unit_interval(p_l, p_s, p_g):
    p = success_probability(np.array([p_l]), np.array([p_s]), np.array([p_g]))[0]
    assert 0.0 <= p <= 1.0


@given(p=floats01, p_star=floats01, tau=st.floats(min_value=0.01, max_value=5.0))
def test_desirable_difficulty_match_in_unit_interval(p, p_star, tau):
    m = desirable_difficulty_match(np.array([p]), p_star=p_star, tau=tau)[0]
    assert 0.0 <= m <= 1.0


@given(
    sum_priors=st.floats(min_value=0.01, max_value=100),
    p=floats01, q_learn=st.floats(min_value=0.0, max_value=10.0),
)
def test_reward_desirable_nonnegative(sum_priors, p, q_learn):
    r = reward_desirable(np.array([sum_priors]), np.array([p]), np.array([q_learn]))[0]
    assert r >= 0.0


@given(p_l=floats01, correct=st.booleans(), p_t=floats01, p_s=sub05, p_g=sub05)
def test_bkt_posterior_update_in_unit_interval(p_l, correct, p_t, p_s, p_g):
    post = bkt_posterior_update(p_l, correct, p_t, p_s, p_g)
    assert 0.0 <= post <= 1.0


@given(p_l=floats01, p_s=sub05, p_g=sub05)
def test_predict_correct_prob_in_unit_interval(p_l, p_s, p_g):
    p = predict_correct_prob(p_l, p_s, p_g)
    assert 0.0 <= p <= 1.0


@given(p_l=floats01, p_t=floats01, p_s=sub05, p_g=sub05)
def test_bkt_delta_mastery_bounded(p_l, p_t, p_s, p_g):
    """Delta-mastery (posterior - prior) must stay within [-1, 1] per the
    RQ note's Progress-Tracking invariant."""
    post_correct = bkt_posterior_update(p_l, True, p_t, p_s, p_g)
    post_wrong = bkt_posterior_update(p_l, False, p_t, p_s, p_g)
    assert -1.0 <= post_correct - p_l <= 1.0
    assert -1.0 <= post_wrong - p_l <= 1.0
