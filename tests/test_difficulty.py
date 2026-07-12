import numpy as np
from hypothesis import given, strategies as st

from noxed.difficulty import (
    empirical_difficulty, blend_difficulty, bkt_priors_from_difficulty, sigmoid,
)

floats01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
nonneg = st.floats(min_value=0.0, max_value=1e6, allow_nan=False)


@given(z=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_sigmoid_bounded(z):
    assert 0.0 <= sigmoid(z) <= 1.0


@given(wrong=nonneg, attempted=nonneg, mu=floats01, m=st.floats(min_value=0.01, max_value=1000))
def test_empirical_difficulty_bounded(wrong, attempted, mu, m):
    wrong = min(wrong, attempted)  # wrong can't exceed attempted
    d = empirical_difficulty(np.array([wrong]), np.array([attempted]), mu=mu, m=m)[0]
    assert 0.0 <= d <= 1.0


@given(d_emp=floats01, d_stat=floats01, attempted=nonneg, a0=st.floats(min_value=0.01, max_value=1000))
def test_blend_difficulty_bounded_and_convex(d_emp, d_stat, attempted, a0):
    d = blend_difficulty(np.array([d_emp]), np.array([d_stat]), np.array([attempted]), a0=a0)[0]
    assert 0.0 <= d <= 1.0
    lo, hi = sorted([d_emp, d_stat])
    assert lo - 1e-9 <= d <= hi + 1e-9


def test_blend_difficulty_cold_start_equals_static():
    d = blend_difficulty(np.array([0.9]), np.array([0.3]), np.array([0.0]), a0=30.0)[0]
    assert abs(d - 0.3) < 1e-9


@given(D=floats01, cli=floats01, n_opt=st.floats(min_value=1.0, max_value=10.0))
def test_bkt_priors_ranges(D, cli, n_opt):
    priors = bkt_priors_from_difficulty(np.array([D]), np.array([cli]), np.array([n_opt]))
    assert 0.0 <= priors["P_G"][0] <= 0.3
    assert 0.0 <= priors["P_S"][0] <= 0.3
    assert 0.0 <= priors["P_L0"][0] <= 1.0
    assert 0.0 <= priors["P_T"][0] <= 1.0
    assert priors["P_S"][0] < 0.5 and priors["P_G"][0] < 0.5
