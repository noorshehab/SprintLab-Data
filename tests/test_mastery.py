import numpy as np
from hypothesis import given, settings, strategies as st

from noxed.tracing import mastery_trajectory, detect_mastery_crossing, bkt_posterior_update

floats01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
sub05 = st.floats(min_value=0.0, max_value=0.499, allow_nan=False)


@given(
    seq=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30),
    p_l0=floats01, p_t=floats01, p_s=sub05, p_g=sub05,
)
@settings(deadline=None, max_examples=50)
def test_mastery_trajectory_length_and_range(seq, p_l0, p_t, p_s, p_g):
    traj = mastery_trajectory(np.array(seq), p_l0, p_t, p_s, p_g)
    assert len(traj) == len(seq) + 1
    assert np.all(traj >= 0.0) and np.all(traj <= 1.0)
    assert traj[0] == p_l0


@given(seq=st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=30), p_l0=floats01, p_t=floats01, p_s=sub05, p_g=sub05)
@settings(deadline=None, max_examples=50)
def test_mastery_trajectory_is_idempotent(seq, p_l0, p_t, p_s, p_g):
    """Replaying the same sequence through the same fitted parameters must
    yield the identical trajectory every time (idempotence / traceability
    per the RQ note's Progress-Tracking property test T-3)."""
    a = np.array(seq)
    traj1 = mastery_trajectory(a, p_l0, p_t, p_s, p_g)
    traj2 = mastery_trajectory(a, p_l0, p_t, p_s, p_g)
    assert np.array_equal(traj1, traj2)


def test_detect_mastery_crossing_finds_sustained_threshold():
    traj = np.array([0.1, 0.3, 0.5, 0.96, 0.97, 0.98, 0.99])
    idx = detect_mastery_crossing(traj, threshold=0.95, sustain=3)
    assert idx == 3


def test_detect_mastery_crossing_none_when_never_sustained():
    traj = np.array([0.1, 0.96, 0.5, 0.97, 0.4])
    idx = detect_mastery_crossing(traj, threshold=0.95, sustain=3)
    assert idx is None


@given(p_l=floats01, p_t=floats01, p_s=sub05, p_g=sub05)
def test_bkt_update_deterministic(p_l, p_t, p_s, p_g):
    a = bkt_posterior_update(p_l, True, p_t, p_s, p_g)
    b = bkt_posterior_update(p_l, True, p_t, p_s, p_g)
    assert a == b
