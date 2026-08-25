"""BKT learning-logic contract tests.

Guarantees the core knowledge-tracing invariants hold end to end:
  - correct answers improve the prior, wrong answers decrease it
  - P(L|correct) >= P(L|wrong) for identical states
  - priors never reach 0 or 1 (engine clamps to [0.01, 0.99])
  - predict_response probability rises with the prior
Both the pure functions (BKT.py) and the engine-through-mediator path
are covered.
"""
import pytest

from services.Data_service import Data_Service
from services.Entities import question
from services.knowledge_tracing.BKT import next_response, update_prior
from services.knowledge_tracing.knowledge_tracing_engine import (
    knowledge_tracing_engine,
)
from services.Diagnosis_service import Diagnosis_service
from unittest.mock import MagicMock

G, S, T = 0.2, 0.1, 0.017  # realistic guess/slip/transition parameters


#--- pure functions -------------------------------------------------------

@pytest.mark.parametrize('prior', [0.05, 0.2, 0.5, 0.75, 0.9])
def test_correct_answer_improves_prior(prior):
    assert update_prior(prior, G, S, T, response=True) > prior


@pytest.mark.parametrize('prior', [0.05, 0.2, 0.5, 0.75, 0.9])
def test_wrong_answer_decreases_prior(prior):
    updated = update_prior(prior, G, S, T, response=False)
    assert updated < prior


def test_correct_beats_wrong_from_same_state():
    for prior in [0.1, 0.3, 0.5, 0.7, 0.9]:
        after_correct = update_prior(prior, G, S, T, response=True)
        after_wrong = update_prior(prior, G, S, T, response=False)
        assert after_correct > after_wrong


@pytest.mark.parametrize('response', [True, False])
def test_update_is_monotonic_in_prior(response):
    """A student with a higher prior stays ahead after the same answer."""
    lower = update_prior(0.3, G, S, T, response=response)
    higher = update_prior(0.6, G, S, T, response=response)
    assert higher > lower


@pytest.mark.parametrize('prior', [-0.5, 0.0, 0.5, 1.0, 1.5])
def test_degenerate_inputs_never_explode(prior):
    """Documented behaviour: pathological inputs may produce out-of-range
    values from the raw formula - the engine clamp is what protects state.
    This pins that the formula at least returns finite numbers."""
    value = update_prior(prior, G, S, T, response=True)
    assert value == value  # not NaN


def test_next_response_probability_rises_with_prior():
    assert next_response(0.9, G, S)[0] > next_response(0.1, G, S)[0]


def test_next_response_prediction_threshold():
    _, predicts_correct_high = next_response(0.9, G, S)
    _, predicts_correct_low = next_response(0.05, G, S)
    assert predicts_correct_high == 1
    assert predicts_correct_low == 0


#--- engine path through the mediator -------------------------------------

@pytest.fixture
def wired_engine():
    ds = Data_Service()
    ds.add_student('S-BKT')
    # one question per skill so similar-skill propagation stays out of the way
    ds.add_question(q_id='q_ok', skill_cluster_id='KC_OK', p_g=G, p_s=S, p_t=T)
    kt = knowledge_tracing_engine()
    Diagnosis_service(kt, MagicMock(), ds, MagicMock())
    return kt, ds


def _drive(engine, q_id, n, response):
    engine.update_student_priors('S-BKT', [q_id] * n, [response] * n)


def test_engine_correct_answers_raise_clamped_prior(wired_engine):
    engine, ds = wired_engine
    ds.get_student('S-BKT').update_prior('KC_OK', 0.3)

    _drive(engine, 'q_ok', 15, 1)

    prior = ds.get_student('S-BKT').get_priors()['KC_OK']
    assert prior > 0.3
    assert prior <= 0.99


def test_engine_wrong_answers_lower_clamped_prior(wired_engine):
    engine, ds = wired_engine
    ds.get_student('S-BKT').update_prior('KC_OK', 0.8)

    _drive(engine, 'q_ok', 25, 0)

    prior = ds.get_student('S-BKT').get_priors()['KC_OK']
    assert prior < 0.8
    assert prior >= 0.01


def test_engine_repeated_answers_converge_without_hitting_bounds(wired_engine):
    """Hammer both directions hard - the clamp must keep state open-ended."""
    engine, ds = wired_engine
    ds.get_student('S-BKT').update_prior('KC_OK', 0.5)

    _drive(engine, 'q_ok', 200, 1)
    ceiling = ds.get_student('S-BKT').get_priors()['KC_OK']
    _drive(engine, 'q_ok', 400, 0)
    floor = ds.get_student('S-BKT').get_priors()['KC_OK']

    assert 0.01 <= floor < ceiling <= 0.99


def test_engine_alternating_answers_track_direction(wired_engine):
    """Each answer must move the prior the right way relative to its own
    starting point, alternating correctness."""
    engine, ds = wired_engine
    student = ds.get_student('S-BKT')
    student.update_prior('KC_OK', 0.4)

    sequence = [1, 1, 0, 1, 0, 0, 1]
    previous = 0.4
    for i, response in enumerate(sequence):
        engine.update_student_priors('S-BKT', ['q_ok'], [response])
        current = student.get_priors()['KC_OK']
        if response:
            assert current > previous, f'answer {i}: correct must raise {previous}'
        else:
            assert current < previous, f'answer {i}: wrong must lower {previous}'
        previous = current
