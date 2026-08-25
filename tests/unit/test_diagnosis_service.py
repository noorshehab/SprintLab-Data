from unittest.mock import MagicMock

import pytest

from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.Treatment_service import Treatment_Service


@pytest.fixture
def diag_service():
    ds = Data_Service()
    ds.add_student('S1')
    kt = MagicMock()
    bd = MagicMock()
    ts = MagicMock()  # mock treatment so we can assert call counts
    svc = Diagnosis_service(kt, bd, ds, ts)
    return svc, ds, kt, bd


def _respond(svc, n=1):
    for i in range(n):
        svc.add_student_response('S1', [f'q{i}'], [1], [30.0], [None])


def test_calibration_fires_at_window(diag_service):
    svc, ds, kt, bd = diag_service
    _respond(svc, 10)

    kt.calibrate_priors.assert_called_once_with('S1')


def test_priors_update_after_calibration(diag_service):
    svc, ds, kt, bd = diag_service
    _respond(svc, 11)

    kt.update_student_priors.assert_called_once()
    assert len(ds.get_student('S1').get_priors_history()) == 1


def test_bd_diagnosis_fires_at_100_responses_once(diag_service):
    svc, ds, kt, bd = diag_service
    _respond(svc, 100)

    assert bd.diagnose_student.call_count == 1
    # treatment service was invoked through the mediator
    svc.Treatment_service.set_treatment_plan.assert_called_once_with('S1')


def test_notify_dispatches_to_subscribers(diag_service):
    svc, *_ = diag_service

    subscriber = MagicMock()
    svc.subscribe(subscriber, 'Question_Bandit')
    svc.notify('priors_updated', {'student_id': 'S1', 'q_ids': ['q1', 'q2']})

    # list.update bug fixed: each subscriber receives the event,
    # with q_ids remapped to the key Match_Service expects
    subscriber.update.assert_called_once_with(
        'priors_updated', {'student_id': 'S1', 'questions': ['q1', 'q2']})


def test_notify_unknown_event_is_noop(diag_service):
    svc, *_ = diag_service
    subscriber = MagicMock()
    svc.subscribe(subscriber, 'Question_Bandit')
    svc.notify('not_a_real_event', {})
    subscriber.update.assert_not_called()


def test_request_switch_routes_to_data_service(diag_service):
    svc, ds, *_ = diag_service
    result = svc.request({'type': 'get_student', 'student_id': 'S1'})
    assert result is ds.get_student('S1')


def test_periodic_recheck_every_10_after_diagnosis(diag_service):
    svc, ds, kt, bd = diag_service
    s = ds.get_student('S1')
    # seed an existing diagnosis so the periodic branch is active
    s.add_diagnosis_record('t0', {'language': -0.5}, ['language'])
    _respond(svc, 20)  # periodic branch hits at response 10 and 20

    assert bd.diagnose_student.call_count == 2
    assert svc.Treatment_service.update_treatment_plan.call_count == 2
