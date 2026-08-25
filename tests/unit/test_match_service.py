import pytest

from services.Data_service import Data_Service
from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.Question_Selection_service import Question_Selection_Service
from services.Match_Service import Match_Service


@pytest.fixture
def data_service():
    ds = Data_Service()
    ds.add_student('S1')
    s = ds.get_student('S1')
    for d in ['flexibility']:
        s.add_diagnosis(d)
    # permissive treatment plan so QSS's query matches every seeded question
    s.update_treatment_plan('t', 'treatment_plan',
                            {'general': {'base': {'difficulty_level': 0,
                                                  'Operator': '>='}},
                             'specific': {}})
    # questions across units/difficulties
    ds.add_question(q_id='qA', skill_cluster_id='KC-1', unit_id='U2', difficulty_level=0.9, p_g=0.1)
    ds.add_question(q_id='qB', skill_cluster_id='KC-1', unit_id='U1', difficulty_level=0.2, p_g=0.1)
    return ds


def _make_match(ds):
    qss = Question_Selection_Service(ContextualBandit())
    return Match_Service(qss, ds)


def test_constructor_wires_qss_mediator(data_service):
    match = _make_match(data_service)
    assert match.Q_S.mediator is match


def test_set_match_returns_question_dicts(data_service):
    match = _make_match(data_service)
    # stub the bandit selection so we control the candidate set
    match.Q_S.Bandit.select = lambda *a, **k: __import__('pandas').DataFrame(
        [{'q_id': 'qA'}, {'q_id': 'qB'}])

    result = match.set_match('S1')

    assert isinstance(result, list)
    assert {item['q'].id for item in result} == {'qA', 'qB'}
    assert all('difficulty' in item and 'unit' in item for item in result)


def test_set_match_sorts_by_unit_for_flexibility(data_service):
    match = _make_match(data_service)
    match.Q_S.Bandit.select = lambda *a, **k: __import__('pandas').DataFrame(
        [{'q_id': 'qA'}, {'q_id': 'qB'}])

    result = match.set_match('S1')  # S1 has flexibility

    assert [item['q'].id for item in result] == ['qB', 'qA']  # U1 before U2


def test_request_routes_to_data_service(data_service):
    match = _make_match(data_service)

    assert match.request({'type': 'get_student', 'student_id': 'S1'}).id == 'S1'
    # Data_Service.get_question always returns a list
    assert match.request({'type': 'get_question', 'question_id': 'qA'})[0].id == 'qA'
    assert match.request({'type': 'query',
                          'parameters': {'Topic': None, 'Attribute': 'difficulty_level',
                                         'Operator': '>=', 'Threshold': 0.5}}).keys() == {'qA'}


def test_update_computes_rewards_and_calls_bandit(data_service):
    match = _make_match(data_service)
    student = data_service.get_student('S1')
    student.update_prior('KC-1', 0.5)
    student.add_priors_record('t1', {'KC-1': 0.5})
    student.add_priors_record('t2', {'KC-1': 0.75})

    calls = []
    match.Q_S.update = lambda sc, qc, rw: calls.append((sc, qc, rw))

    match.update('priors_updated', {'student_id': 'S1', 'questions': ['qA']})

    assert len(calls) == 1
    sc, qc, rw = calls[0]
    assert sc['id'] == 'S1'
    assert rw == [0.5]  # (0.75 - 0.5) / 0.5


def test_update_ignores_unknown_student(data_service):
    match = _make_match(data_service)
    called = []
    match.Q_S.update = lambda *a: called.append(a)
    match.update('priors_updated', {'student_id': 'GHOST', 'questions': []})
    assert not called
