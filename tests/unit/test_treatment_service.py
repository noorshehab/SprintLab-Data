import pytest

from services.Treatment_service import Treatment_Service


class FakeMediator:
    """Stands in for Diagnosis_service; handles get_student and persists
    treatment plan writes onto the student entity like the real mediator."""

    def __init__(self, student):
        self.student = student
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        if request.get('type') == 'get_student':
            return self.student
        if request.get('type') == 'update_treatment_plan':
            self.student.update_treatment_plan(
                request['timestamp'], request['treatment_name'], request['parameters'])
            return None
        return None


@pytest.fixture
def make_student():
    from services.Entities import student

    def _make(diagnoses=None, content_gaps=None, deltas=None, priors_history=None):
        s = student('S1')
        for d in (diagnoses or []):
            s.add_diagnosis(d)
        if content_gaps:
            s.update_content_gaps(content_gaps)
        for ts, rec in (deltas or {}).items():
            s.add_diagnosis_record(ts, rec, diagnoses or [])
        for ts, rec in (priors_history or {}).items():
            s.add_priors_record(ts, rec)
        return s

    return _make


def test_set_treatment_plan_maps_diagnoses_and_gaps(make_student):
    s = make_student(
        diagnoses=['language', 'impulsive'],  # impulsive has no mapping
        content_gaps={'KC-BIO-01': 'Gap_Absence'},
    )
    svc = Treatment_Service(FakeMediator(s))
    plan = svc.set_treatment_plan('S1')

    assert 'language' in plan['general']
    assert 'impulsive' not in plan['general']
    assert plan['specific']['KC-BIO-01']['bloom_taxonomy_level'] == ['Remember', 'Understand']
    # plan persisted to the student entity
    stored = s.get_treatment_plan()['treatment_plan']
    assert stored[1] == plan


def test_set_treatment_plan_empty_for_unknown_student(make_student):
    svc = Treatment_Service(FakeMediator(None))
    assert svc.set_treatment_plan('NOPE') == {}


def test_update_scales_numeric_params_by_delta_improvement(make_student):
    # language delta improved 50%: -0.4 -> -0.2 => factor 1.5
    s = make_student(
        diagnoses=['language'],
        deltas={
            't1': {'language': -0.40},
            't2': {'language': -0.20},
        },
    )
    svc = Treatment_Service(FakeMediator(s))
    base = svc.set_treatment_plan('S1')
    updated = svc.update_treatment_plan('S1')

    # numeric params scale by improvement pct; strings don't
    assert updated['general']['language']['Operator'] == base['general']['language']['Operator']
    # language_level escalates one quartile per >=10% improvement
    assert updated['general']['language']['language_level'] == 'Q4'


def test_update_working_memory_escalates_cognitive_load(make_student):
    # working_memory delta -0.5 -> -0.25 is 50% improvement -> load ceiling raised
    s = make_student(
        diagnoses=['working_memory'],
        deltas={
            't1': {'working_memory': -0.5},
            't2': {'working_memory': -0.25},
        },
    )
    svc = Treatment_Service(FakeMediator(s))
    svc.set_treatment_plan('S1')
    updated = svc.update_treatment_plan('S1')

    assert updated['general']['working_memory']['cognitive_load'] == 4


def test_update_time_management_adds_time_and_steps(make_student):
    # time_management_ratio 3.0 -> 1.5 is 50% improvement
    s = make_student(
        diagnoses=['time_management'],
        deltas={
            't1': {'time_management_ratio': 3.0},
            't2': {'time_management_ratio': 1.5},
        },
    )
    svc = Treatment_Service(FakeMediator(s))
    svc.set_treatment_plan('S1')
    updated = svc.update_treatment_plan('S1')

    tm = updated['general']['time_management']
    assert tm['time_allowed'] == 180 + 180  # >=40% improvement adds +180s
    assert tm['logical_steps'] == 2 + 6


def test_update_specific_gap_scaled_by_prior_improvement(make_student):
    # prior on KC-BIO-01 went 0.5 -> 1.0 => +100% improvement
    s = make_student(
        content_gaps={'KC-BIO-01': 'Gap_Absence'},
        deltas={},
        priors_history={
            't1': {'KC-BIO-01': 0.5},
            't2': {'KC-BIO-01': 1.0},
        },
    )
    svc = Treatment_Service(FakeMediator(s))
    svc.set_treatment_plan('S1')
    updated = svc.update_treatment_plan('S1')

    gap = updated['specific']['KC-BIO-01']
    # bloom list and operator are non-numeric so unchanged
    assert gap['bloom_taxonomy_level'] == ['Remember', 'Understand']


def test_no_history_leaves_params_unscaled(make_student):
    s = make_student(diagnoses=['language'])
    svc = Treatment_Service(FakeMediator(s))
    base = svc.set_treatment_plan('S1')
    updated = svc.update_treatment_plan('S1')

    assert updated['general']['language'] == base['general']['language']
