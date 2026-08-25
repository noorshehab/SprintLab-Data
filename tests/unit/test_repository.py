"""Repository seam tests.

Verifies InMemoryRepository behavior and, critically, that Data_Service
works with ANY repository implementation (swap test) - the guarantee the
future SqlAlchemyRepository relies on.
"""
import pytest

from services.Data_service import Data_Service
from services.Entities import question
from services.storage.base import DataRepository
from services.storage.memory import InMemoryRepository


@pytest.fixture
def repo():
    return InMemoryRepository()


def test_student_crud(repo):
    assert repo.get_student('S1') is None
    repo.add_student('S1')
    assert repo.get_student('S1').id == 'S1'
    # idempotent
    repo.add_student('S1')
    assert len(repo.list_students()) == 1


def test_question_crud_and_attribute_update(repo):
    repo.add_question(q_id='q1', skill_cluster_id='KC-1', difficulty_level=0.5)
    found = repo.get_question(['q1', 'missing'])
    assert [q.id if q else None for q in found] == ['q1', None]
    # scalar id returns single-entity list
    assert repo.get_question('q1')[0].id == 'q1'

    repo.update_question_attributes('q1', {'language_level': 'Q2'})
    assert repo.list_questions()[0].language_level == 'Q2'

    with pytest.raises(KeyError):
        repo.update_question_attributes('ghost', {'p_t': 0.1})


def test_query_lives_in_repository(repo):
    repo.add_question(q_id='a', skill_cluster_id='KC-1', difficulty_level=0.9,
                      language_level='Q4')
    repo.add_question(q_id='b', skill_cluster_id='KC-2', difficulty_level=0.1,
                      language_level='Q1')

    hard = repo.query([{'Topic': 'general', 'Attribute': 'difficulty_level',
                        'Operator': '>=', 'Threshold': 0.5}])
    assert set(hard) == {'a'}

    easy_lang = repo.query({'Topic': 'general', 'Attribute': 'lang_difficulty',
                            'Operator': '==', 'Threshold': 'Q1'})  # legacy alias
    assert set(easy_lang) == {'b'}

    with pytest.raises(ValueError):
        repo.query([{'Topic': 'general', 'Attribute': 'not_a_field',
                     'Operator': '==', 'Threshold': 1}])


def test_student_state_writes(repo):
    from datetime import datetime
    repo.add_student('S1')
    ts = str(datetime.now())

    repo.update_responses('S1', [('q1', 1, 10.0, 0, 'tag')])
    assert len(repo.get_student('S1').get_responses()) == 1

    repo.update_priors('S1', 'KC-1', 0.7)
    student = repo.get_student('S1')
    assert student.get_priors()['KC-1'] == 0.7
    assert any('KC-1' in snap for snap in student.get_priors_history().values())

    repo.add_diagnosis('S1', ['stress'], deltas={'t': 1}, timestamp=ts)
    assert 'stress' in student.get_diagnoses()
    assert ts in student.get_deltas()

    repo.update_content_gaps('S1', {'KC-1': 'Gap_Absence'})
    assert student.get_content_gaps() == {'KC-1': 'Gap_Absence'}

    repo.add_priors_history('S1', ts, {'KC-1': 0.5})
    assert ts in student.get_priors_history()

    repo.update_treatment_plan('S1', ts, 'treatment_plan',
                               ({'general': {}}, {'specific': {}}))
    assert 'treatment_plan' in student.get_treatment_plan()


def test_skills(repo):
    repo.add_skill('KC-1', ['KC-2'])
    assert repo.get_skill('KC-1').get_similar() == ['KC-2']
    repo.add_skill('KC-1', ['other'])  # first write wins
    assert repo.get_skill('KC-1').get_similar() == ['KC-2']


class FakePostgresRepository(DataRepository):
    """Minimal stand-in proving any implementation slots under Data_Service."""

    def __init__(self):
        self.rows = {}

    def add_student(self, student_id):
        self.rows.setdefault(student_id, question if False else None)

    def get_student(self, student_id):
        return self.rows.get(student_id)

    def list_students(self):
        return []

    def add_question(self, **kwargs):
        pass

    def get_question(self, q_ids):
        return []

    def list_questions(self):
        return []

    def update_question_attributes(self, q_id, attributes):
        pass

    def add_skill(self, skill_id, similar_skills=None):
        pass

    def get_skill(self, skill_id):
        return None

    def update_priors(self, student_id, skill_id, new_prior):
        pass

    def update_responses(self, student_id, responses):
        pass

    def add_diagnosis(self, student_id, diagnoses, deltas=None, timestamp=None):
        pass

    def update_content_gaps(self, student_id, content_gap_types):
        pass

    def add_priors_history(self, student_id, timestamp, priors):
        pass

    def update_treatment_plan(self, student_id, timestamp, treatment_name, parameters):
        pass

    def query(self, parameters):
        return {}


def test_swap_test_data_service_accepts_any_repository():
    fake = FakePostgresRepository()
    ds = Data_Service(repository=fake)

    ds.add_student('S9')
    assert fake.rows.keys() == {'S9'}          # writes hit the repository...
    assert ds.repository is fake               # ...not the in-memory default


def test_default_repository_is_in_memory():
    ds = Data_Service()
    assert isinstance(ds.repository, InMemoryRepository)
