import pandas as pd
import pytest

from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.Question_Selection_service import Question_Selection_Service


class FakeMatchService:
    """Stands in for Match_Service mediator."""

    def __init__(self, student, candidates):
        self.student = student
        self.candidates = candidates  # dict {q_id: question}
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        if request.get('type') == 'get_student':
            return self.student
        if request.get('type') == 'query':
            return self.candidates
        return None


@pytest.fixture
def bandit():
    return ContextualBandit()


def _make_question(q_id, skill_cluster_id, difficulty=0.5):
    from services.Entities import question
    return question(q_id=q_id, skill_cluster_id=skill_cluster_id,
                    difficulty_level=difficulty, p_g=0.1)


def test_unwrap_treatment_plan_produces_conditions():
    treatment = {
        'general': {'language': {'language_level': 'Q1', 'Operator': '=='},
                    'working_memory': {'cognitive_load': 1, 'Operator': '<='}},
        'specific': {'KC-BIO-01': {'bloom_taxonomy_level': ['Remember'], 'Operator': 'in'}},
    }
    svc = Question_Selection_Service(ContextualBandit(), FakeMatchService(None, {}))
    conditions = svc._unwrap_treatment_plan(treatment)

    assert {'Topic': 'general', 'Attribute': 'language_level',
            'Operator': '==', 'Threshold': 'Q1'} in conditions
    assert {'Topic': 'general', 'Attribute': 'cognitive_load',
            'Operator': '<=', 'Threshold': 1} in conditions
    assert {'Topic': 'KC-BIO-01', 'Attribute': 'bloom_taxonomy_level',
            'Operator': 'in', 'Threshold': ['Remember']} in conditions
    assert len(conditions) == 3


def test_get_candidate_questions_sends_query_to_mediator():
    from services.Entities import student

    s = student('S1')
    s.update_treatment_plan('t', 'treatment_plan',
                            {'general': {}, 'specific': {}})
    mediator = FakeMatchService(s, {})
    svc = Question_Selection_Service(ContextualBandit(), mediator)
    svc.get_candidate_questions('S1')

    assert any(r.get('type') == 'query' for r in mediator.requests)


def test_get_optimal_set_empty_candidates_returns_empty_frame():
    from services.Entities import student

    s = student('S1')
    s.update_treatment_plan('t', 'treatment_plan',
                            {'general': {}, 'specific': {}})
    svc = Question_Selection_Service(ContextualBandit(), FakeMatchService(s, None))

    result = svc.get_optimal_set('S1')
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_bandit_select_ranks_by_expected_reward(bandit):
    contexts = pd.DataFrame([
        {'q_id': 'q_low', 'skill_ids': ['KC-1'], 'difficulty': 0.9, 'learning': 0.9},
        {'q_id': 'q_high', 'skill_ids': ['KC-2'], 'difficulty': 0.1, 'learning': 0.1},
    ])
    student_context = {'id': 'S1', 'priors': {'KC-1': 0.1, 'KC-2': 0.9}}

    scores = bandit.select(student_context, contexts)

    # q_low: low prior on its skill + high difficulty/learning => highest reward
    assert scores.iloc[0]['q_id'] == 'q_low'


def test_bandit_select_handles_empty_contexts(bandit):
    empty = pd.DataFrame({'q_id': [], 'skill_ids': [], 'difficulty': [], 'learning': []})
    result = bandit.select({'id': 'S1', 'priors': {}}, empty)
    assert result.empty


def test_bandit_update_accumulates_rewards_per_context(bandit):
    contexts = pd.DataFrame([
        {'q_id': 'q1', 'skill_ids': ['KC-1'], 'difficulty': 0.5, 'learning': 0.5},
        {'q_id': 'q2', 'skill_ids': ['KC-1'], 'difficulty': 0.5, 'learning': 0.5},
    ])
    rewards = [0.2, 0.6]
    bandit.update({'id': 'S1', 'priors': {'KC-1': 0.5}}, contexts, rewards)

    key = round((1 - 0.5) * 0.5 * 0.5, 2)
    assert sorted(bandit.rewards_by_context[key]) == [0.2, 0.6]
