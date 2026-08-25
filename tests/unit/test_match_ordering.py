"""Ordering contract tests for diagnosis-driven question serving.

stress, flexibility and attention_span do not (primarily) shape treatment
parameters - they dictate HOW questions are served, i.e. their ORDER in
Match_Service.set_match:
  - flexibility      -> grouped by unit (unit ascending)
  - attention_span   -> hardest questions first (difficulty descending)
  - stress           -> easy/hard interleaved after ascending sort
These tests drive the full QSS -> query -> bandit -> arrangement chain.
"""
import pandas as pd
import pytest

from services.Data_service import Data_Service
from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.Question_Selection_service import Question_Selection_Service
from services.Match_Service import Match_Service

# deterministic candidate pool: 6 questions, units U1..U3, difficulties 0.1..0.9
QUESTIONS = [
    # q_id, unit, difficulty
    ('q_easy1',   'U2', 0.10),
    ('q_med',     'U1', 0.50),
    ('q_hard',    'U3', 0.90),
    ('q_easy2',   'U3', 0.20),
    ('q_med_hard','U1', 0.70),
    ('q_veryhard','U2', 0.95),
]


def _pipeline_with_diagnoses(diagnoses):
    ds = Data_Service()
    ds.add_student('S_order')
    s = ds.get_student('S_order')
    for d in diagnoses:
        s.add_diagnosis(d)
    # permissive plan so every seeded question is a candidate
    s.update_treatment_plan('t', 'treatment_plan',
                            {'general': {'base': {'difficulty_level': 0,
                                                  'Operator': '>='}},
                             'specific': {}})
    for q_id, unit, difficulty in QUESTIONS:
        ds.add_question(q_id=q_id, skill_cluster_id='KC-1', unit_id=unit,
                        difficulty_level=difficulty, p_g=0.1)
    match = Match_Service(Question_Selection_Service(ContextualBandit()), ds)
    return match


def _ids(result):
    return [item['q'].id for item in result]


def test_no_diagnosis_preserves_bandit_ranking():
    match = _pipeline_with_diagnoses([])
    result = match.set_match('S_order')
    assert sorted(_ids(result)) == sorted(q[0] for q in QUESTIONS)


def test_flexibility_groups_questions_by_unit():
    match = _pipeline_with_diagnoses(['flexibility'])
    result = match.set_match('S_order')

    ids = _ids(result)
    assert len(ids) == len(QUESTIONS)  # all served, just reordered
    units = [item['unit'] for item in result]
    assert units == sorted(units), f'expected unit-ascending order, got {list(zip(ids, units))}'


def test_attention_span_serves_hardest_first():
    match = _pipeline_with_diagnoses(['attention_span'])
    result = match.set_match('S_order')

    ids = _ids(result)
    assert ids[0] == 'q_veryhard'  # difficulty 0.95 first
    difficulties = [item['difficulty'] for item in result]
    assert difficulties == sorted(difficulties, reverse=True)


def test_stress_interleaves_easy_and_hard():
    match = _pipeline_with_diagnoses(['stress'])
    result = match.set_match('S_order')

    difficulties = [item['difficulty'] for item in result]
    # ascending baseline: 0.1,0.2,0.5 | 0.7,0.9,0.95 -> zip(easy, hard) weave
    expected = [0.10, 0.70, 0.20, 0.90, 0.50, 0.95]
    assert difficulties == expected, f'expected easy/hard weave {expected}, got {difficulties}'


def test_combined_diagnoses_apply_in_documented_sequence():
    """When multiple ordering diagnoses coexist they apply sequentially:
    flexibility (unit) -> attention_span (difficulty desc) -> stress (weave).
    Final order must equal running all three passes by hand."""
    match = _pipeline_with_diagnoses(['flexibility', 'attention_span', 'stress'])
    result = match.set_match('S_order')
    difficulties = [item['difficulty'] for item in result]

    # hand-run the same three passes over the raw pool
    pool = sorted(QUESTIONS, key=lambda q: q[2])           # attention pass sorts desc,
    pool = sorted(pool, key=lambda q: q[2], reverse=True)  # but flexibility ran first on units
    flex_sorted = sorted(QUESTIONS, key=lambda q: q[1])    # unit asc
    att_sorted = sorted(flex_sorted, key=lambda q: q[2], reverse=True)
    asc = sorted(att_sorted, key=lambda q: q[2])
    mid = len(asc) // 2
    easy, hard = asc[:mid], asc[mid:]
    woven = []
    for e, h in zip(easy, hard):
        woven.extend([e, h])
    woven.extend(hard[len(easy):])

    assert difficulties == [q[2] for q in woven]


@pytest.mark.parametrize('diagnoses', [
    ['flexibility'],
    ['attention_span'],
    ['stress'],
], ids=['flexibility', 'attention_span', 'stress'])
def test_ordering_never_drops_questions(diagnoses):
    match = _pipeline_with_diagnoses(diagnoses)
    result = match.set_match('S_order')
    assert set(_ids(result)) == {q[0] for q in QUESTIONS}
