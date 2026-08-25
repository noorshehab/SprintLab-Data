"""Entity -> DTO serializers (whitelisted fields, never raw entities)."""
from __future__ import annotations
from typing import Any

QUESTION_DTO_FIELDS = [
    #identity / taxonomy
    'id', 'skill_cluster_id', 'skill_ids', 'unit', 'lesson_id',
    'subject', 'grade',
    #serving-relevant derived + treated fields
    'difficulty_level', 'language_level', 'reasoning_level',
    'cognitive_load', 'cognitive_load_index', 'logical_steps',
    'variables_count', 'time_pressure_flag', 'visual_dependency',
    'multi_concept_flag', 'bloom_taxonomy_level',
    #BKT parameters
    'p_t', 'p_s', 'p_g',
    #content needed for future reprocessing
    'question_text', 'correct_answer_content', 'language',
]


def question_dto(q) -> dict[str, Any]:
    out = {}
    for field in QUESTION_DTO_FIELDS:
        value = getattr(q, field, None)
        if isinstance(value, set):
            value = list(value)
        out[field] = value
    return out


def match_questions_dto(questions: list[dict]) -> list[dict[str, Any]]:
    """set_match returns [{'q': entity, 'difficulty': x, 'unit': y}]."""
    return [{'question': question_dto(item['q']),
             'difficulty': item['difficulty'],
             'unit': item['unit']} for item in questions]


def student_dto(s) -> dict[str, Any]:
    return {
        'studentId': s.id,
        'priors': s.get_priors(),
        'contentGaps': s.get_content_gaps(),
        'diagnoses': s.get_diagnoses(),
        'treatmentPlan': _plan_dto(s),
        'responsesCount': len(s.get_responses()),
    }


def _plan_dto(s) -> dict[str, Any]:
    stored = s.get_treatment_plan().get('treatment_plan')
    if not stored:
        return {}
    return {'updatedAt': stored[0], 'constraints': stored[1]}
