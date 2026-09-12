"""Entity -> DTO serializers (whitelisted fields, never raw entities)."""
from __future__ import annotations
from typing import Any

OPTION_DTO_FIELDS = [
    'option_id', 'content', 'is_correct', 'display_order',
    'content_gap_type', 'distractor_reading_trap', 'distractor_impulsive',
    'prior_skill_reference',
]

QUESTION_DTO_FIELDS = [
    #identity / taxonomy
    'id', 'question_type', 'question_text', 'question_language',
    'skill_cluster_id', 'skill_ids', 'unit_id', 'lesson_id',
    'subject', 'grade', 'grade_band',
    #serving-relevant derived + treated fields
    'difficulty_level', 'language_level', 'reasoning_level',
    'cognitive_load', 'cognitive_load_index', 'logical_steps',
    'variables_count', 'time_pressure_flag', 'visual_dependency',
    'multi_concept_flag', 'bloom_taxonomy_level',
    #BKT parameters
    'p_t', 'p_s', 'p_g',
    #content needed for future reprocessing
    'question_text', 'correct_answer_content',
    #A-Tags presence flag
    'option_count',
]


def _serialize_option(opt: Any) -> dict[str, Any]:
    out = {}
    for field in OPTION_DTO_FIELDS:
        value = getattr(opt, field, None)
        if isinstance(value, set):
            value = list(value)
        out[field] = value
    return out


def question_dto(q: Any) -> dict[str, Any]:
    out = {}
    for field in QUESTION_DTO_FIELDS:
        if field == 'option_count':
            out['option_count'] = len(getattr(q, 'options', []))
            continue
        value = getattr(q, field, None)
        if isinstance(value, set):
            value = list(value)
        out[field] = value
    # serialize options
    out['options'] = [_serialize_option(opt) for opt in getattr(q, 'options', [])]
    return out


def match_questions_dto(questions: list[dict]) -> list[dict[str, Any]]:
    """set_match returns [{'q': entity, 'difficulty': x, 'unit': y}]."""
    return [{'question': question_dto(item['q']),
             'difficulty': item['difficulty'],
             'unit': item['unit']} for item in questions]


def student_dto(s: Any) -> dict[str, Any]:
    return {
        'studentId': s.id,
        'priors': s.get_priors(),
        'contentGaps': s.get_content_gaps(),
        'diagnoses': s.get_diagnoses(),
        'treatmentPlan': _plan_dto(s),
        'responsesCount': len(s.get_responses()),
    }


def _plan_dto(s: Any) -> dict[str, Any]:
    stored = s.get_treatment_plan().get('treatment_plan')
    if not stored:
        return {}
    return {'updatedAt': stored[0], 'constraints': stored[1]}