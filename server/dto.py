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


def student_dto(s: Any, data_service: Any) -> dict[str, Any]:
    return {
        'studentId': s.id,
        'priors': data_service.get_priors(s.id),
        'contentGaps': data_service.get_content_gaps(s.id),
        'diagnoses': data_service.get_diagnoses(s.id),
        'treatmentPlan': data_service.get_treatment_plan(s.id) or {},
        'responsesCount': data_service.count_responses(s.id),
    }


def response_dto(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'matchId': record['matchId'],
        'date': record.get('date'),
        'questionId': record['questionId'],
        'optionId': record.get('optionId'),
        'response': record.get('response'),
        'error': bool(record.get('error')),
        'timeout': bool(record.get('timeout')),
        'responseTimeSeconds': float(record.get('responseTimeSeconds') or 0.0),
        'stressTrigger': int(record.get('stressTrigger') or 0),
        'hintsUsed': int(record.get('hintsUsed') or 0),
        'answerTag': record.get('answerTag'),
    }


def hint_dto(hint: Any) -> dict[str, Any]:
    out = {'hintId': hint.hint_id, 'hintText': hint.get_text()}
    for field in ('topic', 'content_gap_type', 'diagnosis', 'hint_type',
                  'question_id', 'q_id', 'hint_order'):
        if hasattr(hint, field):
            out[field] = getattr(hint, field)
    return out