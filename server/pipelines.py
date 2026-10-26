"""Request-scoped pipelines that orchestrate the service graph.

Kept out of the route handlers so they are independently testable and
the routes stay thin.
"""
from __future__ import annotations
from typing import Any
from uuid import uuid4

from services.log_setup import get_logger

log = get_logger('pipelines')

#metric name -> its *_diag companion key in the BD result series
_DIAG_KEYS = {
    'processing_speed': 'processing_speed_diag',
    'reasoning': 'reasoning_diag',
    'language': 'language_diag',
    'flexibility': 'flexibility_diag',
    'attention': 'attention_diag',
    'frustration': 'frustration_diag',
    'working_memory': 'working_memory_diag',
    'time_management_ratio': 'time_management_diag',
    'stress_ratio': 'stress_diag',
    'impulse_error_rate': 'impulsivity_diag',
}


def _answer_tuple(answer, match_id: str, date: str = '') -> tuple:
    """Map an API answer onto the Response-table tuple contract.

    Prefer the persisted OptionID when supplied; otherwise pass the raw
    response through so Diagnosis_service can resolve/score it.
    """
    option_id = getattr(answer, 'optionId', None)
    response = option_id if option_id is not None else getattr(answer, 'response', None)
    return (match_id,
            getattr(answer, 'date', None) or date,
            answer.questionId,
            response,
            None,  # error calculated by process_responses
            bool(getattr(answer, 'timeout', False)),
            getattr(answer, 'responseTimeSeconds', 0.0),
            getattr(answer, 'stressTrigger', 0),
            getattr(answer, 'hintsUsed', 0),
            getattr(answer, 'answerTag', None))


def _answer_tuples(answers, match_id: str, date: str = '') -> list[tuple]:
    return [_answer_tuple(a, match_id, date) for a in answers]


def _dict_answer_tuple(answer: dict, match_id: str) -> tuple:
    response = answer.get('optionId')
    if response is None:
        response = answer.get('response')
    return (match_id,
            answer.get('date') or '',
            answer['questionId'],
            response,
            None,
            bool(answer.get('timeOut', False)),
            answer.get('responseTimeSeconds', 0.0),
            answer.get('stressTrigger', 0),
            answer.get('hintsUsed', 0),
            answer.get('answerTag'))


def run_diagnostics(runtime, student_id: str, answers,
                    match_id: str | None = None) -> dict[str, Any]:
    """Diagnostics job body: absorb answers, run KT + BD + treatment.

    Bypasses the incremental response-count gates on purpose: an API
    caller explicitly requests a diagnostics pass.
    """
    data = runtime['data_service']
    diag = runtime['diagnosis_service']
    kt = diag.KT_engine

    match_id = match_id or f"diagnostics-{uuid4()}"
    data.create_match(match_id, 'diagnostics', 1)

    old_prior_count = len(data.get_priors(student_id))
    responses = _answer_tuples(answers, match_id)
    processed = diag.process_responses(student_id, responses)

    kt.calibrate_priors(student_id)
    kt.update_student_priors(student_id,
                             [r[2] for r in processed],
                             [0 if r[4] else 1 for r in processed])
    kt.update_content_gaps(student_id)

    bd_result = diag.BD_engine.diagnose_student(student_id)

    plan = data.get_treatment_plan(student_id)
    if not plan:
        plan = diag.Treatment_service.set_treatment_plan(student_id)
    else:
        plan = diag.Treatment_service.update_treatment_plan(student_id)

    result = {
        'priorsWritten': max(len(data.get_priors(student_id)), old_prior_count),
        'treatmentPlanWritten': plan is not None,
    }
    log.info("diagnostics | student_id=%s diagnoses=%s -> %s",
             student_id,
             list(bd_result.get('diagnoses', [])) if bd_result is not None else None,
             result)
    return result


def complete_match(runtime, match: dict) -> dict[str, Any]:
    """Match completion: progress tracking over accumulated answers.

    Writes updated priors, per-skill mastery deltas, behavioral scores and
    the updated treatment plan; bypasses response-count gates because the
    caller decides when a match ends.
    """
    data = runtime['data_service']
    diag = runtime['diagnosis_service']
    kt = diag.KT_engine
    student_id = match['studentId']
    match_id = match['matchId']

    responses = [_dict_answer_tuple(a, match_id) for a in match['answers']]

    priors_before = dict(data.get_priors(student_id))

    processed = diag.process_responses(student_id, responses)
    kt.update_student_priors(student_id,
                             [r[2] for r in processed],
                             [0 if r[4] else 1 for r in processed])
    kt.update_content_gaps(student_id)

    bd_result = diag.BD_engine.diagnose_student(student_id)

    plan = data.get_treatment_plan(student_id)
    if not plan:
        plan = diag.Treatment_service.set_treatment_plan(student_id)
    else:
        plan = diag.Treatment_service.update_treatment_plan(student_id)

    priors_after = data.get_priors(student_id)
    delta_mastery = {skill: round(priors_after.get(skill, 0.0)
                                  - priors_before.get(skill, 0.0), 4)
                     for skill in set(priors_before) | set(priors_after)}

    behavioral_scores = []
    if bd_result is not None:
        for key, value in bd_result.items():
            diag_key = _DIAG_KEYS.get(key)
            if diag_key is None or diag_key not in bd_result:
                continue  # skips 'diagnoses' and unknown extras
            behavioral_scores.append({
                'metric': key,
                'value': None if _is_nan(value) else round(float(value), 4),
                'diagnosed': bool(bd_result[diag_key]),
            })

    constraints = runtime['selection_service']._unwrap_treatment_plan(plan or {})

    return {
        'deltaMastery': delta_mastery,
        'updatedPriors': {k: float(v) for k, v in priors_after.items()},
        'behavioralScores': behavioral_scores,
        'updatedTreatmentPlan': constraints,
    }


def _is_nan(value) -> bool:
    try:
        import math
        return isinstance(value, float) and math.isnan(value)
    except (TypeError, ValueError):
        return False
