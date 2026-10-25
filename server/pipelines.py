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


def _answer_tuples(answers, match_id: str, date: str = '') -> list[tuple]:
    return [(match_id,
             date,
             a.questionId,
             a.response,
             None,  # error calculated by process_responses
             False,  # timeout (not available in diagnostics)
             a.responseTimeSeconds,
             a.stressTrigger,
             0,  # hints_used (not available in diagnostics)
             a.answerTag)
            for a in answers]


def run_diagnostics(runtime, student_id: str, answers) -> dict[str, Any]:
    """Diagnostics job body: absorb answers, run KT + BD + treatment.

    Bypasses the incremental response-count gates on purpose: an API
    caller explicitly requests a diagnostics pass.
    """
    data = runtime['data_service']
    diag = runtime['diagnosis_service']
    kt = diag.KT_engine

    match_id = f"diagnostics-{uuid4()}"
    data.create_match(match_id, 'diagnostics', 1)

    old_prior_count = len(data.get_priors(student_id))
    responses = _answer_tuples(answers, match_id)

    diag.add_student_response(student_id, responses)

    kt.calibrate_priors(student_id)
    kt.update_student_priors(student_id, [a.questionId for a in answers],
                             [_correctness(diag, a.questionId, a.response) for a in answers])
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

    q_ids = [a['questionId'] for a in match['answers']]
    responses = [(match_id,
                  a.get('date', ''),
                  a['questionId'],
                  a['response'],
                  None,  # error calculated by process_responses
                  a.get('timeOut', False),
                  a.get('responseTimeSeconds'),
                  a.get('stressTrigger'),
                  a.get('hintsUsed', 0),
                  a.get('answerTag')) for a in match['answers']]

    priors_before = dict(data.get_priors(student_id))

    diag.add_student_response(student_id, responses)
    kt.update_student_priors(student_id, q_ids,
                             [_correctness(diag, r[2], r[3]) for r in responses])
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


def _correctness(diag, q_id: str, response) -> int:
    question = diag.Data_service.get_question(q_id)
    if isinstance(question, list):
        question = question[0] if question else None
    if question is None:
        return 1 if bool(response) else 0
    return 0 if diag._calculate_error(question, response) else 1


def _is_nan(value) -> bool:
    try:
        import math
        return isinstance(value, float) and math.isnan(value)
    except (TypeError, ValueError):
        return False
