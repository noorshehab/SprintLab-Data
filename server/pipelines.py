"""Request-scoped pipelines that orchestrate the service graph.

Kept out of the route handlers so they are independently testable and
the routes stay thin.
"""
from __future__ import annotations
from typing import Any

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


def _answer_tuples(answers) -> list[tuple]:
    return [(a.questionId,
             1 if bool(a.response) else 0,
             a.responseTimeSeconds,
             a.stressTrigger,
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

    old_prior_count = len(data.get_student(student_id).get_priors()) \
        if data.get_student(student_id) else 0

    diag.request({'type': 'add_response', 'student_id': student_id,
                  'responses': _answer_tuples(answers)})

    kt.calibrate_priors(student_id)
    kt.update_student_priors(student_id, [a.questionId for a in answers],
                             [1 if bool(a.response) else 0 for a in answers])
    kt.update_content_gaps(student_id)

    bd_result = diag.BD_engine.diagnose_student(student_id)

    student = data.get_student(student_id)
    if not student.get_treatment_plan():
        diag.Treatment_service.set_treatment_plan(student_id)
    else:
        diag.Treatment_service.update_treatment_plan(student_id)

    result = {
        'priorsWritten': max(len(student.get_priors()), old_prior_count),
        'treatmentPlanWritten': 'treatment_plan' in student.get_treatment_plan(),
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

    q_ids = [a['questionId'] for a in match['answers']]
    responses = [(a['questionId'],
                  1 if bool(a['response']) else 0,
                  a.get('responseTimeSeconds'),
                  a.get('stressTrigger'),
                  a.get('answerTag')) for a in match['answers']]

    priors_before = dict(data.get_student(student_id).get_priors())

    diag.request({'type': 'add_response', 'student_id': student_id,
                  'responses': responses})
    kt.update_student_priors(student_id, q_ids,
                             [1 if bool(r[1]) else 0 for r in responses])
    kt.update_content_gaps(student_id)

    bd_result = diag.BD_engine.diagnose_student(student_id)

    student = data.get_student(student_id)
    if not student.get_treatment_plan():
        diag.Treatment_service.set_treatment_plan(student_id)
    else:
        diag.Treatment_service.update_treatment_plan(student_id)

    priors_after = student.get_priors()
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

    stored = student.get_treatment_plan().get('treatment_plan')
    constraints = []
    if stored:
        constraints = runtime['selection_service']._unwrap_treatment_plan(stored[1])

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
