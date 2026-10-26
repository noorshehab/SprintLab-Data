"""SprintLab API - FastAPI shell over the in-process service graph.

Single uvicorn worker only: Data_Service is an in-memory singleton and
the stores here are process-local. Multi-worker requires the storage
seam backed by a real database (Phase B).
Run:
    ./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException

from server import dto, pipelines, reprocessing
from server.schemas import (
    AnswerAccepted, BatchResult, ContentGapsOut, CurriculumCreated,
    DiagnosesHistoryOut, DiagnosesOut, DiagnosticsAccepted, DiagnosticsRun,
    DiagnosticsStatus, HintsOut, MatchAnswer, MatchCompleteResult, MatchCreate,
    MatchCreated, PriorsHistoryOut, PriorsOut, QuestionBatch, QuestionQueryIn,
    QuestionsOut, QuestionsQuery, ResponseCountOut, ResponsesOut, SkillIn,
    StudentCreated, StudentIn, TreatmentPlanIn, TreatmentPlanOut,
    UnitIn, LessonIn, LearningObjectiveIn,
)
from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.Treatment_service import Treatment_Service
from services.behavioral_diagnosis.behavioral_diagnosis_engine import (
    behavioral_diagnosis_engine,
)
from services.knowledge_tracing.knowledge_tracing_engine import (
    knowledge_tracing_engine,
)
from services.log_setup import get_logger
from services.storage.match_store import JobStore, MatchStore
from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.Question_Selection_service import (
    Question_Selection_Service,
)
from services.Match_Service import Match_Service

log = get_logger('api')

#process-wide singletons (one worker!)
_match_store = MatchStore()
_job_store = JobStore()
runtime: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_service = Data_Service()
    #Diagnosis_service.__init__ rewires each component's mediator to itself,
    #so engines are constructed bare and handed over
    kt_engine = knowledge_tracing_engine()
    bd_engine = behavioral_diagnosis_engine()
    treatment_service = Treatment_Service()
    diagnosis_service = Diagnosis_service(kt_engine, bd_engine,
                                          data_service, treatment_service)

    selection_service = Question_Selection_Service(ContextualBandit())
    match_service = Match_Service(selection_service, data_service)  # wires Q_S.mediator

    runtime.update({
        'data_service': data_service,
        'diagnosis_service': diagnosis_service,
        'treatment_service': treatment_service,
        'selection_service': selection_service,
        'match_service': match_service,
        'match_store': _match_store,
        'job_store': _job_store,
    })
    log.info("API runtime constructed | questions=%d students=%d",
             len(data_service.list_questions()), len(data_service.list_students()))
    yield
    runtime.clear()


app = FastAPI(title='SprintLab API', version='0.1.0', lifespan=lifespan)


def _rt() -> dict:
    if not runtime:
        raise HTTPException(503, 'runtime not initialised')
    return runtime

#--- curriculum ------------------------------------------------------------

@app.post('/curriculum/units', status_code=201, response_model=CurriculumCreated)
def add_unit(body: UnitIn):
    rt = _rt()
    from services.Entities import Unit
    unit = Unit(**body.model_dump())
    rt['data_service'].add_unit(unit)
    return {'id': unit.unit_id}

@app.post('/curriculum/lessons', status_code=201, response_model=CurriculumCreated)
def add_lesson(body: LessonIn):
    rt = _rt()
    from services.Entities import Lesson
    lesson = Lesson(**body.model_dump())
    rt['data_service'].add_lesson(lesson)
    return {'id': lesson.lesson_id}

@app.post('/curriculum/objectives', status_code=201, response_model=CurriculumCreated)
def add_objective(body: LearningObjectiveIn):
    rt = _rt()
    from services.Entities import LearningObjective
    lo = LearningObjective(**body.model_dump())
    rt['data_service'].add_learning_objective(lo)
    return {'id': lo.learning_objective_id}

@app.post('/curriculum/skills', status_code=201, response_model=CurriculumCreated)
def add_skill(body: SkillIn):
    rt = _rt()
    # Data_service.add_skill takes (skill_id, similar_skills)
    rt['data_service'].add_skill(body.skill_cluster_id, body.similar_skills)
    return {'id': body.skill_cluster_id}

#--- diagnostics --------------------------------------------------------

@app.post('/diagnostics/run', status_code=202, response_model=DiagnosticsAccepted)
def run_diagnostics(body: DiagnosticsRun, background: BackgroundTasks):
    rt = _rt()
    if rt['data_service'].get_student(body.studentId) is None:
        raise HTTPException(404, f'student {body.studentId} not found')
    job = rt['job_store'].create()

    def _run():
        try:
            result = pipelines.run_diagnostics(
                rt, body.studentId, body.Answers, body.matchId)
            rt['job_store'].finish(job['jobId'], result)
        except Exception as exc:  # surface failure through the job record
            rt['job_store'].fail(job['jobId'], str(exc))

    background.add_task(_run)
    return {'jobId': job['jobId'], 'status': job['status']}


@app.get('/diagnostics/{job_id}', response_model=DiagnosticsStatus)
def diagnostics_status(job_id: str):
    rt = _rt()
    job = rt['job_store'].get(job_id)
    if job is None:
        raise HTTPException(404, f'job {job_id} not found')
    return {'status': job['status'], 'result': job['result']}


#--- matches ------------------------------------------------------------

@app.post('/matches', status_code=201, response_model=MatchCreated)
def create_match(body: MatchCreate):
    rt = _rt()
    if rt['data_service'].get_student(body.studentId) is None:
        raise HTTPException(404, f'student {body.studentId} not found')
    served = rt['match_service'].set_match(body.studentId)
    match = rt['match_store'].create(
        body.studentId, dto.match_questions_dto(served[0]))
    rt['data_service'].create_match(match['matchId'], 'match', 1)
    return {'matchId': match['matchId'], 'questions': match['questions'],
            'pre_hints': served[1], 'post_hints': served[2], 'play_hints': served[3],
            'general_music': served[4], 'question_music': served[5]}


@app.post('/matches/{match_id}/answers', response_model=AnswerAccepted)
def append_answer(match_id: str, body: MatchAnswer):
    rt = _rt()
    updated = rt['match_store'].append_answer(match_id, {
        'questionId': body.questionId,
        'response': body.response,
        'optionId': body.optionId,
        'responseTimeSeconds': body.responseTimeSeconds,
        'stressTrigger': body.stressTrigger,
        'timeOut': body.timeout,
        'hintsUsed': body.hintsUsed,
        'date': body.date,
        'answerTag': body.answerTag,
    })
    if updated is None:
        raise HTTPException(404, f'match {match_id} not found')
    return {'accepted': True}


@app.post('/matches/{match_id}/complete', response_model=MatchCompleteResult)
def complete_match(match_id: str):
    rt = _rt()
    match = rt['match_store'].get(match_id)
    if match is None:
        raise HTTPException(404, f'match {match_id} not found')
    if not match['answers']:
        raise HTTPException(400, 'match has no answers to score')
    result = pipelines.complete_match(rt, match)
    rt['match_store'].complete(match_id)
    return result


#--- questions ----------------------------------------------------------

@app.get('/questions', response_model=QuestionsOut)
def get_questions(body: QuestionsQuery):
    rt = _rt()
    want_all = str(body.allQs).strip().lower() == 'true'
    if want_all:
        selected = rt['data_service'].list_questions()
    else:
        if not body.ids:
            raise HTTPException(400, "provide ids or set allQs='True'")
        selected = [q for q in rt['data_service'].get_question(body.ids) if q]
    return {'Questions': [dto.question_dto(q) for q in selected]}


@app.post('/questions/batch', status_code=201, response_model=BatchResult)
def insert_questions(body: QuestionBatch, background: BackgroundTasks):
    rt = _rt()
    inserted = 0
    for q in body.Questions:
        before = len(rt['data_service'].list_questions())
        payload = q.model_dump(exclude_none=True)
        rt['data_service'].add_question(**payload)
        if len(rt['data_service'].list_questions()) > before:
            inserted += 1

    #relative measures shift with new content: recompute corpus-wide and
    #rebuild skill embeddings/similar-skills map (embedding cache makes
    #this cheap for incremental batches)
    summary = reprocessing.reprocess_corpus(
        rt['data_service'], rt['diagnosis_service'])

    log.info('question batch | inserted=%d reprocessed=%s',
             inserted, summary)
    return {'NumInserted': inserted}


#--- students -----------------------------------------------------------

@app.post('/students', status_code=201, response_model=StudentCreated)
def create_student(body: StudentIn):
    rt = _rt()
    rt['data_service'].add_student(body.studentId)
    return {'studentId': body.studentId}


@app.get('/students/{student_id}')
def get_student(student_id: str):
    rt = _rt()
    s = rt['data_service'].get_student(student_id)
    if s is None:
        raise HTTPException(404, f'student {student_id} not found')
    return dto.student_dto(s, rt['data_service'])


@app.get('/students/{student_id}/responses', response_model=ResponsesOut)
def get_student_responses(student_id: str):
    rt = _rt()
    records = rt['data_service'].get_response_records(student_id)
    return {'responses': [dto.response_dto(r) for r in records]}


@app.get('/students/{student_id}/responses/count', response_model=ResponseCountOut)
def get_student_response_count(student_id: str):
    rt = _rt()
    return {'count': rt['data_service'].count_responses(student_id)}


@app.get('/students/{student_id}/priors', response_model=PriorsOut)
def get_student_priors(student_id: str):
    rt = _rt()
    return {'priors': rt['data_service'].get_priors(student_id)}


@app.get('/students/{student_id}/priors/history', response_model=PriorsHistoryOut)
def get_student_priors_history(student_id: str, window: int = 2):
    rt = _rt()
    raw = rt['data_service'].get_priors_history(student_id, window)
    history = [{'timestamp': str(ts), 'priors': priors}
               for ts, priors in sorted(raw.items(), key=lambda item: str(item[0]))]
    return {'history': history}


@app.get('/students/{student_id}/diagnoses', response_model=DiagnosesOut)
def get_student_diagnoses(student_id: str):
    rt = _rt()
    return {'diagnoses': rt['data_service'].get_diagnoses(student_id)}


@app.get('/students/{student_id}/diagnoses/history', response_model=DiagnosesHistoryOut)
def get_student_diagnoses_history(student_id: str, window: int = 2):
    rt = _rt()
    history = rt['data_service'].get_diagnoses_history(student_id, window)
    return {'history': history}


@app.get('/students/{student_id}/content-gaps', response_model=ContentGapsOut)
def get_student_content_gaps(student_id: str):
    rt = _rt()
    return {'contentGaps': rt['data_service'].get_content_gaps(student_id)}


@app.get('/students/{student_id}/treatment-plan', response_model=TreatmentPlanOut)
def get_student_treatment_plan(student_id: str):
    rt = _rt()
    return {'treatmentPlan': rt['data_service'].get_treatment_plan(student_id) or {}}


@app.put('/students/{student_id}/treatment-plan', response_model=TreatmentPlanOut)
def put_student_treatment_plan(student_id: str, body: TreatmentPlanIn):
    rt = _rt()
    if rt['data_service'].get_student(student_id) is None:
        raise HTTPException(404, f'student {student_id} not found')
    rt['data_service'].replace_treatment_plan(
        student_id, datetime.now().isoformat(), body.treatmentPlan)
    return {'treatmentPlan': rt['data_service'].get_treatment_plan(student_id) or {}}


@app.get('/skills/{skill_id}')
def get_skill(skill_id: str):
    rt = _rt()
    skill = rt['data_service'].get_skill(skill_id)
    if skill is None:
        raise HTTPException(404, f'skill {skill_id} not found')
    return {
        'skillClusterId': skill.skill_cluster_id,
        'skillClusterVersionId': getattr(skill, 'skill_cluster_version_id', None),
        'content': getattr(skill, 'content', None),
        'similarSkills': skill.get_similar(),
    }


@app.post('/questions/query', response_model=QuestionsOut)
def query_questions(body: QuestionQueryIn):
    rt = _rt()
    result = rt['data_service'].query(body.conditions)
    return {'Questions': [dto.question_dto(q) for q in result.values()]}


@app.get('/hints/pre', response_model=HintsOut)
def get_pre_hints(topic: Optional[str] = None,
                  content_gap_type: Optional[str] = None,
                  diagnosis: Optional[str] = None):
    rt = _rt()
    ds = rt['data_service']
    if content_gap_type is not None and topic is not None:
        hints = ds.get_pre_hints_by_content_gap_type(topic, content_gap_type)
    elif diagnosis is not None:
        hints = ds.get_pre_hints_by_diagnosis(diagnosis)
    elif topic is not None:
        hints = ds.get_pre_hints_by_topic(topic)
    else:
        hints = ds.list_pre_hints()
    return {'hints': [dto.hint_dto(h) for h in hints]}


@app.get('/hints/pre/{hint_id}')
def get_pre_hint(hint_id: str):
    rt = _rt()
    hint = rt['data_service'].get_pre_hint(hint_id)
    if hint is None:
        raise HTTPException(404, f'pre hint {hint_id} not found')
    return dto.hint_dto(hint)


@app.get('/hints/post', response_model=HintsOut)
def get_post_hints(question_id: Optional[str] = None):
    rt = _rt()
    ds = rt['data_service']
    hints = (ds.get_post_hints_by_question(question_id)
             if question_id is not None else ds.list_post_hints())
    return {'hints': [dto.hint_dto(h) for h in hints]}


@app.get('/hints/post/{hint_id}')
def get_post_hint(hint_id: str):
    rt = _rt()
    hint = rt['data_service'].get_post_hint(hint_id)
    if hint is None:
        raise HTTPException(404, f'post hint {hint_id} not found')
    return dto.hint_dto(hint)


@app.get('/hints/play', response_model=HintsOut)
def get_play_hints(question_id: Optional[str] = None):
    rt = _rt()
    ds = rt['data_service']
    hints = (ds.get_play_hints_by_question(question_id)
             if question_id is not None else ds.list_play_hints())
    return {'hints': [dto.hint_dto(h) for h in hints]}


@app.get('/hints/play/{hint_id}')
def get_play_hint(hint_id: str):
    rt = _rt()
    hint = rt['data_service'].get_play_hint(hint_id)
    if hint is None:
        raise HTTPException(404, f'play hint {hint_id} not found')
    return dto.hint_dto(hint)


@app.get('/health')
def health():
    return {'status': 'ok', 'runtimeReady': bool(runtime)}
