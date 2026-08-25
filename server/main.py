"""SprintLab API - FastAPI shell over the in-process service graph.

Single uvicorn worker only: Data_Service is an in-memory singleton and
the stores here are process-local. Multi-worker requires the storage
seam backed by a real database (Phase B).
Run:
    ./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException

from server import dto, pipelines, reprocessing
from server.schemas import (
    AnswerAccepted, BatchResult, DiagnosticsAccepted, DiagnosticsRun,
    DiagnosticsStatus, MatchAnswer, MatchCompleteResult, MatchCreate,
    MatchCreated, QuestionBatch, QuestionsOut, QuestionsQuery,
    StudentCreated,
)
from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.Entities import student as StudentEntity
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


#--- diagnostics --------------------------------------------------------

@app.post('/diagnostics/run', status_code=202, response_model=DiagnosticsAccepted)
def run_diagnostics(body: DiagnosticsRun, background: BackgroundTasks):
    rt = _rt()
    if rt['data_service'].get_student(body.studentId) is None:
        raise HTTPException(404, f'student {body.studentId} not found')
    job = rt['job_store'].create()

    def _run():
        try:
            result = pipelines.run_diagnostics(rt, body.studentId, body.Answers)
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
        body.studentId, dto.match_questions_dto(served))
    return {'matchId': match['matchId'], 'questions': match['questions']}


@app.post('/matches/{match_id}/answers', response_model=AnswerAccepted)
def append_answer(match_id: str, body: MatchAnswer):
    rt = _rt()
    updated = rt['match_store'].append_answer(match_id, {
        'questionId': body.questionId,
        'response': body.response,
        'responseTimeSeconds': body.responseTimeSeconds,
        'stressTrigger': body.stressTrigger,
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
def create_student(body: StudentCreated):
    rt = _rt()
    rt['data_service'].add_student(body.studentId)
    return {'studentId': body.studentId}


@app.get('/students/{student_id}')
def get_student(student_id: str):
    rt = _rt()
    s = rt['data_service'].get_student(student_id)
    if s is None:
        raise HTTPException(404, f'student {student_id} not found')
    return dto.student_dto(s)


@app.get('/health')
def health():
    return {'status': 'ok', 'runtimeReady': bool(runtime)}
