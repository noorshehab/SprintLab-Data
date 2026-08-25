"""Demo runtime: wires the service graph, ingests the science-question CSV,
and scaffolds the S101 profile from scripts/generate.py.

S101 semantics preserved, priors/gaps remapped onto the real skill
clusters present in the candidate CSV (KC-BIO-01 -> SC-CELL-01,
KC-PHYS-01 -> SC-KINEMATICS-01, KC-CHEM-01 -> SC-BOND-01).
"""
from __future__ import annotations
import os

from server import reprocessing
from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.Treatment_service import Treatment_Service
from services.behavioral_diagnosis.behavioral_diagnosis_engine import (
    behavioral_diagnosis_engine)
from services.data_ingestion import file_loader
from services.data_ingestion.data_ingestion_service import data_ingestion_service
from services.knowledge_tracing.knowledge_tracing_engine import (
    knowledge_tracing_engine)
from services.log_setup import get_logger
from services.Match_Service import Match_Service
from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.Question_Selection_service import (
    Question_Selection_Service)

log = get_logger('demo')

QUESTION_CSV = os.getenv(
    'DEMO_QUESTIONS_CSV',
    'E://projects//sprintlabfiles//sprintlab_candidate_science_questions.csv')

DEMO_STUDENT = 'S101'

#generate.py S101 profile, keyed onto real CSV skill clusters
PROFILE = {
    'name': 'Student A - low prior, high impulsivity & language gap',
    'diagnoses': ['language', 'impulsive', 'processing_speed'],
    'priors': {
        'SC-CELL-01': 0.25,        # was KC-BIO-01
        'SC-KINEMATICS-01': 0.30,  # was KC-PHYS-01
        'SC-BOND-01': 0.35,        # was KC-CHEM-01
    },
    'content_gaps': {
        'SC-CELL-01': 'Gap_Absence',
        'SC-KINEMATICS-01': 'Gap_Prior',
    },
    'delta_windows': [
        {'language': -0.35, 'processing_speed': -0.25},
        {'language': -0.20, 'processing_speed': -0.15},
    ],
    'prior_windows': [
        {'SC-CELL-01': 0.25, 'SC-KINEMATICS-01': 0.30},
        {'SC-CELL-01': 0.40, 'SC-KINEMATICS-01': 0.48},
    ],
}

#small window so priors visibly move within a short demo session
CALIBRATION_WINDOW = 3

#behavioral diagnosis checkpoint cadence during the demo
CHECKPOINT_EVERY = 5


def build_runtime() -> dict:
    """Construct the graph once per process (single-worker demo)."""
    data = Data_Service()
    kt = knowledge_tracing_engine(calibration_window=CALIBRATION_WINDOW)
    bd = behavioral_diagnosis_engine()
    treatment = Treatment_Service()
    diagnosis = Diagnosis_service(kt, bd, data, treatment)
    #the response-window gate lives on the mediator as well as the engine
    diagnosis.calibration_window = CALIBRATION_WINDOW

    selection = Question_Selection_Service(ContextualBandit())
    match_service = Match_Service(selection, data)  # wires Q_S.mediator

    _ingest_questions(data, diagnosis)
    _scaffold_student(data, diagnosis, treatment)

    log.info('demo runtime ready | questions=%d student=%s',
             len(data.list_questions()), DEMO_STUDENT)
    return {
        'data': data, 'diagnosis': diagnosis, 'treatment': treatment,
        'bd_engine': bd, 'selection': selection, 'match': match_service,
        'answered': [],           # question ids served this session
        'answer_count': 0,        # drives checkpoint cadence
    }


def _ingest_questions(data, mediator) -> int:
    records = file_loader.load_records(QUESTION_CSV)
    if not records:
        raise RuntimeError(f'demo question CSV not found: {QUESTION_CSV}')
    ingestion = data_ingestion_service()
    ingestion.mediator = mediator
    inserted, skipped = ingestion.populate_questions(records)
    #relative features (quartiles/BKT params/science entities) corpus-wide
    reprocessing.reprocess_corpus(data, mediator)
    log.info('demo ingestion | inserted=%d skipped=%d', inserted, skipped)
    return inserted


def _scaffold_student(data, diagnosis, treatment) -> None:
    data.add_student(DEMO_STUDENT)
    student = data.get_student(DEMO_STUDENT)
    for skill_id, prior in PROFILE['priors'].items():
        student.update_prior(skill_id, prior)
    student.update_content_gaps(PROFILE['content_gaps'])
    for d in PROFILE['diagnoses']:
        student.add_diagnosis(d)
    for i, deltas in enumerate(PROFILE['delta_windows']):
        student.add_diagnosis_record(f'2026-08-{15 + i:02d}T10:00:00',
                                     deltas, PROFILE['diagnoses'])
    for i, priors in enumerate(PROFILE['prior_windows']):
        student.add_priors_record(f'2026-08-{15 + i:02d}T10:00:00', priors)
    treatment.set_treatment_plan(DEMO_STUDENT)
