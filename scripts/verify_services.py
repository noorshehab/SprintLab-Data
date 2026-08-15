import os
import random
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'services'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'knowledge_tracing'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'behavioral_diagnosis'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'data_ingestion'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'question_processing'))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.knowledge_tracing.knowledge_tracing_engine import knowledge_tracing_engine
from services.behavioral_diagnosis.behavioral_diagnosis_engine import behavioral_diagnosis_engine
from services.data_ingestion import file_loader
from services.data_ingestion.data_ingestion_service import data_ingestion_service
from services.question_processing.embedding_service import embedding_service
from services.question_processing.question_processing_service import question_processing_service

N_QUESTIONS = 100
N_STUDENTS = 50


def select_subset(metadata_path, responses_path, n_questions=N_QUESTIONS, n_students=N_STUDENTS, seed=42):
    """100 most-solved questions (top by 'attempted'), 50 random uids, only their
    responses to those questions."""
    qmd = pd.read_csv(metadata_path)
    top = qmd.nlargest(n_questions, 'attempted')

    responses = pd.read_csv(responses_path)
    if 'fold' in responses.columns:
        responses = responses.drop(columns=['fold'])

    rng = random.Random(seed)
    uids = list(responses['uid'].unique())
    selected_uids = rng.sample(uids, min(n_students, len(uids)))

    selected_qids = set(top['question_id'].tolist())
    subset = responses[responses['uid'].isin(selected_uids) & responses['questions'].isin(selected_qids)]

    #ensures at least one candidate row per selected uid for stable reproduction
    subset = subset.sort_values('timestamps').reset_index(drop=True)
    print(f'Subset: {len(selected_qids)} questions, {len(selected_uids)} uids, {len(subset)} responses')
    return top, subset, selected_qids, selected_uids


def main():
    metadata_path = os.getenv('QUESTION_METADATA_PATH')
    responses_path = os.getenv('TEST_SET')

    top, subset, selected_qids, selected_uids = select_subset(metadata_path, responses_path)

    #--------------------------------------------------------------------------
    # wire services (mediator) + embedding fallback-friendly embedder
    #--------------------------------------------------------------------------
    data_service = Data_Service()
    kt = knowledge_tracing_engine(calibration_window=10)
    bd = behavioral_diagnosis_engine()
    mediator = Diagnosis_service(kt, bd, data_service)

    embedder = embedding_service(cache_path=os.path.join(
        os.path.dirname(os.getenv('EMBEDDING_CACHE_PATH', 'experiments/embedding_cache.json')),
        'skill_embeddings_cache.json'))
    qproc = question_processing_service(embedder=embedder)
    qproc.mediator = mediator
    ingest = data_ingestion_service()
    ingest.mediator = mediator

    #--------------------------------------------------------------------------
    # question processing: embed skills (subset) + skills map + features/probs
    #--------------------------------------------------------------------------
    questions_json = file_loader.load_records(os.getenv('QUESTIONS_PATH'))

    qmd_100 = top.copy()
    #merge language/task-specific optional fields from the raw source if present
    qmd_100['content'] = [questions_json.get(str(q), {}).get('content', '') for q in qmd_100['question_id']]
    qmd_100['analysis'] = [questions_json.get(str(q), {}).get('analysis', '') for q in qmd_100['question_id']]

    all_skill_texts = file_loader.load_skill_texts(os.getenv('KC_METADATA_PATH'))
    subset_skills = set()
    for raw in top['kc_ids']:
        subset_skills.update(qproc._parse_skill_list(raw))
    subset_skills = {int(s) for s in subset_skills}
    skill_texts = {k: v for k, v in all_skill_texts.items() if int(k) in subset_skills}
    print(f'Subset skills for embedding: {len(skill_texts)}')
    enriched = qproc.process(qmd_100, questions_json=questions_json, skill_texts=skill_texts)

    #--------------------------------------------------------------------------
    # ingestion: students from subset, questions from enriched metadata
    #--------------------------------------------------------------------------
    ingest.populate_students(file_loader.load_records(responses_path), id_field='uid')
    selected_uids_set = set(int(u) for u in selected_uids)
    data_service.students = {k: v for k, v in data_service.students.items() if int(k) in selected_uids_set}

    #register skills (clustered by embedding similarity) + their similar-skill lists
    skill_ids = qproc.register_skills()

    records = enriched.to_dict('records')
    n_added, n_skipped = ingest.populate_questions(records)

    #--------------------------------------------------------------------------
    # assertions
    #--------------------------------------------------------------------------
    print(f'\nData service -> students: {len(data_service.students)} | questions: {len(data_service.questions)}')
    assert len(data_service.students) == min(N_STUDENTS, len(selected_uids)), 'student count mismatch'
    assert n_added == N_QUESTIONS, f'expected {N_QUESTIONS} questions, added {n_added}'

    missing_attrs = []
    for q in data_service.questions.values():
        for attr, dtype in [('language', str), ('num_unknowns', int), ('num_operations', int),
                            ('cognitive_load_index', int)]:
            v = getattr(q, attr, None)
            if v is None or (dtype is int and not isinstance(v, (int, np.integer))):
                missing_attrs.append((q.id, attr, v))

    assert not missing_attrs, f'missing/new-type attributes: {missing_attrs[:10]}'

    p_t_vals = {q.p_t for q in data_service.questions.values()}
    assert p_t_vals == {0.017}, f'p_t should be fixed 0.017 for all, got {p_t_vals}'

    cli = {getattr(q, 'cognitive_load_index', None) for q in data_service.questions.values()}
    assert all(isinstance(x, (int, np.integer)) and 1 <= x <= 5 for x in cli), f'cognitive_load_index out of range: {cli}'

    assert qproc.similar_skills_map, 'similar-skills map is empty'
    n_skills = len(data_service.skills)
    print(f'Skill embeddings: {len(qproc.skill_embeddings)} | '
          f'Skills: {n_skills} | similar-skills map entries: {len(qproc.similar_skills_map)}')
    assert n_skills == len(qproc.similar_skills_map), 'skills not all registered'
    sample_skill = next(iter(data_service.skills.items()))
    print('Sample skill:', sample_skill[0], '-> similar:', sample_skill[1].get_similar()[:5])

    sample_q = next(iter(data_service.questions.values()))
    print('\nSample question attrs:')
    for k, v in sample_q.__dict__.items():
        print(f'  {k}: {v}')

    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()