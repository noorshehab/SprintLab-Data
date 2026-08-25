"""Corpus reprocessing: after batch question inserts, relative measures
(quartiles, BKT probabilities) must be recomputed across ALL questions,
and the skill embeddings / similar-skills map rebuilt.
"""
from __future__ import annotations

from services.log_setup import get_logger
from services.question_processing.question_processing_service import (
    question_processing_service,
)

log = get_logger('reprocessing')


def _corpus_frame(data_service):
    """Rebuild the source-style metadata frame from stored questions."""
    rows = []
    for q in data_service.list_questions():
        rows.append({
            'question_id': q.id,
            'Question_Text': q.question_text or '',
            'Correct_Answer_Content': q.correct_answer_content or '',
            'Question_Language': {'en': 'English', 'ar': 'Arabic'}.get(q.language, 'English'),
            'Skill_Cluster_ID': q.skill_cluster_id,
            'Difficulty_Level': q.difficulty_level,
            'Logical_Steps': q.logical_steps,
            'Time_Pressure_Flag': q.time_pressure_flag,
            'Cognitive_Load_Index': q.cognitive_load_index,
        })
    return rows


def _write_back(data_service, enriched) -> int:
    """Push derived fields onto every question entity via the repository."""
    by_id = {str(row['question_id']): row for _, row in enriched.iterrows()}
    written = 0
    for q in data_service.list_questions():
        row = by_id.get(str(q.id))
        if row is None:
            continue
        fields = {
            'language_level': row.get('language_level'),
            'reasoning_level': row.get('reasoning_level'),
            'cognitive_load_index': int(row.get('cognitive_load_index') or 0),
            'cognitive_load': float(row.get('cognitive_load') or 0.0),
            'num_unknowns': int(row.get('num_unknowns') or 0),
            'num_operations': int(row.get('num_operations') or 0),
            'p_t': float(row.get('p_t') or 0.0),
            'p_s': float(row.get('p_s') or 0.0),
            'p_g': float(row.get('p_g') or 0.0),
        }
        data_service.update_question_attributes(q.id, fields)
        written += 1
    return written


def _skill_ids(data_service) -> list[str]:
    ids = set()
    for q in data_service.list_questions():
        if q.skill_cluster_id:
            ids.add(str(q.skill_cluster_id))
        ids.update(str(s) for s in (q.skill_ids or []))
    return sorted(ids)


def reprocess_corpus(data_service, mediator) -> dict:
    """Recompute derived fields corpus-wide and rebuild the skills map.

    mediator: any Mediator whose request() handles add_skill
    (Diagnosis_service). Returns a summary dict for logging/response.
    """
    qp = question_processing_service()
    rows = _corpus_frame(data_service)
    if not rows:
        return {'questionsReprocessed': 0, 'skillsRegistered': 0}

    import pandas as pd
    df = pd.DataFrame(rows)

    #quartiles + probabilities across the whole corpus
    skill_map = {str(q.id): [str(s) for s in (q.skill_ids or [])]
                 for q in data_service.list_questions()}
    enriched = qp.derive_question_attributes(df, question_skills=skill_map)
    written = _write_back(data_service, enriched)

    #skills: embeddings are cached on disk; only new skill texts get embedded
    skill_ids = _skill_ids(data_service)
    if skill_ids:
        qp.mediator = mediator
        qp.embed_skills(skill_texts={sid: sid.replace('-', ' ') for sid in skill_ids})
        qp.build_similar_skills_map()
        qp.register_skills()

    log.info("corpus reprocessed | questions=%d skills=%d",
             written, len(skill_ids))
    return {'questionsReprocessed': written, 'skillsRegistered': len(skill_ids)}
