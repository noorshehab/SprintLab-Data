"""Canonical question-field schema.

Single source of truth for question attribute names used across querying,
treatment plans and bandit contexts. Legacy/alternate spellings are listed
as aliases and resolved transparently by Data_service.query - new code
should always use the canonical name.

To add a queryable question field:
    1. add the attribute to services.Entities.question
    2. add an entry here (with any legacy aliases)
Everything else (query resolution, validation, contracts) follows.
"""
from __future__ import annotations

# canonical field -> known aliases (lowercase compare)
FIELD_ALIASES: dict = {
    'q_id':                 ['question_id', 'id'],
    'skill_cluster_id':     ['cluster_id', 'kc_cluster'],
    'skill_ids':            ['skills', 'kc_ids'],
    'unit':                 ['unit_id', 'supertopic_id'],
    'lesson_id':            [],
    'grade':                [],
    'subject':              [],
    'question_type':        [],
    'question_text':        [],
    'language':             ['question_language'],
    'difficulty_level':     ['difficulty'],
    #derived / treated fields (unified canonical names)
    'language_level':       ['lang_difficulty', 'language_quartile', 'lang_level'],
    'cognitive_load':       ['max_cognitive_load', 'wm_score', 'working_memory_load'],
    'cognitive_load_index': ['load_index'],
    'visual_dependency':    ['has_image', 'image_dependency'],
    'bloom_taxonomy_level': ['bloom_types', 'bloom_level'],
    'time_pressure_flag':   ['time_pressure'],
    'time_allowed':         ['time'],
    'logical_steps':        ['steps', 'max_steps'],
    'variables_count':      ['num_variables', 'variables'],
    'num_unknowns':         [],
    'num_operations':       [],
    'reasoning_level':      ['reasoning_quartile'],
    'multi_concept_flag':   [],
    'real_world_context':   [],
    'trick_question':       [],
    'p_t':                  [],
    'p_s':                  [],
    'p_g':                  ['learning'],
}


def _norm(name):
    return str(name).strip().lower() if name is not None else ''


# reverse lookup: alias -> canonical (built once)
_ALIAS_TO_CANONICAL: dict = {}
for _canonical, _aliases in FIELD_ALIASES.items():
    _ALIAS_TO_CANONICAL[_norm(_canonical)] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_norm(_alias)] = _canonical


def resolve_attribute(name):
    """Map a raw attribute name (or legacy alias) to its canonical field.

    Returns None when the name is not part of the schema at all.
    """
    key = _norm(name)
    return _ALIAS_TO_CANONICAL.get(key)


def is_known_attribute(name) -> bool:
    return resolve_attribute(name) is not None
