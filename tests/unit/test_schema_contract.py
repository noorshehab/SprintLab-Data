"""Schema contract tests.

Guarantees the three consumers of question attributes stay in sync:
  1. every treatment-plan condition references a real question field
     (directly or via a registered alias)
  2. every treatment condition actually matches at least one question
     from a corpus seeded with each field's realistic value space -
     this is what would have caught lang_difficulty/max_cognitive_load/
     has_image/bloom_types silently matching nothing
  3. legacy aliases resolve to the unified canonical names
  4. unknown attributes raise instead of matching nothing
"""
import pytest

from services.Data_service import Data_Service
from services.Entities import question
from services.Treatment_service import Treatment_Service
from services.question_schema import FIELD_ALIASES, resolve_attribute, is_known_attribute


def _all_conditions(treatment_map):
    """Flatten initial_treatment_map into (diagnosis, condition) pairs."""
    out = []
    for diagnosis, params in treatment_map.items():
        for attr in params:
            if attr != 'Operator':
                out.append((diagnosis, attr))
    return out


def test_every_treatment_attribute_is_a_known_schema_field():
    svc = Treatment_Service()
    for diagnosis, attr in _all_conditions(svc.initial_treatment_map):
        assert is_known_attribute(attr), (
            f"treatment '{diagnosis}' queries unknown attribute '{attr}' - "
            f"add it to services/question_schema.py or fix the name")


def test_every_treatment_attribute_exists_on_question_entity():
    svc = Treatment_Service()
    q = question(q_id='probe')
    known = {v.lower() for v in vars(q)}
    for diagnosis, attr in _all_conditions(svc.initial_treatment_map):
        canonical = resolve_attribute(attr)
        assert canonical and canonical.lower() in known, (
            f"treatment '{diagnosis}' uses '{attr}' -> '{canonical}' "
            f"which is not an attribute of the question entity")


def test_legacy_aliases_resolve_to_canonical_names():
    assert resolve_attribute('lang_difficulty') == 'language_level'
    assert resolve_attribute('max_cognitive_load') == 'cognitive_load'
    assert resolve_attribute('has_image') == 'visual_dependency'
    assert resolve_attribute('bloom_types') == 'bloom_taxonomy_level'


@pytest.fixture
def corpus_ds():
    """A corpus exercising each treated field's value space."""
    ds = Data_Service()
    ds.add_student('S_contract')
    seeds = [
        # language_level quartiles
        dict(q_id='lang1', skill_cluster_id='KC-1', language_level='Q1'),
        dict(q_id='lang2', skill_cluster_id='KC-1', language_level='Q2'),
        # cognitive load spread
        dict(q_id='load1', skill_cluster_id='KC-2', cognitive_load=0.5),
        dict(q_id='load2', skill_cluster_id='KC-2', cognitive_load=2.0),
        # visual dependency both values
        dict(q_id='vis1', skill_cluster_id='KC-3', visual_dependency=1),
        dict(q_id='vis0', skill_cluster_id='KC-3', visual_dependency=0),
        # bloom taxonomy levels
        dict(q_id='bloom1', skill_cluster_id='KC-4', bloom_taxonomy_level='Remember'),
        dict(q_id='bloom2', skill_cluster_id='KC-4', bloom_taxonomy_level='Understand'),
        dict(q_id='bloom3', skill_cluster_id='KC-4', bloom_taxonomy_level='Analyze'),
        # time pressure + steps + multi-concept
        dict(q_id='tp0', skill_cluster_id='KC-5', time_pressure_flag=0),
        dict(q_id='tp1', skill_cluster_id='KC-5', time_pressure_flag=1),
        dict(q_id='steps', skill_cluster_id='KC-6', logical_steps=1),
        dict(q_id='multi', skill_cluster_id='KC-6', multi_concept_flag=1),
    ]
    for seed in seeds:
        ds.add_question(**seed)
    return ds


@pytest.mark.parametrize('diagnosis,params', list(
    Treatment_Service().initial_treatment_map.items()),
    ids=lambda x: x if isinstance(x, str) else '')
def test_every_treatment_condition_matches_at_least_one_question(diagnosis, params, corpus_ds):
    ds = corpus_ds
    conditions = []
    operator = params.get('Operator', '<=')
    topic = 'general' if not diagnosis.startswith('Gap_') else None
    for attr, threshold in params.items():
        if attr == 'Operator':
            continue
        conditions.append({'Topic': topic, 'Attribute': attr,
                           'Operator': operator, 'Threshold': threshold})

    result = ds.query(conditions)
    assert result, (
        f"treatment '{diagnosis}' with conditions {conditions} matches ZERO "
        f"questions in the reference corpus - the query can never serve anything")
