"""Science-entity recognition tests.

The real SciBERT model is never downloaded in the suite (conftest sets
USE_SCIBERT_NER=0); the semantic layer is exercised with stubs.
"""
import pandas as pd
import pytest

from services.question_processing.feature_derivation import (
    derive_features, derive_variables_count_from_text, get_variable_ner)
from services.question_processing.scibert_ner import (
    CONCEPT_LEXICON, ScienceEntityNER,
    _lexicon_spans, _unit_anchored_quantities)


#--- lexicon + unit-anchored layers (model-free) -------------------------

def test_lexicon_catches_biology_traits():
    spans = _lexicon_spans('the animal has fur and nurses its young')
    assert 'body covering' in spans      # fur
    assert 'reproduction' in spans       # nurses


def test_unit_anchored_quantity():
    spans = _unit_anchored_quantities('the car is moving at velocity 6 km/s')
    assert 'velocity' in spans


def test_unit_family_names_the_concept_for_unknown_words():
    # '300 m' is distance regardless of the surrounding sentence
    assert 'distance' in _unit_anchored_quantities('the car travels 300 m')
    assert 'time' in _unit_anchored_quantities('it took 45 seconds')
    assert 'mass' in _unit_anchored_quantities('the sample weighs 20 g')


def test_lexicon_word_refines_unit_concept():
    # generic unit (J) refined by the lexicon phrase right before it
    spans = _unit_anchored_quantities('kinetic energy of 40 J')
    assert 'energy' in spans


def test_extract_without_model_is_lexicon_plus_units():
    ner = ScienceEntityNER()  # USE_SCIBERT_NER=0 via conftest
    result = ner.extract(
        'a car accelerates from rest until its velocity reaches 30 m/s '
        'while an animal with fur watches')
    assert {'velocity', 'body covering'} <= set(result)
    assert ner._pipeline is None and ner._model is None


def test_count_and_empty_inputs():
    ner = ScienceEntityNER()
    assert ner.count('the animal has fur') >= 1
    assert ner.extract('') == []
    assert ner.extract(None) == []


def test_lexicon_is_extensible_registry():
    assert 'fur' in CONCEPT_LEXICON['body covering']
    assert 'velocity' in CONCEPT_LEXICON['velocity']


#--- semantic extension (stubbed embeddings) ------------------------------

class StubSemanticNER(ScienceEntityNER):
    """Model load bypassed; embedding similarity fully controlled."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        self._loaded = True

    def _embed(self, texts):
        out = {}
        for text in texts:
            #prototype phrasings look like 'the <concept> of the subject'
            if text.startswith('the ') and text.endswith('of the subject'):
                concept = text[4:-len(' of the subject')]
                out[text] = {'nutrition': [1.0, 0.0],
                             'habitat': [0.0, 1.0]}.get(concept, [0.5, 0.5])
            elif 'chlorophyll' in text.lower():
                out[text] = [0.99, 0.05]   # near nutrition prototype
            else:
                out[text] = [0.2, 0.2]     # matches nothing
        return out


def test_semantic_layer_maps_uncovered_word_to_concept():
    ner = StubSemanticNER()
    ner._prototypes = {}
    # 'chlorophyll' is not in the lexicon; its context embedding lands
    # near the nutrition prototype, so it joins under that concept
    result = ner._extract_semantic('leaves use chlorophyll for energy')
    assert 'nutrition' in result


def test_semantic_layer_respects_threshold():
    ner = StubSemanticNER()
    ner.sim_threshold = 0.99  # nothing can match ([0.2,0.2] scores ~0.28)
    result = ner._extract_semantic('widgets everywhere')
    assert result == []


#--- fine-tuned NER checkpoint path ---------------------------------------

def test_fine_tuned_pipeline_used_when_configured():
    ner = ScienceEntityNER(ner_model='fake/science-ner')

    class FakePipeline:
        def __call__(self, text):
            return [{'word': 'Fur', 'entity_group': 'TRAIT'},
                    {'word': 'nurses', 'entity_group': 'TRAIT'}]

    ner._pipeline = FakePipeline()
    ner._loaded = True

    assert ner.extract('the animal has fur and nurses its young') == \
        ['fur', 'nurses']


def test_ner_failure_falls_back_to_lexicon():
    ner = ScienceEntityNER(ner_model='fake/science-ner')

    class ExplodingPipeline:
        def __call__(self, text):
            raise RuntimeError('gpu on fire')

    ner._pipeline = ExplodingPipeline()
    ner._loaded = True

    result = ner.extract('the animal has fur')
    assert 'body covering' in result


#--- feature_derivation integration ---------------------------------------

def test_derive_features_uses_ner_when_source_lacks_num_variables(monkeypatch):
    monkeypatch.setattr(ScienceEntityNER, 'count', lambda self, text: 4)
    df = pd.DataFrame({
        'question_id': ['q1'],
        'Question_Text': ['some science question'],
    })
    enriched = derive_features(df)
    assert enriched['num_variables'].tolist() == [4]


def test_source_provided_num_variables_wins(monkeypatch):
    monkeypatch.setattr(ScienceEntityNER, 'count',
                        lambda self, text: pytest.fail(
                            'must not run NER when source provides the column'))
    df = pd.DataFrame({
        'question_id': ['q1'],
        'Question_Text': ['some question'],
        'Variables_Count': [7],
    })
    enriched = derive_features(df)
    assert enriched['num_variables'].tolist() == [7]


def test_shared_instance_is_lazy_and_cached():
    assert get_variable_ner() is get_variable_ner()


def test_derive_helper_handles_empty_texts():
    counts = derive_variables_count_from_text([
        'the animal has fur', '', None])
    assert isinstance(counts, list) and len(counts) == 3
