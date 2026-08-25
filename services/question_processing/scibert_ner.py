"""Science-entity recognition over question text.

Purpose: count distinct science quantities/concepts mentioned in a
question - the `variables_count` feature. Examples of what counts:

    "the car is moving at velocity 6 km/s"      -> velocity   (unit-anchored)
    "the animal has fur and nurses its young"   -> fur, nursing (lexicon)

Recognition layers, tried in order:

1. Fine-tuned NER checkpoint (SCIBERT_NER_MODEL env): a token-classification
   head trained to tag science entities is authoritative when provided.
2. Unit-anchored quantities + concept lexicon (always available, no model):
   any "<quantity-word> <number> <unit>" span ("velocity 6 km/s", "20 g")
   plus matches against a curated cross-domain lexicon (physics/chemistry/
   biology traits). This layer works even with transformers unavailable.
3. SciBERT semantic extension (when the base model loads): content words
   NOT covered by the lexicon are embedded in context and compared against
   lexicon prototypes; close matches join the result under their matched
   concept. Catches synonyms the lexicon missed ("nurses its young" ->
   nursing/reproduction-like traits).

Fallback behaviour: layers 3 degrades gracefully to 2 whenever
transformers/torch are unavailable or USE_SCIBERT_NER=0.
"""
from __future__ import annotations
import os
import re

from services.log_setup import get_logger

log = get_logger('scibert_ner')

DEFAULT_MODEL = 'allenai/scibert_scivocab_uncased'

#--------------------------------------------------------------------------
# lexicon: canonical concept -> surface forms recognised in text
# (extendable; keep forms lowercase)
#--------------------------------------------------------------------------
CONCEPT_LEXICON: dict[str, list[str]] = {
    #physics quantities
    'velocity':     ['velocity', 'speed'],
    'acceleration': ['acceleration', 'deceleration'],
    'distance':     ['distance', 'displacement'],
    'time':         ['time', 'duration', 'period'],
    'mass':         ['mass', 'weight'],
    'force':        ['force', 'thrust', 'friction', 'tension', 'gravity'],
    'temperature':  ['temperature', 'heat'],
    'pressure':     ['pressure'],
    'volume':       ['volume'],
    'density':      ['density'],
    'energy':       ['energy', 'work', 'kinetic energy', 'potential energy'],
    'power':        ['power'],
    'current':      ['current', 'voltage', 'resistance'],
    #chemistry
    'concentration':['concentration', 'molarity'],
    'reaction':     ['reaction', 'reactant', 'product'],
    'substance':    ['acid', 'base', 'salt', 'compound', 'element', 'molecule',
                     'atom', 'ion', 'solution', 'mixture'],
    #biology traits/processes
    'body covering':['fur', 'hair', 'feathers', 'scales', 'skin'],
    'reproduction': ['nurses', 'nursing', 'lays eggs', 'gives birth',
                     'reproduce', 'reproduction', 'pollination'],
    'thermoregulation': ['warm-blooded', 'cold-blooded', 'endothermic',
                         'ectothermic'],
    'locomotion':   ['walks', 'swims', 'flies', 'crawl', 'wings', 'fins',
                     'legs', 'limbs'],
    'nutrition':    ['eats', 'herbivore', 'carnivore', 'omnivore', 'prey',
                     'predator', 'photosynthesis'],
    'habitat':      ['habitat', 'aquatic', 'terrestrial', 'ocean', 'forest',
                     'desert'],
    'anatomy':      ['backbone', 'spine', 'skeleton', 'cell', 'nucleus',
                     'membrane', 'roots', 'leaves', 'stem', 'petals'],
}

#phrases scanned longest-first so multi-word forms win
_PHRASES: list[tuple[str, str]] = sorted(
    ((form, concept)
     for concept, forms in CONCEPT_LEXICON.items() for form in forms),
    key=lambda item: len(item[0]), reverse=True)

#number + unit spans anchor quantities to their physical family
_UNIT_RE = re.compile(r'\d+(?:\.\d+)?\s*([a-zA-Z°/%]+)')

#unit -> canonical concept (family wins over surrounding wording)
UNIT_CONCEPTS: dict[str, str] = {
    'km/s': 'velocity', 'm/s': 'velocity', 'km/h': 'velocity',
    'km': 'distance', 'm': 'distance', 'cm': 'distance', 'mm': 'distance',
    'kg': 'mass', 'mg': 'mass', 'g': 'mass', 'lbs': 'mass',
    '°c': 'temperature', '°f': 'temperature', 'k': 'temperature',
    'min': 'time', 'ms': 'time', 'h': 'time', 'hr': 'time',
    'hour': 'time', 'hours': 'time', 'second': 'time', 'seconds': 'time',
    'sec': 'time', 'secs': 'time', 'minute': 'time', 'minutes': 'time',
    'day': 'time', 'days': 'time',
    '%': 'concentration', 'mol': 'concentration', 'ml': 'volume', 'l': 'volume',
    'n': 'force', 'j': 'energy', 'kj': 'energy',
    'w': 'power', 'kw': 'power',
    'atm': 'pressure', 'pa': 'pressure', 'kpa': 'pressure',
}

_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
    'its', 'his', 'her', 'their', 'and', 'or', 'of', 'to', 'in', 'on',
    'at', 'for', 'with', 'from', 'by', 'this', 'that', 'which', 'how',
    'what', 'when', 'where', 'why', 'does', 'do', 'did', 'can', 'will',
    'not', 'it', 'they', 'them', 'then', 'than', 'into', 'about',
}


def _lexicon_spans(text: str) -> dict[str, str]:
    """{canonical concept: first surface form found}."""
    lowered = text.lower()
    found: dict[str, str] = {}
    for form, concept in _PHRASES:
        if concept in found:
            continue
        if re.search(r'\b' + re.escape(form) + r'\b', lowered):
            found[concept] = form
    return found


def _unit_anchored_quantities(text: str) -> dict[str, str]:
    """Quantities implied by '<something> <number><unit>' spans.

    The unit's physical family names the concept ('300 m' -> distance);
    a lexicon word directly before the number refines it when present
    ('kinetic energy of 40 J' stays 'energy').
    """
    found: dict[str, str] = {}
    lowered = text.lower()
    covered_forms = {form: concept
                     for concept, forms in CONCEPT_LEXICON.items()
                     for form in forms}
    for match in _UNIT_RE.finditer(lowered):
        unit = match.group(1).strip().rstrip('.')
        concept = UNIT_CONCEPTS.get(unit)
        if concept is None:
            continue  # bare number or unrecognised unit

        #allow an immediately preceding lexicon word to refine the concept
        prefix_words = [w for w in re.findall(r'[a-z]+', lowered[:match.start()])
                        if w not in _STOPWORDS]
        for word in reversed(prefix_words[-3:]):
            if word in covered_forms:
                concept = covered_forms[word]
                break
        found.setdefault(concept, concept)
    return found


class ScienceEntityNER:
    """Recognise science entities in question text (see module docstring)."""

    def __init__(self, ner_model: str | None = None,
                 base_model: str | None = None,
                 sim_threshold: float | None = None):
        self.ner_model_name = ner_model or os.getenv('SCIBERT_NER_MODEL')
        self.base_model_name = base_model or os.getenv('SCIBERT_MODEL', DEFAULT_MODEL)
        self.sim_threshold = float(sim_threshold or os.getenv('SCIBERT_SIM_THRESHOLD', '0.55'))
        self.enabled = os.getenv('USE_SCIBERT_NER', '1') not in ('0', 'false', 'False')
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._prototypes: dict[str, list] | None = None
        self._loaded = False

    #------------------------------------------------------------------
    # public API
    #------------------------------------------------------------------
    def extract(self, text: str) -> list[str]:
        """Distinct canonical science concepts found in the text."""
        if not isinstance(text, str) or not text.strip():
            return []

        if self._load():
            if self._pipeline is not None:
                return self._extract_with_ner(text)
            return self._extract_semantic(text)
        #no model: lexicon + unit anchoring only
        return sorted({**_unit_anchored_quantities(text),
                       **_lexicon_spans(text)}.keys())

    def count(self, text: str) -> int:
        return len(self.extract(text))

    #------------------------------------------------------------------
    # model lifecycle (lazy)
    #------------------------------------------------------------------
    def _load(self) -> bool:
        """Load models once; True when a semantic layer is available."""
        if self._loaded:
            return self._pipeline is not None or self._model is not None
        self._loaded = True
        if not self.enabled:
            log.info('USE_SCIBERT_NER=0 -> lexicon-only science entity '
                     'extraction')
            return False
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer, pipeline
        except ImportError:
            log.warning('transformers/torch unavailable -> lexicon-only '
                        'science entity extraction')
            return False

        try:
            if self.ner_model_name:
                self._pipeline = pipeline(
                    'token-classification', model=self.ner_model_name,
                    aggregation_strategy='simple')
                log.info('loaded fine-tuned NER checkpoint %s',
                         self.ner_model_name)
                return True

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self._model = AutoModel.from_pretrained(self.base_model_name)
            self._model.eval()
            log.info('loaded %s for science-entity semantic matching',
                     self.base_model_name)
            return True
        except Exception as exc:
            log.warning('scibert load failed (%s) -> lexicon-only science '
                        'entity extraction', exc)
            self._pipeline = None
            self._model = None
            return False

    #------------------------------------------------------------------
    # layer 1: fine-tuned token classification
    #------------------------------------------------------------------
    def _extract_with_ner(self, text: str) -> list[str]:
        try:
            entities = self._pipeline(text)
        except Exception as exc:
            log.warning('NER inference failed (%s); lexicon fallback', exc)
            return sorted({**_unit_anchored_quantities(text),
                           **_lexicon_spans(text)}.keys())
        return sorted({e['word'].strip().lower() for e in entities
                       if isinstance(e, dict) and e.get('word')})

    #------------------------------------------------------------------
    # layer 3: semantic extension over the lexicon
    #------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> dict[str, list[float]]:
        import torch
        encoded = self._tokenizer(texts, padding=True, truncation=True,
                                  max_length=128, return_tensors='pt')
        with torch.no_grad():
            output = self._model(**encoded)
        mask = encoded['attention_mask'].unsqueeze(-1).float()
        pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return {text: pooled[i].tolist() for i, text in enumerate(texts)}

    def _concept_prototype(self, concept: str) -> list[float]:
        """Embedding of the canonical concept phrasing (cached)."""
        prototypes = self._prototypes if self._prototypes is not None else {}
        if concept not in prototypes:
            vectors = self._embed([f'the {concept} of the subject'])
            prototypes[concept] = next(iter(vectors.values()))
        return prototypes[concept]

    @staticmethod
    def _candidate_words(text: str) -> list[str]:
        return [w for w in re.findall(r'[a-z][a-z-]+', text.lower())
                if len(w) > 2 and w not in _STOPWORDS]

    def _extract_semantic(self, text: str) -> list[str]:
        import numpy as np

        found = {**_unit_anchored_quantities(text), **_lexicon_spans(text)}
        covered_forms = {form for forms in CONCEPT_LEXICON.values()
                         for form in forms}
        remaining = [w for w in self._candidate_words(text)
                     if w not in covered_forms]

        if remaining:
            window_chars = 60
            contexts = []
            for word in remaining:
                pos = text.lower().find(word)
                start = max(0, pos - window_chars)
                end = min(len(text), pos + len(word) + window_chars)
                contexts.append(text[start:end])
            embedded = self._embed(contexts)

            for word, context in zip(remaining, contexts):
                vec = np.asarray(embedded[context])
                best_concept, best_score = None, self.sim_threshold
                for concept in CONCEPT_LEXICON:
                    proto = np.asarray(self._concept_prototype(concept))
                    score = float(np.dot(vec, proto))
                    if score >= best_score:
                        best_concept, best_score = concept, score
                if best_concept:
                    found.setdefault(best_concept, word)

        return sorted(found.keys())


#backwards-compatible alias (module was born as variable counting)
SciBERTVariableNER = ScienceEntityNER
