import re
import spacy
from spacy.matcher import PhraseMatcher
try:
    import curriculum_vocab
except ImportError:
    from . import curriculum_vocab
from typing import Dict
nlp = spacy.load("en_core_web_sm")



#Arabic unicode block ranges
ARABIC_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
#any CJK char (placeholder detection for the current Chinese corpus)
CJK_RE = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]')
#rough "word" split for english
EN_WORD_RE = re.compile(r"[A-Za-z0-9']+")
#approximate arabic word split on whitespace/punct
AR_WORD_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9]+')




def detect_language(text, default='en'):
    """Return 'ar' if the text contains Arabic script, else 'en'."""
    if text is None:
        return default
    if ARABIC_RE.search(str(text)):
        return 'ar'
    return default


def tokenize(text, language='en'):
    """Language-aware tokenization used for vocabulary richness."""
    if not text:
        return []
    text = str(text).lower()
    if language == 'ar':
        return AR_WORD_RE.findall(text)
    #english (and non-arabic placeholders like CJK) -> match letter/digit runs
    return EN_WORD_RE.findall(text)


def vocabulary_richness(text, language='en'):
    """Type-token ratio of the question text (0 if empty)."""
    tokens = tokenize(text, language)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


#word lists for the language-challenge flag (configurable, extendable)
NEGATION_WORDS = {
    'en': {'not', 'no', 'never', 'neither', "don't", "doesn't", "didn't", "can't", "cannot",
           "won't", "isn't", "aren't", "wasn't", "weren't", 'without', "ain't", "n't"},
    'ar': {'لا', 'ليس', 'ليست', 'لم', 'لن', 'غير', 'عدا', 'ماعدا', 'بلا'},
}
EXCEPTION_WORDS = {
    'en': {'except', 'unless', 'but', 'however', 'rather than', 'other than', 'otherwise',
           'apart from', 'excluding', 'exclude', 'if and only if'},
    'ar': {'ما عدا', 'إلا', 'إذا', 'لكن', 'باستثناء', 'غير ذلك', 'بخلاف'},
}
INDIRECT_WORDS = {
    'en': {'approximately', 'about', 'around', 'roughly', 'nearly', 'almost', 'at most',
           'at least', 'at least', 'more than', 'less than', 'up to', 'no more than'},
    'ar': {'تقريبا', 'حوالي', 'نحو', 'على الأقل', 'على الأكثر', 'لا يقل', 'لا يزيد', 'أكثر من', 'أقل من'},
}


def get_tree_depth(token):
    if not list(token.children):
        return 1
    return 1 + max(get_tree_depth(child) for child in token.children)

def evaluate_sentence_complexity(sent_text, threshold=60.0):
    doc = nlp(sent_text)
    sent = next(doc.sents)
    
    # 1. Identify Root & Depth
    root = [tok for tok in sent if tok.head == tok][0]
    depth = get_tree_depth(root)
    
    # 2. Mean Dependency Distance (MDD)
    distances = [abs(tok.i - tok.head.i) for tok in sent if tok != root]
    mdd = (sum(distances) / len(distances)) if distances else 1.0
    
    # 3. Subordinate clauses & 4. Length
    sub_clauses = sum(1 for tok in sent if tok.dep_ in ("advcl", "relcl", "acl", "ccomp"))
    length = len([tok for tok in sent if not tok.is_punct])

    # Normalized component scores (capped at 100)
    # Typical simple: depth <= 3, mdd <= 2.2, sub_clauses = 0, length <= 12
    # Typical complex: depth >= 6, mdd >= 4.0, sub_clauses >= 2, length >= 28
    score_depth = min(100, max(0, (depth - 2) / (6 - 2) * 100))
    score_mdd = min(100, max(0, (mdd - 1.5) / (4.0 - 1.5) * 100))
    score_sub = min(100, (sub_clauses / 2.0) * 100)
    score_len = min(100, max(0, (length - 10) / (30 - 10) * 100))
    
    final_score = (
        0.35 * score_depth +
        0.25 * score_mdd +
        0.25 * score_sub +
        0.15 * score_len
    )
    
    is_complex = final_score >= threshold
    
    return is_complex

class CurriculumDifficultyScorer:

    def __init__(self, taxonomy: Dict[str, float]):
        self.taxonomy = taxonomy
        # Sort terms descending by number of tokens so 3-grams match before 2-grams, then 1-grams
        self.sorted_terms = sorted(
            taxonomy.keys(), key=lambda t: len(t.split()), reverse=True
        )

    def score_text(self, raw_text: str) -> Dict[str, object]:
        """Calculates difficulty score, term counts, and matched entities."""
        # Step A: Preprocess document to sequence of lemmas
        doc = nlp(raw_text.lower())
        token_lemmas = [
            token.lemma_
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        total_tokens = len(token_lemmas)

        if total_tokens == 0:
            return {
                "score": 0.0,
                "matches": [],
                "term_count": 0,
                "density": 0.0,
            }

        # Step B: Greedy span-marking to prevent nested/sub-token double counting
        occupied_indices = set()
        matched_records = []
        total_weighted_score = 0.0

        for term in self.sorted_terms:
            term_tokens = term.split()
            term_len = len(term_tokens)

            # Slide over document looking for consecutive matches
            for i in range(len(token_lemmas) - term_len + 1):
                # Check if current span overlaps with an already-matched longer term
                span_indices = set(range(i, i + term_len))
                if span_indices & occupied_indices:
                    continue

                # Check if tokens match
                if token_lemmas[i : i + term_len] == term_tokens:
                    weight = self.taxonomy[term]
                    total_weighted_score += weight
                    matched_records.append(
                        {
                            "term": term,
                            "token_span": (i, i + term_len),
                            "weight": weight,
                        }
                    )
                    occupied_indices.update(span_indices)

        # Step C: Metrics calculation
        raw_term_count = len(matched_records)
        # Vocabulary density: proportion of document made up of curriculum terms
        matched_tokens_count = len(occupied_indices)
        density = matched_tokens_count / total_tokens

        # Final Normalized Difficulty Score (density scaled by average weight)
        # Score increases both when domain vocabulary is dense and when terms are advanced
        normalized_difficulty = (
            (total_weighted_score / total_tokens) * 100
            if total_tokens > 0
            else 0.0
        )
      
        return normalized_difficulty

def _flatten_vocab(vocab_dict):
    flat = {}
    for topic_data in vocab_dict.values():
        if isinstance(topic_data, dict):
            flat.update(topic_data)
    return flat


def language_challenge_flag(text, language, grade_band, subject, curriculum):
    """Flag questions whose language is challenging: negation, exceptions,
    uncommon vocabulary or indirect/hedged phrasing.

    Returns a dictionary with challenge flag, type, and curriculum score.
    """
    if not text:
        return {'language_challenge': False, 'language_challenge_type': None, 'curriculum_vocab_score': 0.0}
    
    text_lower = str(text).lower()
    curr_vocab_score = 0.0
    taxonomy = {}
    if subject == 'chemistry':
        taxonomy = _flatten_vocab(curriculum_vocab.BR_789_SCIENCE_VOCAB.get('chemistry', {}))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_11_CHEMISTRY_VOCABULARY))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_12_CHEMISTRY_VOCABULARY))
    elif subject == 'physics':
        taxonomy = _flatten_vocab(curriculum_vocab.BR_789_SCIENCE_VOCAB.get('physics', {}))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_11_PHYSICS_VOCABULARY))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_12_PHYSICS_VOCABULARY))
    elif subject == 'biology':
        taxonomy = _flatten_vocab(curriculum_vocab.BR_789_SCIENCE_VOCAB.get('biology', {}))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_11_BIOLOGY_VOCABULARY))
        taxonomy.update(_flatten_vocab(curriculum_vocab.BR_12_BIOLOGY_VOCABULARY))
    else:
        taxonomy = _flatten_vocab(curriculum_vocab.BR_789_SCIENCE_VOCAB)
    
    neg = any(w in text_lower for w in NEGATION_WORDS.get(language, set()))
    exc = any(w in text_lower for w in EXCEPTION_WORDS.get(language, set()))
    indirect = any(w in text_lower for w in INDIRECT_WORDS.get(language, set()))

    challenge_type = []
    if neg:
        challenge_type.append('Negation')
    if exc:
        challenge_type.append('Exception')
    if indirect:
        challenge_type.append('Indirect_Phrasing')
    if evaluate_sentence_complexity(text_lower):
        challenge_type.append('Grammatical_Complexity')

    # Curriculum Difficulty Scoring
    if taxonomy:
        scorer = CurriculumDifficultyScorer(taxonomy)
        curr_vocab_score = scorer.score_text(text_lower)
        if curr_vocab_score > 20:
            challenge_type.append('Uncommon_Vocabulary')

    res = {
        'language_challenge': len(challenge_type) > 0,
        'language_challenge_type': 'Multiple' if len(challenge_type) > 1 else (challenge_type[0] if challenge_type else None),
        'curriculum_vocab_score': curr_vocab_score
    }
    return res

