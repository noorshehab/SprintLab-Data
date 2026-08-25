import re
import numpy as np
import pandas as pd
from scipy import stats
from services.question_processing import language_tools
from services.question_processing.scibert_ner import ScienceEntityNER

#shared instance - model loads lazily on first extraction, never at import
_science_ner: ScienceEntityNER | None = None


def get_variable_ner() -> ScienceEntityNER:
    global _science_ner
    if _science_ner is None:
        _science_ner = ScienceEntityNER()
    return _science_ner


def derive_variables_count_from_text(question_texts):
    """Number of distinct science entities/quantities mentioned in the
    question text (unit-anchored quantities + concept lexicon, extended
    by SciBERT semantic matching when the model is available).
    """
    ner = get_variable_ner()
    return [ner.count(text) for text in question_texts]

#math symbols used to detect distinct operations in a solution
MATH_SYMBOLS = set('+-×÷=≠<>±')

#source column names -> canonical internal names (question-processing schema)
COLUMN_ALIASES = {
    'Question_ID': 'question_id',
    'Question_Text': 'question_text',
    'Question_Language': 'question_language',
    'Variables_Count': 'num_variables',
    'Logical_Steps': 'num_steps',
    'Language_Challenging': 'language_challenging',
    'Language_Challenge_Type': 'language_challenge_type',
    'Time_Pressure_Flag': 'time_pressure_flag',
    'Cognitive_Load_Index': 'cognitive_load_index',
}


def _minmax(series):
    """Min-max normalize to [0, 1]."""
    s = pd.to_numeric(pd.Series(series), errors='coerce')
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return s.fillna(0.0)
    return (s - mn) / (mx - mn)


def _z(series):
    s = pd.to_numeric(pd.Series(series), errors='coerce')
    if s.std(skipna=True) == 0 or pd.isna(s.std(skipna=True)):
        return s.fillna(0.0)
    return pd.Series(stats.zscore(s.fillna(s.mean())), index=s.index)


def derive_num_unknowns(solution_texts):
    """Number of unknowns derived from the solution; minimum is one.

    Heuristic: count distinct single-letter variables mentioned in the solution,
    floored at 1. Overridable when the source data provides its own column.
    """
    out = []
    for text in solution_texts:
        if not isinstance(text, str) or not text.strip():
            out.append(1)
            continue
        vars_found = set(re.findall(r'\b([a-z])\b', text.lower()))
        #arabic/cjk don't use latin letters; still floor at 1
        out.append(max(1, len(vars_found)))
    return out


def derive_num_operations(solution_texts):
    """Number of different operations.

    Combines two signals:
    - distinct math symbols used in the solution text
    - distinct steps: consecutive solution sentences with no shared vocabulary
      count as separate steps, so step count = (# of vocab-disjoint boundaries)+1
    """
    out = []
    for text in solution_texts:
        if not isinstance(text, str) or not text.strip():
            out.append(0)
            continue
        symbols = set(ch for ch in text if ch in MATH_SYMBOLS)
        sentences = re.split(r'[。.!?；;\n]', text)
        sentences = [s for s in sentences if s.strip()]
        steps = 1
        for i in range(1, len(sentences)):
            prev = set(re.findall(r'\w+', sentences[i - 1].lower()))
            cur = set(re.findall(r'\w+', sentences[i].lower()))
            if prev and not (prev & cur):
                steps += 1
        out.append(max(len(symbols), steps))
    return out


def derive_cognitive_load_index(variables_count, num_unknowns):
    """Cognitive load index, integer 1-5.

    zscore(variables_count) + zscore(number_of_unknowns), min-max scaled to 1-5.
    """
    score = _z(variables_count) + _z(num_unknowns)
    scaled = _minmax(score) * 4 + 1  # [1, 5]
    return np.rint(scaled).astype(int)


def _safe_quartile(series, q=4):
    """Quartile labels, falling back to all-Q1 when the score is constant."""
    s = pd.to_numeric(series, errors='coerce')
    if s.nunique(dropna=True) < 2:
        return pd.Series('Q1', index=series.index)
    try:
        return pd.qcut(s, q=q, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        return pd.Series('Q1', index=series.index)


def derive_language_level(question_length, vocabulary_richness, num_clauses, num_sentences):
    """language_level graded Q1-Q4 from the same composite z-score formula."""
    score = _z(question_length) + _z(vocabulary_richness) + _z(num_clauses) + _z(num_sentences)
    scaled = _minmax(score)
    return _safe_quartile(scaled)


def derive_reasoning_level(num_steps, solution_vocab):
    """Reasoning difficulty labeled from number of solution steps + solution vocab richness."""
    score = _z(num_steps) + _z(solution_vocab)
    scaled = _minmax(score)
    return _safe_quartile(scaled)


def derive_language_challenge(texts, languages):
    """Flag questions with negation/exceptions/indirect phrasing via the word lists."""
    out = []
    for text, lang in zip(texts, languages):
        out.append(int(language_tools.language_challenge_flag(text, lang)))
    return out


def derive_vocabulary_richness(texts, languages):
    out = []
    for text, lang in zip(texts, languages):
        out.append(language_tools.vocabulary_richness(text, lang))
    return out


def derive_features(df, text_col=None, solutions_col=None, language_col='Question_Language'):
    """Apply all derivations to a question metadata frame and return enriched copy.

    Works with the generated question-set columns (Question_Text, Question_Language,
    Variables_Count, Logical_Steps... ) as well as the legacy processed-schema names.
    Any raw feature missing from the source set is fall-backed to a text/solution
    derived proxy. Derived columns that are not part of the source set are still
    produced: language_level, reasoning_level, cognitive_load, num_unknowns,
    num_operations, p_s/p_g/p_t (the last three via prob_calcs).
    """
    out = df.copy()

    #collapse source column names into canonical internal names
    rename = {}
    if text_col and text_col in out.columns:
        rename[text_col] = 'question_text'
    if solutions_col and solutions_col in out.columns:
        rename[solutions_col] = 'solutions_text'
    for src, internal in COLUMN_ALIASES.items():
        if src in out.columns and src not in rename:
            rename[src] = internal
    if rename:
        out = out.rename(columns=rename)

    #language
    if language_col in out.columns:
        out['question_language'] = out[language_col]
    if 'question_language' in out.columns:
        langs = out['question_language'].astype(str).str.lower().map(
            {'english': 'en', 'en': 'en', 'eng': 'en',
             'arabic': 'ar', 'ar': 'ar', 'ara': 'ar'}).fillna('en')
    elif 'question_text' in out.columns:
        langs = out['question_text'].map(lambda t: language_tools.detect_language(t))
    else:
        langs = pd.Series(['en'] * len(out), index=out.index)
    out['language'] = langs

    #texts
    if 'question_text' in out.columns:
        texts = out['question_text'].fillna('')
    else:
        texts = pd.Series([''] * len(out), index=out.index)
    if 'solutions_text' in out.columns:
        sol_texts = out['solutions_text'].fillna('')
    else:
        sol_texts = pd.Series([''] * len(out), index=out.index)

    #vocabulary richness & language challenge from text
    out['vocabulary_richness'] = pd.Series(derive_vocabulary_richness(texts, langs), index=out.index)
    if 'language_challenging' in out.columns:
        out['language_challenging'] = out['language_challenging'].astype(str).str.lower().isin(
            ['1', 'true', 'yes', 'y', 't']).astype(int)
        out['language_challenge'] = out['language_challenging']
    else:
        out['language_challenge'] = pd.Series(derive_language_challenge(texts, langs), index=out.index)
        out['language_challenging'] = out['language_challenge']

    #unknowns / operations from the solution
    out['num_unknowns'] = pd.Series(derive_num_unknowns(sol_texts), index=out.index)
    out['num_operations'] = pd.Series(derive_num_operations(sol_texts), index=out.index)

    #raw-feature fallbacks (only when the source set does not provide them)
    if 'question_length' not in out.columns:
        out['question_length'] = texts.map(lambda t: len(re.findall(r'\w+', str(t))) if isinstance(t, str) else 0)
    if 'num_sentences' not in out.columns:
        out['num_sentences'] = texts.map(lambda t: len(re.findall(r'[.!?。؟]', str(t))) if isinstance(t, str) else 0)
    if 'num_clauses' not in out.columns:
        out['num_clauses'] = texts.map(lambda t: len(re.findall(r'[,،;؛:]', str(t))) if isinstance(t, str) else 0)
    if 'num_variables' not in out.columns:
        out['num_variables'] = pd.Series(
            derive_variables_count_from_text(texts), index=out.index)
    if 'num_steps' not in out.columns:
        out['num_steps'] = 0
    if 'solution_vocab' not in out.columns:
        out['solution_vocab'] = pd.Series(derive_vocabulary_richness(sol_texts, langs), index=out.index)
    else:
        out['solution_vocab'] = pd.to_numeric(out['solution_vocab'], errors='coerce').fillna(0.0)

    #quartiles
    out['language_level'] = derive_language_level(
        out['question_length'], out['vocabulary_richness'], out['num_clauses'], out['num_sentences'])
    out['reasoning_level'] = derive_reasoning_level(out['num_steps'], out['solution_vocab'])

    #cognitive load: use the provided index when present, otherwise derive it
    if 'cognitive_load_index' in out.columns:
        out['cognitive_load_index'] = pd.to_numeric(out['cognitive_load_index'], errors='coerce').fillna(0).astype(int)
    else:
        out['cognitive_load_index'] = derive_cognitive_load_index(out['num_variables'], out['num_unknowns'])
    #working-memory proxy used by the behavioral diagnosis engine
    out['cognitive_load'] = pd.to_numeric(out['cognitive_load_index'], errors='coerce').fillna(0.0)

    return out