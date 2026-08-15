import re
import numpy as np
import pandas as pd
from scipy import stats
from services.question_processing import language_tools

#math symbols used to detect distinct operations in a solution
MATH_SYMBOLS = set('+-×÷=≠<>±')


def _minmax(series):
    """Min-max normalize to [0, 1]."""
    s = pd.to_numeric(series, errors='coerce')
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return s.fillna(0.0)
    return (s - mn) / (mx - mn)


def _z(series):
    s = pd.to_numeric(series, errors='coerce')
    if s.std(skipna=True) == 0 or pd.isna(s.std(skipna=True)):
        return s.fillna(0.0)
    return stats.zscore(s.fillna(s.mean()))


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


def derive_language_level(question_length, vocabulary_richness, num_clauses, num_sentences):
    """language_level graded Q1-Q4 from the same composite z-score formula."""
    score = _z(question_length) + _z(vocabulary_richness) + _z(num_clauses) + _z(num_sentences)
    scaled = _minmax(score)
    return pd.qcut(scaled, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')


def derive_reasoning_level(num_steps, solution_vocab):
    """Reasoning difficulty labeled from number of solution steps + solution vocab richness."""
    score = _z(num_steps) + _z(solution_vocab)
    scaled = _minmax(score)
    return pd.qcut(scaled, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')


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


def derive_features(df, text_col='content', solutions_col='analysis', language_col=None):
    """Apply all derivations to a question metadata frame and return enriched copy.

    Expected raw columns (any may be absent; fallbacks used):
    question_length, num_clauses, num_sentences, num_variables, num_steps,
    solution_vocab. Text for vocab richness -> `text_col`; solution -> `solutions_col`.
    """
    out = df.copy()

    #language
    if language_col and language_col in out.columns:
        langs = out[language_col].astype(str).str.lower().map({'arabic': 'ar', 'arabic language': 'ar', 'ar': 'ar'}).fillna('en')
    else:
        langs = out[text_col].map(lambda t: language_tools.detect_language(t))
    out['language'] = langs

    #vocabulary richness & language challenge from text
    texts = out[text_col].fillna('') if text_col in out.columns else pd.Series([''] * len(out), index=out.index)
    out['vocabulary_richness'] = pd.Series(derive_vocabulary_richness(texts, langs), index=out.index)
    out['language_challenge'] = pd.Series(derive_language_challenge(texts, langs), index=out.index)

    #unknowns / operations from the solution
    if solutions_col in out.columns:
        sol_texts = out[solutions_col].fillna('')
    else:
        sol_texts = pd.Series([''] * len(out), index=out.index)
    out['num_unknowns'] = pd.Series(derive_num_unknowns(sol_texts), index=out.index)
    out['num_operations'] = pd.Series(derive_num_operations(sol_texts), index=out.index)

    #quartiles
    for col in ['question_length', 'num_clauses', 'num_sentences', 'num_variables', 'num_steps', 'solution_vocab']:
        if col not in out.columns:
            out[col] = 0.0
    out['language_level'] = derive_language_level(
        out['question_length'], out['vocabulary_richness'], out['num_clauses'], out['num_sentences'])
    out['reasoning_level'] = derive_reasoning_level(out['num_steps'], out['solution_vocab'])
    out['cognitive_load_index'] = derive_cognitive_load_index(out['num_variables'], out['num_unknowns'])

    return out