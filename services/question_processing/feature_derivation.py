import re
import numpy as np
import pandas as pd
from scipy import stats
from services.question_processing import language_tools
from services.question_processing.scibert_ner import ScienceEntityNER

#shared instance - model loads lazily on first extraction, never at import
_science_ner: ScienceEntityNER | None = None

#delete this
def get_variable_ner() -> ScienceEntityNER:
    global _science_ner
    if _science_ner is None:
        _science_ner = ScienceEntityNER()
    return _science_ner


#source column names -> canonical internal names (question-processing schema)
COLUMN_ALIASES = {
    'Question_ID': 'question_id',
    'Question_Text': 'question_text',
    'Variables_Count': 'num_variables',
    'Logical_Steps': 'num_steps',
}
BLOOM_TAX={
    'Remember':1,
    'Understand':2,
    'Apply':3,
    'Analyze':4,
    'Evaluate':5,
    'Create':6
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


def derive_cognitive_load_index(variables_count, bloom_tax):
    """Cognitive load index, integer 1-5.

    zscore(variables_count) , min-max scaled to 1-5. and the bloom tax level mapped 1-6
    """
    score = _z(variables_count)+BLOOM_TAX[bloom_tax]
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


def derive_language_level(question_length, vocabulary_richness, num_clauses, num_sentences,curr_vocab):
    """language_level graded Q1-Q4 from the same composite z-score formula."""
    score = _z(question_length) + _z(vocabulary_richness) + _z(num_clauses) + _z(num_sentences)+_z(curr_vocab)
    scaled = _minmax(score)
    return _safe_quartile(scaled)


def derive_reasoning_level(logical_steps,bloom_tax):
    """Reasoning difficulty labeled from number of solution steps and bloom taxonomy level mapped 1-6 """
    score = _z(logical_steps) +BLOOM_TAX[bloom_tax]
    scaled = _minmax(score)
    return _safe_quartile(scaled)


def derive_language_challenge(texts, languages, grade_bands, subjects, curriculums):
    """Flag questions with negation/exceptions/indirect phrasing via the word lists."""
    out = []
    for text, lang, gb, sub, curr in zip(texts, languages, grade_bands, subjects, curriculums):
        res = language_tools.language_challenge_flag(text, lang, gb, sub, curr)
        out.append(res)
    return out

#derive language challenge type


def derive_vocabulary_richness(texts, languages):
    out = []
    for text, lang in zip(texts, languages):
        out.append(language_tools.vocabulary_richness(text, lang))
    return out



#change this so that theres not all these fallbacks 
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
    for src, internal in COLUMN_ALIASES.items():
        if src in out.columns and src not in rename:
            rename[src] = internal
    if rename:
        out = out.rename(columns=rename)

    #language
    if 'question_text' in out.columns:
        langs = out['question_text'].map(lambda t: language_tools.detect_language(t))
    else:
        langs = pd.Series(['en'] * len(out), index=out.index)
    out['language'] = langs

    #texts
    if 'question_text' in out.columns:
        texts = out['question_text'].fillna('')
    else:
        texts = pd.Series([''] * len(out), index=out.index)
  

    #vocabulary richness & language challenge from text
    out['vocabulary_richness'] = pd.Series(derive_vocabulary_richness(texts, langs), index=out.index)
 
    gb = out.get('grade_band', pd.Series([None] * len(out), index=out.index))
    sub = out.get('subject', pd.Series([None] * len(out), index=out.index))
    curr = out.get('curriculum', pd.Series([None] * len(out), index=out.index))
    
    challenge_results = derive_language_challenge(texts, langs, gb, sub, curr)
    out['language_challenging'] = pd.Series([r['language_challenge'] for r in challenge_results], index=out.index)
    out['language_challenge_type'] = pd.Series([r['language_challenge_type'] for r in challenge_results], index=out.index)
    out['curriculum_vocab_score'] = pd.Series([r['curriculum_vocab_score'] for r in challenge_results], index=out.index)



    #raw-feature fallbacks (only when the source set does not provide them)
    out['question_length'] = texts.map(lambda t: len(re.findall(r'\w+', str(t))) if isinstance(t, str) else 0)
    out['num_sentences'] = texts.map(lambda t: len(re.findall(r'[.!?。؟]', str(t))) if isinstance(t, str) else 0)
    out['num_clauses'] = texts.map(lambda t: len(re.findall(r'[,،;؛:]', str(t))) if isinstance(t, str) else 0)
    

    #quartiles
    out['language_level'] = derive_language_level(out['question_length'], out['vocabulary_richness'], out['num_clauses'], out['num_sentences'],out['curriculum_vocab_score'])
    out['reasoning_level'] = derive_reasoning_level(out['num_steps'], out['solution_vocab'])
    out['cognitive_load_index'] = derive_cognitive_load_index(out['num_variables'], out['num_unknowns'])
    

    return out