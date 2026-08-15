import re


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


def language_challenge_flag(text, language='en', uncommon_threshold=None):
    """Flag questions whose language is challenging: negation, exceptions,
    uncommon vocabulary or indirect/hedged phrasing.

    Returns True if any signal fires, False otherwise.
    """
    if not text:
        return False
    text = str(text).lower()
    neg = any(w in text for w in NEGATION_WORDS.get(language, set()))
    exc = any(w in text for w in EXCEPTION_WORDS.get(language, set()))
    indirect = any(w in text for w in INDIRECT_WORDS.get(language, set()))
    return bool(neg or exc or indirect)