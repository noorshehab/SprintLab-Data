import numpy as np
import pandas as pd
from scipy import stats

#features used for p_s / p_g, mirroring data_preprocess.set_probs() list with the
#two new derived features (num_unknowns, num_operations) substituted in
PROB_FEATURES = [
    'question_length', 'num_variables', 'vocabulary_richness',
    'num_unknowns', 'num_operations', 'solution_length',
    'num_equations', 'num_steps',
]
DEFAULT_P_T = 0.017  # fixed 1.7%


def compute_probs(df, p_t=DEFAULT_P_T):
    """Per-question BKT probabilities from the 8-feature body.

    p_s: min-max scaled collective z score * 0.1 (zeros -> 0.1)
    p_g: inverted collective z score * 0.3 (zeros -> 0.3)
    p_t: fixed 0.017
    """
    out = df[['question_id']].copy()
    for feat in PROB_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    z_scores = []
    for feat in PROB_FEATURES:
        s = pd.to_numeric(df[feat], errors='coerce')
        s = s.fillna(s.mean() if s.notna().any() else 0.0)
        if s.std() == 0 or pd.isna(s.std()):
            z_scores.append(s * 0.0)
        else:
            z_scores.append(stats.zscore(s))
    collective = np.mean(np.array(z_scores), axis=0)
    cmin, cmax = collective.min(), collective.max()
    scaled = (collective - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(collective)
    inverted = np.abs(collective - cmax) / (cmax - cmin) if cmax > cmin else np.zeros_like(collective)

    p_s = scaled * 0.1
    p_g = inverted * 0.3
    p_s = np.where(p_s == 0, 0.1, p_s)
    p_g = np.where(p_g == 0, 0.3, p_g)

    out['p_s'] = p_s
    out['p_g'] = p_g
    out['p_t'] = p_t

    #parameter constraints (p(s)+p(g)<1 and p(g)<0.3, p(s)<0.1 kept ~intact)
    out['p_t_constraint'] = out['p_t'] < 1 - out['p_s'] / (1 - out['p_g'])
    return out