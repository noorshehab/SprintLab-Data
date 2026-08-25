"""Shared question-context construction for the bandit.

Single definition of the DataFrame schema the ContextualBandit expects;
Match_Service and Question_Selection_Service must not build their own.
"""
from __future__ import annotations
import pandas as pd


def build_question_context(questions) -> pd.DataFrame:
    """Build the bandit context frame from question entities.

    Columns (the contract with MAB.select/update):
        q_id       - question id
        skill_ids  - list of skill-cluster ids the question tests
        difficulty - difficulty_level
        learning   - p_g (learning rate parameter)
    """
    rows = [{'q_id': q.id,
             'skill_ids': q.get_skills(),
             'difficulty': q.difficulty_level,
             'learning': q.p_g}
            for q in questions if q]
    return pd.DataFrame(rows, columns=['q_id', 'skill_ids', 'difficulty', 'learning'])
