import ast
import numpy as np
import pandas as pd
from services.Interfaces import Component
from services.data_ingestion import file_loader


def parse_list(value):
    """Parse a value that may be a string like \"[1, 2]\" into a list of ints."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def parse_unit(value):
    """Parse a super_topic_ids value into a single unit (first element)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        return parsed[0] if isinstance(parsed, list) and parsed else (parsed if isinstance(parsed, (list, tuple)) else parsed)
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else None
    return value


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class data_ingestion_service(Component):
    """Populates the Data_Service (students, questions, skills) from raw files.

    Reads JSON/CSV sources and pushes entities into the data service through the
    mediator, mapping source field names to the entity attributes.
    """

    def __init__(self):
        self.mediator = None
        #default mapping from question entity fields to source columns
        self.default_question_field_map = {
            'q_id': 'question_id',
            'skill_ids': 'kc_ids',
            'unit_id': 'super_topic_ids',
            'text': 'content',
            'time': 'time',
            'time_pressure': 'time_pressure',
            'level': 'difficulty_level',
            'cognitive_load': 'cognitive_load',
            'variables_count': 'num_variables',
            'steps': 'num_steps',
            'language_challenge': 'language_challenge',
            'language_level': 'language_level',
            'reasoning_level': 'reasoning_level',
            'p_t': 'p_t',
            'p_s': 'p_s',
            'p_g': 'p_g',
            'language': 'language',
            'num_unknowns': 'num_unknowns',
            'num_operations': 'num_operations',
            'cognitive_load_index': 'cognitive_load_index',
        }

    def populate_students(self, records, id_field='uid'):
        """Register every unique student id found in the records."""
        ids = file_loader.extract_ids(records, id_field)
        for sid in ids:
            self.mediator.request(self, {'type': 'add_student', 'student_id': sid})
        return ids

    def populate_skills(self, skills_map):
        """Register skills with their lists of similar skills.

        skills_map: {skill_id: [similar_skill_id, ...]}
        """
        for skill_id, similar in skills_map.items():
            self.mediator.request(self, {
                'type': 'add_skill',
                'skill_id': skill_id,
                'similar_skills': list(similar),
            })
        return list(skills_map.keys())

    def populate_questions(self, records, field_map=None):
        """Register questions. field_map overrides the default source columns."""
        field_map = {**self.default_question_field_map, **(field_map or {})}
        n = 0
        skipped = 0
        for row in records:
            qatt = self._build_question_kwargs(row, field_map)
            if qatt is None:
                skipped += 1
                continue
            self.mediator.request(self, {'type': 'add_question', 'question': qatt})
            n += 1
        return n, skipped

    def _build_question_kwargs(self, row, field_map):
        """Map a single raw row to the question entity constructor kwargs."""
        def get(target):
            return row.get(field_map.get(target))

        q_id = get('q_id')
        if q_id is None or pd.isna(q_id):
            return None

        kwargs = {
            'q_id': to_int(q_id),
            'skill_ids': parse_list(get('skill_ids')),
            'unit_id': parse_unit(get('unit_id')),
            'text': str(get('text')) if get('text') is not None else str(q_id),
            'time': to_float(get('time')),
            'time_pressure': to_int(get('time_pressure')),
            'level': to_float(get('level')),
            'cognitive_load': to_float(get('cognitive_load')),
            'variables_count': to_int(get('variables_count')),
            'steps': to_int(get('steps')),
            'language_challenge': to_float(get('language_challenge')),
            'language_level': str(get('language_level')) if get('language_level') is not None else None,
            'reasoning_level': str(get('reasoning_level')) if get('reasoning_level') is not None else None,
            'p_t': to_float(get('p_t')),
            'p_s': to_float(get('p_s')),
            'p_g': to_float(get('p_g')),
            'language': str(get('language')) if get('language') is not None else 'en',
            'num_unknowns': to_int(get('num_unknowns')),
            'num_operations': to_int(get('num_operations')),
            'cognitive_load_index': to_int(get('cognitive_load_index')) if get('cognitive_load_index') is not None else None,
        }
        return kwargs

    def load_question_records(self, metadata_path, assignments_df=None):
        """Load question metadata records and, if given, attach assignments.

        assignments_df: DataFrame with the derived per-question features
        (output of the question processing service) keyed by question_id.
        """
        records = file_loader.load_records(metadata_path)
        if assignments_df is not None:
            assignments = assignments_df.set_index('question_id')
            merged = []
            for row in records:
                r = dict(row)
                qid = r.get('question_id')
                if qid is not None and qid in assignments.index:
                    r.update(assignments.loc[qid].to_dict())
                merged.append(r)
            return merged
        return records