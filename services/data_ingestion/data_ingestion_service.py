import ast
import pandas as pd
from services.Interfaces import Component
from services.data_ingestion import file_loader
from services.Entities import Question, PreHint, PostHint, PlayHint


def parse_list(value):
    """Parse a value that may be a string like \"[1, 2]\" into a list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, (list, tuple, pd.Series)):
        return list(value)
    return [value] if value is not None else []


def to_language(value, default='en'):#change this to something that detects language
    """Map a 'Question_Language' value (English/Arabic) to 'en'/'ar'."""
    if value is None:
        return default
    s = str(value).strip().lower()
    mapping = {'english': 'en', 'en': 'en', 'eng': 'en',
               'arabic': 'ar', 'ar': 'ar', 'ara': 'ar'}
    return mapping.get(s, default)


class data_ingestion_service(Component):
    """Populates the Data_Service (students, questions, skills) from raw files.

    Reads JSON/CSV sources and pushes entities into the data service through the
    mediator. The primary path is JSON (parse_questions_from_json); CSV is
    retained for backward compatibility.
    """

    def __init__(self):
        self.mediator = None

    def populate_students(self, records, id_field='uid'):
        """Register every unique student id found in the records."""
        ids = file_loader.extract_ids(records, id_field)
        for sid in ids:
            self.mediator.request({'type': 'add_student', 'student_id': sid})
        return ids

    def populate_skills(self, skills_map):
        """Register skills with their lists of similar skills."""
        for skill_id, similar in skills_map.items():
            self.mediator.request({
                'type': 'add_skill',
                'skill_id': skill_id,
                'similar_skills': list(similar),
            })
        return list(skills_map.keys())

    def populate_questions_from_json(self, path):
        """Load questions from  JSON file.

        Returns (inserted_count, skipped_count).
        """
        questions = file_loader.load_questions_from_json(path)
        n = 0
        for q in questions:
            self.mediator.request({'type': 'add_question', 'question': q})
            n += 1
        return n, 0

    def populate_questions(self, records, field_map=None):
        """Register questions from flat CSV rows (backward compatibility)."""
        n = 0
        skipped = 0
        for row in records:
            qatt = self._build_question_kwargs(row, field_map or {})
            if qatt is None:
                skipped += 1
                continue
            self.mediator.request({'type': 'add_question', 'question': qatt})
            n += 1
        return n, skipped


    def _build_question_kwargs(self, row, field_map):
        """Map a single raw CSV row to question entity constructor kwargs."""
        def get(target):
            source = field_map.get(target)
            if source in row and pd.notna(row.get(source)):
                return row[source]
            return row.get(target)

        q_id = row.get('Question_ID', row.get('question_id'))
        if q_id is None or pd.isna(q_id):
            return None

        skill_ids = parse_list(get('skill_ids'))
        return {
            'q_id': q_id,
            'grade': get('grade'),
            'grade_band': get('grade_band'),
            'subject': get('subject'),
            'unit_id': get('unit_id'),
            'lesson_id': get('lesson_id'),
            'skill_cluster_id': get('skill_cluster_id'),
            'skill_ids': skill_ids,
            'question_type': get('question_type'),
            'question_text': str(get('question_text')) if get('question_text') is not None else str(q_id),
            'question_media_url': get('question_media_url'),
            'question_language': to_language(get('language')),
            'difficulty_level': float(get('difficulty_level') or 0),
            'bloom_taxonomy_level': get('bloom_taxonomy_level'),
            'population_difficulty': float(get('population_difficulty') or 0),
            'discrimination_index': float(get('discrimination_index') or 0),
            'time_allowed': float(get('time_allowed') or 0),
            'theoretical_solving_time': float(get('theoretical_solving_time') or 0),
            'dynamic_avg_time': float(get('dynamic_avg_time') or 0),
            'cognitive_load_index': int(get('cognitive_load_index') or 0),
            'variables_count': int(get('variables_count') or 0),
            'logical_steps': int(get('logical_steps') or 0),
            'language_challenging': int(bool(get('language_challenging'))),
            'language_challenge_type': get('language_challenge_type'),
            'trick_question': bool(get('trick_question')),
            'time_pressure_flag': bool(get('time_pressure_flag')),
            'visual_dependency': bool(get('visual_dependency')),
            'real_world_context': bool(get('real_world_context')),
            'multi_concept_flag': bool(get('multi_concept_flag')),
            'prerequisite_concept_ids': parse_list(get('prerequisite_concept_id')),
            'correct_answer_content': get('correct_answer_content'),
            'tag_source': get('tag_source'),
            'tag_confidence_score': float(get('tag_confidence_score') or 0),
            'question_status': get('question_status'),
            'usage_count': int(get('usage_count') or 0),
            'date_created': get('date_created'),
            'last_updated': get('last_updated'),
            'cognitive_load': float(get('cognitive_load') or 0),
            'language_level': str(get('language_level')) if get('language_level') is not None else None,
            'reasoning_level': str(get('reasoning_level')) if get('reasoning_level') is not None else None,
            'p_t': float(get('p_t') or 0),
            'p_s': float(get('p_s') or 0),
            'p_g': float(get('p_g') or 0),
        }

    def load_question_records(self, metadata_path, assignments_df=None):
        """Load question metadata records and, if given, attach assignments."""
        records = file_loader.load_records(metadata_path)
        if assignments_df is not None:
            id_col = 'question_id' if 'question_id' in assignments_df.columns else 'Question_ID'
            assignments = assignments_df.set_index(id_col)
            merged = []
            for row in records:
                r = dict(row)
                qid = r.get('question_id', r.get('Question_ID'))
                if qid is not None and qid in assignments.index:
                    r.update(assignments.loc[qid].to_dict())
                merged.append(r)
            return merged
        return records

    def populate_pre_hints(self, records) -> int:
        """Register pre-hints from records (CSV rows or dicts).

        Expected columns: hint_id, hint_text, topic, content_gap_type, diagnosis, hint_type
        """
        n = 0
        for row in records:
            hint_id = row.get('hint_id', row.get('Hint_ID'))
            if hint_id is None or pd.isna(hint_id):
                continue
            pre_hint = PreHint(
                hint_id=str(hint_id),
                hint_text=str(row.get('hint_text', row.get('Hint_Text', ''))),
                topic=str(row['topic']) if pd.notna(row.get('topic')) else None,
                content_gap_type=str(row['content_gap_type']) if pd.notna(row.get('content_gap_type')) else None,
                diagnosis=str(row['diagnosis']) if pd.notna(row.get('diagnosis')) else None,
                hint_type=str(row.get('hint_type', 'content_gap')),
            )
            self.mediator.request({'type': 'add_pre_hint', 'pre_hint': pre_hint})
            n += 1
        return n

    def populate_post_hints(self, records) -> int:
        """Register post-hints from records (CSV rows or dicts).

        Expected columns: hint_id, hint_text, question_id, hint_order
        """
        n = 0
        for row in records:
            hint_id = row.get('hint_id', row.get('Hint_ID'))
            question_id = row.get('question_id', row.get('Question_ID'))
            if hint_id is None or pd.isna(hint_id) or question_id is None or pd.isna(question_id):
                continue
            post_hint = PostHint(
                hint_id=str(hint_id),
                hint_text=str(row.get('hint_text', row.get('Hint_Text', ''))),
                question_id=str(question_id),
                hint_order=int(row.get('hint_order', row.get('Hint_Order', 0)) or 0),
            )
            self.mediator.request({'type': 'add_post_hint', 'post_hint': post_hint})
            n += 1
        return n

    def populate_play_hints(self, records) -> int:
        """Register play-hints from records (CSV rows or dicts).

        Expected columns: hint_id, hint_text, q_id, hint_order
        """
        n = 0
        for row in records:
            hint_id = row.get('hint_id', row.get('Hint_ID'))
            q_id = row.get('q_id', row.get('Q_ID'))
            if hint_id is None or pd.isna(hint_id) or q_id is None or pd.isna(q_id):
                continue
            play_hint = PlayHint(
                hint_id=str(hint_id),
                hint_text=str(row.get('hint_text', row.get('Hint_Text', ''))),
                q_id=str(q_id),
                hint_order=int(row.get('hint_order', row.get('Hint_Order', 0)) or 0),
            )
            self.mediator.request({'type': 'add_play_hint', 'play_hint': play_hint})
            n += 1
        return n