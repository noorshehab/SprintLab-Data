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


def to_flag(value):
    """Coerce a boolean-like value (True, 'TRUE', 1, 'yes') to 0/1."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    s = str(value).strip().lower()
    return int(s in ('1', 'true', 'yes', 'y', 't'))


def to_language(value, default='en'):
    """Map a 'Question_Language' value (English/Arabic) to 'en'/'ar'."""
    if value is None:
        return default
    s = str(value).strip().lower()
    mapping = {'english': 'en', 'en': 'en', 'eng': 'en', 'arabic': 'ar', 'ar': 'ar', 'ara': 'ar'}
    return mapping.get(s, default)


class data_ingestion_service(Component):
    """Populates the Data_Service (students, questions, skills) from raw files.

    Reads JSON/CSV sources and pushes entities into the data service through the
    mediator, mapping source field names to the entity attributes.
    """

    def __init__(self):
        self.mediator = None
        #default mapping from question entity fields to source columns
        self.default_question_field_map = {
            'q_id': 'Question_ID',
            'grade': 'Grade',
            'grade_band': 'Grade_Band',
            'subject': 'Subject',
            'unit_id': 'Unit_ID',
            'lesson_id': 'Lesson_ID',
            'skill_cluster_id': 'Skill_Cluster_ID',
            'skill_ids': 'Skill_Cluster_ID',
            'question_type': 'Question_Type',
            'question_text': 'Question_Text',
            'question_media_url': 'Question_Media_URL',
            'language': 'Question_Language',
            'difficulty_level': 'Difficulty_Level',
            'bloom_taxonomy_level': 'Bloom_Taxonomy_Level',
            'population_difficulty': 'Population_Difficulty',
            'discrimination_index': 'Discrimination_Index',
            'time_allowed': 'Time_Allowed',
            'theoretical_solving_time': 'Theoretical_Solving_Time',
            'dynamic_avg_time': 'Dynamic_Avg_Time',
            'cognitive_load_index': 'Cognitive_Load_Index',
            'variables_count': 'Variables_Count',
            'logical_steps': 'Logical_Steps',
            'language_challenging': 'Language_Challenging',
            'language_challenge_type': 'Language_Challenge_Type',
            'trick_question': 'Trick_Question',
            'time_pressure_flag': 'Time_Pressure_Flag',
            'visual_dependency': 'Visual_Dependency',
            'real_world_context': 'Real_World_Context',
            'multi_concept_flag': 'Multi_Concept_Flag',
            'prerequisite_concept_id': 'Prerequisite_Concept_ID',
            'correct_answer_id': 'Correct_Answer_ID',
            'correct_answer_content': 'Correct_Answer_Content',
            'distractor_1_content': 'Distractor_1_Content',
            'distractor_2_content': 'Distractor_2_Content',
            'distractor_3_content': 'Distractor_3_Content',
            'content_gap_type': 'Content_Gap_Type',
            'tag_source': 'Tag_Source',
            'tag_confidence_score': 'Tag_Confidence_Score',
            'question_status': 'Question_Status',
            'usage_count': 'Usage_Count',
            'date_created': 'Date_Created',
            'last_updated': 'Last_Updated',
            #derived features, written by the question processing service
            'cognitive_load': 'cognitive_load',
            'language_level': 'language_level',
            'reasoning_level': 'reasoning_level',
            'num_unknowns': 'num_unknowns',
            'num_operations': 'num_operations',
            'p_t': 'p_t',
            'p_s': 'p_s',
            'p_g': 'p_g',
        }

    def populate_students(self, records, id_field='uid'):
        """Register every unique student id found in the records."""
        ids = file_loader.extract_ids(records, id_field)
        for sid in ids:
            self.mediator.request({'type': 'add_student', 'student_id': sid})
        return ids

    def populate_skills(self, skills_map):
        """Register skills with their lists of similar skills.

        skills_map: {skill_id: [similar_skill_id, ...]}
        """
        for skill_id, similar in skills_map.items():
            self.mediator.request({
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
            self.mediator.request( {'type': 'add_question', 'question': qatt})
            n += 1
        return n, skipped

    def _build_question_kwargs(self, row, field_map):
        """Map a single raw row to the question entity constructor kwargs."""
        def get(target):
            source = field_map.get(target)
            if source in row and pd.notna(row.get(source)):
                return row[source]
            return row.get(target)

        q_id = row.get('Question_ID', row.get('question_id'))
        if q_id is None or pd.isna(q_id):
            return None

        skill_ids = parse_list(get('skill_ids'))
        kwargs = {
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
            'difficulty_level': to_float(get('difficulty_level')),
            'bloom_taxonomy_level': get('bloom_taxonomy_level'),
            'population_difficulty': to_float(get('population_difficulty')),
            'discrimination_index': to_float(get('discrimination_index')),
            'time_allowed': to_float(get('time_allowed')),
            'theoretical_solving_time': to_float(get('theoretical_solving_time')),
            'dynamic_avg_time': to_float(get('dynamic_avg_time')),
            'cognitive_load_index': get('cognitive_load_index'),
            'variables_count': to_int(get('variables_count')),
            'logical_steps': to_int(get('logical_steps')),
            'language_challenging': to_flag(get('language_challenging')),
            'language_challenge_type': get('language_challenge_type'),
            'trick_question': to_flag(get('trick_question')),
            'time_pressure_flag': to_flag(get('time_pressure_flag')),
            'visual_dependency': to_flag(get('visual_dependency')),
            'real_world_context': to_flag(get('real_world_context')),
            'multi_concept_flag': to_flag(get('multi_concept_flag')),
            'prerequisite_concept_id': get('prerequisite_concept_id'),
            'correct_answer_id': get('correct_answer_id'),
            'correct_answer_content': get('correct_answer_content'),
            'distractor_1_content': get('distractor_1_content'),
            'distractor_2_content': get('distractor_2_content'),
            'distractor_3_content': get('distractor_3_content'),
            'content_gap_type': get('content_gap_type'),
            'tag_source': get('tag_source'),
            'tag_confidence_score': to_float(get('tag_confidence_score')),
            'question_status': get('question_status'),
            'usage_count': to_int(get('usage_count')),
            'date_created': get('date_created'),
            'last_updated': get('last_updated'),
            #derived features (kept)
            'cognitive_load': to_float(get('cognitive_load')),
            'language_level': str(get('language_level')) if get('language_level') is not None else None,
            'reasoning_level': str(get('reasoning_level')) if get('reasoning_level') is not None else None,
            'num_unknowns': to_int(get('num_unknowns')),
            'num_operations': to_int(get('num_operations')),
            'p_t': to_float(get('p_t')),
            'p_s': to_float(get('p_s')),
            'p_g': to_float(get('p_g')),
        }
        return kwargs

    def load_question_records(self, metadata_path, assignments_df=None):
        """Load question metadata records and, if given, attach assignments.

        assignments_df: DataFrame with the derived per-question features
        (output of the question processing service) keyed by question_id.
        """
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