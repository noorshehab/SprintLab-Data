"""Data_Service - thin facade over a DataRepository.

Owns the public surface every service already codes against; storage
mechanics live in the injected repository (default: InMemoryRepository).
Swap in SqlAlchemyRepository later without changing any caller.

Singleton pattern retained for the process-wide instance.
"""
from __future__ import annotations
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.log_setup import get_logger
from services.Entities import Question, SkillCluster, Student, PreHint, PostHint, PlayHint, Unit, Lesson, LearningObjective
from services.Interfaces import SigletonMeta
from services.storage.base import DataRepository
from services.storage.memory import InMemoryRepository
from services.storage.mysql_repo import SqlAlchemyRepository
from services.question_processing.question_processing_service import question_processing_service
import os
from dotenv import load_dotenv

load_dotenv()

log = get_logger('Data_Service')


class Data_Service(metaclass=SigletonMeta):

    def __init__(self, repository: DataRepository | None = None) -> None:
        if repository:
            self.repository = repository
        else:
            mode = os.getenv('STORAGE_MODE', 'memory').lower()
            if mode == 'mysql':
                self.repository = SqlAlchemyRepository()
            else:
                self.repository = InMemoryRepository()
        
        log.info("constructed Data_Service (singleton, repo=%s) | students=%d questions=%d skills=%d",
                 type(self.repository).__name__,
                 len(self.repository.list_students()),
                 len(self.repository.list_questions()),
                 0)

    #--- students -------------------------------------------------------
    def add_student(self, student_id: str) -> None:
        self.repository.add_student(student_id)

    def get_student(self, student_id: str) -> Student | None:
        return self.repository.get_student(student_id)

    def list_students(self) -> list[Student]:
        return self.repository.list_students()

        
    #--- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        # 1. Add to repository
        self.repository.add_question(**kwargs)
        q_id = kwargs.get('q_id')
        if not q_id:
            return

        # 2. Track which attributes were explicitly provided
        explicit_attrs = set(kwargs.keys())

        # 3. Immediately derive attributes and probabilities
        q = self.repository.get_question(q_id)[0]
        if q:
            # Build a minimal metadata frame for the single question
            metadata = {
                'question_id': [q.id],
                'Question_Text': [q.question_text],
                'Question_Language': [q.language],
                'Logical_Steps': [q.logical_steps],
            }
            # Include options content for solution-based derivations
            if q.options:
                correct_opt = next((opt for opt in q.options if opt.is_correct), None)
                metadata['Correct_Answer_Content'] = [correct_opt.content if correct_opt else '']
            else:
                metadata['Correct_Answer_Content'] = ['']

            df = pd.DataFrame(metadata)
            qp = question_processing_service()
            enriched = qp.derive_question_attributes(df)
            
            # Extract derived values for this question, but only overwrite
            # attributes that were NOT explicitly provided by the caller.
            row = enriched.iloc[0]
            derived_attrs = {}
            for key in ('difficulty_level', 'language_level', 'reasoning_level',
                        'cognitive_load', 'cognitive_load_index', 'p_g', 'p_s', 'p_t'):
                if key not in explicit_attrs:
                    derived_attrs[key] = row.get(key) or (0.0 if 'load' in key or 'g' in key or 's' in key or 't' in key else 'Q1')
            if derived_attrs:
                self.repository.update_question_attributes(q.id, derived_attrs)

    def get_question(self, q_ids: list | str) -> list[Question | None]:
        return self.repository.get_question(q_ids)

    def list_questions(self) -> list[Question]:
        return self.repository.list_questions()

    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        self.repository.update_question_attributes(q_id, attributes)

    #--- curriculum ---------------------------------------------------------
    def add_unit(self, unit: Unit) -> None:
        self.repository.add_unit(unit)

    def add_lesson(self, lesson: Lesson) -> None:
        self.repository.add_lesson(lesson)

    def add_learning_objective(self, lo: LearningObjective) -> None:
        self.repository.add_learning_objective(lo)

    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None:
        self.repository.add_skill(skill_id, similar_skills)

    def get_skill(self, skill_id: str) -> SkillCluster | None:
        return self.repository.get_skill(skill_id)

    #--- student state ---------------------------------------------------
    def update_priors(self, student_id: str, skill_id: str, new_prior: float) -> None:
        self.repository.update_priors(student_id, skill_id, new_prior)

    def update_responses(self, student_id: str, responses: list[tuple]) -> None:
        self.repository.update_responses(student_id, responses)

    def add_diagnosis(self, student_id: str, diagnoses: list,
                      deltas: dict | None = None,
                      timestamp: str | None = None) -> None:
        self.repository.add_diagnosis(student_id, diagnoses, deltas, timestamp)

    def update_content_gaps(self, student_id: str, content_gap_types: dict) -> None:
        self.repository.update_content_gaps(student_id, content_gap_types)

    def add_priors_history(self, student_id: str, timestamp: str, priors: dict) -> None:
        self.repository.add_priors_history(student_id, timestamp, priors)

    def update_treatment_plan(self, student_id: str, timestamp: str,
                              treatment_name: str, parameters: dict) -> None:
        self.repository.update_treatment_plan(student_id, timestamp, treatment_name, parameters)

    #--- Crystal text / hints --------------------------------------------
    def add_pre_hint(self, pre_hint: PreHint) -> None:
        self.repository.add_pre_hint(pre_hint)

    def get_pre_hint(self, hint_id: str) -> PreHint | None:
        return self.repository.get_pre_hint(hint_id)

    def get_pre_hints_by_topic(self, topic: str) -> list[PreHint]:
        return self.repository.get_pre_hints_by_topic(topic)

    def get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> list[PreHint]:
        return self.repository.get_pre_hints_by_content_gap_type(topic, content_gap_type)

    def get_pre_hints_by_diagnosis(self, diagnosis: str) -> list[PreHint]:
        return self.repository.get_pre_hints_by_diagnosis(diagnosis)

    def list_pre_hints(self) -> list[PreHint]:
        return self.repository.list_pre_hints()

    def add_post_hint(self, post_hint: PostHint) -> None:
        self.repository.add_post_hint(post_hint)

    def get_post_hint(self, hint_id: str) -> PostHint | None:
        return self.repository.get_post_hint(hint_id)

    def get_post_hints_by_question(self, question_id: str) -> list[PostHint]:
        return self.repository.get_post_hints_by_question(question_id)

    def list_post_hints(self) -> list[PostHint]:
        return self.repository.list_post_hints()

    def add_play_hint(self, play_hint: PlayHint) -> None:
        self.repository.add_play_hint(play_hint)

    def get_play_hint(self, hint_id: str) -> PlayHint | None:
        return self.repository.get_play_hint(hint_id)

    def get_play_hints_by_question(self, q_id: str) -> list[PlayHint]:
        return self.repository.get_play_hints_by_question(q_id)

    def list_play_hints(self) -> list[PlayHint]:
        return self.repository.list_play_hints()

    #--- query ----------------------------------------------------------
    def query(self, parameters: list | dict) -> dict:
        """Query questions by AND-conjoined conditions.

        See services/storage/memory.py (InMemoryRepository.query) and
        services/question_schema.py for the condition/attribute contract.
        Returns {question_id: question}.
        """
        return self.repository.query(parameters)
