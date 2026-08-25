"""Data_Service - thin facade over a DataRepository.

Owns the public surface every service already codes against; storage
mechanics live in the injected repository (default: InMemoryRepository).
Swap in SqlAlchemyRepository later without changing any caller.

Singleton pattern retained for the process-wide instance.
"""
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.log_setup import get_logger
from services.Entities import question, skill, student
from services.Interfaces import SigletonMeta
from services.storage.base import DataRepository
from services.storage.memory import InMemoryRepository

log = get_logger('Data_Service')


class Data_Service(metaclass=SigletonMeta):

    def __init__(self, repository: DataRepository | None = None) -> None:
        self.repository: DataRepository = repository or InMemoryRepository()
        log.info("constructed Data_Service (singleton, repo=%s) | students=%d questions=%d skills=%d",
                 type(self.repository).__name__,
                 len(self.repository.list_students()),
                 len(self.repository.list_questions()),
                 0)

    #--- students -------------------------------------------------------
    def add_student(self, student_id: str) -> None:
        self.repository.add_student(student_id)

    def get_student(self, student_id: str) -> student | None:
        return self.repository.get_student(student_id)

    def list_students(self) -> list[student]:
        return self.repository.list_students()

    #--- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        self.repository.add_question(**kwargs)

    def get_question(self, q_ids: list | str) -> list[question | None]:
        return self.repository.get_question(q_ids)

    def list_questions(self) -> list[question]:
        return self.repository.list_questions()

    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        self.repository.update_question_attributes(q_id, attributes)

    #--- skills ---------------------------------------------------------
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None:
        self.repository.add_skill(skill_id, similar_skills)

    def get_skill(self, skill_id: str) -> skill | None:
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

    #--- query ----------------------------------------------------------
    def query(self, parameters: list | dict) -> dict:
        """Query questions by AND-conjoined conditions.

        See services/storage/memory.py (InMemoryRepository.query) and
        services/question_schema.py for the condition/attribute contract.
        Returns {question_id: question}.
        """
        return self.repository.query(parameters)
