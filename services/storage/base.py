"""Repository contract for the data layer.

Data_Service delegates storage to an implementation of this interface.
InMemoryRepository ships today; SqlAlchemyRepository (PostgreSQL) is the
target implementation - it must hydrate/convert rows <-> entities
internally so nothing above the seam changes.

Entities (services.Entities.student/question/skill) remain the currency
of the interface, not raw row dicts.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from services.Entities import question, skill, student


class DataRepository(ABC):
    #--- students -------------------------------------------------------
    @abstractmethod
    def add_student(self, student_id: str) -> None: ...

    @abstractmethod
    def get_student(self, student_id: str) -> student | None: ...

    @abstractmethod
    def list_students(self) -> list[student]: ...

    #--- questions ------------------------------------------------------
    @abstractmethod
    def add_question(self, **kwargs) -> None: ...

    @abstractmethod
    def get_question(self, q_ids: list | str) -> list[question | None]: ...

    @abstractmethod
    def list_questions(self) -> list[question]: ...

    @abstractmethod
    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        """Set derived/processed fields on a stored question."""

    #--- skills ---------------------------------------------------------
    @abstractmethod
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None: ...

    @abstractmethod
    def get_skill(self, skill_id: str) -> skill | None: ...

    #--- student state ---------------------------------------------------
    @abstractmethod
    def update_priors(self, student_id: str, skill_id: str, new_prior: float) -> None: ...

    @abstractmethod
    def update_responses(self, student_id: str, responses: list[tuple]) -> None: ...

    @abstractmethod
    def add_diagnosis(self, student_id: str, diagnoses: list,
                      deltas: dict | None = None,
                      timestamp: str | None = None) -> None: ...

    @abstractmethod
    def update_content_gaps(self, student_id: str, content_gap_types: dict) -> None: ...

    @abstractmethod
    def add_priors_history(self, student_id: str, timestamp: str, priors: dict) -> None: ...

    @abstractmethod
    def update_treatment_plan(self, student_id: str, timestamp: str,
                              treatment_name: str, parameters: dict) -> None: ...

    #--- query ----------------------------------------------------------
    @abstractmethod
    def query(self, parameters: list | dict) -> dict:
        """AND-conjoined condition query over questions.

        See services/question_schema.py for the attribute registry.
        Returns {question_id: question}.
        """
