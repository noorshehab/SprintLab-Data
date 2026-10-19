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

from services.Entities import Question, SkillCluster, Student, PreHint, PostHint, PlayHint, Unit, Lesson, LearningObjective


class DataRepository(ABC):
    #--- students -------------------------------------------------------
    @abstractmethod
    def add_student(self, student_id: str) -> None: ...

    @abstractmethod
    def get_student(self, student_id: str) -> Student | None: ...

    @abstractmethod
    def list_students(self) -> list[Student]: ...

    #--- questions ------------------------------------------------------
    @abstractmethod
    def add_question(self, **kwargs) -> None: ...

    @abstractmethod
    def get_question(self, q_ids: list | str) -> list[Question | None]: ...

    @abstractmethod
    def list_questions(self) -> list[Question]: ...

    @abstractmethod
    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        """Set derived/processed fields on a stored question.

        Attributes are applied to the QuestionAttributes sub-object.
        """

    #--- curriculum ---------------------------------------------------------
    @abstractmethod
    def add_unit(self, unit: Unit) -> None: ...
    @abstractmethod
    def get_unit(self, unit_id: str) -> Unit | None: ...

    @abstractmethod
    def add_lesson(self, lesson: Lesson) -> None: ...
    @abstractmethod
    def get_lesson(self, lesson_id: str) -> Lesson | None: ...

    @abstractmethod
    def add_learning_objective(self, lo: LearningObjective) -> None: ...
    @abstractmethod
    def get_learning_objective(self, lo_id: str) -> LearningObjective | None: ...

    #--- skills ---------------------------------------------------------
    @abstractmethod
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None: ...

    @abstractmethod
    def get_skill(self, skill_id: str) -> SkillCluster | None: ...

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

    #--- pre-hints --------------------------------------------------------
    @abstractmethod
    def add_pre_hint(self, pre_hint: PreHint) -> None: ...

    @abstractmethod
    def get_pre_hint(self, hint_id: str) -> PreHint | None: ...

    @abstractmethod
    def get_pre_hints_by_topic(self, topic: str) -> list[PreHint]: ...

    @abstractmethod
    def get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> list[PreHint]: ...

    @abstractmethod
    def get_pre_hints_by_diagnosis(self, diagnosis: str) -> list[PreHint]: ...

    @abstractmethod
    def list_pre_hints(self) -> list[PreHint]: ...

    #--- post-hints -------------------------------------------------------
    @abstractmethod
    def add_post_hint(self, post_hint: PostHint) -> None: ...

    @abstractmethod
    def get_post_hint(self, hint_id: str) -> PostHint | None: ...

    @abstractmethod
    def get_post_hints_by_question(self, question_id: str) -> list[PostHint]: ...

    @abstractmethod
    def list_post_hints(self) -> list[PostHint]: ...

    #--- play-hints -------------------------------------------------------
    @abstractmethod
    def add_play_hint(self, play_hint: PlayHint) -> None: ...

    @abstractmethod
    def get_play_hint(self, hint_id: str) -> PlayHint | None: ...

    @abstractmethod
    def get_play_hints_by_question(self, q_id: str) -> list[PlayHint]: ...

    @abstractmethod
    def list_play_hints(self) -> list[PlayHint]: ...
