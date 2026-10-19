"""Pydantic request/response schemas - mirrors the API spec bodies."""
from __future__ import annotations
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuestionsQuery(BaseModel):
    allQs: str = "False"
    ids: list[str] = Field(default_factory=list)


class OptionIn(BaseModel):
    option_id: str = ""
    content: str = ""
    is_correct: bool = False
    display_order: int = 0
    content_gap_type: Optional[str] = None
    distractor_reading_trap: bool = False
    distractor_impulsive: bool = False
    prior_skill_reference: list[str] = Field(default_factory=list)


class QuestionAttributesIn(BaseModel):
    difficulty_level: float = 0.0
    bloom_taxonomy_level: Optional[str] = None
    variables_count: int = 0
    logical_steps: int = 0
    trick_question: bool = False
    visual_dependency: bool = False
    real_world_context: bool = False
    multi_concept_flag: bool = False
    prerequisite_concept_ids: list[str] = Field(default_factory=list)


class QuestionIn(BaseModel):
    """One question in a batch. Supports nested structure from JSON."""
    model_config = {'extra': 'allow'}
    q_id: str
    question_type: Optional[str] = None
    question_text: Optional[str] = None
    grade: Optional[int] = None
    subject: Optional[str] = None
    curriculum=Optional[str]=None
    unit_id: Optional[str] = None
    lesson_id: Optional[str] = None
    skill_cluster_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    tag_source: Optional[str] = None
    tag_confidence_score: Optional[float] = None
    question_status: Optional[str] = None
    options: list[OptionIn] = Field(default_factory=list)
    attrs: Optional[QuestionAttributesIn] = None


class QuestionBatch(BaseModel):
    Questions: list[QuestionIn]


class BatchResult(BaseModel):
    NumInserted: int


class QuestionsOut(BaseModel):
    Questions: list[dict[str, Any]]


#--- diagnostics --------------------------------------------------------

class Answer(BaseModel):
    questionId: str
    response: Any
    responseTimeSeconds: float = 0.0
    stressTrigger: int = 0
    answerTag: Optional[str] = None


class DiagnosticsRun(BaseModel):
    studentId: str
    Answers: list[Answer]


class DiagnosticsAccepted(BaseModel):
    jobId: str
    status: str


class DiagnosticsStatus(BaseModel):
    status: str
    result: Optional[dict[str, Any]] = None


#--- matches ------------------------------------------------------------

class MatchCreate(BaseModel):
    studentId: str


class MatchCreated(BaseModel):
    matchId: str
    questions: list[dict[str, Any]]


class MatchAnswer(BaseModel):
    questionId: str
    response: Any
    responseTimeSeconds: float = 0.0
    stressTrigger: int = 0
    timeout:bool
    hintsUsed:int
    date:str


class AnswerAccepted(BaseModel):
    accepted: bool


class MatchCompleteResult(BaseModel):
    deltaMastery: dict[str, float]
    updatedPriors: dict[str, float]
    behavioralScores: list[dict[str, Any]]
    updatedTreatmentPlan: list[dict[str, Any]]


#--- curriculum ------------------------------------------------------------

class UnitIn(BaseModel):
    unit_id: str
    language: str = "en"
    curriculum: str = ""
    grade: str = ""
    unit_name: str = ""
    unit_content: str = ""

class LessonIn(BaseModel):
    lesson_id: str
    unit_id: str
    lesson_name: str = ""
    lesson_content: str = ""

class LearningObjectiveIn(BaseModel):
    learning_objective_id: str
    lesson_id: str
    learning_objective_content: str = ""

class SkillIn(BaseModel):
    skill_cluster_id: str
    lesson_id: str
    unit_id: str
    learning_objective_id: Optional[str] = None
    skill_cluster_content: str = ""
    similar_skills: list[str] = Field(default_factory=list)

class CurriculumCreated(BaseModel):
    id: str

#--- curriculum ------------------------------------------------------------

class UnitIn(BaseModel):
    unit_id: str
    language: str = "en"
    curriculum: str = ""
    grade: str = ""
    unit_name: str = ""
    unit_content: str = ""

class LessonIn(BaseModel):
    lesson_id: str
    unit_id: str
    lesson_name: str = ""
    lesson_content: str = ""

class LearningObjectiveIn(BaseModel):
    learning_objective_id: str
    lesson_id: str
    learning_objective_content: str = ""

class SkillIn(BaseModel):
    skill_cluster_id: str
    lesson_id: str
    unit_id: str
    learning_objective_id: Optional[str] = None
    skill_cluster_content: str = ""
    similar_skills: list[str] = Field(default_factory=list)

class CurriculumCreated(BaseModel):
    id: str
