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
    cognitive_load_index: Optional[int] = None
    variables_count: int = 0
    logical_steps: int = 0
    language_challenging: int = 0
    language_challenge_type: Optional[str] = None
    trick_question: bool = False
    time_pressure_flag: bool = False
    visual_dependency: bool = False
    real_world_context: bool = False
    multi_concept_flag: bool = False
    prerequisite_concept_ids: list[str] = Field(default_factory=list)
    cognitive_load: float = 0.0
    language_level: Optional[str] = None
    reasoning_level: Optional[str] = None
    num_unknowns: int = 1
    num_operations: int = 0
    p_t: float = 0.0
    p_s: float = 0.0
    p_g: float = 0.0


class QuestionIn(BaseModel):
    """One question in a batch. Supports nested structure from JSON."""
    model_config = {'extra': 'allow'}
    q_id: str
    question_type: Optional[str] = None
    question_text: Optional[str] = None
    question_media_url: Optional[str] = None
    question_language: str = "en"
    grade: Optional[int] = None
    grade_band: Optional[str] = None
    subject: Optional[str] = None
    unit_id: Optional[str] = None
    lesson_id: Optional[str] = None
    skill_cluster_id: Optional[str] = None
    skill_ids: list[str] = Field(default_factory=list)
    tag_source: Optional[str] = None
    tag_confidence_score: Optional[float] = None
    question_status: Optional[str] = None
    usage_count: int = 0
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


#--- students / misc ----------------------------------------------------

class StudentCreated(BaseModel):
    studentId: str