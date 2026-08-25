"""Pydantic request/response schemas - mirrors the API spec bodies."""
from __future__ import annotations
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


#--- questions ----------------------------------------------------------

class QuestionsQuery(BaseModel):
    allQs: str = "False"
    ids: list[str] = Field(default_factory=list)


class QuestionIn(BaseModel):
    """One question in a batch. Accepts entity-constructor field names."""
    model_config = {'extra': 'allow'}
    q_id: str
    skill_cluster_id: Optional[str] = None
    unit_id: Optional[str] = None
    difficulty_level: float = 0.0
    p_t: float = 0.0
    p_s: float = 0.0
    p_g: float = 0.0
    question_text: Optional[str] = None
    correct_answer_content: Optional[str] = None


class QuestionBatch(BaseModel):
    Questions: list[QuestionIn]


class BatchResult(BaseModel):
    NumInserted: int


class QuestionsOut(BaseModel):
    Questions: list[dict[str, Any]]


#--- diagnostics --------------------------------------------------------

class Answer(BaseModel):
    questionId: str
    response: Any  # 1/0 correct-incorrect (or richer payload)
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
    status: str  # processing | completed | failed
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
    answerTag: Optional[str] = None


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
