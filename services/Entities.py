from __future__ import annotations
from typing import Any, List, Optional
from sqlalchemy import String, Float, Integer, Boolean, ForeignKey, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from services.Interfaces import CrystalText 

class Base(DeclarativeBase):
    pass

class Unit(Base):
    __tablename__ = "units"
    unit_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    language: Mapped[str] = mapped_column(String(10), default="en")
    curriculum: Mapped[str] = mapped_column(String(100), default="")
    grade: Mapped[str] = mapped_column(String(20), default="")
    unit_name: Mapped[str] = mapped_column(String(255), default="")
    unit_content: Mapped[str] = mapped_column(String(1000), default="")

    def __init__(
        self,
        unit_id: str = "",
        language: str = "en",
        curriculum: str = "",
        grade: str = "",
        unit_name: str = "",
        unit_content: str = "",
    ) -> None:
        self.unit_id = unit_id
        self.language = language
        self.curriculum = curriculum
        self.grade = grade
        self.unit_name = unit_name
        self.unit_content = unit_content

    def __repr__(self) -> str:
        return f"Unit({self.unit_id!r}, name={self.unit_name!r})"


class Lesson(Base):
    __tablename__ = "lessons"
    lesson_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    unit_id: Mapped[str] = mapped_column(ForeignKey("units.unit_id"))
    lesson_name: Mapped[str] = mapped_column(String(255), default="")
    lesson_content: Mapped[str] = mapped_column(String(1000), default="")

    def __init__(
        self,
        lesson_id: str = "",
        unit_id: str = "",
        lesson_name: str = "",
        lesson_content: str = "",
    ) -> None:
        self.lesson_id = lesson_id
        self.unit_id = unit_id
        self.lesson_name = lesson_name
        self.lesson_content = lesson_content

    def __repr__(self) -> str:
        return f"Lesson({self.lesson_id!r}, name={self.lesson_name!r})"


class LearningObjective(Base):
    __tablename__ = "learning_objectives"
    learning_objective_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    lesson_id: Mapped[str] = mapped_column(ForeignKey("lessons.lesson_id"))
    learning_objective_content: Mapped[str] = mapped_column(String(1000), default="")

    def __init__(
        self,
        learning_objective_id: str = "",
        lesson_id: str = "",
        learning_objective_content: str = "",
    ) -> None:
        self.learning_objective_id = learning_objective_id
        self.lesson_id = lesson_id
        self.learning_objective_content = learning_objective_content

    def __repr__(self) -> str:
        return f"LearningObjective({self.learning_objective_id!r})"


class SkillCluster(Base):
    __tablename__ = "skill_clusters"
    skill_cluster_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    lesson_id: Mapped[str] = mapped_column(ForeignKey("lessons.lesson_id"))
    unit_id: Mapped[str] = mapped_column(ForeignKey("units.unit_id"))
    learning_objective_id: Mapped[Optional[str]] = mapped_column(ForeignKey("learning_objectives.learning_objective_id"))
    skill_cluster_content: Mapped[str] = mapped_column(String(1000), default="")
    
    def __init__(
        self,
        skill_cluster_id: str = "",
        lesson_id: str = "",
        unit_id: str = "",
        learning_objective_id: str | None = None,
        skill_cluster_content: str = "",
        similar_skills: list[str] | None = None,
    ) -> None:
        self.skill_cluster_id = skill_cluster_id
        self.id = skill_cluster_id
        self.lesson_id = lesson_id
        self.unit_id = unit_id
        self.learning_objective_id = learning_objective_id
        self.skill_cluster_content = skill_cluster_content
        self.similar = similar_skills or []

    def get_similar(self) -> list[str]:
        return self.similar

    def __repr__(self) -> str:
        return f"SkillCluster({self.skill_cluster_id!r})"

# Association table for SkillCluster similarities
skill_similarities = Table(
    "skill_similarities",
    Base.metadata,
    Column("skill_id", String(50), ForeignKey("skill_clusters.skill_cluster_id"), primary_key=True),
    Column("similar_skill_id", String(50), ForeignKey("skill_clusters.skill_cluster_id"), primary_key=True),
)

class Option(Base):
    __tablename__ = "options"
    option_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.q_id"))
    content: Mapped[str] = mapped_column(String(1000), default="")
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    content_gap_type: Mapped[Optional[str]] = mapped_column(String(100))
    distractor_reading_trap: Mapped[bool] = mapped_column(Boolean, default=False)
    distractor_impulsive: Mapped[bool] = mapped_column(Boolean, default=False)
    
    def __init__(
        self,
        option_id: str = "",
        content: str = "",
        is_correct: bool = False,
        display_order: int = 0,
        content_gap_type: str | None = None,
        distractor_reading_trap: bool = False,
        distractor_impulsive: bool = False,
        prior_skill_reference: list[str] | None = None,
    ) -> None:
        self.option_id = option_id
        self.content = content
        self.is_correct = is_correct
        self.display_order = display_order
        self.content_gap_type = content_gap_type
        self.distractor_reading_trap = distractor_reading_trap
        self.distractor_impulsive = distractor_impulsive
        self.prior_skill_reference = prior_skill_reference or []

    def __repr__(self) -> str:
        return f"Option({self.option_id!r}, correct={self.is_correct})"


class QuestionAttributes(Base):
    __tablename__ = "question_attributes"
    q_id: Mapped[str] = mapped_column(ForeignKey("questions.q_id"), primary_key=True)
    difficulty_level: Mapped[float] = mapped_column(Float, default=0.0)
    bloom_taxonomy_level: Mapped[Optional[str]] = mapped_column(String(50))
    population_difficulty: Mapped[Optional[float]] = mapped_column(Float)
    discrimination_index: Mapped[Optional[float]] = mapped_column(Float)
    time_allowed: Mapped[Optional[float]] = mapped_column(Float)
    theoretical_solving_time: Mapped[Optional[float]] = mapped_column(Float)
    dynamic_avg_time: Mapped[Optional[float]] = mapped_column(Float)
    cognitive_load_index: Mapped[Optional[int]] = mapped_column(Integer)
    variables_count: Mapped[int] = mapped_column(Integer, default=0)
    logical_steps: Mapped[int] = mapped_column(Integer, default=0)
    language_challenging: Mapped[int] = mapped_column(Integer, default=0)
    language_challenge_type: Mapped[Optional[str]] = mapped_column(String(100))
    trick_question: Mapped[bool] = mapped_column(Boolean, default=False)
    visual_dependency: Mapped[bool] = mapped_column(Boolean, default=False)
    real_world_context: Mapped[bool] = mapped_column(Boolean, default=False)
    multi_concept_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    cognitive_load: Mapped[float] = mapped_column(Float, default=0.0)
    language_level: Mapped[Optional[str]] = mapped_column(String(20))
    reasoning_level: Mapped[Optional[str]] = mapped_column(String(20))
    p_t: Mapped[float] = mapped_column(Float, default=0.0)
    p_s: Mapped[float] = mapped_column(Float, default=0.0)
    p_g: Mapped[float] = mapped_column(Float, default=0.0)

    def __init__(
        self,
        difficulty_level: float = 0.0,
        bloom_taxonomy_level: str | None = None,
        population_difficulty: float | None = None,
        discrimination_index: float | None = None,
        time_allowed: float | None = None,
        theoretical_solving_time: float | None = None,
        dynamic_avg_time: float | None = None,
        cognitive_load_index: int | None = None,
        variables_count: int = 0,
        logical_steps: int = 0,
        language_challenging: int = 0,
        language_challenge_type: str | None = None,
        trick_question: bool = False,
        visual_dependency: bool = False,
        real_world_context: bool = False,
        multi_concept_flag: bool = False,
        prerequisite_concept_ids: list[str] | None = None,
        cognitive_load: float = 0.0,
        language_level: str | None = None,
        reasoning_level: str | None = None,
        p_t: float = 0.0,
        p_s: float = 0.0,
        p_g: float = 0.0,
    ) -> None:
        self.difficulty_level = difficulty_level
        self.bloom_taxonomy_level = bloom_taxonomy_level
        self.population_difficulty = population_difficulty
        self.discrimination_index = discrimination_index
        self.time_allowed = time_allowed
        self.theoretical_solving_time = theoretical_solving_time
        self.dynamic_avg_time = dynamic_avg_time
        self.cognitive_load_index = cognitive_load_index
        self.variables_count = variables_count
        self.logical_steps = logical_steps
        self.language_challenging = language_challenging
        self.language_challenge_type = language_challenge_type
        self.trick_question = trick_question
        self.visual_dependency = visual_dependency
        self.real_world_context = real_world_context
        self.multi_concept_flag = multi_concept_flag
        self.cognitive_load = cognitive_load
        self.language_level = language_level
        self.reasoning_level = reasoning_level
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g

    def get_atts(self) -> dict[str, Any]:
            return {
                "difficulty_level": self.difficulty_level,
                "bloom_tax": self.bloom_taxonomy_level,
                "theoretical_time": self.theoretical_solving_time,
                "variables_count": self.variables_count,
                "logical_steps": self.logical_steps,
                "cognitive_load_index": self.cognitive_load_index,
                "reasoning_level": self.reasoning_level,
                "language_level": self.language_level,
                "language_challenge_type": self.language_challenge_type,
                "p_g": self.p_g,
                "p_t": self.p_t,
                "p_s": self.p_s,
            }
    def __repr__(self) -> str:
        return f"QuestionAttributes(diff={self.difficulty_level}, load={self.cognitive_load_index})"


class Question(Base):
    __tablename__ = "questions"
    q_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    question_type: Mapped[str] = mapped_column(String(100), default="")
    question_text: Mapped[str] = mapped_column(String(2000), default="")
    question_media_url: Mapped[Optional[str]] = mapped_column(String(500))
    question_language: Mapped[str] = mapped_column(String(10), default="en")
    grade: Mapped[Optional[int]] = mapped_column(Integer)
    grade_band: Mapped[Optional[str]] = mapped_column(String(20))
    subject: Mapped[Optional[str]] = mapped_column(String(100))
    unit_id: Mapped[Optional[str]] = mapped_column(ForeignKey("units.unit_id"))
    lesson_id: Mapped[Optional[str]] = mapped_column(ForeignKey("lessons.lesson_id"))
    skill_cluster_id: Mapped[Optional[str]] = mapped_column(ForeignKey("skill_clusters.skill_cluster_id"))
    tag_source: Mapped[Optional[str]] = mapped_column(String(100))
    tag_confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    question_status: Mapped[Optional[str]] = mapped_column(String(50))
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    date_created: Mapped[Optional[str]] = mapped_column(String(50))
    last_updated: Mapped[Optional[str]] = mapped_column(String(50))

    options = relationship("Option", backref="question", cascade="all, delete-orphan")
    attrs = relationship("QuestionAttributes", backref="question", uselist=False, cascade="all, delete-orphan")

    def __init__(
        self,
        q_id: str = "",
        question_type: str = "",
        question_text: str = "",
        question_media_url: str | None = None,
        question_language: str = "en",
        language: str = "en",
        grade: int | None = None,
        grade_band: str | None = None,
        subject: str | None = None,
        unit_id: str | None = None,
        lesson_id: str | None = None,
        skill_cluster_id: str | None = None,
        skill_ids: list[str] | None = None,
        tag_source: str | None = None,
        tag_confidence_score: float | None = None,
        question_status: str | None = None,
        usage_count: int = 0,
        date_created: str | None = None,
        last_updated: str | None = None,
        options: list[Option] | None = None,
        attrs: QuestionAttributes | None = None,
        **kwargs,
    ) -> None:
        self.q_id = q_id
        self.question_type = question_type
        self.question_text = question_text
        self.question_media_url = question_media_url
        self.question_language = question_language
        self.language = question_language
        self.grade = grade
        self.grade_band = grade_band
        self.subject = subject
        self.unit_id = unit_id
        self.lesson_id = lesson_id
        self.skill_cluster_id = skill_cluster_id
        self.skill_ids = skill_ids or ([skill_cluster_id] if skill_cluster_id is not None else [])
        self.tag_source = tag_source
        self.tag_confidence_score = tag_confidence_score
        self.question_status = question_status
        self.usage_count = usage_count
        self.date_created = date_created
        self.last_updated = last_updated
        self.options = []
        if options:
            for opt in options:
                if isinstance(opt, dict):
                    self.options.append(Option(**opt))
                elif isinstance(opt, Option):
                    self.options.append(opt)
                else:
                    self.options.append(Option(**vars(opt)))
        self.attrs = attrs or QuestionAttributes()
        if isinstance(attrs, dict):
            self.attrs = QuestionAttributes(**attrs)
        for key, val in kwargs.items():
            if hasattr(self.attrs, key):
                setattr(self.attrs, key, val)

    def __getattr__(self, name: str):
        attr_fields = {
            "difficulty_level", "bloom_taxonomy_level", "population_difficulty",
            "discrimination_index", "time_allowed", "theoretical_solving_time",
            "dynamic_avg_time", "cognitive_load_index", "variables_count",
            "logical_steps", "language_challenging", "language_challenge_type",
            "trick_question", "visual_dependency", "real_world_context",
            "multi_concept_flag", "cognitive_load", "language_level",
            "reasoning_level", "p_t", "p_s", "p_g",
        }
        if name in attr_fields and self.attrs:
            return getattr(self.attrs, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")
    
    def get_atts(self) -> dict:
        atts = self.attrs.get_atts()
        atts["question_text"] = self.question_text
        return atts

    def get_unit(self) -> str | None:
        return self.unit_id

    def get_skills(self) -> list[str]:
        return self.skill_ids

    @property
    def correct_answer_content(self) -> str | None:
        for opt in self.options:
            if opt.is_correct:
                return opt.content
        return None

    @property
    def correct_answer_id(self) -> str | None:
        for opt in self.options:
            if opt.is_correct:
                return opt.option_id
        return None

    def add_option(self, option: Option) -> None:
        self.options.append(option)

    def __repr__(self) -> str:
        return f"Question({self.q_id!r}, type={self.question_type!r}, options={len(self.options)})"


class PreHint(Base, CrystalText):
    __tablename__ = "pre_hints"
    hint_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    hint_text: Mapped[str] = mapped_column(String(2000))
    topic: Mapped[Optional[str]] = mapped_column(String(100))
    content_gap_type: Mapped[Optional[str]] = mapped_column(String(100))
    diagnosis: Mapped[Optional[str]] = mapped_column(String(100))
    hint_type: Mapped[str] = mapped_column(String(50), default="content_gap")

    def __init__(
        self,
        hint_id: str,
        hint_text: str,
        topic: str | None = None,
        content_gap_type: str | None = None,
        diagnosis: str | None = None,
        hint_type: str = "content_gap",
    ) -> None:
        super().__init__(hint_text, hint_id)
        self.topic = topic
        self.content_gap_type = content_gap_type
        self.diagnosis = diagnosis
        self.hint_type = hint_type

    def get_topic(self) -> str | None:
        return self.topic

    def get_content_gap_type(self) -> str | None:
        return self.content_gap_type

    def get_diagnosis(self) -> str | None:
        return self.diagnosis

    def get_hint_type(self) -> str:
        return self.hint_type


class PostHint(Base, CrystalText):
    __tablename__ = "post_hints"
    hint_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    hint_text: Mapped[str] = mapped_column(String(2000))
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.q_id"))
    hint_order: Mapped[int] = mapped_column(Integer, default=0)

    def __init__(
        self,
        hint_id: str,
        hint_text: str,
        question_id: str,
        hint_order: int = 0,
    ) -> None:
        super().__init__(hint_text, hint_id)
        self.question_id = question_id
        self.hint_order = hint_order

    def get_question_id(self) -> str:
        return self.question_id

    def get_hint_order(self) -> int:
        return self.hint_order


class PlayHint(Base, CrystalText):
    __tablename__ = "play_hints"
    hint_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    hint_text: Mapped[str] = mapped_column(String(2000))
    q_id: Mapped[str] = mapped_column(ForeignKey("questions.q_id"))
    hint_order: Mapped[int] = mapped_column(Integer, default=0)

    def __init__(
        self,
        hint_id: str,
        hint_text: str,
        q_id: str,
        hint_order: int = 0,
    ) -> None:
        super().__init__(hint_text, hint_id)
        self.q_id = q_id
        self.hint_order = hint_order

    def get_q_id(self) -> str:
        return self.q_id

    def get_hint_order(self) -> int:
        return self.hint_order


class Match(Base):
    __tablename__ = "matches"
    match_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    created_timestamp: Mapped[str] = mapped_column(String(50), default="")
    match_type: Mapped[str] = mapped_column(String(100), default="")
    num_participants: Mapped[int] = mapped_column(Integer, default=1)

    def __init__(
        self,
        match_id: str = "",
        created_timestamp: str = "",
        match_type: str = "",
        num_participants: int = 1,
    ) -> None:
        self.match_id = match_id
        self.created_timestamp = created_timestamp
        self.match_type = match_type
        self.num_participants = num_participants

    def __repr__(self) -> str:
        return f"Match({self.match_id!r}, type={self.match_type!r})"


class Response(Base):
    __tablename__ = "responses"
    response_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String(50), ForeignKey("matches.match_id"))
    student_id: Mapped[str] = mapped_column(String(50), ForeignKey("students.student_id"))
    question_id: Mapped[str] = mapped_column(String(50), ForeignKey("questions.q_id"))
    stress_triggers: Mapped[int] = mapped_column(Integer, default=0)
    option_id: Mapped[Optional[str]] = mapped_column(String(50))
    error: Mapped[bool] = mapped_column(Boolean, default=False)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    time_out: Mapped[bool] = mapped_column(Boolean, default=False)
    response_time: Mapped[float] = mapped_column(Float, default=0.0)

    def __init__(
        self,
        match_id: str = "",
        student_id: str = "",
        question_id: str = "",
        stress_triggers: int = 0,
        option_id: str | None = None,
        error: bool = False,
        hints_used: int = 0,
        time_out: bool = False,
        response_time: float = 0.0,
    ) -> None:
        self.match_id = match_id
        self.student_id = student_id
        self.question_id = question_id
        self.stress_triggers = stress_triggers
        self.option_id = option_id
        self.error = error
        self.hints_used = hints_used
        self.time_out = time_out
        self.response_time = response_time

    def __repr__(self) -> str:
        return f"Response(m={self.match_id!r}, s={self.student_id!r}, q={self.question_id!r})"


class StudentClass(Base):
    __tablename__ = "student_classes"
    student_id: Mapped[str] = mapped_column(String(50), ForeignKey("students.student_id"), primary_key=True)
    class_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    def __init__(self, student_id: str = "", class_id: str = "") -> None:
        self.student_id = student_id
        self.class_id = class_id

    def __repr__(self) -> str:
        return f"StudentClass(s={self.student_id!r}, c={self.class_id!r})"


class StudentMasteryLevel(Base):
    __tablename__ = "student_mastery_levels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(50), default="")
    current: Mapped[bool] = mapped_column(Boolean, default=True)
    student_id: Mapped[str] = mapped_column(String(50), ForeignKey("students.student_id"))
    skill_cluster_id: Mapped[str] = mapped_column(String(50), ForeignKey("skill_clusters.skill_cluster_id"))
    mastery: Mapped[float] = mapped_column(Float, default=0.0)
    content_gap: Mapped[bool] = mapped_column(Boolean, default=False)
    content_gap_type: Mapped[Optional[str]] = mapped_column(String(100))

    def __init__(
        self, 
        timestamp: str="",
        current:bool =True,
        student_id: str = "",
        skill_cluster_id: str = "",
        mastery: float = 0.0,
        content_gap: bool = False,
        content_gap_type: str | None = None,
    ) -> None:
        self.student_id = student_id
        self.skill_cluster_id = skill_cluster_id
        self.mastery = mastery
        self.content_gap = content_gap
        self.content_gap_type = content_gap_type
        self.timestamp=timestamp
        self.current=True

    def __repr__(self) -> str:
        return f"StudentMasteryLevel(s={self.student_id!r}, skill={self.skill_cluster_id!r}, m={self.mastery})"

class Treatment(Base):
    __tablename__ = "Treatment"

    student_id = Column("StudentID", String(64), ForeignKey("Student.StudentID"), primary_key=True)
    diagnosis_id = Column("DiagnosisID", String(64), primary_key=True)
    question_attribute = Column("QuestionAttribute", String(64), primary_key=True)
    value = Column("Value", String(128), nullable=False)
    query_operator = Column("QueryOperator", String(16), nullable=False, default="=")


class DiagnosticThreshold(Base):
    __tablename__ = "diagnostic_thresholds"
    diagnosis_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    diagnosis_name: Mapped[str] = mapped_column(String(255), default="")
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[str] = mapped_column(String(50), default="")
    current: Mapped[bool] = mapped_column(Boolean, default=True)

    def __init__(
        self,
        diagnosis_id: str = "",
        diagnosis_name: str = "",
        threshold: float = 0.0,
        timestamp:str="",
        current:bool =True
    ) -> None:
        self.diagnosis_id = diagnosis_id
        self.diagnosis_name = diagnosis_name
        self.threshold = threshold
        self.timestamp=timestamp
        self.current=current

    def __repr__(self) -> str:
        return f"DiagnosticThreshold({self.diagnosis_id!r}, name={self.diagnosis_name!r})"


class StudentDiagnostic(Base):
    __tablename__ = "student_diagnostics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[str] = mapped_column(String(50), ForeignKey("students.student_id"))
    diagnosis_id: Mapped[str] = mapped_column(String(50))
    delta: Mapped[float] = mapped_column(Float, default=0.0)

    def __init__(
        self,
        student_id: str = "",
        diagnosis_id: str = "",
        delta: float = 0.0,
    ) -> None:
        self.student_id = student_id
        self.diagnosis_id = diagnosis_id
        self.delta = delta

    def __repr__(self) -> str:
        return f"StudentDiagnostic(s={self.student_id!r}, d={self.diagnosis_id!r}, delta={self.delta})"


class Student(Base):
    __tablename__ = "students"
    student_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    grade_band: Mapped[str] = mapped_column(String(20), default="")
    curriculum: Mapped[str] = mapped_column(String(100), default="")

    def __init__(
        self,
        student_id: str = "",
        grade_band: str = "",
        curriculum: str = "",
    ) -> None:
        self.student_id = student_id
        self.id = student_id
        self.grade_band = grade_band
        self.curriculum = curriculum
        
        # In-memory tracking structures for runtime algorithms
        # These will now be backed by queries to related tables
        self.priors: dict[str, float] = {}
        self.responses: list[Any] = []
        self.diagnoses: list[str] = []
        self.deltas_history: dict[str, dict] = {}
        self.diagnoses_history: dict[str, list] = {}
        self.priors_history: dict[str, dict] = {}
        self.treatment_plan: dict[str, tuple[str, dict]] = {}
        self.content_gap_types: dict[str, str] = {}

    def update_prior(self, kc_id: str, new_prior: float) -> None:
        self.priors[kc_id] = new_prior

    def add_response(
        self,
        match_id: str,
        date: str,
        q_id: str,
        response: int | str,
        error: bool,
        timeout: bool,
        time: float | None,
        stress_triggers: list | int | None,
        hints_used: int,
        atag: str | None,
    ) -> None:
        self.responses.append((match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag))

    def add_diagnosis(self, diagnosis: str) -> None:
        self.diagnoses.append(diagnosis)

    def add_diagnosis_record(self, timestamp: str, deltas: dict, diagnoses: list) -> None:
        self.deltas_history[timestamp] = deltas
        self.diagnoses_history[timestamp] = diagnoses

    def add_priors_record(self, timestamp: str, priors: dict) -> None:
        self.priors_history[timestamp] = dict(priors)

    def get_priors_history(self) -> dict:
        return self.priors_history

    def add_treatment_plan(self, timestamp: str, treatment_name: str, parameters: dict) -> None:
        self.treatment_plan[treatment_name] = (timestamp, parameters)

    def update_treatment_plan(self, timestamp: str, treatment_name: str, parameters: dict) -> None:
        self.treatment_plan[treatment_name] = (timestamp, parameters)

    def get_treatment_plan(self) -> dict:
        return self.treatment_plan

    def update_content_gaps(self, content_gap_types: dict) -> None:
        self.content_gap_types = dict(content_gap_types)

    def get_content_gaps(self) -> dict:
        return self.content_gap_types

    def get_responses(self) -> list:
        return self.responses

    def get_priors(self) -> dict:
        return self.priors

    def get_diagnoses(self) -> list:
        return self.diagnoses

    def get_deltas(self) -> dict:
        return self.deltas_history

    def get_diagnoses_history(self) -> dict:
        return self.diagnoses_history

    def __repr__(self) -> str:
        return f"Student({self.student_id!r}, grade={self.grade_band!r})"
