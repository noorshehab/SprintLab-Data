from __future__ import annotations
from services.Interfaces import CrystalText 


class Option:
    def __init__(
        self,
        option_id: str = "",
        content: str = "",
        is_correct: bool = False,
        display_order: int = 0,
        answer_media_url: str | None = None,
        content_gap_type: str | None = None,
        distractor_reading_trap: bool = False,
        distractor_impulsive: bool = False,
        prior_skill_reference: list[str] | None = None,
    ) -> None:
        self.option_id = option_id
        self.content = content
        self.is_correct = is_correct
        self.display_order = display_order
        self.answer_media_url = answer_media_url
        self.content_gap_type = content_gap_type
        self.distractor_reading_trap = distractor_reading_trap
        self.distractor_impulsive = distractor_impulsive
        self.prior_skill_reference = prior_skill_reference or []

    def __repr__(self) -> str:
        return f"Option({self.option_id!r}, correct={self.is_correct})"


class QuestionAttributes:
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
        time_pressure_flag: bool = False,
        visual_dependency: bool = False,
        real_world_context: bool = False,
        multi_concept_flag: bool = False,
        prerequisite_concept_ids: list[str] | None = None,
        cognitive_load: float = 0.0,
        language_level: str | None = None,
        reasoning_level: str | None = None,
        num_unknowns: int = 1,
        num_operations: int = 0,
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
        self.time_pressure_flag = time_pressure_flag
        self.visual_dependency = visual_dependency
        self.real_world_context = real_world_context
        self.multi_concept_flag = multi_concept_flag
        self.prerequisite_concept_ids = prerequisite_concept_ids or []
        self.cognitive_load = cognitive_load
        self.language_level = language_level
        self.reasoning_level = reasoning_level
        self.num_unknowns = num_unknowns
        self.num_operations = num_operations
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g

    def get_atts(self) -> dict:
        return {
            "time_allowed": self.time_allowed,
            "difficulty_level": self.difficulty_level,
            "language_level": self.language_level,
            "cognitive_load": self.cognitive_load,
            "variables_count": self.variables_count,
            "logical_steps": self.logical_steps,
            "time_pressure_flag": self.time_pressure_flag,
            "language_challenging": self.language_challenging,
            "reasoning_level": self.reasoning_level,
            "num_unknowns": self.num_unknowns,
            "num_operations": self.num_operations,
            "cognitive_load_index": self.cognitive_load_index,
        }

    def __repr__(self) -> str:
        return f"QuestionAttributes(diff={self.difficulty_level}, load={self.cognitive_load_index})"


class PreHint(CrystalText):
    """General hints shown before the match.

    Can be for content gaps (topic/skill_cluster + content_gap_type)
    or for behavioral/cognitive/exam skill diagnoses (diagnosis identifier).
    """
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


class PostHint(CrystalText):
    """Hint specific to a question, has an attached question_id."""
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


class PlayHint(CrystalText):
    """Hint specific to a question, has an attached question_id (q_id)."""
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


class Question:
    # attrs that live on QuestionAttributes but can be passed here for convenience
    _ATTRS_FIELDS = {
        "difficulty_level", "bloom_taxonomy_level", "population_difficulty",
        "discrimination_index", "time_allowed", "theoretical_solving_time",
        "dynamic_avg_time", "cognitive_load_index", "variables_count",
        "logical_steps", "language_challenging", "language_challenge_type",
        "trick_question", "time_pressure_flag", "visual_dependency",
        "real_world_context", "multi_concept_flag", "prerequisite_concept_ids",
        "cognitive_load", "language_level", "reasoning_level", "num_unknowns",
        "num_operations", "p_t", "p_s", "p_g",
    }

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
        self.id = q_id
        self.question_type = question_type
        self.question_text = question_text
        self.question_media_url = question_media_url
        self.question_language = question_language
        self.language = language
        self.language = question_language
        self.grade = grade
        self.grade_band = grade_band
        self.subject = subject
        self.unit_id = unit_id
        self.lesson_id = lesson_id
        self.skill_cluster_id = skill_cluster_id
        self.skill_ids = skill_ids or (
            [skill_cluster_id] if skill_cluster_id is not None else []
        )
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
        # forward attrs-level kwargs for backward compatibility
        for key, val in kwargs.items():
            if key in self._ATTRS_FIELDS:
                setattr(self.attrs, key, val)

    # --- proxy to QuestionAttributes so existing callers work ---

    def __getattr__(self, name: str):
        if name in (
            "difficulty_level",
            "bloom_taxonomy_level",
            "population_difficulty",
            "discrimination_index",
            "time_allowed",
            "theoretical_solving_time",
            "dynamic_avg_time",
            "cognitive_load_index",
            "variables_count",
            "logical_steps",
            "language_challenging",
            "language_challenge_type",
            "trick_question",
            "time_pressure_flag",
            "visual_dependency",
            "real_world_context",
            "multi_concept_flag",
            "prerequisite_concept_ids",
            "cognitive_load",
            "language_level",
            "reasoning_level",
            "num_unknowns",
            "num_operations",
            "p_t",
            "p_s",
            "p_g",
        ):
            return getattr(self.attrs, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

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
        return f"Question({self.id!r}, type={self.question_type!r}, options={len(self.options)})"


class student():
    def __init__(self, student_id: str) -> None:
        self.id = student_id
        self.priors = {}  # {skill_id: prior}
        self.responses = []  # list of (match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag)
        self.diagnoses = []
        self.deltas_history = {}  # {timestamp: {metric: delta}}
        self.diagnoses_history = {}  # {timestamp: [diagnoses]}
        self.priors_history = {}  # {timestamp: {skill_id: prior}}
        self.treatment_plan = {}
        self.content_gap_types = {}  # skill_id:gap_type

    def update_prior(self, kc_id: str, new_prior: float) -> None:
        self.priors[kc_id] = new_prior

    def add_response(self, match_id: str, date: str, q_id: str, response: int, error: bool, timeout: bool, time: float | None, stress_triggers: list | None, hints_used: int, atag: str | None) -> None:
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


class skill():
    def __init__(self, skill_id: str, similar_skills: list) -> None:
        self.id = skill_id
        self.similar = similar_skills

    def get_similar(self) -> list:
        return self.similar
