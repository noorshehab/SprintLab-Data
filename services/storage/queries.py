from __future__ import annotations
from typing import Any
from sqlalchemy import Select, and_, text, TextClause, bindparam

from services.Entities import (
    PreHint,
    PostHint,
    PlayHint,
    Unit,
    Lesson,
    LearningObjective,
)

# Mapping from treatment/query dictionary keys to QuestionDiagnostics SQL columns
ATTRIBUTE_COLUMN_MAP = {
    "language_level": "qd.LanguageLevel",
    "cognitive_load": "qd.CognitiveLoadIndex",
    "cognitive_load_index": "qd.CognitiveLoadIndex",
    "bloom_taxonomy_level": "qd.BloomTax",
    "bloom_tax": "qd.BloomTax",
    "visual_dependency": "qd.VisualDependancy",
    "logical_steps": "qd.LogicalSteps",
    "multi_concept_flag": "qd.MultiConceptFlag",
    "difficulty_level": "qd.DifficultyLevel",
    "variables_count": "qd.VariablesCount",
    "trick_question": "qd.TrickQuestion",
    "real_world_context": "qd.RealWorldContext",
    "reasoning_level": "qd.ReasoningLevel",
    "time_pressure_flag": "qd.TheoreticalTime",
    "theoretical_time": "qd.TheoreticalTime",
}

OPERATOR_MAP = {
    "==": "=",
    "=": "=",
    "!=": "!=",
    "<=": "<=",
    ">=": ">=",
    "<": "<",
    ">": ">",
    "in": "IN",
    "IN": "IN",
}

_QUESTION_SELECT = """
    SELECT qc.QuestionID, qc.SkillClusterID, qc.QuestionText, qc.QuestionType,
           qd.BloomTax, qd.TheoreticalTime, qd.VariablesCount, qd.LogicalSteps,
           qd.TrickQuestion, qd.VisualDependancy, qd.RealWorldContext,
           qd.MultiConceptFlag, qd.CognitiveLoadIndex, qd.LanguageChallengeType,
           qd.ReasoningLevel, qd.LanguageLevel, qd.DifficultyLevel,
           qd.p_g, qd.p_t, qd.p_s
    FROM QuestionContent qc
    LEFT JOIN QuestionDiagnostics qd
           ON qd.QuestionID = qc.QuestionID AND qd.Current = TRUE
    WHERE qc.Current = TRUE
"""


class QueryBuilder:
    # =========================================================================
    # 1. Students
    # =========================================================================

    def build_get_student(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT StudentID, GradeBand, Curriculum
            FROM Student
            WHERE StudentID = :student_id;
        """)
        return sql, {"student_id": student_id}

    def build_list_students(self) -> tuple[TextClause, dict[str, Any]]:
        return text("SELECT StudentID, GradeBand, Curriculum FROM Student;"), {}

    def build_add_student(self, student_id: str, grade_band: str = "", curriculum: str = "") -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT IGNORE INTO Student (StudentID, GradeBand, Curriculum)
            VALUES (:student_id, :grade_band, :curriculum);
        """)
        return sql, {"student_id": student_id, "grade_band": grade_band, "curriculum": curriculum}

    # =========================================================================
    # 2. Curriculum (SCD Type 2 Stored Procedures & Reads)
    # =========================================================================

    def build_upsert_unit_scd2(self, unit: Unit) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            CALL sp_update_unit_scd2(
                :unit_id, :language, :curriculum, :grade, :unit_name, :unit_content
            );
        """)
        params = {
            "unit_id": unit.unit_id,
            "language": getattr(unit, "language", "en"),
            "curriculum": unit.curriculum,
            "grade": unit.grade,
            "unit_name": unit.unit_name,
            "unit_content": unit.unit_content,
        }
        return sql, params

    def build_get_unit(self, unit_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT UnitID, Language, Curriculum, Grade, UnitName, UnitContent
            FROM Unit WHERE UnitID = :unit_id AND Current = TRUE
            ORDER BY ValidFrom DESC LIMIT 1;
        """)
        return sql, {"unit_id": unit_id}

    def build_upsert_lesson_scd2(self, lesson: Lesson) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            CALL sp_update_lesson_scd2(
                :lesson_id, :unit_id, :lesson_name, :lesson_content
            );
        """)
        params = {
            "lesson_id": lesson.lesson_id,
            "unit_id": lesson.unit_id,
            "lesson_name": lesson.lesson_name,
            "lesson_content": lesson.lesson_content,
        }
        return sql, params

    def build_get_lesson(self, lesson_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT LessonID, UnitID, LessonName, LessonContent
            FROM Lesson WHERE LessonID = :lesson_id AND Current = TRUE
            ORDER BY ValidFrom DESC LIMIT 1;
        """)
        return sql, {"lesson_id": lesson_id}

    def build_upsert_learning_objective_scd2(self, lo: LearningObjective) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            CALL sp_update_learning_objective_scd2(
                :lo_id, :lesson_id, :lo_content
            );
        """)
        params = {
            "lo_id": lo.learning_objective_id,
            "lesson_id": lo.lesson_id,
            "lo_content": lo.learning_objective_content,
        }
        return sql, params

    def build_get_learning_objective(self, lo_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT LearningObjectiveID, LessonID, LearningObjectiveContent
            FROM LearningObjective WHERE LearningObjectiveID = :lo_id AND Current = TRUE
            ORDER BY ValidFrom DESC LIMIT 1;
        """)
        return sql, {"lo_id": lo_id}

    # =========================================================================
    # 3. Skills & Similarities
    # =========================================================================

    def build_upsert_skill_scd2(
        self,
        skill_id: str,
        lesson_id: str = "",
        unit_id: str | None = None,
        learning_objective_id: str | None = None,
        skill_cluster_content: str = "",
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            CALL sp_update_skill_cluster_scd2(
                :skill_id, :lesson_id, :unit_id, :lo_id, :content
            );
        """)
        params = {
            "skill_id": skill_id,
            "lesson_id": lesson_id,
            "unit_id": unit_id,
            "lo_id": learning_objective_id,
            "content": skill_cluster_content,
        }
        return sql, params

    def build_get_skill(self, skill_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT SkillClusterID, LessonID, UnitID, LearningObjectiveID, SkillClusterContent
            FROM SkillCluster
            WHERE SkillClusterID = :skill_id AND Current = TRUE
            ORDER BY ValidFrom DESC LIMIT 1;
        """)
        return sql, {"skill_id": skill_id}

    def build_get_skills_by_ids(self, skill_ids: list[str]) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT SkillClusterID, LessonID, UnitID, LearningObjectiveID, SkillClusterContent
            FROM SkillCluster
            WHERE Current = TRUE AND SkillClusterID IN :skill_ids
            ORDER BY ValidFrom DESC;
        """).bindparams(bindparam("skill_ids", expanding=True))
        return sql, {"skill_ids": skill_ids}

    def build_delete_skill_similarities(self, skill_id: str) -> tuple[TextClause, dict[str, Any]]:
        return text("DELETE FROM skill_similarities WHERE skill_id = :id"), {"id": skill_id}

    def build_insert_skill_similarity(self, skill_id: str, sim_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("INSERT INTO skill_similarities (skill_id, similar_skill_id) VALUES (:id, :sim)")
        return sql, {"id": skill_id, "sim": sim_id}

    def build_get_skill_similarities(self, skill_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("SELECT similar_skill_id FROM skill_similarities WHERE skill_id = :id")
        return sql, {"id": skill_id}

    # =========================================================================
    # 4. Questions, Options & Diagnostics (SCD Type 2)
    # =========================================================================

    def build_get_questions_by_ids(self, q_ids: list[str]) -> tuple[TextClause, dict[str, Any]]:
        sql = text(_QUESTION_SELECT + " AND qc.QuestionID IN :q_ids ORDER BY qc.QuestionID;") \
            .bindparams(bindparam("q_ids", expanding=True))
        return sql, {"q_ids": q_ids}

    def build_list_questions(self) -> tuple[TextClause, dict[str, Any]]:
        return text(_QUESTION_SELECT + " ORDER BY qc.QuestionID;"), {}

    def build_get_options(self, q_ids: list[str]) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT OptionID, QuestionID, OptionText, IsCorrect, DisplayOrder,
                   ContentGapType, DistractorReadingTrap, DistractorImpulsive
            FROM Options
            WHERE QuestionID IN :q_ids
            ORDER BY QuestionID, DisplayOrder, OptionID;
        """).bindparams(bindparam("q_ids", expanding=True))
        return sql, {"q_ids": q_ids}

    def build_insert_question_content(self, q) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT IGNORE INTO QuestionContent
                (QuestionID, SkillClusterID, QuestionText, QuestionType, Current)
            VALUES
                (:q_id, :skill_cluster_id, :question_text, :question_type, TRUE);
        """)
        return sql, {
            "q_id": q.q_id,
            "skill_cluster_id": q.skill_cluster_id,
            "question_text": q.question_text,
            "question_type": q.question_type,
        }

    def build_insert_option(self, q_id: str, opt) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT INTO Options
                (OptionID, QuestionID, OptionText, IsCorrect, DisplayOrder,
                 ContentGapType, DistractorReadingTrap, DistractorImpulsive)
            VALUES
                (:option_id, :question_id, :option_text, :is_correct, :display_order,
                 :content_gap_type, :reading_trap, :impulsive)
            ON DUPLICATE KEY UPDATE
                OptionText = VALUES(OptionText),
                IsCorrect = VALUES(IsCorrect),
                DisplayOrder = VALUES(DisplayOrder),
                ContentGapType = VALUES(ContentGapType),
                DistractorReadingTrap = VALUES(DistractorReadingTrap),
                DistractorImpulsive = VALUES(DistractorImpulsive);
        """)
        return sql, {
            "option_id": opt.option_id,
            "question_id": q_id,
            "option_text": opt.content,
            "is_correct": bool(opt.is_correct),
            "display_order": opt.display_order,
            "content_gap_type": opt.content_gap_type,
            "reading_trap": str(opt.distractor_reading_trap) if opt.distractor_reading_trap is not None else None,
            "impulsive": bool(opt.distractor_impulsive),
        }

    def build_update_question_diagnostics_scd2(
        self, q_id: str, attributes: dict[str, Any]
    ) -> list[tuple[TextClause, dict[str, Any]]]:
        expire_sql = text("""
            UPDATE QuestionDiagnostics
            SET Current = FALSE, ValidTo = UTC_TIMESTAMP()
            WHERE QuestionID = :q_id AND Current = TRUE;
        """)

        insert_sql = text("""
            INSERT INTO QuestionDiagnostics (
                DiagnosticsVersion, QuestionID, BloomTax, TheoreticalTime,
                VariablesCount, LogicalSteps, TrickQuestion, VisualDependancy,
                RealWorldContext, MultiConceptFlag, CognitiveLoadIndex,
                LanguageChallengeType, ReasoningLevel, LanguageLevel,
                DifficultyLevel, p_g, p_t, p_s, ValidFrom, ValidTo, Current
            ) VALUES (
                UUID(), :q_id, :bloom_tax, :theoretical_time,
                :variables_count, :logical_steps, :trick_question, :visual_dependancy,
                :real_world_context, :multi_concept_flag, :cognitive_load_index,
                :language_challenge_type, :reasoning_level, :language_level,
                :difficulty_level, :p_g, :p_t, :p_s, UTC_TIMESTAMP(), NULL, TRUE
            );
        """)

        params = {
            "q_id": q_id,
            "bloom_tax": attributes.get("bloom_tax") or attributes.get("bloom_taxonomy_level"),
            "theoretical_time": attributes.get("theoretical_time") or attributes.get("theoretical_solving_time"),
            "variables_count": attributes.get("variables_count", 0),
            "logical_steps": attributes.get("logical_steps", 0),
            "trick_question": attributes.get("trick_question", False),
            "visual_dependancy": attributes.get("visual_dependancy", attributes.get("visual_dependency", False)),
            "real_world_context": attributes.get("real_world_context", False),
            "multi_concept_flag": attributes.get("multi_concept_flag", False),
            "cognitive_load_index": attributes.get("cognitive_load_index", attributes.get("cognitive_load")),
            "language_challenge_type": attributes.get("language_challenge_type"),
            "reasoning_level": attributes.get("reasoning_level"),
            "language_level": attributes.get("language_level"),
            "difficulty_level": attributes.get("difficulty_level", 0.0),
            "p_g": attributes.get("p_g", 0.0),
            "p_t": attributes.get("p_t", 0.0),
            "p_s": attributes.get("p_s", 0.0),
        }
        return [(expire_sql, {"q_id": q_id}), (insert_sql, params)]

    def build_dynamic_question_query(self, parameters: list[dict[str, Any]]) -> tuple[TextClause, dict[str, Any]]:
        where_clauses = ["qc.Current = TRUE", "qd.Current = TRUE"]
        bind_params: dict[str, Any] = {}

        for idx, cond in enumerate(parameters):
            attr = cond.get("Attribute")
            op_raw = cond.get("Operator", "==")
            threshold = cond.get("Threshold")

            if not attr or threshold is None:
                continue

            if attr in ("skill_cluster_id", "SkillClusterID"):
                col = "qc.SkillClusterID"
            elif attr in ("question_type", "QuestionType"):
                col = "qc.QuestionType"
            else:
                col = ATTRIBUTE_COLUMN_MAP.get(attr, f"qd.{attr}")

            sql_op = OPERATOR_MAP.get(op_raw, "=")
            param_key = f"param_{idx}"

            if attr == "time_pressure_flag" and threshold is False:
                where_clauses.append(f"{col} <= :{param_key}")
                bind_params[param_key] = 30.0
            elif sql_op == "IN" and isinstance(threshold, (list, tuple, set)):
                in_keys = []
                for sub_idx, item in enumerate(threshold):
                    sub_key = f"{param_key}_{sub_idx}"
                    in_keys.append(f":{sub_key}")
                    bind_params[sub_key] = item
                where_clauses.append(f"{col} IN ({', '.join(in_keys)})")
            else:
                where_clauses.append(f"{col} {sql_op} :{param_key}")
                bind_params[param_key] = threshold

        sql = text(f"""
            SELECT qc.QuestionID
            FROM QuestionContent qc
            JOIN QuestionDiagnostics qd ON qc.QuestionID = qd.QuestionID
            WHERE {' AND '.join(where_clauses)};
        """)
        return sql, bind_params

    # =========================================================================
    # 5. Student State (Stored Procedures, Responses, Treatments, History)
    # =========================================================================

    def build_update_priors(
        self,
        student_id: str,
        skill_id: str,
        new_prior: float,
        content_gap: bool = False,
        content_gap_type: str | None = None,
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            CALL sp_record_student_mastery(
                :student_id, :skill_id, :mastery, :content_gap, :content_gap_type
            );
        """)
        params = {
            "student_id": student_id,
            "skill_id": skill_id,
            "mastery": new_prior,
            "content_gap": content_gap,
            "content_gap_type": content_gap_type,
        }
        return sql, params

    def build_upsert_response(
        self,
        match_id: str,
        student_id: str,
        question_id: str,
        option_id: str | None,
        error: bool,
        time_out: bool,
        response_time: float,
        stress_triggers: int,
        hints_used: int,
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT INTO Response (
                MatchID, StudentID, QuestionID, OptionID, Error,
                TimeOut, ResponseTime, StressTriggers, HintsUsed
            ) VALUES (
                :match_id, :student_id, :question_id, :option_id, :error,
                :time_out, :response_time, :stress_triggers, :hints_used
            )
            ON DUPLICATE KEY UPDATE
                OptionID = VALUES(OptionID),
                Error = VALUES(Error),
                TimeOut = VALUES(TimeOut),
                ResponseTime = VALUES(ResponseTime),
                StressTriggers = VALUES(StressTriggers),
                HintsUsed = VALUES(HintsUsed);
        """)
        params = {
            "match_id": match_id,
            "student_id": student_id,
            "question_id": question_id,
            "option_id": option_id,
            "error": error,
            "time_out": time_out,
            "response_time": response_time,
            "stress_triggers": stress_triggers,
            "hints_used": hints_used,
        }
        return sql, params

    def build_add_diagnosis(
        self, student_id: str, diagnosis_id: str, delta: float = 0.0
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("CALL sp_record_student_diagnostic(:student_id, :diagnosis_id, :delta);")
        return sql, {"student_id": student_id, "diagnosis_id": diagnosis_id, "delta": delta}

    def build_update_content_gaps(
        self,
        student_id: str,
        skill_id: str,
        content_gap: bool,
        content_gap_type: str | None,
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            UPDATE StudentMasteryLevels
            SET ContentGap = :content_gap,
                ContentGapType = :content_gap_type
            WHERE StudentID = :student_id
              AND SkillClusterID = :skill_id
              AND Current = TRUE;
        """)
        params = {
            "content_gap": content_gap,
            "content_gap_type": content_gap_type,
            "student_id": student_id,
            "skill_id": skill_id,
        }
        return sql, params

    def build_update_treatment_plan(
        self,
        student_id: str,
        diagnosis_id: str,
        question_attribute: str,
        value: str | int | float | bool,
        query_operator: str = "=",
    ) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT INTO Treatment (
                StudentID, DiagnosisID, QuestionAttribute, Value, QueryOperator
            ) VALUES (
                :student_id, :diagnosis_id, :question_attribute, :value, :query_operator
            )
            ON DUPLICATE KEY UPDATE
                Value = VALUES(Value),
                QueryOperator = VALUES(QueryOperator);
        """)
        params = {
            "student_id": student_id,
            "diagnosis_id": diagnosis_id,
            "question_attribute": question_attribute,
            "value": str(value),
            "query_operator": query_operator,
        }
        return sql, params

    def build_delete_treatment_plan(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        return text("DELETE FROM Treatment WHERE StudentID = :student_id;"), {"student_id": student_id}

    def build_get_priors(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT SkillClusterID, Mastery, ContentGap, ContentGapType, Timestamp
            FROM StudentMasteryLevels
            WHERE StudentID = :student_id
              AND Current = TRUE;
        """)
        return sql, {"student_id": student_id}

    def build_get_diagnosis(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT DiagnosisID, Delta, Timestamp
            FROM StudentDiagnostics
            WHERE StudentID = :student_id
              AND Current = TRUE;
        """)
        return sql, {"student_id": student_id}

    def build_get_treatment(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT DiagnosisID, QuestionAttribute, Value, QueryOperator
            FROM Treatment
            WHERE StudentID = :student_id;
        """)
        return sql, {"student_id": student_id}

    def build_get_priors_history(self, student_id: str, window: int = 5) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            WITH windows AS (
                SELECT DISTINCT Timestamp
                FROM StudentMasteryLevels
                WHERE StudentID = :student_id AND Current = FALSE
                ORDER BY Timestamp DESC
                LIMIT :window
            )
            SELECT sml.SkillClusterID, sml.Mastery, sml.ContentGap,
                   sml.ContentGapType, sml.Timestamp
            FROM StudentMasteryLevels sml
            JOIN windows w ON w.Timestamp = sml.Timestamp
            WHERE sml.StudentID = :student_id AND sml.Current = FALSE
            ORDER BY sml.Timestamp, sml.SkillClusterID;
        """)
        return sql, {"student_id": student_id, "window": int(window)}

    def build_get_diagnoses_history(self, student_id: str, window: int = 5) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            WITH windows AS (
                SELECT DISTINCT Timestamp
                FROM StudentDiagnostics
                WHERE StudentID = :student_id AND Current = FALSE
                ORDER BY Timestamp DESC
                LIMIT :window
            )
            SELECT sd.DiagnosisID, sd.Delta, sd.Timestamp
            FROM StudentDiagnostics sd
            JOIN windows w ON w.Timestamp = sd.Timestamp
            WHERE sd.StudentID = :student_id AND sd.Current = FALSE
            ORDER BY sd.Timestamp, sd.DiagnosisID;
        """)
        return sql, {"student_id": student_id, "window": int(window)}

    def build_get_responses(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            SELECT r.MatchID, m.CreatedTimeStamp AS Date, r.QuestionID,
                   r.OptionID, r.Error, r.TimeOut, r.ResponseTime,
                   r.StressTriggers, r.HintsUsed
            FROM Response r
            LEFT JOIN `Match` m ON m.MatchID = r.MatchID
            WHERE r.StudentID = :student_id
            ORDER BY m.CreatedTimeStamp, r.MatchID, r.QuestionID;
        """)
        return sql, {"student_id": student_id}

    def build_count_responses(self, student_id: str) -> tuple[TextClause, dict[str, Any]]:
        sql = text("SELECT COUNT(*) FROM Response WHERE StudentID = :student_id;")
        return sql, {"student_id": student_id}

    def build_create_match(self, match_id: str, match_type: str, num_participants: int = 1) -> tuple[TextClause, dict[str, Any]]:
        sql = text("""
            INSERT IGNORE INTO `Match`
                (MatchID, CreatedTimeStamp, MatchType, NumParticipants)
            VALUES (:match_id, UTC_TIMESTAMP(), :match_type, :num_participants);
        """)
        return sql, {
            "match_id": match_id,
            "match_type": match_type,
            "num_participants": num_participants,
        }

    # =========================================================================
    # 6. Hints
    # =========================================================================

    def build_get_pre_hint(self, hint_id: str) -> Select:
        return select(PreHint).where(PreHint.hint_id == hint_id)

    def build_get_pre_hints_by_topic(self, topic: str) -> Select:
        return select(PreHint).where(PreHint.topic == topic)

    def build_get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> Select:
        return select(PreHint).where(
            and_(PreHint.topic == topic, PreHint.content_gap_type == content_gap_type)
        )

    def build_get_pre_hints_by_diagnosis(self, diagnosis: str) -> Select:
        return select(PreHint).where(PreHint.diagnosis == diagnosis)

    def build_list_pre_hints(self) -> Select:
        return select(PreHint)

    def build_get_post_hint(self, hint_id: str) -> Select:
        return select(PostHint).where(PostHint.hint_id == hint_id)

    def build_get_post_hints_by_question(self, question_id: str) -> Select:
        return select(PostHint).where(PostHint.question_id == question_id).order_by(PostHint.hint_order)

    def build_list_post_hints(self) -> Select:
        return select(PostHint)

    def build_get_play_hint(self, hint_id: str) -> Select:
        return select(PlayHint).where(PlayHint.hint_id == hint_id)

    def build_get_play_hints_by_question(self, q_id: str) -> Select:
        return select(PlayHint).where(PlayHint.q_id == q_id).order_by(PlayHint.hint_order)

    def build_list_play_hints(self) -> Select:
        return select(PlayHint)
