from __future__ import annotations
import json
import os
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from services.Entities import (
    Question,
    SkillCluster,
    Student,
    PreHint,
    PostHint,
    PlayHint,
    Unit,
    Lesson,
    LearningObjective,
    Option,
    QuestionAttributes,
)
from services.storage.base import DataRepository
from services.storage.queries import QueryBuilder

load_dotenv()


class SqlAlchemyRepository(DataRepository):
    def __init__(self) -> None:
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "password")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "sprintlab_db")

        self.engine = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}",
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.query_builder = QueryBuilder()

    def _get_session(self) -> Session:
        return self.SessionLocal()

    # --- hydration helpers ---------------------------------------------
    @staticmethod
    def _student_from_row(row) -> Student:
        return Student(
            student_id=row["StudentID"],
            grade_band=row.get("GradeBand") or "",
            curriculum=row.get("Curriculum") or "",
        )

    @staticmethod
    def _option_from_row(row) -> Option:
        return Option(
            option_id=row["OptionID"],
            content=row["OptionText"],
            is_correct=bool(row["IsCorrect"]),
            display_order=row["DisplayOrder"] or 0,
            content_gap_type=row["ContentGapType"],
            distractor_reading_trap=bool(row["DistractorReadingTrap"])
            if row["DistractorReadingTrap"] not in (None, "", "False", "false", "0", 0)
            else False,
            distractor_impulsive=bool(row["DistractorImpulsive"]),
        )

    def _question_from_row(self, row, options: list[Option], skill: SkillCluster | None) -> Question:
        attrs = QuestionAttributes(
            difficulty_level=float(row["DifficultyLevel"] or 0.0),
            bloom_taxonomy_level=row["BloomTax"],
            theoretical_solving_time=float(row["TheoreticalTime"] or 0.0),
            cognitive_load_index=row["CognitiveLoadIndex"],
            variables_count=int(row["VariablesCount"] or 0),
            logical_steps=int(row["LogicalSteps"] or 0),
            language_challenge_type=row["LanguageChallengeType"],
            language_challenging=1 if row["LanguageChallengeType"] else 0,
            trick_question=bool(row["TrickQuestion"]),
            visual_dependency=bool(row["VisualDependancy"]),
            real_world_context=bool(row["RealWorldContext"]),
            multi_concept_flag=bool(row["MultiConceptFlag"]),
            cognitive_load=float(row["CognitiveLoadIndex"] or 0.0),
            language_level=row["LanguageLevel"],
            reasoning_level=row["ReasoningLevel"],
            p_g=float(row["p_g"] or 0.0),
            p_t=float(row["p_t"] or 0.0),
            p_s=float(row["p_s"] or 0.0),
        )
        return Question(
            q_id=row["QuestionID"],
            question_type=row["QuestionType"],
            question_text=row["QuestionText"],
            skill_cluster_id=row["SkillClusterID"],
            skill_ids=[row["SkillClusterID"]],
            unit_id=skill.unit_id if skill else None,
            lesson_id=skill.lesson_id if skill else None,
            options=options,
            attrs=attrs,
        )

    def _hydrate_questions(self, session: Session, rows: list[Any]) -> list[Question]:
        if not rows:
            return []
        q_ids = [row["QuestionID"] for row in rows]
        skill_ids = list({row["SkillClusterID"] for row in rows if row["SkillClusterID"]})

        options_by_q: dict[str, list[Option]] = {qid: [] for qid in q_ids}
        if q_ids:
            opt_sql, opt_params = self.query_builder.build_get_options(q_ids)
            for opt_row in session.execute(opt_sql, opt_params).mappings().all():
                options_by_q.setdefault(opt_row["QuestionID"], []).append(
                    self._option_from_row(opt_row)
                )

        skills: dict[str, SkillCluster] = {}
        if skill_ids:
            skill_sql, skill_params = self.query_builder.build_get_skills_by_ids(skill_ids)
            for skill_row in session.execute(skill_sql, skill_params).mappings().all():
                sid = skill_row["SkillClusterID"]
                if sid in skills:
                    continue
                skills[sid] = SkillCluster(
                    skill_cluster_id=sid,
                    lesson_id=skill_row["LessonID"] or "",
                    unit_id=skill_row["UnitID"] or "",
                    learning_objective_id=skill_row["LearningObjectiveID"],
                    skill_cluster_content=skill_row["SkillClusterContent"] or "",
                )

        return [
            self._question_from_row(
                row,
                options_by_q.get(row["QuestionID"], []),
                skills.get(row["SkillClusterID"]),
            )
            for row in rows
        ]

    @staticmethod
    def _decode_value(value: str) -> Any:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _resolve_option_id(question: Question | None, response: Any) -> str | None:
        if question is None or not question.options:
            return None
        options = question.options

        if isinstance(response, bool):
            idx = 1 if response else 0
            return options[idx].option_id if 0 <= idx < len(options) else None
        if isinstance(response, int):
            return options[response].option_id if 0 <= response < len(options) else None
        if isinstance(response, str):
            for opt in options:
                if opt.option_id == response:
                    return opt.option_id
            for opt in options:
                if opt.option_id.endswith(f"_{response}") or opt.option_id.endswith(response):
                    return opt.option_id
            for opt in options:
                if opt.content.strip().lower() == response.strip().lower():
                    return opt.option_id
        return None

    def _atag_for_response(self, q_id: str, option_id: str | None, error: bool) -> str | None:
        if not error:
            return "Correct"
        question = self.get_question(q_id)[0]
        if question and option_id:
            for opt in question.options:
                if opt.option_id == option_id and opt.content_gap_type:
                    return opt.content_gap_type
        return "Gap_Prior"

    # --- students -------------------------------------------------------
    def add_student(self, student_id: str, grade_band: str = "", curriculum: str = "") -> None:
        sql, params = self.query_builder.build_add_student(student_id, grade_band, curriculum)
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    def get_student(self, student_id: str) -> Student | None:
        sql, params = self.query_builder.build_get_student(student_id)
        with self._get_session() as session:
            row = session.execute(sql, params).mappings().first()
            return self._student_from_row(row) if row else None

    def list_students(self) -> list[Student]:
        sql, params = self.query_builder.build_list_students()
        with self._get_session() as session:
            return [self._student_from_row(r)
                    for r in session.execute(sql, params).mappings().all()]

    # --- curriculum (SCD Type 2 via Stored Procedures) ------------------
    def add_unit(self, unit: Unit) -> None:
        sql, params = self.query_builder.build_upsert_unit_scd2(unit)
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    def update_unit(self, unit: Unit) -> None:
        self.add_unit(unit)

    def get_unit(self, unit_id: str) -> Unit | None:
        sql, params = self.query_builder.build_get_unit(unit_id)
        with self._get_session() as session:
            row = session.execute(sql, params).mappings().first()
            if not row:
                return None
            return Unit(
                unit_id=row["UnitID"],
                language=row["Language"],
                curriculum=row["Curriculum"],
                grade=row["Grade"],
                unit_name=row["UnitName"],
                unit_content=row["UnitContent"] or "",
            )

    def add_lesson(self, lesson: Lesson) -> None:
        sql, params = self.query_builder.build_upsert_lesson_scd2(lesson)
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    def update_lesson(self, lesson: Lesson) -> None:
        self.add_lesson(lesson)

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        sql, params = self.query_builder.build_get_lesson(lesson_id)
        with self._get_session() as session:
            row = session.execute(sql, params).mappings().first()
            if not row:
                return None
            return Lesson(
                lesson_id=row["LessonID"],
                unit_id=row["UnitID"],
                lesson_name=row["LessonName"],
                lesson_content=row["LessonContent"] or "",
            )

    def add_learning_objective(self, lo: LearningObjective) -> None:
        sql, params = self.query_builder.build_upsert_learning_objective_scd2(lo)
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    def update_learning_objective(self, lo: LearningObjective) -> None:
        self.add_learning_objective(lo)

    def get_learning_objective(self, lo_id: str) -> LearningObjective | None:
        sql, params = self.query_builder.build_get_learning_objective(lo_id)
        with self._get_session() as session:
            row = session.execute(sql, params).mappings().first()
            if not row:
                return None
            return LearningObjective(
                learning_objective_id=row["LearningObjectiveID"],
                lesson_id=row["LessonID"],
                learning_objective_content=row["LearningObjectiveContent"] or "",
            )

    # --- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        q = Question(**kwargs)
        if not q.q_id:
            return
        options = q.options or []
        attrs = q.attrs

        with self._get_session() as session:
            existing = session.execute(
                *self.query_builder.build_get_questions_by_ids([q.q_id])
            ).first()
            if existing:
                return

            sql, params = self.query_builder.build_insert_question_content(q)
            session.execute(sql, params)

            for opt in options:
                opt_sql, opt_params = self.query_builder.build_insert_option(q.q_id, opt)
                session.execute(opt_sql, opt_params)

            attr_dict = {}
            if attrs is not None:
                if isinstance(attrs, dict):
                    attr_dict = dict(attrs)
                else:
                    attr_dict = {
                        "difficulty_level": attrs.difficulty_level,
                        "bloom_taxonomy_level": attrs.bloom_taxonomy_level,
                        "theoretical_solving_time": attrs.theoretical_solving_time,
                        "variables_count": attrs.variables_count,
                        "logical_steps": attrs.logical_steps,
                        "trick_question": attrs.trick_question,
                        "visual_dependency": attrs.visual_dependency,
                        "real_world_context": attrs.real_world_context,
                        "multi_concept_flag": attrs.multi_concept_flag,
                        "cognitive_load_index": attrs.cognitive_load_index,
                        "language_challenge_type": attrs.language_challenge_type,
                        "language_level": attrs.language_level,
                        "reasoning_level": attrs.reasoning_level,
                        "p_g": attrs.p_g,
                        "p_t": attrs.p_t,
                        "p_s": attrs.p_s,
                    }
            for dsql, dparams in self.query_builder.build_update_question_diagnostics_scd2(
                q.q_id, attr_dict
            ):
                session.execute(dsql, dparams)
            session.commit()

    def get_question(self, q_ids: list[str] | str) -> list[Question | None]:
        if isinstance(q_ids, str):
            q_ids = [q_ids]
        if not q_ids:
            return []
        sql, params = self.query_builder.build_get_questions_by_ids(q_ids)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            hydrated = self._hydrate_questions(session, rows)
            q_map = {q.q_id: q for q in hydrated}
            return [q_map.get(qid) for qid in q_ids]

    def list_questions(self) -> list[Question]:
        sql, params = self.query_builder.build_list_questions()
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return self._hydrate_questions(session, rows)

    def update_question_attributes(self, q_id: str, attributes: dict[str, Any]) -> None:
        queries = self.query_builder.build_update_question_diagnostics_scd2(q_id, attributes)
        with self._get_session() as session:
            for sql, params in queries:
                session.execute(sql, params)
            session.commit()

    # --- skills (SCD Type 2 via Stored Procedure) -----------------------
    def add_skill(
        self,
        skill_id: str,
        similar_skills: list[str] | None = None,
        lesson_id: str = "",
        unit_id: str | None = None,
        learning_objective_id: str | None = None,
        skill_cluster_content: str = "",
    ) -> None:
        with self._get_session() as session:
            sql, params = self.query_builder.build_upsert_skill_scd2(
                skill_id=skill_id,
                lesson_id=lesson_id,
                unit_id=unit_id,
                learning_objective_id=learning_objective_id,
                skill_cluster_content=skill_cluster_content,
            )
            session.execute(sql, params)

            if similar_skills is not None:
                del_sql, del_params = self.query_builder.build_delete_skill_similarities(skill_id)
                session.execute(del_sql, del_params)
                for sim_id in similar_skills:
                    ins_sql, ins_params = self.query_builder.build_insert_skill_similarity(skill_id, sim_id)
                    session.execute(ins_sql, ins_params)
            session.commit()

    def get_skill(self, skill_id: str) -> SkillCluster | None:
        sql, params = self.query_builder.build_get_skill(skill_id)
        with self._get_session() as session:
            row = session.execute(sql, params).mappings().first()
            if not row:
                return None
            sim_sql, sim_params = self.query_builder.build_get_skill_similarities(skill_id)
            sim_ids = list(session.execute(sim_sql, sim_params).scalars().all())
            return SkillCluster(
                skill_cluster_id=row["SkillClusterID"],
                lesson_id=row["LessonID"] or "",
                unit_id=row["UnitID"] or "",
                learning_objective_id=row["LearningObjectiveID"],
                skill_cluster_content=row["SkillClusterContent"] or "",
                similar_skills=sim_ids,
            )

    # --- student state (Stored Procedures & History) --------------------
    def update_priors(self, student_id: str, skill_id: str, new_prior: float) -> None:
        sql, params = self.query_builder.build_update_priors(student_id, skill_id, new_prior)
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    def add_priors_history(self, student_id: str, timestamp: str, priors: dict[str, float]) -> None:
        # StudentMasteryLevels is versioned by sp_record_student_mastery on
        # every update_priors call; writing the current snapshot again here
        # would create duplicate same-second versions.
        return None

    def get_priors(self, student_id: str) -> dict[str, float]:
        sql, params = self.query_builder.build_get_priors(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return {row["SkillClusterID"]: float(row["Mastery"] or 0.0) for row in rows}

    def get_priors_history(self, student_id: str, window: int = 5) -> dict:
        sql, params = self.query_builder.build_get_priors_history(student_id, window)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
        snapshots: dict[str, dict[str, float]] = {}
        for row in rows:
            ts = str(row["Timestamp"])
            snapshots.setdefault(ts, {})[row["SkillClusterID"]] = float(row["Mastery"] or 0.0)
        return snapshots

    def update_responses(self, student_id: str, responses: list[tuple]) -> None:
        with self._get_session() as session:
            for resp in responses:
                question = self.get_question(resp[2])[0]
                option_id = self._resolve_option_id(question, resp[3])
                sql, params = self.query_builder.build_upsert_response(
                    match_id=resp[0],
                    student_id=student_id,
                    question_id=resp[2],
                    option_id=option_id,
                    error=bool(resp[4]),
                    time_out=bool(resp[5]),
                    response_time=float(resp[6]) if resp[6] is not None else 0.0,
                    stress_triggers=int(resp[7]) if resp[7] is not None else 0,
                    hints_used=int(resp[8]) if resp[8] is not None else 0,
                )
                session.execute(sql, params)
            session.commit()

    def get_responses(self, student_id: str) -> list[tuple]:
        sql, params = self.query_builder.build_get_responses(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
        out = []
        for row in rows:
            error = bool(row["Error"])
            option_id = row["OptionID"]
            date = row["Date"]
            if hasattr(date, "isoformat"):
                date = date.isoformat()
            out.append((
                row["MatchID"],
                date or "",
                row["QuestionID"],
                0 if error else 1,
                error,
                bool(row["TimeOut"]),
                float(row["ResponseTime"] or 0.0),
                row["StressTriggers"] or 0,
                row["HintsUsed"] or 0,
                self._atag_for_response(row["QuestionID"], option_id, error),
            ))
        return out

    def count_responses(self, student_id: str) -> int:
        sql, params = self.query_builder.build_count_responses(student_id)
        with self._get_session() as session:
            return int(session.execute(sql, params).scalar() or 0)

    def add_diagnosis(
        self,
        student_id: str,
        diagnoses: list[str],
        deltas: dict[str, float] | None = None,
        timestamp: str | None = None,
    ) -> None:
        deltas = deltas or {}
        with self._get_session() as session:
            for diagnosis_id in diagnoses:
                delta_val = float(deltas.get(diagnosis_id, 0.0))
                sql, params = self.query_builder.build_add_diagnosis(student_id, diagnosis_id, delta_val)
                session.execute(sql, params)
            session.commit()

    def get_diagnoses(self, student_id: str) -> list[str]:
        sql, params = self.query_builder.build_get_diagnosis(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return [row["DiagnosisID"] for row in rows]

    def get_diagnosis(self, student_id: str) -> dict[str, float]:
        sql, params = self.query_builder.build_get_diagnosis(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return {row["DiagnosisID"]: float(row["Delta"] or 0.0) for row in rows}

    def get_diagnoses_history(self, student_id: str, window: int = 5) -> list[dict]:
        sql, params = self.query_builder.build_get_diagnoses_history(student_id, window)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
        grouped: dict[str, dict] = {}
        for row in rows:
            ts = str(row["Timestamp"])
            snap = grouped.setdefault(ts, {"timestamp": ts, "diagnoses": [], "deltas": {}})
            snap["diagnoses"].append(row["DiagnosisID"])
            snap["deltas"][row["DiagnosisID"]] = float(row["Delta"] or 0.0)
        return [grouped[k] for k in sorted(grouped)]

    def update_content_gaps(self, student_id: str, content_gap_types: dict[str, str]) -> None:
        with self._get_session() as session:
            for skill_id, gap_type in content_gap_types.items():
                sql, params = self.query_builder.build_update_content_gaps(
                    student_id=student_id,
                    skill_id=skill_id,
                    content_gap=bool(gap_type),
                    content_gap_type=gap_type,
                )
                session.execute(sql, params)
            session.commit()

    def get_content_gaps(self, student_id: str) -> dict[str, str]:
        sql, params = self.query_builder.build_get_priors(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return {
                row["SkillClusterID"]: row["ContentGapType"]
                for row in rows
                if row["ContentGap"] and row["ContentGapType"]
            }

    def update_treatment_plan(
        self,
        student_id: str,
        timestamp: str,
        treatment_name: str,
        parameters: dict[str, Any],
    ) -> None:
        if treatment_name == "treatment_plan":
            self.replace_treatment_plan(student_id, timestamp, parameters)
            return
        op = parameters.get("Operator", "==") if isinstance(parameters, dict) else "=="
        with self._get_session() as session:
            for attr, val in parameters.items():
                if attr == "Operator":
                    continue
                sql, params = self.query_builder.build_update_treatment_plan(
                    student_id=student_id,
                    diagnosis_id=treatment_name,
                    question_attribute=attr,
                    value=json.dumps(val),
                    query_operator=op,
                )
                session.execute(sql, params)
            session.commit()

    def replace_treatment_plan(self, student_id: str, timestamp: str,
                               treatment_plan: dict) -> None:
        with self._get_session() as session:
            del_sql, del_params = self.query_builder.build_delete_treatment_plan(student_id)
            session.execute(del_sql, del_params)
            for topic in ("general", "specific"):
                for diagnosis_id, conditions in (treatment_plan.get(topic) or {}).items():
                    if not isinstance(conditions, dict):
                        continue
                    op = conditions.get("Operator", "==")
                    for attr, value in conditions.items():
                        if attr == "Operator":
                            continue
                        sql, params = self.query_builder.build_update_treatment_plan(
                            student_id=student_id,
                            diagnosis_id=diagnosis_id,
                            question_attribute=attr,
                            value=json.dumps(value),
                            query_operator=op,
                        )
                        session.execute(sql, params)
            session.commit()

    def get_treatment(self, student_id: str) -> list[dict[str, Any]]:
        sql, params = self.query_builder.build_get_treatment(student_id)
        with self._get_session() as session:
            rows = session.execute(sql, params).mappings().all()
            return [dict(row) for row in rows]

    def get_treatment_plan(self, student_id: str) -> dict | None:
        rows = self.get_treatment(student_id)
        if not rows:
            return None
        diagnoses = set(self.get_diagnoses(student_id))
        plan = {"general": {}, "specific": {}}
        for row in rows:
            topic = "general" if row["DiagnosisID"] in diagnoses else "specific"
            diagnosis_id = row["DiagnosisID"]
            clause = plan[topic].setdefault(diagnosis_id, {})
            clause[row["QuestionAttribute"]] = self._decode_value(row["Value"])
            clause["Operator"] = row["QueryOperator"]
        return plan

    def create_match(self, match_id: str, match_type: str,
                     num_participants: int = 1) -> None:
        sql, params = self.query_builder.build_create_match(
            match_id, match_type, num_participants
        )
        with self._get_session() as session:
            session.execute(sql, params)
            session.commit()

    # --- query ----------------------------------------------------------
    def query(self, parameters: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Question]:
        if isinstance(parameters, dict):
            parameters = [parameters]

        sql, bind_params = self.query_builder.build_dynamic_question_query(parameters)
        with self._get_session() as session:
            matching_q_ids = list(session.execute(sql, bind_params).scalars().all())
        if not matching_q_ids:
            return {}
        questions = [q for q in self.get_question(matching_q_ids) if q]
        return {q.q_id: q for q in questions}

    # --- hints ----------------------------------------------------------
    def add_pre_hint(self, pre_hint: PreHint) -> None:
        with self._get_session() as session:
            session.add(pre_hint)
            session.commit()

    def get_pre_hint(self, hint_id: str) -> PreHint | None:
        stmt = self.query_builder.build_get_pre_hint(hint_id)
        with self._get_session() as session:
            return session.execute(stmt).scalars().first()

    def get_pre_hints_by_topic(self, topic: str) -> list[PreHint]:
        stmt = self.query_builder.build_get_pre_hints_by_topic(topic)
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> list[PreHint]:
        stmt = self.query_builder.build_get_pre_hints_by_content_gap_type(topic, content_gap_type)
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def get_pre_hints_by_diagnosis(self, diagnosis: str) -> list[PreHint]:
        stmt = self.query_builder.build_get_pre_hints_by_diagnosis(diagnosis)
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def list_pre_hints(self) -> list[PreHint]:
        stmt = self.query_builder.build_list_pre_hints()
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def add_post_hint(self, post_hint: PostHint) -> None:
        with self._get_session() as session:
            session.add(post_hint)
            session.commit()

    def get_post_hint(self, hint_id: str) -> PostHint | None:
        stmt = self.query_builder.build_get_post_hint(hint_id)
        with self._get_session() as session:
            return session.execute(stmt).scalars().first()

    def get_post_hints_by_question(self, question_id: str) -> list[PostHint]:
        stmt = self.query_builder.build_get_post_hints_by_question(question_id)
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def list_post_hints(self) -> list[PostHint]:
        stmt = self.query_builder.build_list_post_hints()
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def add_play_hint(self, play_hint: PlayHint) -> None:
        with self._get_session() as session:
            session.add(play_hint)
            session.commit()

    def get_play_hint(self, hint_id: str) -> PlayHint | None:
        stmt = self.query_builder.build_get_play_hint(hint_id)
        with self._get_session() as session:
            return session.execute(stmt).scalars().first()

    def get_play_hints_by_question(self, q_id: str) -> list[PlayHint]:
        stmt = self.query_builder.build_get_play_hints_by_question(q_id)
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def list_play_hints(self) -> list[PlayHint]:
        stmt = self.query_builder.build_list_play_hints()
        with self._get_session() as session:
            return list(session.execute(stmt).scalars().all())
