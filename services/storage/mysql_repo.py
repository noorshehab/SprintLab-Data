from __future__ import annotations
import os
from typing import Any, List, Optional
from sqlalchemy import create_engine, select, update, insert, and_, or_, text
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from services.Entities import (
    Base, Question, SkillCluster, Student, PreHint, PostHint, PlayHint, 
    Unit, Lesson, LearningObjective, Option, QuestionAttributes, Response,StudentMasteryLevel,StudentDiagnostic
)
from services.storage.base import DataRepository

load_dotenv()

class SqlAlchemyRepository(DataRepository):
    def __init__(self) -> None:
        user = os.getenv('DB_USER', 'root')
        password = os.getenv('DB_PASSWORD', 'password')
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '3306')
        db_name = os.getenv('DB_NAME', 'sprintlab_db')
        
        self.engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}")
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _get_session(self) -> Session:
        return self.SessionLocal()

    #--- students -------------------------------------------------------
    def add_student(self, student_id: str) -> None:
        with self._get_session() as session:
            if not session.get(Student, student_id):
                session.add(Student(student_id=student_id))
                session.commit()

    def get_student(self, student_id: str) -> Student | None:
        with self._get_session() as session:
            return session.get(Student, student_id)

    def list_students(self) -> list[Student]:
        with self._get_session() as session:
            return session.execute(select(Student)).scalars().all()

    #--- curriculum ---------------------------------------------------------
    def add_unit(self, unit: Unit) -> None:
        with self._get_session() as session:
            session.add(unit)
            session.commit()

    def get_unit(self, unit_id: str) -> Unit | None:
        with self._get_session() as session:
            return session.get(Unit, unit_id)

    def add_lesson(self, lesson: Lesson) -> None:
        with self._get_session() as session:
            session.add(lesson)
            session.commit()

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        with self._get_session() as session:
            return session.get(Lesson, lesson_id)

    def add_learning_objective(self, lo: LearningObjective) -> None:
        with self._get_session() as session:
            session.add(lo)
            session.commit()

    def get_learning_objective(self, lo_id: str) -> LearningObjective | None:
        with self._get_session() as session:
            return session.get(LearningObjective, lo_id)

    #--- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        with self._get_session() as session:
            q_id = kwargs.get('q_id')
            if q_id and session.get(Question, q_id):
                return
            
            # Handle Options
            options_data = kwargs.pop('options', [])
            attrs_data = kwargs.pop('attrs', None)
            
            q = Question(**kwargs)
            session.add(q)
            
            if options_data:
                for opt_data in options_data:
                    opt = Option(**opt_data, question_id=q_id)
                    session.add(opt)
            
            if attrs_data:
                attrs = QuestionAttributes(**attrs_data, q_id=q_id)
                session.add(attrs)
            
            session.commit()

    def get_question(self, q_ids: list | str) -> list[Question | None]:
        if isinstance(q_ids, str):
            q_ids = [q_ids]
        with self._get_session() as session:
            # Eager load options and attrs
            stmt = select(Question).where(Question.q_id.in_(q_ids))
            results = session.execute(stmt).scalars().all()
            # Ensure we return a list of the same length as q_ids, with Nones where missing
            q_map = {q.q_id: q for q in results}
            return [q_map.get(qid) for qid in q_ids]

    def list_questions(self) -> list[Question]:
        with self._get_session() as session:
            return session.execute(select(Question)).scalars().all()

    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        with self._get_session() as session:
            attr = session.get(QuestionAttributes, q_id)
            if not attr:
                attr = QuestionAttributes(q_id=q_id)
                session.add(attr)
            for key, value in attributes.items():
                setattr(attr, key, value)
            session.commit()

    #--- skills ---------------------------------------------------------
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None:
        with self._get_session() as session:
            skill = session.get(SkillCluster, skill_id)
            if not skill:
                skill = SkillCluster(skill_cluster_id=skill_id)
                session.add(skill)
            
            if similar_skills:
                # Using a separate mapping table for similarities
                # First, clear old similarities if any
                session.execute(
                    text("DELETE FROM skill_similarities WHERE skill_id = :id"),
                    {"id": skill_id}
                )
                for sim_id in similar_skills:
                    session.execute(
                        text("INSERT INTO skill_similarities (skill_id, similar_skill_id) VALUES (:id, :sim)"),
                        {"id": skill_id, "sim": sim_id}
                    )
            session.commit()

    def get_skill(self, skill_id: str) -> SkillCluster | None:
        with self._get_session() as session:
            skill = session.get(SkillCluster, skill_id)
            if skill:
                # Hydrate similar_skills list
                res = session.execute(
                    text("SELECT similar_skill_id FROM skill_similarities WHERE skill_id = :id"),
                    {"id": skill_id}
                ).scalars().all()
                skill.similar = list(res)
            return skill

    #--- student state ---------------------------------------------------
    def update_priors(self, student_id: str, skill_id: str, new_prior: float) -> None:
        with self._get_session() as session:
            # We use StudentMasteryLevel as the storage for priors
            mastery = session.query(StudentMasteryLevel).filter_by(
                student_id=student_id, skill_cluster_id=skill_id, current=True
            ).first()
            if mastery:
                mastery.mastery = new_prior
            else:
                # Mark previous as not current
                session.execute(
                    update(StudentMasteryLevel).where(
                        StudentMasteryLevel.student_id == student_id,
                        StudentMasteryLevel.skill_cluster_id == skill_id
                    ).values(current=False)
                )
                session.add(StudentMasteryLevel(
                    student_id=student_id, skill_cluster_id=skill_id, mastery=new_prior, current=True
                ))
            session.commit()

    def update_responses(self, student_id: str, responses: list[tuple]) -> None:
        with self._get_session() as session:
            for resp in responses:
                # resp = (match_id, date, q_id, response_value, error, timeout, response_time, stress_triggers, hints_used, atag)
                # Response entity only takes a subset
                response_obj = Response(
                    match_id=resp[0],
                    student_id=student_id,
                    question_id=resp[2],
                    response_time=resp[6] if resp[6] is not None else 0.0,
                    stress_triggers=resp[7] if resp[7] is not None else 0,
                    error=resp[4],
                    hints_used=resp[8],
                    time_out=resp[5]
                )
                session.add(response_obj)
            session.commit()

    def add_diagnosis(self, student_id: str, diagnoses: list,
                      deltas: dict | None = None,
                      timestamp: str | None = None) -> None:
        with self._get_session() as session:
            for diagnosis in diagnoses:
                session.add(StudentDiagnostic(student_id=student_id, diagnosis_id=diagnosis))
            # History is handled by snapshots in actual implementation
            # For now, just commit diagnostics
            session.commit()

    def update_content_gaps(self, student_id: str, content_gap_types: dict) -> None:
        with self._get_session() as session:
            # content_gap_types = {skill_id: gap_type}
            # We store this in StudentMasteryLevel
            for skill_id, gap_type in content_gap_types.items():
                mastery = session.query(StudentMasteryLevel).filter_by(
                    student_id=student_id, skill_cluster_id=skill_id, current=True
                ).first()
                if mastery:
                    mastery.content_gap_type = gap_type
                    mastery.content_gap = True
            session.commit()

    def add_priors_history(self, student_id: str, timestamp: str, priors: dict) -> None:
        # In a real DB, this would be a snapshot table
        pass

    def update_treatment_plan(self, student_id: str, timestamp: str,
                               treatment_name: str, parameters: dict) -> None:
        # Plan would be stored in a dedicated table
        pass

    #--- query ----------------------------------------------------------
    def query(self, parameters: list | dict) -> dict:
        if isinstance(parameters, dict):
            parameters = [parameters]
        
        with self._get_session() as session:
            stmt = select(Question)
            for cond in parameters:
                attr = cond.get('Attribute')
                op = cond.get('Operator', '==')
                threshold = cond.get('Threshold')
                
                # This is a simplified implementation. In a real one, 
                # we would resolve the attribute to the correct column.
                # For brevity, we assume it's a column on Question or its attributes.
                # We'll use a raw SQL approach for the complex logic or SQLAlchemy filters.
                pass 
            
            results = session.execute(stmt).scalars().all()
            return {q.q_id: q for q in results}

    #--- hints --------------------------------------------------------
    def add_pre_hint(self, pre_hint: PreHint) -> None:
        with self._get_session() as session:
            session.add(pre_hint)
            session.commit()

    def get_pre_hint(self, hint_id: str) -> PreHint | None:
        with self._get_session() as session:
            return session.get(PreHint, hint_id)

    def get_pre_hints_by_topic(self, topic: str) -> list[PreHint]:
        with self._get_session() as session:
            return session.execute(select(PreHint).where(PreHint.topic == topic)).scalars().all()

    def get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> list[PreHint]:
        with self._get_session() as session:
            return session.execute(
                select(PreHint).where(and_(PreHint.topic == topic, PreHint.content_gap_type == content_gap_type))
            ).scalars().all()

    def get_pre_hints_by_diagnosis(self, diagnosis: str) -> list[PreHint]:
        with self._get_session() as session:
            return session.execute(select(PreHint).where(PreHint.diagnosis == diagnosis)).scalars().all()

    def list_pre_hints(self) -> list[PreHint]:
        with self._get_session() as session:
            return session.execute(select(PreHint)).scalars().all()

    def add_post_hint(self, post_hint: PostHint) -> None:
        with self._get_session() as session:
            session.add(post_hint)
            session.commit()

    def get_post_hint(self, hint_id: str) -> PostHint | None:
        with self._get_session() as session:
            return session.get(PostHint, hint_id)

    def get_post_hints_by_question(self, question_id: str) -> list[PostHint]:
        with self._get_session() as session:
            return session.execute(
                select(PostHint).where(PostHint.question_id == question_id).order_by(PostHint.hint_order)
            ).scalars().all()

    def list_post_hints(self) -> list[PostHint]:
        with self._get_session() as session:
            return session.execute(select(PostHint)).scalars().all()

    def add_play_hint(self, play_hint: PlayHint) -> None:
        with self._get_session() as session:
            session.add(play_hint)
            session.commit()

    def get_play_hint(self, hint_id: str) -> PlayHint | None:
        with self._get_session() as session:
            return session.get(PlayHint, hint_id)

    def get_play_hints_by_question(self, q_id: str) -> list[PlayHint]:
        with self._get_session() as session:
            return session.execute(
                select(PlayHint).where(PlayHint.q_id == q_id).order_by(PlayHint.hint_order)
            ).scalars().all()

    def list_play_hints(self) -> list[PlayHint]:
        with self._get_session() as session:
            return session.execute(select(PlayHint)).scalars().all()
