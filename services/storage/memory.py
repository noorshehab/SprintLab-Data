"""In-memory DataRepository - the original Data_Service dict logic, verbatim.

Single-process reference implementation; swap for SqlAlchemyRepository
(PostgreSQL) without touching anything above the seam.
"""
from __future__ import annotations
import time

from services.log_setup import get_logger
from services.Entities import Question, SkillCluster, Student, PreHint, PostHint, PlayHint, Unit, Lesson, LearningObjective
from services.question_schema import resolve_attribute
from services.storage.base import DataRepository

log = get_logger('memory_repository')


class InMemoryRepository(DataRepository):

    def __init__(self) -> None:
        self.students: dict[str, Student] = {}
        self.questions: dict[str, Question] = {}
        self.skills: dict[str, SkillCluster] = {}
        self.units: dict[str, Unit] = {}
        self.lessons: dict[str, Lesson] = {}
        self.learning_objectives: dict[str, LearningObjective] = {}
        self.pre_hints: dict[str, PreHint] = {}
        self.post_hints: dict[str, PostHint] = {}
        self.play_hints: dict[str, PlayHint] = {}

    #--- students -------------------------------------------------------
    def add_student(self, student_id: str) -> None:
        if student_id not in self.students:
             self.students[student_id] = Student(student_id)


    def get_student(self, student_id: str) -> Student | None:
        return self.students.get(student_id, None)

    def list_students(self) -> list[Student]:
        return list(self.students.values())

    #--- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        q_id = kwargs.get('q_id')
        if q_id is not None and q_id not in self.questions:
            self.questions[q_id] = Question(**kwargs)

    def get_question(self, q_ids: list | str) -> list[Question | None]:
        if isinstance(q_ids, str):
            q_ids = [q_ids]
        return [self.questions.get(q_id, None) for q_id in q_ids]

    def list_questions(self) -> list[Question]:
        return list(self.questions.values())

    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        q = self.questions.get(q_id)
        if q is None:
            raise KeyError(f'question {q_id} not found')
        for name, value in attributes.items():
            setattr(q.attrs, name, value)

    #--- curriculum ---------------------------------------------------------
    def add_unit(self, unit: Unit) -> None:
        self.units[unit.unit_id] = unit

    def get_unit(self, unit_id: str) -> Unit | None:
        return self.units.get(unit_id)

    def add_lesson(self, lesson: Lesson) -> None:
        self.lessons[lesson.lesson_id] = lesson

    def get_lesson(self, lesson_id: str) -> Lesson | None:
        return self.lessons.get(lesson_id)

    def add_learning_objective(self, lo: LearningObjective) -> None:
        self.learning_objectives[lo.learning_objective_id] = lo

    def get_learning_objective(self, lo_id: str) -> LearningObjective | None:
        return self.learning_objectives.get(lo_id)

    #--- skills ---------------------------------------------------------
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None:
        if skill_id not in self.skills:
             self.skills[skill_id] = SkillCluster(
                skill_cluster_id=skill_id,
                similar_skills=similar_skills if similar_skills is not None else [])

    def get_skill(self, skill_id: str) -> SkillCluster | None:
        return self.skills.get(skill_id, None)

    #--- student state ---------------------------------------------------
    def update_priors(self, student_id: str, skill_id: str, new_prior: float) -> None:
        stu = self.students[student_id]
        old_prior = stu.get_priors().get(skill_id)
        stu.update_prior(skill_id, new_prior)
        timestamp = time.time()
        self.add_priors_history(student_id, timestamp, {skill_id: old_prior})

    def update_responses(self, student_id: str, responses: list[tuple]) -> None:
        stu = self.students[student_id]
        for response in responses:
            match_id = response[0] if len(response) > 0 else None
            date = response[1] if len(response) > 1 else None
            q_id = response[2] if len(response) > 2 else None
            response_value = response[3] if len(response) > 3 else None
            error = response[4] if len(response) > 4 else None
            timeout = response[5] if len(response) > 5 else False
            response_time = response[6] if len(response) > 6 else None
            stress_triggers = response[7] if len(response) > 7 else None
            hints_used = response[8] if len(response) > 8 else 0
            atag = response[9] if len(response) > 9 else None
            stu.add_response(match_id, date, q_id, response_value, error, timeout,
                             response_time, stress_triggers, hints_used, atag)

    def add_diagnosis(self, student_id: str, diagnoses: list,
                      deltas: dict | None = None,
                      timestamp: str | None = None) -> None:
        stu = self.students[student_id]
        for diagnosis in diagnoses:
            stu.add_diagnosis(diagnosis)
        if deltas is not None and timestamp is not None:
            stu.add_diagnosis_record(timestamp, deltas, diagnoses)

    def update_content_gaps(self, student_id: str, content_gap_types: dict) -> None:
        self.students[student_id].update_content_gaps(content_gap_types)

    def add_priors_history(self, student_id: str, timestamp: str, priors: dict) -> None:
        self.students[student_id].add_priors_record(timestamp, priors)

    def update_treatment_plan(self, student_id: str, timestamp: str,
                              treatment_name: str, parameters: dict) -> None:
        self.students[student_id].update_treatment_plan(timestamp, treatment_name, parameters)

    def get_responses(self, student_id: str) -> list[tuple]:
        stu = self.students.get(student_id)
        return list(stu.get_responses()) if stu else []

    def count_responses(self, student_id: str) -> int:
        stu = self.students.get(student_id)
        return len(stu.get_responses()) if stu else 0

    def get_priors(self, student_id: str) -> dict[str, float]:
        stu = self.students.get(student_id)
        return dict(stu.get_priors()) if stu else {}

    def get_priors_history(self, student_id: str, window: int = 2) -> dict:
        stu = self.students.get(student_id)
        if not stu:
            return {}
        history = stu.get_priors_history()
        keys = list(history.keys())[-window:]
        return {k: dict(history[k]) for k in keys}

    def get_diagnoses(self, student_id: str) -> list[str]:
        stu = self.students.get(student_id)
        return list(stu.get_diagnoses()) if stu else []

    def get_diagnoses_history(self, student_id: str, window: int = 2) -> list[dict]:
        stu = self.students.get(student_id)
        if not stu:
            return []
        deltas = stu.get_deltas()
        history = stu.get_diagnoses_history()
        keys = list(history.keys())[-window:]
        return [
            {'timestamp': k, 'diagnoses': list(history[k]), 'deltas': dict(deltas.get(k, {}))}
            for k in keys
        ]

    def get_content_gaps(self, student_id: str) -> dict[str, str]:
        stu = self.students.get(student_id)
        return dict(stu.get_content_gaps()) if stu else {}

    def get_treatment_plan(self, student_id: str) -> dict | None:
        stu = self.students.get(student_id)
        if not stu:
            return None
        stored = stu.get_treatment_plan().get('treatment_plan')
        return stored[1] if stored else None

    def replace_treatment_plan(self, student_id: str, timestamp: str,
                               treatment_plan: dict) -> None:
        self.students[student_id].update_treatment_plan(timestamp, 'treatment_plan', treatment_plan)

    def create_match(self, match_id: str, match_type: str,
                     num_participants: int = 1) -> None:
        return None

    #--- query ----------------------------------------------------------
    def query(self, parameters: list | dict) -> dict:
        """AND-conjoined condition query over questions.

        parameters: a single condition dict or a list of them. Each condition:

            {'Topic': 'general' | <skill_cluster_id>,
             'Attribute': <question attribute name>,
             'Operator': '==' | '!=' | '>' | '>=' | '<' | '<=' | 'in',
             'Threshold': <value to compare against>}

        Topic 'general' (or None/'all') searches without a skill-cluster filter;
        any other value restricts to that skill cluster. All conditions must
        hold (AND). Attributes resolve via services/question_schema.py aliases.
        """
        if isinstance(parameters, dict):
            parameters = [parameters]
        if not parameters:
            return {}

        def _norm(v):
            if isinstance(v, str):
                s = v.strip()
                low = s.lower()
                if low in ('true', 'yes', 'y', 't'):
                    return True
                if low in ('false', 'no', 'n', 'f'):
                    return False
                try:
                    return int(s)
                except ValueError:
                    pass
                try:
                    return float(s)
                except ValueError:
                    return v
            return v

        def _apply(operator, value, threshold):
            v, t = _norm(value), _norm(threshold)
            op = operator.strip()
            if op == '==':
                if isinstance(v, str) and isinstance(t, str):
                    return v.lower() == t.lower()
                return v == t
            if op == '!=':
                return not _apply('==', value, threshold)
            if op == 'in':
                if isinstance(t, str):
                    t = [x.strip() for x in t.split(',')]
                if isinstance(value, str):
                    return value.lower() in {str(x).lower() for x in t}
                return value in t
            numeric_ops = {'>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
                           '<': lambda a, b: a < b, '<=': lambda a, b: a <= b}
            if op in numeric_ops:
                try:
                    v_num, t_num = float(v), float(t)
                except (TypeError, ValueError):
                    return False
                return numeric_ops[op](v_num, t_num)
            return False

        def _validate_attributes() -> None:
            """Fail fast on condition attributes that resolve to nothing."""
            sample = next(iter(self.questions.values()), None)
            for cond in parameters:
                name = cond.get('Attribute')
                if resolve_attribute(name) is not None:
                    continue
                if sample is not None and any(
                        existing.lower() == str(name).lower()
                        for existing in vars(sample)):
                    continue
                raise ValueError(
                    f"query condition references unknown question attribute "
                    f"'{name}'; see services/question_schema.py")

        def _attribute(q, name):
            canonical = resolve_attribute(name)
            attr_name = canonical if canonical is not None else str(name)
            if hasattr(q, attr_name):
                val = getattr(q, attr_name)
                # Ensure numeric attributes are never None to prevent crashes in comparisons
                if attr_name in ('difficulty_level', 'p_g', 'p_s', 'p_t', 'cognitive_load', 'cognitive_load_index'):
                    return val if val is not None else 0.0
                return val
            #case-insensitive fallback (validated in _validate_attributes)
            for existing in vars(q):
                if existing.lower() == str(name).lower():
                    val = getattr(q, existing)
                    if attr_name in ('difficulty_level', 'p_g', 'p_s', 'p_t', 'cognitive_load', 'cognitive_load_index'):
                        return val if val is not None else 0.0
                    return val
            return None

        _validate_attributes()

        def match(q, cond):
            topic = cond.get('Topic')
            if topic is not None and str(topic).lower() not in (
                    'general', 'skill_cluster_id'):
                cluster = str(topic)
                cluster_ids = [str(s) for s in (q.skill_ids or [])]
                if str(q.skill_cluster_id) != cluster and cluster not in cluster_ids:
                    return False
            value = _attribute(q, cond.get('Attribute'))
            operator = str(cond.get('Operator', '=='))
            threshold = cond.get('Threshold')
            return _apply(operator, value, threshold)

        matches = {q.id: q for q in self.questions.values()
                   if all(match(q, c) for c in parameters)}
        log.info("query | %d condition(s) over %d questions -> %d match(es)",
                 len(parameters), len(self.questions), len(matches))
        if not matches:
            log.warning("query matched nothing | conditions=%s", parameters)
        return matches

    #--- pre-hints --------------------------------------------------------
    def add_pre_hint(self, pre_hint: PreHint) -> None:
        self.pre_hints[pre_hint.id] = pre_hint

    def get_pre_hint(self, hint_id: str) -> PreHint | None:
        return self.pre_hints.get(hint_id)

    def get_pre_hints_by_topic(self, topic: str) -> list[PreHint]:
        return [h for h in self.pre_hints.values() if h.topic == topic]

    def get_pre_hints_by_content_gap_type(self, topic: str, content_gap_type: str) -> list[PreHint]:
        return [h for h in self.pre_hints.values()
                if h.topic == topic and h.content_gap_type == content_gap_type]

    def get_pre_hints_by_diagnosis(self, diagnosis: str) -> list[PreHint]:
        return [h for h in self.pre_hints.values() if h.diagnosis == diagnosis]

    def list_pre_hints(self) -> list[PreHint]:
        return list(self.pre_hints.values())

    #--- post-hints -------------------------------------------------------
    def add_post_hint(self, post_hint: PostHint) -> None:
        self.post_hints[post_hint.id] = post_hint

    def get_post_hint(self, hint_id: str) -> PostHint | None:
        return self.post_hints.get(hint_id)

    def get_post_hints_by_question(self, question_id: str) -> list[PostHint]:
        hints = [h for h in self.post_hints.values() if h.question_id == question_id]
        return sorted(hints, key=lambda h: h.hint_order)

    def list_post_hints(self) -> list[PostHint]:
        return list(self.post_hints.values())

    #--- play-hints -------------------------------------------------------
    def add_play_hint(self, play_hint: PlayHint) -> None:
        self.play_hints[play_hint.id] = play_hint

    def get_play_hint(self, hint_id: str) -> PlayHint | None:
        return self.play_hints.get(hint_id)

    def get_play_hints_by_question(self, q_id: str) -> list[PlayHint]:
        hints = [h for h in self.play_hints.values() if h.q_id == q_id]
        return sorted(hints, key=lambda h: h.hint_order)

    def list_play_hints(self) -> list[PlayHint]:
        return list(self.play_hints.values())
