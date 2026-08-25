"""In-memory DataRepository - the original Data_Service dict logic, verbatim.

Single-process reference implementation; swap for SqlAlchemyRepository
(PostgreSQL) without touching anything above the seam.
"""
from __future__ import annotations
import time

from services.log_setup import get_logger
from services.Entities import question, skill, student
from services.question_schema import resolve_attribute
from services.storage.base import DataRepository

log = get_logger('memory_repository')


class InMemoryRepository(DataRepository):

    def __init__(self) -> None:
        self.students: dict[str, student] = {}
        self.questions: dict[str, question] = {}
        self.skills: dict[str, skill] = {}

    #--- students -------------------------------------------------------
    def add_student(self, student_id: str) -> None:
        if student_id not in self.students:
            self.students[student_id] = student(student_id)

    def get_student(self, student_id: str) -> student | None:
        return self.students.get(student_id, None)

    def list_students(self) -> list[student]:
        return list(self.students.values())

    #--- questions ------------------------------------------------------
    def add_question(self, **kwargs) -> None:
        q_id = kwargs.get('q_id')
        if q_id is not None and q_id not in self.questions:
            self.questions[q_id] = question(**kwargs)

    def get_question(self, q_ids: list | str) -> list[question | None]:
        if isinstance(q_ids, str):
            q_ids = [q_ids]
        return [self.questions.get(q_id, None) for q_id in q_ids]

    def list_questions(self) -> list[question]:
        return list(self.questions.values())

    def update_question_attributes(self, q_id: str, attributes: dict) -> None:
        q = self.questions.get(q_id)
        if q is None:
            raise KeyError(f'question {q_id} not found')
        for name, value in attributes.items():
            setattr(q, name, value)

    #--- skills ---------------------------------------------------------
    def add_skill(self, skill_id: str, similar_skills: list | None = None) -> None:
        if skill_id not in self.skills:
            self.skills[skill_id] = skill(
                skill_id, similar_skills if similar_skills is not None else [])

    def get_skill(self, skill_id: str) -> skill | None:
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
            q_id, response_value = response[0], response[1]
            response_time = response[2] if len(response) > 2 else None
            stress_triggers = response[3] if len(response) > 3 else None
            atag = response[4] if len(response) > 4 else None
            stu.add_response(q_id, response_value, response_time,
                             stress_triggers, atag)

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
                return getattr(q, attr_name)
            #case-insensitive fallback (validated in _validate_attributes)
            for existing in vars(q):
                if existing.lower() == str(name).lower():
                    return getattr(q, existing)
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
