from __future__ import annotations
import numpy as np
from datetime import datetime
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.log_setup import get_logger
from services.knowledge_tracing import knowledge_tracing_engine
from services.Data_service import Data_Service
from services.behavioral_diagnosis.behavioral_diagnosis_engine import behavioral_diagnosis_engine
from services.Treatment_service import Treatment_Service
from services.Entities import student
from services.Interfaces import Mediator,Publisher

log = get_logger('Diagnosis_service')

#mediator
class Diagnosis_service(Mediator,Publisher):
    def __init__(self,KT_engine:knowledge_tracing_engine,BD_engine:behavioral_diagnosis_engine,
                 Data_service:Data_Service,Treatment_Service:Treatment_Service)->None:
        self.KT_engine=KT_engine
        self.KT_engine.mediator = self
        self.BD_engine=BD_engine
        self.BD_engine.mediator = self
        self.Data_service=Data_service
        self.Treatment_service=Treatment_Service
        self.Treatment_service.mediator=self
        self.calibration_window:int=10
        self.subscribers:defaultdict=defaultdict(list)
        log.info("constructed Diagnosis_service | KT=%s BD=%s treatment=%s data=%s",
                 type(KT_engine).__name__,type(BD_engine).__name__,
                 type(Treatment_Service).__name__,type(Data_service).__name__)

    def subscribe(self, subscriber, subscriber_type:str)->None:
        self.subscribers[subscriber_type].append(subscriber)
        log.info("subscribed %s to topic '%s' (%d subscribers)",
                 type(subscriber).__name__,subscriber_type,len(self.subscribers[subscriber_type]))

    def unsubscribe(self, subscriber, subscriber_type:str)->None:
        if subscriber in self.subscribers[subscriber_type]:
            self.subscribers[subscriber_type].remove(subscriber)
            log.info("unsubscribed %s from topic '%s'",type(subscriber).__name__,subscriber_type)

    def notify(self, event_type:str, event_data:dict)->None:
        event_subscriber_map:dict={
            'priors_updated':'Question_Bandit'
        }
        subscriber_type = event_subscriber_map.get(event_type)
        if not subscriber_type:
            log.debug("notify('%s') has no subscriber topic; ignored",event_type)
            return
        # Match_Service expects 'questions' key instead of 'q_ids'
        payload = dict(event_data)
        if 'q_ids' in payload:
            payload['questions'] = payload.pop('q_ids')

        targets=self.subscribers[subscriber_type]
        log.info("notify '%s' -> topic '%s', fanning out to %d subscriber(s) | data=%s",
                 event_type,subscriber_type,len(targets),payload)
        for subscriber in targets:
            subscriber.update(event_type,payload)

    _REQUEST_SUMMARIES:dict={
        # request_type -> formatter over the request dict (for concise logs)
        'get_student':      lambda r:f"student_id={r.get('student_id')}",
        'get_question':     lambda r:f"question_id={r.get('question_id')}",
        'get_skill':        lambda r:f"skill_id={r.get('skill_id')}",
        'add_diagnosis':    lambda r:f"student_id={r.get('student_id')} diagnoses={r.get('diagnosis')}",
        'add_response':     lambda r:f"student_id={r.get('student_id')} n={len(r.get('responses') or [])}",
        'update_prior':     lambda r:(f"student_id={r.get('student_id')} skill={r.get('skill_id')} "
                                      f"new_prior={r.get('new_prior')}"),
        'update_content_gaps':lambda r:f"student_id={r.get('student_id')}",
        'add_priors_history':lambda r:f"student_id={r.get('student_id')} ts={r.get('timestamp')}",
        'update_treatment_plan':lambda r:f"student_id={r.get('student_id')} name={r.get('treatment_name')}",
        'add_student':      lambda r:f"student_id={r.get('student_id')}",
        'add_question':     lambda r:f"q_id={(r.get('question') or {}).get('q_id')}",
        'add_skill':        lambda r:f"skill_id={r.get('skill_id')}",
    }

    @staticmethod
    def _summarize(request:dict)->str:
        fmt=Diagnosis_service._REQUEST_SUMMARIES.get(request.get('type'))
        try:
            return fmt(request) if fmt else str(request)
        except Exception:
            return str(request)

    def request(self,request:dict)->None:
        # the request has a request_type
        #theres a switch case that directs control flow to a bunch of functions
        rtype=request.get('type','none')
        if rtype=='get_student':
            return self.Data_service.get_student(request.get('student_id'))
        if rtype=='get_question':
            result=self.Data_service.get_question(request.get('question_id'))
            #scalar id -> single entity (engines expect one question, not a list)
            if isinstance(request.get('question_id'),str):
                result=result[0] if result else None
            log.debug("request 'get_question' | question_id=%s",
                      request.get('question_id'))
            return result
        if rtype=='get_skill':
            log.debug("request 'get_skill' | skill_id=%s",request.get('skill_id'))
            return self.Data_service.get_skill(request.get('skill_id'))
        if rtype=='add_diagnosis':
            self.Data_service.add_diagnosis(request.get('student_id'),request.get('diagnosis'),deltas=request.get('deltas'),timestamp=request.get('timestamp'))
        elif rtype=='add_response':
            self.Data_service.update_responses(request.get('student_id'),request.get('responses'))
        elif rtype=='update_prior':
            self.Data_service.update_priors(request.get('student_id'),request.get('skill_id'),request.get('new_prior'))
        elif rtype=='update_content_gaps':
            self.Data_service.update_content_gaps(request.get('student_id'),request.get('content_gap_types'))
        elif rtype=='add_priors_history':
            self.Data_service.add_priors_history(request.get('student_id'),request.get('timestamp'),request.get('priors'))
        elif rtype=='update_treatment_plan':
            self.Data_service.update_treatment_plan(request.get('student_id'),request.get('timestamp'),request.get('treatment_name'),request.get('parameters'))
        elif rtype=='add_student':
            self.Data_service.add_student(request.get('student_id'))
        elif rtype=='add_question':
            self.Data_service.add_question(**request['question'])
        elif rtype=='add_skill':
            self.Data_service.add_skill(request.get('skill_id'),request.get('similar_skills'))
        elif rtype=='add_pre_hint':
            self.Data_service.add_pre_hint(request.get('pre_hint'))
        elif rtype=='add_post_hint':
            self.Data_service.add_post_hint(request.get('post_hint'))
        elif rtype=='add_play_hint':
            self.Data_service.add_play_hint(request.get('play_hint'))
        else:
            log.warning("request with unknown type '%s' dropped | request=%s",rtype,request)
            return
        log.info("request '%s' handled | %s",rtype,self._summarize(request))

           

    def add_student_response(self, student_id, responses):
        """
        Add student responses with error and atag calculation.

        Args:
            student_id: The student's ID
            responses: List of 10-tuples:
                (match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag)
        """
        new_responses = []
        for response_data in responses:
            processed = self._process_response(response_data)
            new_responses.append(processed)

        self.Data_service.update_responses(student_id, new_responses)

        student = self.Data_service.get_student(student_id)
        n_responses = len(student.get_responses())
        q_ids = [r[2] for r in new_responses]  # q_id is at index 2

        log.info("add_student_response | student_id=%s +%d responses (total=%d)",
                 student_id, len(new_responses), n_responses)

        if n_responses == self.calibration_window:
            log.info("calibration window reached (%d) | calibrating priors for %s",
                     self.calibration_window, student_id)
            self.KT_engine.calibrate_priors(student_id)
        elif n_responses > self.calibration_window:
            binary_responses = [1 if bool(r[3]) else 0 for r in new_responses]  # response at index 3
            self.KT_engine.update_student_priors(student_id, q_ids, binary_responses)
            self.KT_engine.update_content_gaps(student_id)
            self.Data_service.add_priors_history(student_id,
                                 datetime.now().isoformat(), student.get_priors().copy())

        if n_responses >= 100 and not student.get_diagnoses_history():
            log.info("first behavioral diagnosis triggered at %d responses | student_id=%s",
                     n_responses, student_id)
            self.BD_engine.diagnose_student(student_id)
            self.Treatment_service.set_treatment_plan(student_id)

        if student.get_diagnoses_history() and (n_responses % 10 == 0) and n_responses > 0:
            log.info("periodic re-diagnosis at %d responses | student_id=%s",
                     n_responses, student_id)
            self.BD_engine.diagnose_student(student_id)
            self.Treatment_service.update_treatment_plan(student_id)
            self.notify('priors_updated', {'student_id': student_id, 'q_ids': q_ids})

        return f"Added {len(new_responses)} responses to {student_id}"

    def _process_response(self, response_data: tuple) -> tuple:
        """Process a single response, calculating error and atag if not already set.

        Args:
            response_data: 10-tuple (match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag)

        Returns:
            Processed 10-tuple with error and atag calculated
        """
        match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag = response_data

        result = self.Data_service.get_question(q_id)
        if isinstance(result, list):
            question = result[0] if result else None
        else:
            question = result

        if question is None:
            log.warning("_process_response: question %s not found", q_id)
            return response_data

        if error is None:
            error = self._calculate_error(question, response)

        if atag is None:
            atag = self._calculate_atag(question, response, error)

        return (match_id, date, q_id, response, error, timeout, time, stress_triggers, hints_used, atag)

    def _calculate_error(self, question, response) -> bool:
        """Determine if response is correct by comparing to correct option(s)."""
        question_type = getattr(question, 'question_type', 'mcq')

        if question_type == 'mcq':
            if isinstance(response, int) and 0 <= response < len(question.options):
                return bool(question.options[response].is_correct)
            return False

        elif question_type == 'true_false':
            for opt in question.options:
                if opt.is_correct:
                    correct_str = str(opt.content).lower()
                    response_str = str(response).lower()
                    return correct_str == response_str
            return False

        elif question_type == 'ordering':
            return self._check_ordering_correct(question, response)

        elif question_type == 'matching':
            return self._check_matching_correct(question, response)

        elif question_type == 'fill_blank':
            return self._check_fillblank_correct(question, response)

        elif question_type == 'drag_drop':
            return self._check_drag_drop_correct(question, response)

        return False

    def _check_ordering_correct(self, question, response) -> bool:
        """Check if ordering response matches correct order."""
        if not isinstance(response, list):
            return False
        correct_order = [opt.content for opt in sorted(question.options, key=lambda x: getattr(x, 'display_order', 0))]
        return response == correct_order

    def _check_matching_correct(self, question, response) -> bool:
        """Check if matching response is correct.

        Response is a dict with 'left_items' and 'right_items' lists.
        Correct answer is when left_items[i] matches right_items[i] in the same order.
        """
        if not isinstance(response, dict):
            return False
        left_items = response.get('left_items', [])
        right_items = response.get('right_items', [])

        for opt in question.options:
            if opt.is_correct:
                parts = str(opt.content).split(' - ')
                if len(parts) == 2:
                    correct_left = parts[0].strip()
                    correct_right = parts[1].strip()
                    for i, (left, right) in enumerate(zip(left_items, right_items)):
                        if left == correct_left and right != correct_right:
                            return False
        return True

    def _check_fillblank_correct(self, question, response) -> bool:
        """Check if fill blank response matches any correct answer."""
        if not isinstance(response, str):
            return False
        response_lower = response.lower().strip()
        for opt in question.options:
            if opt.is_correct:
                correct_answers = [a.strip().lower() for a in str(opt.content).split(',')]
                if response_lower in correct_answers:
                    return True
        return False

    def _check_drag_drop_correct(self, question, response) -> bool:
        """Check if drag and drop response matches correct answers."""
        if not isinstance(response, list):
            return False
        for opt in question.options:
            if opt.is_correct:
                correct_answers = [a.strip().lower() for a in str(opt.content).split(',')]
                response_lower = [str(r).lower().strip() for r in response]
                return response_lower == correct_answers
        return False

    def _calculate_atag(self, question, response, error: bool) -> str:
        """Calculate atag based on question type and error category.

        Rules:
            Question Type       Rule
            Ordering            Only two adjacent steps swapped → Gap_Prior
                                Fully scrambled order → Gap_Absence
            Matching            Term matched to different, closely related term from same lesson → Gap_Concept
                                Term matched to definition from older, unrelated lesson → Gap_Prior
            Fill in the Blank  Answer text matches term from completely different, unrelated unit → Gap_Absence
            Drag & Drop        Element placed in logically opposite position → Gap_Misconception
                                Fully random placement → Gap_Absence
        """
        if not error:
            return "Correct"

        question_type = getattr(question, 'question_type', 'mcq')

        if question_type == 'mcq':
            if isinstance(response, int) and 0 <= response < len(question.options):
                opt = question.options[response]
                if hasattr(opt, 'content_gap_type') and opt.content_gap_type:
                    return opt.content_gap_type
            return "Gap_Prior"

        elif question_type == 'true_false':
            for opt in question.options:
                if opt.is_correct:
                    if hasattr(opt, 'content_gap_type') and opt.content_gap_type:
                        return opt.content_gap_type
            return "Gap_Prior"

        elif question_type == 'ordering':
            return self._atag_ordering(question, response)

        elif question_type == 'matching':
            return self._atag_matching(question, response)

        elif question_type == 'fill_blank':
            return self._atag_fill_blank(question, response)

        elif question_type == 'drag_drop':
            return self._atag_drag_drop(question, response)

        return "Gap_Prior"

    def _atag_ordering(self, question, response) -> str:
        """Determine atag for ordering question based on error type."""
        if not isinstance(response, list):
            return "Gap_Prior"

        correct_order = [opt.content for opt in sorted(question.options, key=lambda x: getattr(x, 'display_order', 0))]
        if len(response) != len(correct_order):
            return "Gap_Prior"

        swap_count = 0
        for i, (r, c) in enumerate(zip(response, correct_order)):
            if r != c:
                swap_count += 1

        if swap_count == 1:
            return "Gap_Prior"
        else:
            return "Gap_Absence"

    def _atag_matching(self, question, response) -> str:
        """Determine atag for matching question based on error type."""
        if not isinstance(response, dict):
            return "Gap_Prior"

        left_items = response.get('left_items', [])
        right_items = response.get('right_items', [])

        for i, (left, right) in enumerate(zip(left_items, right_items)):
            for opt in question.options:
                if opt.is_correct:
                    parts = str(opt.content).split(' - ')
                    if len(parts) == 2:
                        correct_left = parts[0].strip()
                        correct_right = parts[1].strip()
                        if left == correct_left and right != correct_right:
                            return "Gap_Concept"

        return "Gap_Prior"

    def _atag_fill_blank(self, question, response) -> str:
        """Determine atag for fill blank question based on error type."""
        if not isinstance(response, str):
            return "Gap_Absence"

        return "Gap_Absence"

    def _atag_drag_drop(self, question, response) -> str:
        """Determine atag for drag and drop question based on error type."""
        if not isinstance(response, list):
            return "Gap_Absence"

        for opt in question.options:
            if opt.is_correct:
                correct_answers = [a.strip().lower() for a in str(opt.content).split(',')]
                response_lower = [str(r).lower().strip() for r in response]

                if len(response_lower) != len(correct_answers):
                    return "Gap_Absence"

                opposite_count = 0
                for r, c in zip(response_lower, correct_answers):
                    if r != c:
                        opposite_count += 1

                if opposite_count == 1:
                    return "Gap_Misconception"
                else:
                    return "Gap_Absence"

        return "Gap_Absence"


    
    


