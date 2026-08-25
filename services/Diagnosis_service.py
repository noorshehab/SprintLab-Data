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
from services.Entities import student, question
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
        else:
            log.warning("request with unknown type '%s' dropped | request=%s",rtype,request)
            return
        log.info("request '%s' handled | %s",rtype,self._summarize(request))

           

    def add_student_response(self, student_id, q_ids, responses,timings,stress_triggers):
        response_index = 0
        new_responses = []
        for q_id, response,time,stress_trigger in zip(q_ids,responses,timings,stress_triggers):
            new_responses.append((q_id,response,time,stress_trigger))
            response_index += 1
        self.Data_service.update_responses(student_id,new_responses)

        student = self.Data_service.get_student(student_id)
        n_responses=len(student.get_responses())
        log.info("add_student_response | student_id=%s +%d responses (total=%d)",
                 student_id,response_index,n_responses)
        if n_responses==self.calibration_window:
            log.info("calibration window reached (%d) | calibrating priors for %s",
                     self.calibration_window,student_id)
            self.KT_engine.calibrate_priors(student_id)
        elif n_responses>self.calibration_window:
            self.KT_engine.update_student_priors(student_id,q_ids,responses)
            self.KT_engine.update_content_gaps(student_id)
            self.Data_service.add_priors_history(student_id,
                                 datetime.now().isoformat(),student.get_priors().copy())

        #behavioral diagnosis is only run once the student has 100 responses
        if n_responses>=100 and not student.get_diagnoses_history():
            log.info("first behavioral diagnosis triggered at %d responses | student_id=%s",
                     n_responses,student_id)
            self.BD_engine.diagnose_student(student_id)
            self.Treatment_service.set_treatment_plan(student_id)

        #runs every 10 responses after initial diagnosis
        if student.get_diagnoses_history() and (n_responses%10==0) and n_responses>0:
            log.info("periodic re-diagnosis at %d responses | student_id=%s",
                     n_responses,student_id)
            self.BD_engine.diagnose_student(student_id)
            self.Treatment_service.update_treatment_plan(student_id)
            self.notify('priors_updated',{'student_id':student_id,'q_ids':q_ids})


        return f"Added {response_index} responses to {student_id}"    
  
    def predict_response(self, student_id, q_id):
        p_C, pred = self.KT_engine.predict_response(student_id,q_id)
        return p_C, pred

    def diagnose_student(self,student_id):
        return self.BD_engine.diagnose_student(student_id)

    
    
    
    


