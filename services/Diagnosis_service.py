import numpy as np
from knowledge_tracing import knowledge_tracing_engine
from services.Data_service import Data_Service
from services.behavioral_diagnosis.behavioral_diagnosis_engine import behavioral_diagnosis_engine
from services.Interfaces import Mediator

#mediator
class Diagnosis_service(Mediator):
    def __init__(self,KT_engine:knowledge_tracing_engine,BD_engine:behavioral_diagnosis_engine,Data_service:Data_Service)->None:
        self.KT_engine=KT_engine
        self.KT_engine.mediator = self
        self.BD_engine=BD_engine
        self.BD_engine.mediator = self
        self.Data_service=Data_service
        self.calibration_window=10
        self.f100_diagnoses = {}

    #what are the requests sent to the mediator from the components
    #kt: get student or just responses?,get skill, get question, update priors, add predicted responses to student? 
    def request(self,sender:object,request:dict[str:any]):
        # the request has a request_type
        #theres a switch case that directs control flow to a bunch of functions
        type=request.get('type','none')
        if type=='get_student':
            return self.Data_service.get_student(request.get('student_id'))
        if type=='get_question':
            return self.Data_service.get_question(request.get('question_id'))
        if type=='get_skill':
            return self.Data_service.get_skill(request.get('skill_id'))
        if type=='add_diagnosis':
            self.Data_service.add_diagnosis(request.get('student_id'),request.get('diagnosis'))
        if type=='add_response':
            self.Data_service.update_responses(request.get('student_id'),request.get('responses'))
        if type=='update_prior':
            self.Data_service.update_priors(request.get('student_id'),request.get('skill_id'),request.get('new_prior'))
        if type=='add_student':
            self.Data_service.add_student(request.get('student_id'))
        if type=='add_question':
            self.Data_service.add_question(**request['question'])
        if type=='add_skill':
            self.Data_service.add_skill(request.get('skill_id'),request.get('similar_skills'))
       

    def add_student_response(self, student_id, q_ids, responses,timings,stress_triggers):
        response_index = 0
        new_responses = []
        for q_id, response,time,stress_trigger in zip(q_ids,responses,timings,stress_triggers):
            new_responses.append((q_id,response,time,stress_trigger))
            response_index += 1
        self.Data_service.update_responses(student_id,new_responses)

        student = self.request(self, {'type':'get_student','student_id':student_id})
        if len(student.get_responses())==self.calibration_window:
            self.KT_engine.calibrate_priors(student_id)
        elif len(student.get_responses())>self.calibration_window:
            self.KT_engine.update_student_priors(student_id,q_ids,responses)

        #behavioral diagnosis is only run once the student has 100 responses
        if len(student.get_responses())==100 and student_id not in self.f100_diagnoses:
            self.f100_diagnoses[student_id] = self.BD_engine.diagnose_student(student_id)

        return f"Added {response_index} responses to {student_id}"    
  
    def predict_response(self, student_id, q_id):
        p_C, pred = self.KT_engine.predict_response(student_id,q_id)
        return p_C, pred

    def diagnose_student(self,student_id):
        return self.BD_engine.diagnose_student(student_id)

    
    
    
    


