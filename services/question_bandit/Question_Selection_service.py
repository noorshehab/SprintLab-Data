#mediated by the match service
#takes the treatment plan and generates query that gets the candidate set of questions from the data service
#runs the mab on the candidate questions to find the optimal set for the match
#arranges the optimal set of questions according to the treatment plan
from __future__ import annotations
from services.log_setup import get_logger
from services.Interfaces import Component
from services.question_bandit.MAB import ContextualBandit
from services.question_bandit.context import build_question_context
import pandas as pd

log=get_logger('Question_Selection_Service')

class Question_Selection_Service(Component):
    def __init__(self,CB:ContextualBandit,M_S=None)->None:
        self.mediator=M_S
        self.Bandit=CB
        log.info("constructed Question_Selection_Service | bandit=%s mediator=%s",
                 type(CB).__name__,type(M_S).__name__ if M_S else None)

    def get_candidate_questions(self,student_id:str)->dict:
        #get the student and their treatment plan
        student=self.mediator.request({'type': 'get_student', 'student_id': student_id})
        stored=student.get_treatment_plan()
        treatment=stored['treatment_plan'][1] if 'treatment_plan' in stored else {}
        #unwrap the treatment plan for the student to get query parameters
        parameters=self._unwrap_treatment_plan(treatment)
        #send a query request to the match service
        candidates=self.mediator.request({'type': 'query', 'parameters': parameters}) or {}
        log.info("get_candidate_questions | student_id=%s conditions=%d matches=%d",
                 student_id,len(parameters),len(candidates))
        if not candidates:
            log.warning("no candidate questions matched | student_id=%s conditions=%s",
                        student_id,parameters)
        return candidates

    def _unwrap_treatment_plan(self,treatment:dict)->list[dict]:
        #format: {'general':{<diagnosis>:{param:value,'Operator':op}},
        #         'specific':{<skill_id>:{param:value,'Operator':op}}}
        conditions=[]
        #general entries are behavioural/cognitive constraints, never skill clusters
        for topic,params in treatment.get('general',{}).items():
            if not isinstance(params,dict):
                params={'Attribute':topic,'Threshold':params}
            operator=params.get('Operator','<=')
            conditions+=[{'Topic':'general','Attribute':k,'Operator':operator,'Threshold':v}
                         for k,v in params.items() if k!='Operator']
        #specific entries are keyed by skill id and must filter on that cluster
        for skill_id,params in treatment.get('specific',{}).items():
            if not isinstance(params,dict):
                params={'Attribute':str(skill_id),'Threshold':params}
            operator=params.get('Operator','<=')
            conditions+=[{'Topic':str(skill_id),'Attribute':k,'Operator':operator,'Threshold':v}
                         for k,v in params.items() if k!='Operator']
        return conditions

    def get_optimal_set(self,student_id:str)->pd.DataFrame:
        candidates=self.get_candidate_questions(student_id)
        if not candidates:
            return pd.DataFrame({'q_id':[],'score':[]})
        student=self.mediator.request({'type': 'get_student', 'student_id': student_id})
        student_context={'id':student.id,'priors':student.get_priors()}
        question_contexts=build_question_context(candidates.values())
        selected=self.Bandit.select(student_context,question_contexts)
        log.info("get_optimal_set | student_id=%s scored=%d selected=%d",
                 student_id,len(question_contexts),len(selected))
        return selected

    def update(self,student_context:dict,question_contexts:pd.DataFrame,rewards:list)->None:
        self.Bandit.update(student_context,question_contexts,rewards)
