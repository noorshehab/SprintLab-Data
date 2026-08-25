from __future__ import annotations
#the function that takes the first n responses and outputs priors for the skills and all their similar counterparts
#the call to predict

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.log_setup import get_logger
from services.knowledge_tracing.BKT import next_response,update_prior
from services.Interfaces import Component
from collections import defaultdict

log=get_logger('knowledge_tracing_engine')


class knowledge_tracing_engine(Component):
    def __init__(self,D_S=None,calibration_window:int=10)->None:
        self.calibration_window:int = calibration_window
        self.mediator = D_S  
        log.info("constructed knowledge_tracing_engine | calibration_window=%d mediator=%s",
                 calibration_window,type(D_S).__name__ if D_S else None)

    
    
    def calibrate_priors(self, student_id:str)->None:
        direct_correct_multiplier = 1.1
        direct_incorrect_multiplier = 0.90
        similar_correct_multiplier = 1.05
        similar_incorrect_multiplier = 0.95

        student = self.mediator.request( {'type': 'get_student', 'student_id': student_id})
        responses = student.get_responses()[: self.calibration_window]

        log.info("calibrate_priors | student_id=%s responses=%d",student_id,len(responses))
        for q_id, response, *_ in responses:
            question = self.mediator.request( {'type': 'get_question', 'question_id': q_id})
            for kc_id in question.get_skills():
                prior = student.priors.get(kc_id, 0.5)
                if response:  # correct
                    new_prior = prior * direct_correct_multiplier
                else:         # incorrect
                    new_prior = prior * direct_incorrect_multiplier

                new_prior = max(0.01, min(0.99, new_prior))
                log.debug("calibrate_priors | student_id=%s skill=%s prior %.3f -> %.3f (q_id=%s)",
                          student_id,kc_id,prior,new_prior,q_id)
                self.mediator.request({'type': 'update_prior', 'student_id': student_id, 'skill_id': kc_id, 'new_prior': new_prior})

                # apply proportional change to similar skills (with smaller multiplier)
                skill = self.mediator.request({'type': 'get_skill', 'skill_id': kc_id})
                if skill:
                    for sk in skill.get_similar():
                        if response:
                            adjusted = prior * similar_correct_multiplier
                        else:
                            adjusted = prior * similar_incorrect_multiplier
                        adjusted = max(0.01, min(0.99, adjusted))
                        # update similar skill `sk` (not kc_id) via mediator
                        self.mediator.request( {'type': 'update_prior', 'student_id': student_id, 'skill_id': sk, 'new_prior': adjusted})
        log.info("calibrate_priors complete | student_id=%s skills=%d",
                 student_id,len(student.get_priors()))

    
    def update_student_priors(self,student_id:str,q_ids:list,responses:list)->None:
        #how do we update similar skills
        #take the %change in the prior for a kc for example +1% or -5% and apply to similar skills
        student = self.mediator.request({'type': 'get_student', 'student_id': student_id})
        response_index=0
        improvements=defaultdict(list)
        for q_id in q_ids:
            question = self.mediator.request({'type': 'get_question', 'question_id': q_id})
            
            if not question:
                response_index += 1
                continue
            response=responses[response_index]
            for kc_id in question.get_skills():
                prior = student.priors.get(kc_id, 0.5)
                new_prior = update_prior(prior, question.p_g, question.p_s, question.p_t, response)
                new_prior = max(0.01, min(0.99, new_prior))
                log.debug("update_student_priors | student_id=%s skill=%s prior %.3f -> %.3f",
                          student_id,kc_id,prior,new_prior)
                #single write path: route through the mediator so the
                #repository persists it (direct entity mutation would not)
                if self.mediator:
                    self.mediator.request({'type': 'update_prior', 'student_id': student_id, 'skill_id': kc_id, 'new_prior': new_prior})
                perc_change = (new_prior - prior) / prior 
                self.update_similar_skills(student_id, kc_id, perc_change)

            response_index+=1
    
    def update_similar_skills(self, student_id:str, skill_id:str, percentage_change:float)->None:
        skill = self.mediator.request( {'type': 'get_skill', 'skill_id': skill_id})
        if not skill:
            return
        student = self.mediator.request({'type': 'get_student', 'student_id': student_id})
        
        for sk in skill.get_similar():
            old_prior=student.priors.get(sk, 0.5)
            updated = old_prior*(1+percentage_change)
            new_prior = max(0.01, min(0.99, updated))
            #write via mediator only - keeps repository persistence guaranteed
            self.mediator.request({'type': 'update_prior', 'student_id': student_id, 'skill_id': sk, 'new_prior': new_prior})
    
    def predict_response(self, student_id:str, q_id:str)->tuple:
        question = self.mediator.request({'type': 'get_question', 'question_id': q_id})
        if not question:
            raise KeyError(f"Question {q_id} not found")
        student = self.mediator.request({'type': 'get_student', 'student_id': student_id})
        
        # Average priors across all KCs in the question
        avg_prior = np.mean([student.priors.get(kc_id, 0.5) for kc_id in question.get_skills()])
        
        p_C, pred = next_response(avg_prior, question.p_g, question.p_s)
        log.debug("predict_response | student_id=%s q_id=%s avg_prior=%.3f p_C=%.3f",
                  student_id,q_id,avg_prior,p_C)
        return p_C, pred

    def update_content_gaps(self,student_id:str)->dict:
        """Gap_Absence	Answer is linguistically plausible but scientifically nonsensical — catches a student who is guessing.
            Gap_Prior	Answer reflects an error in an earlier foundational skill (e.g. a math mistake in squaring a distance).
            Gap_Concept	Answer uses the right vocabulary from the lesson but does not actually solve the problem — catches rote memorization.
            Gap_Misconception	Answer reflects a logical conclusion built on a well-known, widespread scientific misconception.
        """
        student = self.mediator.request( {'type': 'get_student', 'student_id': student_id})
        if not student:
            return {}

        #for every skill-id with priors below 0.5 collect the atags of the error responses
        weak_skills = {skill_id for skill_id, prior in student.priors.items() if prior < 0.5}

        skill_atags = {}
        for response in student.get_responses():
            q_id, response_value = response[0], response[1]
            if response_value:  #correct -> not an error
                continue
            atag = response[4] if len(response) > 4 else None
            question = self.mediator.request( {'type': 'get_question', 'question_id': q_id})
            if not question:
                continue
            if atag is None:
                atag = question.content_gap_type
            for skill_id in question.get_skills():
                if skill_id in weak_skills:
                    skill_atags.setdefault(skill_id, []).append(atag)

        content_gap_types = dict(getattr(student, 'content_gap_types', {}) or {})

        for skill_id, prior in student.priors.items():
            if prior < 0.5:
                #find the most frequent atag in these errors
                atags = [g for g in skill_atags.get(skill_id, []) if g]
                if atags:
                    content_gap_types[skill_id] = max(set(atags), key=atags.count)
            else:
                #remove skills with prior>0.5 from the content gap types
                content_gap_types.pop(skill_id, None)

        self.mediator.request({'type': 'update_content_gaps', 'student_id': student_id, 'content_gap_types': content_gap_types})
        log.info("update_content_gaps | student_id=%s weak_skills=%d gaps=%s",
                 student_id,len(weak_skills),content_gap_types)
        return content_gap_types
