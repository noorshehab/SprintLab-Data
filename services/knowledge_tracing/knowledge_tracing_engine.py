#"db" of questions and their item specific probabilities
#"db" of students and their personalized priors
#the function that takes the first n responses and outputs priors for the skills and all their similar counterparts
#the call to predict

import pandas as pd
import numpy as np
from BKT import next_response,update_prior
from services.Interfaces import Component


class knowledge_tracing_engine(Component):
    def __init__(self,D_S=None ,calibration_window=10):
        self.calibration_window = calibration_window
        self.mediator = D_S       

    def calibrate_priors(self, student_id):
        direct_correct_multiplier = 1.1
        direct_incorrect_multiplier = 0.90
        similar_correct_multiplier = 1.05
        similar_incorrect_multiplier = 0.95

        student = self.mediator.request(self, {'type': 'get_student', 'student_id': student_id})
        responses = student.get_responses()[: self.calibration_window]

        for q_id, response, *_ in responses:
            question = self.mediator.request(self, {'type': 'get_question', 'question_id': q_id})
            for kc_id in question.get_skills():
                prior = student.priors.get(kc_id, 0.5)
                if response:  # correct
                    new_prior = prior * direct_correct_multiplier
                else:         # incorrect
                    new_prior = prior * direct_incorrect_multiplier

                new_prior = max(0.01, min(0.99, new_prior))
                self.mediator.request(self, {'type': 'update_prior', 'student_id': student_id, 'skill_id': kc_id, 'new_prior': new_prior})

                # apply proportional change to similar skills (with smaller multiplier)
                skill = self.mediator.request(self, {'type': 'get_skill', 'skill_id': kc_id})
                if skill:
                    for sk in skill.get_similar():
                        if response:
                            adjusted = prior * similar_correct_multiplier
                        else:
                            adjusted = prior * similar_incorrect_multiplier
                        adjusted = max(0.01, min(0.99, adjusted))
                        # update similar skill `sk` (not kc_id) via mediator
                        self.mediator.request(self, {'type': 'update_prior', 'student_id': student_id, 'skill_id': sk, 'new_prior': adjusted})
    

   
    def update_student_priors(self,student_id,q_ids,responses):
        #how do we update similar skills
        #take the %change in the prior for a kc for example +1% or -5% and apply to similar skills
        student = self.mediator.request(self, {'type': 'get_student', 'student_id': student_id})
        response_index=0
        for q_id in q_ids:
            question = self.mediator.request(self, {'type': 'get_question', 'question_id': q_id})
            if not question:
                response_index += 1
                continue
            response=responses[response_index]
            for kc_id in question.get_skills():
                prior = student.priors.get(kc_id, 0.5)
                new_prior = update_prior(prior, question.p_g, question.p_s, question.p_t, response)
                new_prior = max(0.01, min(0.99, new_prior))
                student.update_prior(kc_id, new_prior)
                # notify mediator of the prior change as well
                if self.mediator:
                    self.mediator.request(self, {'type': 'update_prior', 'student_id': student_id, 'skill_id': kc_id, 'new_prior': new_prior})
                perc_change = (new_prior - prior) / prior if prior != 0 else 0
                self.update_similar_skills(student_id, kc_id, perc_change)
            response_index+=1
    
    def update_similar_skills(self, student_id,skill_id,percentage_change):
        skill = self.mediator.request(self, {'type': 'get_skill', 'skill_id': skill_id})
        if not skill:
            return
        student = self.mediator.request(self, {'type': 'get_student', 'student_id': student_id})
        
        for sk in skill.get_similar():
            old_prior=student.priors.get(sk, 0.5)
            updated = old_prior*(1+percentage_change)
            new_prior = max(0.01, min(0.99, updated))
            student.update_prior(sk, new_prior)
            self.mediator.request(self, {'type': 'update_prior', 'student_id': student_id, 'skill_id': sk, 'new_prior': new_prior})
    
    def predict_response(self, student_id, q_id):
        question = self.mediator.request(self, {'type': 'get_question', 'question_id': q_id})
        if not question:
            raise KeyError(f"Question {q_id} not found")
        student = self.mediator.request(self, {'type': 'get_student', 'student_id': student_id})
        
        # Average priors across all KCs in the question
        avg_prior = np.mean([student.priors.get(kc_id, 0.5) for kc_id in question.get_skills()])
        
        p_C, pred = next_response(avg_prior, question.p_g, question.p_s)
        return p_C, pred
    

    




