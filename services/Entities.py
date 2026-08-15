import pandas as pd
import numpy as np


class question():
    def __init__(self, q_id, skill_ids,unit_id,text,time,time_pressure,level,
                  cognitive_load,variables_count,steps,language_challenge,language_level,reasoning_level
                  ,p_t, p_s, p_g, language='en', num_unknowns=1, num_operations=0, cognitive_load_index=None):
        self.id = q_id
        self.skill_ids = skill_ids
        self.unit=unit_id #supertopic_id
        self.question_text=text
        self.time=time #recommended_solving_time
        self.difficulty_level=level 
        self.cognitive_load=cognitive_load #wm_score
        self.reasoning_level=reasoning_level#reasoning_quartile
        self.variables_count=variables_count
        self.steps=steps
        self.time_pressure=time_pressure
        self.language_challenge=language_challenge
        self.language_level=language_level #language_quartile
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g
        self.language=language
        self.num_unknowns=num_unknowns
        self.num_operations=num_operations
        self.cognitive_load_index=cognitive_load_index
    
    def get_params(self):
        return self.p_t, self.p_s, self.p_g
    def get_skills(self):
        return self.skill_ids
    def get_unit(self):
        return self.unit
    def get_atts(self):
        return {
            'question_text':self.question_text,
            'time':self.time,
            'difficulty_level':self.difficulty_level,
            'language_level':self.language_level,
            'cognitive_load':self.cognitive_load,
            'variables_count':self.variables_count,
            'steps':self.steps,
            'time_pressure':self.time_pressure,
            'language_challenge':self.language_challenge,
            'reasoning_level':self.reasoning_level,
            'language':self.language,
            'num_unknowns':self.num_unknowns,
            'num_operations':self.num_operations,
            'cognitive_load_index':self.cognitive_load_index }

    
class student():
    def __init__(self, student_id):
        self.id = student_id
        self.priors = {}  # {kc_id: prior}
        self.responses = []  # list of (q_id, response,time,stress_triggers)
        self.diagnoses=[]

    def update_prior(self, kc_id, new_prior):
        self.priors[kc_id] = new_prior
    
    def add_response(self, q_id,response,time,stress_triggers):
        self.responses.append((q_id,response,time,stress_triggers))

    def add_diagnosis(self,diagnosis):
        self.diagnoses.append(diagnosis)
    
    def get_responses(self):
        return self.responses
    
    def get_priors(self):
        return self.priors

    def get_diagnoses(self):
        return self.diagnoses

class skill():
    def __init__(self, skill_id,similar_skills):
            self.id = skill_id
            self.similar=similar_skills  

    def get_similar(self):
        return self.similar
