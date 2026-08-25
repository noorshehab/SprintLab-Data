from __future__ import annotations
import pandas as pd
import numpy as np
import datetime


class question():
    def __init__(self, q_id, grade=None, grade_band=None, subject=None, unit_id=None,
                 lesson_id=None, skill_cluster_id=None, skill_ids=None, question_type=None,
                 question_text=None, question_media_url=None, question_language='en',
                 difficulty_level=0.0, bloom_taxonomy_level=None, population_difficulty=None,
                 discrimination_index=None, time_allowed=0.0, theoretical_solving_time=None,
                 dynamic_avg_time=None, cognitive_load_index=None, variables_count=0,
                 logical_steps=0, language_challenging=0, language_challenge_type=None,
                 trick_question=0, time_pressure_flag=0, visual_dependency=0,
                 real_world_context=0, multi_concept_flag=0, prerequisite_concept_id=None,
                 correct_answer_id=None, correct_answer_content=None,
                 distractor_1_content=None, distractor_2_content=None, distractor_3_content=None,
                 content_gap_type=None, tag_source=None, tag_confidence_score=None,
                 question_status=None, usage_count=0, date_created=None, last_updated=None,
                 cognitive_load=0.0, language_level=None, reasoning_level=None,
                 num_unknowns=1, num_operations=0, p_t=0.0, p_s=0.0, p_g=0.0,
                 language='en'):
        self.id = q_id
        self.grade = grade
        self.grade_band = grade_band
        self.subject = subject
        self.unit = unit_id #supertopic_id
        self.lesson_id = lesson_id
        self.skill_cluster_id = skill_cluster_id
        self.skill_ids = skill_ids if skill_ids is not None else (
            [skill_cluster_id] if skill_cluster_id is not None else [])
        self.question_type = question_type
        self.question_text = question_text
        self.question_media_url = question_media_url
        self.language = question_language if question_language else language
        self.difficulty_level = difficulty_level
        self.bloom_taxonomy_level = bloom_taxonomy_level
        self.population_difficulty = population_difficulty
        self.discrimination_index = discrimination_index
        self.time_allowed = time_allowed 
        self.theoretical_solving_time = theoretical_solving_time
        self.dynamic_avg_time = dynamic_avg_time
        self.cognitive_load_index = cognitive_load_index
        self.variables_count = variables_count
        self.logical_steps = logical_steps
        self.language_challenging = language_challenging
        self.language_challenge_type = language_challenge_type
        self.trick_question = trick_question
        self.time_pressure_flag = time_pressure_flag
        self.visual_dependency = visual_dependency
        self.real_world_context = real_world_context
        self.multi_concept_flag = multi_concept_flag
        self.prerequisite_concept_id = prerequisite_concept_id
        self.correct_answer_id = correct_answer_id
        self.correct_answer_content = correct_answer_content
        self.distractor_1_content = distractor_1_content
        self.distractor_2_content = distractor_2_content
        self.distractor_3_content = distractor_3_content
        self.content_gap_type = content_gap_type
        self.tag_source = tag_source
        self.tag_confidence_score = tag_confidence_score
        self.question_status = question_status
        self.usage_count = usage_count
        self.date_created = date_created
        self.last_updated = last_updated
        #derived features (kept)
        self.cognitive_load = cognitive_load #wm_score proxy
        self.language_level = language_level #language_quartile
        self.reasoning_level = reasoning_level #reasoning_quartile
        self.num_unknowns = num_unknowns
        self.num_operations = num_operations
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g

    def get_params(self)->tuple:
        return self.p_t, self.p_s, self.p_g
    def get_skills(self)->list:
        return self.skill_ids
    def get_unit(self):
        return self.unit
    def get_atts(self)->dict:
        return {
            'question_text':self.question_text,
            'time_allowed':self.time_allowed,
            'difficulty_level':self.difficulty_level,
            'language_level':self.language_level,
            'cognitive_load':self.cognitive_load,
            'variables_count':self.variables_count,
            'logical_steps':self.logical_steps,
            'time_pressure_flag':self.time_pressure_flag,
            'language_challenging':self.language_challenging,
            'reasoning_level':self.reasoning_level,
            'language':self.language,
            'num_unknowns':self.num_unknowns,
            'num_operations':self.num_operations,
            'cognitive_load_index':self.cognitive_load_index }

    
class student():
    def __init__(self, student_id:str)->None:
        self.id = student_id
        self.priors = {}  # {skill_id: prior}
        self.responses = []  # list of (q_id, response,time,stress_triggers)
        self.diagnoses=[]
        self.deltas_history = {}  # {timestamp: {metric: delta}}
        self.diagnoses_history = {}  # {timestamp: [diagnoses]}
        self.priors_history = {}  # {timestamp: {skill_id: prior}}
        self.treatment_plan={}
        self.content_gap_types={} #skill_id:gap_type

    def update_prior(self, kc_id:str, new_prior:float)->None:
        self.priors[kc_id] = new_prior
    
    def add_response(self, q_id:str, response:int, time:float|None, stress_triggers:list|None, atag:str|None)->None:
        self.responses.append((q_id,response,time,stress_triggers,atag))

    def add_diagnosis(self, diagnosis:str)->None:
        self.diagnoses.append(diagnosis)

    def add_diagnosis_record(self, timestamp:str, deltas:dict, diagnoses:list)->None:
        self.deltas_history[timestamp] = deltas
        self.diagnoses_history[timestamp] = diagnoses

    def add_priors_record(self, timestamp:str, priors:dict)->None:
        self.priors_history[timestamp] = dict(priors)

    def get_priors_history(self)->dict:
        return self.priors_history

    def add_treatment_plan(self, timestamp:str, treatment_name:str, parameters:dict)->None:
        self.treatment_plan[treatment_name]=(timestamp,parameters)

    def update_treatment_plan(self, timestamp:str, treatment_name:str, parameters:dict)->None:
         self.treatment_plan[treatment_name]=(timestamp,parameters)

    def get_treatment_plan(self)->dict:
        return self.treatment_plan

    def update_content_gaps(self, content_gap_types:dict)->None:
        self.content_gap_types = dict(content_gap_types)

    def get_content_gaps(self)->dict:
        return self.content_gap_types
    
    def get_responses(self)->list:
        return self.responses
    
    def get_priors(self)->dict:
        return self.priors

    def get_diagnoses(self)->list:
        return self.diagnoses

    def get_deltas(self)->dict:
        return self.deltas_history

    def get_diagnoses_history(self)->dict:
        return self.diagnoses_history

class skill():
    def __init__(self, skill_id:str, similar_skills:list)->None:
            self.id = skill_id
            self.similar=similar_skills  

    def get_similar(self)->list:
        return self.similar
