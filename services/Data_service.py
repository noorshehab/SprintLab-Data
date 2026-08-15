from Entities import question,skill,student
from services.Interfaces import SigletonMeta
#singleton pattern for data service


class Data_Service(metaclass=SigletonMeta):
    
    def __init__(self):
        self.students = {}  # {student_id: student object}
        self.questions = {}  # {question_id: question object}
        self.skills = {}  # {skill_id: skill object}

    def add_student(self, student_id):
        if student_id not in self.students:
            self.students[student_id] = student(student_id)

    def add_question(self, q_id, skill_ids,unit_id,text,time,time_pressure,level,
                  cognitive_load,variables_count,steps,language_challenge,language_level,
                  reasoning_level,p_t, p_s, p_g, language='en', num_unknowns=1, num_operations=0,
                  cognitive_load_index=None):
        if q_id not in self.questions:
            self.questions[q_id] = question(q_id, skill_ids,unit_id,text,time,time_pressure,level,
                  cognitive_load,variables_count,steps,language_challenge,language_level,
                  reasoning_level,p_t, p_s, p_g, language, num_unknowns, num_operations,
                  cognitive_load_index)

    def add_skill(self, skill_id, similar_skills=None):
        if skill_id not in self.skills:
            self.skills[skill_id] = skill(skill_id, similar_skills if similar_skills is not None else [])

    #getters
    def get_student(self, student_id):
        return self.students.get(student_id, None)

    def get_question(self, q_id):
        return self.questions.get(q_id, None)

    def get_skill(self, skill_id):
        return self.skills.get(skill_id, None)
    
    def update_priors(self,student_id,skill_id,new_prior):
        self.students[student_id].update_prior(skill_id,new_prior)

    def update_responses(self,student_id,responses):
        for response in responses:
            q_id, response_value = response[0], response[1]
            response_time = response[2] if len(response) > 2 else None
            stress_triggers = response[3] if len(response) > 3 else None
            self.students[student_id].add_response(q_id,response_value,response_time,stress_triggers)

    def add_diagnosis(self,student_id,diagnoses):
        for diagnosis in diagnoses:
            self.students[student_id].add_diagnosis(diagnosis)

    #search functions
    #get all questions for skill
    #get all questions for a unit

