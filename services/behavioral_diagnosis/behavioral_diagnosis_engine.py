import pandas as pd
import numpy as np
from behavioral_diagnosis import diagnosis
from services.Interfaces import Component


class behavioral_diagnosis_engine(Component):
    def __init__(self):
        self.mediator = None

    def diagnose_student(self, student_id):
        student = self.mediator.request(self, {'type': 'get_student', 'student_id': student_id})
        if not student:
            return None

        df = self.build_dataframe(student.get_responses())
        if df.empty:
            return None

        result = diagnosis(df)

        self.mediator.request(self, {'type': 'add_diagnosis', 'student_id': student_id, 'diagnosis': result['diagnoses']})
        return result

    def build_dataframe(self, responses):
        rows = []
        for i, (q_id, response, response_time, stress_triggers) in enumerate(responses):
            question = self.mediator.request(self, {'type': 'get_question', 'question_id': q_id})
            if not question:
                continue

            row = question.get_atts()
            row.pop('question_text', None)

            row['question_id'] = q_id
            row['unit'] = question.get_unit()
            row['response'] = response
            row['error'] = 1 - response
            row['response_time'] = response_time
            row['stress_triggers'] = stress_triggers
            row['timestamps'] = i

            rows.append(row)

        return pd.DataFrame(rows)
