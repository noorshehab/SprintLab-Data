import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.log_setup import get_logger
from services.behavioral_diagnosis.behavioral_diagnosis import diagnosis
from services.Interfaces import Component

log=get_logger('behavioral_diagnosis_engine')


class behavioral_diagnosis_engine(Component):
    def __init__(self)->None:
        self.mediator = None
        log.info("constructed behavioral_diagnosis_engine")

    def diagnose_student(self, student_id:str):
        log.info("diagnose_student | running behavioral diagnosis for %s",student_id)
        responses = self.mediator.request({'type': 'get_responses', 'student_id': student_id})
        if not responses:
            return None

        df = self.build_dataframe(responses)
        if df.empty:
            return None

        result = diagnosis(df)

        deltas = {k: result[k] for k in result.index if not k.endswith('_diag') and k != 'diagnoses'}
        timestamp = datetime.now().isoformat()

        self.mediator.request({'type': 'add_diagnosis', 'student_id': student_id, 'diagnosis': result['diagnoses'], 'deltas': deltas, 'timestamp': timestamp})
        log.info("diagnose_student complete | student_id=%s diagnoses=%s deltas=%s",
                 student_id,result.get('diagnoses'),deltas)
        return result

    def build_dataframe(self, responses:list)->pd.DataFrame:
        rows = []
        for i, response in enumerate(responses):
            _, _, q_id, response_value, _, _, response_time, stress_triggers, _, atag = response
            question = self.mediator.request({'type': 'get_question', 'question_id': q_id})
            if not question:
                continue

            row = question.get_atts()
            row.pop('question_text', None)

            row['question_id'] = q_id
            row['unit'] = question.get_unit()
            row['response'] = response_value
            row['error'] = 1 - response_value
            row['response_time'] = response_time
            row['stress_triggers'] = stress_triggers
            row['timestamps'] = i
            row['atag']=atag

            rows.append(row)

        return pd.DataFrame(rows)
