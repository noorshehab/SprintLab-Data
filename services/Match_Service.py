from __future__ import annotations
#mediator for the question bandit has an instance of the data service
import numpy as np
from datetime import datetime
from services.log_setup import get_logger
from services.Data_service import Data_Service
from services.question_bandit.Question_Selection_service import Question_Selection_Service
from services.question_bandit.context import build_question_context
from services.Interfaces import Mediator,Subscriber

log=get_logger('Match_Service')

class Match_Service(Mediator,Subscriber):
    def __init__(self,Q_S:Question_Selection_Service,D_S:Data_Service)->None:
        self.Q_S=Q_S
        self.D_S=D_S
        self.Q_S.set_mediator(self)
        log.info("constructed Match_Service | selection=%s data=%s (wired Q_S.mediator=self)",
                 type(Q_S).__name__,type(D_S).__name__)

    def update(self, event:str, event_data:dict)->None:
        # event_data: {'student_id': sid, 'questions': [qid1, qid2, ...]}
        sid = event_data.get('student_id')
        q_ids = event_data.get('questions', [])
        student = self.D_S.get_student(sid)
        if not student: return

        # 1. Collect Student Context
        student_context = {'id': sid, 'priors': student.get_priors()}

        # 2. Collect Question Contexts
        q_objs = self.D_S.get_question(q_ids)
        q_contexts = build_question_context(q_objs)

        # 3. Compute Rewards from Priors History
        # Reward = mean percentage change in priors for skills in the question
        history = student.get_priors_history()
        timestamps = sorted(history.keys())
        if len(timestamps) < 2: return
        
        # Last two snapshots to compute delta
        old_p, new_p = history[timestamps[-2]], history[timestamps[-1]]
        rewards = []
        for q in q_objs:
            if not q: 
                rewards.append(0.0)
                continue
            skills = q.get_skills()
            deltas = []
            for s in skills:
                o, n = old_p.get(s, 0.0), new_p.get(s, 0.0)
                deltas.append((n - o) / o if o != 0 else 0.0)
            rewards.append(np.mean(deltas) if deltas else 0.0)

        self.Q_S.update(student_context, q_contexts, rewards)
        log.info("bandit update | student_id=%s questions=%d rewards_absorbed=%d",
                 sid,len(q_contexts),len(rewards))

    def request(self,request:dict)->None:
    #theres a switch case that directs control flow to a bunch of functions
        rtype=request.get('type','none')
        if rtype=='get_student':
            return self.D_S.get_student(request.get('student_id'))
        #query request
        if rtype=='query':
            result=self.D_S.query(request.get('parameters'))
            log.debug("query | %d condition(s) -> %d match(es)",
                      len(request.get('parameters') or []),len(result or {}))
            return result
        if rtype=='get_question':
            return self.D_S.get_question(request.get('question_id'))
        log.warning("request with unknown type '%s' dropped | request=%s",rtype,request)
        return

    def set_match(self,student_id:str)->list:
        ideal_set=self.Q_S.get_optimal_set(student_id)
        student=self.D_S.get_student(student_id)
        diagnoses=student.get_diagnoses()

        #fetch all selected questions at once
        q_objs=self.D_S.get_question(ideal_set['q_id'].tolist())
        questions=[]
        for q in q_objs:
            if q:
                questions.append({'q':q,'difficulty':q.get_atts()['difficulty_level'],'unit':q.get_unit()})

        #diagnosis-driven serving order: flexibility -> unit grouping,
        #attention_span -> hard-first, stress -> easy/hard interleaved
        order_applied:str='none'
        if 'flexibility' in diagnoses:
            questions.sort(key=lambda x: x['unit'])
            order_applied='flexibility:unit_asc'
        if 'attention_span' in diagnoses:
            questions.sort(key=lambda x: x['difficulty'], reverse=True)
            order_applied='attention_span:difficulty_desc'
        if 'stress' in diagnoses:
            questions.sort(key=lambda x: x['difficulty'])
            mid=len(questions)//2
            easy,hard=questions[:mid],questions[mid:]
            arranged=[]
            for e,h in zip(easy,hard): arranged.extend([e,h])
            arranged.extend(hard[len(easy):])
            questions=arranged
            order_applied='stress:interleaved'

        log.info("set_match | student_id=%s candidates=%d returned=%d diagnoses=%s order=%s",
                 student_id,len(ideal_set),len(questions),diagnoses,order_applied)
        return questions
