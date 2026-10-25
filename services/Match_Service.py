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
        student_context = {'id': sid, 'priors': self.D_S.get_priors(sid)}

        # 2. Collect Question Contexts
        q_objs = self.D_S.get_question(q_ids)
        q_contexts = build_question_context(q_objs)

        # 3. Compute Rewards from Priors History
        # Reward = mean percentage change in priors for skills in the question
        history = self.D_S.get_priors_history(sid, window=2)
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
        if rtype=='get_priors':
            return self.D_S.get_priors(request.get('student_id'))
        if rtype=='get_priors_history':
            return self.D_S.get_priors_history(request.get('student_id'), request.get('window', 2))
        if rtype=='get_treatment_plan':
            return self.D_S.get_treatment_plan(request.get('student_id'))
        if rtype=='get_diagnoses':
            return self.D_S.get_diagnoses(request.get('student_id'))
        if rtype=='get_content_gaps':
            return self.D_S.get_content_gaps(request.get('student_id'))
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

    #add the music selection,prehint selection and posthint selection
    def set_match(self,student_id:str)->list:
        ideal_set=self.Q_S.get_optimal_set(student_id)
        student=self.D_S.get_student(student_id)
        if not student:
            return [], [], {}, {}, '', {}
        diagnoses=self.D_S.get_diagnoses(student_id)

        #fetch all selected questions at once
        q_objs=self.D_S.get_question(ideal_set['q_id'].tolist())
        questions=[]
        for q in q_objs:
            if q:
                questions.append({'q':q,'difficulty':q.get_atts()['difficulty_level'],'unit':q.get_unit()})

        #bundle hint sets for the match
        pre_hints=self.set_pre_hints(student_id, questions)
        post_hints=self.set_post_hints(questions)
        play_hints=self.set_play_hints(questions)

        #music selections for the match
        general_music, question_music=self.set_music(student_id, questions)

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
        return questions, pre_hints, post_hints, play_hints, general_music, question_music

    def set_pre_hints(self, student_id: str, questions: list) -> list:
        """Bundle pre-hints for a student based on diagnoses and content gaps.

        Returns a list of pre-hint dicts for the match.
        """
        student = self.D_S.get_student(student_id)
        if not student:
            return []

        diagnoses = self.D_S.get_diagnoses(student_id)
        content_gaps = self.D_S.get_content_gaps(student_id)
        pre_hints = []

        for diagnosis in diagnoses:
            hints = self.D_S.get_pre_hints_by_diagnosis(diagnosis)
            for h in hints:
                pre_hints.append({
                    'hint_id': h.id,
                    'hint_text': h.text,
                    'diagnosis': h.diagnosis,
                    'hint_type': 'behavioral',
                })

        for skill_id, gap_type in content_gaps.items():
            hints = self.D_S.get_pre_hints_by_content_gap_type(skill_id, gap_type)
            for h in hints:
                pre_hints.append({
                    'hint_id': h.id,
                    'hint_text': h.text,
                    'topic': h.topic,
                    'content_gap_type': h.content_gap_type,
                    'hint_type': 'content_gap',
                })

        log.info("set_pre_hints | student_id=%s diagnoses=%s gaps=%d hints=%d",
                 student_id, diagnoses, len(content_gaps), len(pre_hints))
        return pre_hints

    def set_post_hints(self, questions: list) -> dict:
        """Bundle post-hints for all questions in a match.

        Returns a dict mapping question_id -> list of post-hint dicts.
        """
        post_hints_by_q = {}
        for q_info in questions:
            q = q_info.get('q')
            if not q:
                continue
            q_id = q.id
            hints = self.D_S.get_post_hints_by_question(q_id)
            post_hints_by_q[q_id] = [
                {
                    'hint_id': h.id,
                    'hint_text': h.text,
                    'hint_order': h.hint_order,
                }
                for h in hints
            ]

        log.info("set_post_hints | questions=%d total_hints=%d",
                 len(questions), sum(len(v) for v in post_hints_by_q.values()))
        return post_hints_by_q

    def set_play_hints(self, questions: list) -> dict:
        """Bundle play-hints for all questions in a match.

        Returns a dict mapping q_id -> list of play-hint dicts.
        """
        play_hints_by_q = {}
        for q_info in questions:
            q = q_info.get('q')
            if not q:
                continue
            q_id = q.id
            hints = self.D_S.get_play_hints_by_question(q_id)
            play_hints_by_q[q_id] = [
                {
                    'hint_id': h.id,
                    'hint_text': h.text,
                    'hint_order': h.hint_order,
                }
                for h in hints
            ]

        log.info("set_play_hints | questions=%d total_hints=%d",
                 len(questions), sum(len(v) for v in play_hints_by_q.values()))
        return play_hints_by_q

    def check_options(q):
        if q and q.options:
            for opt in q.options:
                if opt.distractor_reading_trap:
                    return 'reading_trap'
                if opt.distractor_impulsive:
                    return 'impulse'
    
    #not currently of use
    def set_music(self, student_id: str, questions: list) -> tuple:
        """Bundle music treatments for a match based on student diagnoses.

        Returns (general_music, question_music_dict) where:
          - general_music: situational label for the match overall
          - question_music_dict: {q_id: music_description} per-question assignments
        """
        student = self.D_S.get_student(student_id)
        if not student:
            return ('', {})

        diagnoses = self.D_S.get_diagnoses(student_id)
        

        # Map diagnosis types to audio treatments per the constraints table
        MUSIC_MAP = {
            'stress': 'heartbeat',
            'impulsiveness': 'tempo_slow',
            'working_memory': 'white_noise',
            'attention': 'shatter',
            'time_management': 'tick_tock',
            'language': 'language_music',
            'misconception': 'shatter',
        }

        # General situational label: collect all applicable diagnoses
        applicable = [MUSIC_MAP[d] for d in diagnoses if d in MUSIC_MAP]
        general_music = ', '.join(applicable) if applicable else ''

        # Per-question music: check question attributes for matching
        question_music: dict = {}
        for q_info in questions:
            q = q_info.get('q')
            if not q:
                continue
            q_id = q.id
            music_assignments = []

            # Check question attributes against diagnosis map
            if q.attrs and q.attrs.language_level and 'language' in MUSIC_MAP:#language challenging= true
                music_assignments.append(MUSIC_MAP['language'])

            # Check for trick question attribute
            if q.question_type == 'trick_question' and 'impulsiveness' in MUSIC_MAP:
                music_assignments.append(MUSIC_MAP['impulsiveness'])

            # Working memory check via cognitive_load_index
            if q.attrs and q.attrs.cognitive_load_index and q.attrs.cognitive_load_index > 3 and 'working_memory' in MUSIC_MAP:
                music_assignments.append(MUSIC_MAP['working_memory'])

            # Attention lapses: check for distractor-related flags
            if q.attrs and q.attrs.multi_concept_flag and 'attention' in MUSIC_MAP:
                music_assignments.append(MUSIC_MAP['attention'])

            if music_assignments:
                question_music[q_id] = '; '.join(music_assignments)

        log.info("set_music | student_id=%s diagnoses=%s general=%s q_count=%d music_qs=%d",
                 student_id, diagnoses, general_music, len(questions), len(question_music))
        return general_music, question_music