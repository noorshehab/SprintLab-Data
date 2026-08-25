#reward = (1 - mean(priors_for_skills)) * q_difficulty * q_learning
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict
from services.log_setup import get_logger

log=get_logger('MAB')

class ContextualBandit():
    def __init__(self)->None:
        self.top_n:int=10
        self.rewards_by_context=defaultdict(list)# predicted_reward -> [observed rewards]
        log.info("constructed ContextualBandit | top_n=%d",self.top_n)

    def reward(self,priors:list,difficulty:float,learning:float)->float:
        return (1-np.mean(priors))*difficulty*learning

    def select(self,student_context:dict,question_contexts:pd.DataFrame)->pd.DataFrame:
        """
        student_context:{id:id,priors:{'skill1':prior1,'skill2':prior2}}
        question_contexts: DataFrame with columns q_id,difficulty,learning,skill_ids
        """
        if question_contexts is None or len(question_contexts)==0:
            log.info("select | student_id=%s no candidates -> empty selection",
                     student_context.get('id'))
            return pd.DataFrame({'q_id':[],'score':[]})

        scores=pd.DataFrame({
            'q_id':question_contexts['q_id'],
            'score':[-np.inf]*len(question_contexts)
        })

        for question in question_contexts.itertuples():
            skills=question.skill_ids
            student_skills=[student_context['priors'].get(skill,0.0) for skill in skills]
            predicted_reward=self.reward(student_skills,question.difficulty,question.learning)

            context_key=round(predicted_reward,2)
            actual_reward=self.rewards_by_context[context_key]

            if len(actual_reward)>0:
                mean_actual=np.mean(actual_reward)
                std_actual=np.std(actual_reward)
            else:
                mean_actual=predicted_reward
                std_actual=0.045

            sample_reward=np.random.normal(mean_actual,std_actual)

            confidence_bonus=1.96*std_actual/np.sqrt(max(1,len(actual_reward)))
            upper_bound=sample_reward+confidence_bonus

            scores.loc[scores['q_id']==question.q_id,'score']=upper_bound

        selected=scores.sort_values('score',ascending=False)[['q_id']].head(self.top_n)
        best=scores['score'].max()
        log.info("select | student_id=%s scored=%d selected=%d best_score=%.3f "
                 "(known contexts=%d)",
                 student_context.get('id'),len(scores),len(selected),
                 best if np.isfinite(best) else float('-inf'),len(self.rewards_by_context))
        return selected


    def update(self,student_context:dict,question_contexts:pd.DataFrame,rewards:list)->None:
        if question_contexts is None or len(question_contexts)==0:
            log.debug("update | nothing to absorb")
            return
        #rewards_by_context stores a list of observed rewards per context bucket;
        #select() computes mean/std over that list
        absorbed=0
        for question,reward in zip(question_contexts.itertuples(),rewards):
            skills=question.skill_ids
            student_skills=[student_context['priors'].get(skill,0.0) for skill in skills]
            predicted_reward=self.reward(student_skills,question.difficulty,question.learning)

            context_key=round(predicted_reward,2)
            self.rewards_by_context[context_key].append(float(reward))
            absorbed+=1

        log.info("update | student_id=%s absorbed=%d reward(s) across %d context bucket(s) "
                 "| total observations=%d",
                 student_context.get('id'),absorbed,len(self.rewards_by_context),
                 sum(len(v) for v in self.rewards_by_context.values()))
        return
