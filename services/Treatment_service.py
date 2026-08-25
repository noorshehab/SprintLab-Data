#is mediated by the Diagnosis service takes the diagnoses and priors and constructs the treatment plan and appends it to the student
#every diagnosis window it updates the treatment parameters according to the new deltas
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from services.log_setup import get_logger
from services.Interfaces import Component

log=get_logger('Treatment_Service')

class Treatment_Service(Component):
    def __init__(self,D_S=None)->None:
       self.mediator=D_S
       self.initial_treatment_map:dict={

            'language':{'language_level':'Q1','Operator':'=='},
            'working_memory':{'cognitive_load':1,'Operator':'<='},
            'processing_speed':{'time_pressure_flag':False,'Operator':'=='},
            'time_management':{'time_allowed':180,'logical_steps':2,'Operator':'<='},
            'Gap_Absence':{'bloom_taxonomy_level':["Remember" , "Understand"],'Operator':'in'},
            'Gap_Concept':{'visual_dependency':True,'Operator':'=='},
            #diagnoses with delta metrics but previously missing initial plans
            'attention_span': {'logical_steps': 3, 'Operator': '<='},
            'flexibility': {'multi_concept_flag': True, 'Operator': '=='},
            'stress': {'time_pressure_flag': False, 'Operator': '=='}
        }
           

    def set_treatment_plan(self,student_id:str)->dict:
        student=self.mediator.request({'type':'get_student','student_id':student_id})
        if not student:
            return {}
        diagnoses=student.get_diagnoses()
        content_gaps=student.get_content_gaps() or {}
        treatment_plan={'general':{},'specific':{}} #format {'general':{'threshold1':x},'specific':{'skill_id':{'threshold':x}}} 

        #behavioral,exam and cognitive skills
        #set the treatments according to the active diagnoses
        for diagnosis in diagnoses:
            if diagnosis in self.initial_treatment_map:
                treatment_plan['general'][diagnosis]=self.initial_treatment_map[diagnosis]

        #content gaps
        for skill_id,gap_type in content_gaps.items():
            if gap_type in self.initial_treatment_map:
                treatment_plan['specific'][str(skill_id)]=self.initial_treatment_map[gap_type]

        #append the treatment plan to the student
        self.mediator.request({'type':'update_treatment_plan','student_id':student_id,
                                    'timestamp':datetime.now().isoformat(),
                                    'treatment_name':'treatment_plan','parameters':treatment_plan})

        log.info("set_treatment_plan | student_id=%s general=%s specific=%s (unmapped diagnoses skipped)",
                 student_id,list(treatment_plan['general']),list(treatment_plan['specific']))

        return treatment_plan

    def update_treatment_plan(self,student_id:str)->dict:
        student=self.mediator.request({'type':'get_student','student_id':student_id})
        if not student:
            return {}
        diagnoses=student.get_diagnoses_history()
        content_gaps=student.get_content_gaps() or {}

        #diagnosis name -> delta metric recorded in the deltas_history
        delta_metric_map={
            'language':'language',
            'attention_span':'attention',
            'flexibility':'flexibility',
            'working_memory':'working_memory',
            'processing_speed':'processing_speed',
            'time_management':'time_management_ratio',
            'stress':'stress_ratio'
        }

        #the current treatment plan that gets scaled up
        stored=student.get_treatment_plan()
        treatment_plan={k:dict(v) for k,v in stored['treatment_plan'][1].items()}
    
        #percentage improvement of each delta between the last 2 diagnosis windows
        #deltas move towards zero -> (|old|-|new|)/|old| is the fraction improved
        #timestamps may mix ISO strings and epoch floats - sort by string form
        deltas_timestamps=sorted(student.get_deltas().keys(),key=str)
        improvement={}
        if len(deltas_timestamps)>=2:
            old_deltas=student.get_deltas()[deltas_timestamps[-2]]
            new_deltas=student.get_deltas()[deltas_timestamps[-1]]
            for metric in set([*old_deltas.keys(),*new_deltas.keys()]):
                o=old_deltas.get(metric)
                n=new_deltas.get(metric)
                if o is None or n is None or o==0:
                    continue
                improvement[metric]=(abs(o)-abs(n))/abs(o)*100.0

        #general treatments: parameters increase by the same percentage as the matching delta improved
        for treatment_name,parameters in treatment_plan.get('general',{}).items():
            metric=delta_metric_map.get(treatment_name)
            pct=improvement.get(metric,0.0)
            scaled=self._scale_parameters(parameters,pct)
            #language: from the delta history each 10% delta improvement moves language_level up a quartile
            #10% -> Q2, 20% -> Q3, 40% -> Q4 (never moves down)
            if treatment_name=='language' and 'language_level' in parameters:
                scaled['language_level']=self._escalate__language(parameters.get('language_level'),pct)
            if treatment_name=='working_memory' and 'cognitive_load' in parameters:
                scaled['cognitive_load']=self._escalate__working_memory(parameters.get('cognitive_load'),pct)
            if treatment_name=='time_management':
                scaled['time_allowed'],scaled['logical_steps']=self._escalate__time_management(
                    (parameters.get('time_allowed'),parameters.get('logical_steps')),pct)
            treatment_plan['general'][treatment_name]=scaled

        #content gap treatments: parameters scale by the per-skill prior improvement
        priors_history=student.get_priors_history()
        priors_timestamps=sorted(priors_history.keys(),key=str)
        prior_improvement={}
        if len(priors_timestamps)>=2:
            old_priors=priors_history[priors_timestamps[-2]]
            new_priors=priors_history[priors_timestamps[-1]]
            for skill_id in set([*old_priors.keys(),*new_priors.keys()]):
                o=old_priors.get(skill_id,0.0)
                n=new_priors.get(skill_id,0.0)
                if o==0:
                    continue
                prior_improvement[skill_id]=(n-o)/o*100.0

        for skill_id,parameters in treatment_plan.get('specific',{}).items():
            pct=prior_improvement.get(str(skill_id),0.0)
            treatment_plan['specific'][str(skill_id)]=self._scale_parameters(parameters,pct)

        #write back the updated treatment plan
        self.mediator.request({'type':'update_treatment_plan','student_id':student_id,
                                    'timestamp':datetime.now().isoformat(),
                                    'treatment_name':'treatment_plan','parameters':treatment_plan})

        log.info("update_treatment_plan | student_id=%s general=%s specific=%s",
                 student_id,list(treatment_plan['general']),list(treatment_plan['specific']))

        return treatment_plan

    def _scale_parameters(self,parameters:dict,pct:float)->dict:
        factor=1.0+pct/100.0
        scaled={}
        for k,v in parameters.items():
            scaled[k]=round(v*factor,4) if isinstance(v,(int,float)) else v
        return scaled

    def _escalate__language(self,current:str,pct:float)->str:
        #improvements of 10/20/40% move the difficulty up to Q2/Q3/Q4, never down
        if pct>=40: return 'Q4'
        if pct>=20: return 'Q3'
        if pct>=10: return 'Q2'
        return current

    def _escalate__working_memory(self,current:int,pct:float)->int:
        #improvements of 10/20/40% raise the cognitive load ceiling by 0.1/0.2/0.4, never down
        if pct>=40: return 4
        if pct>=20: return 3
        if pct>=10: return 2
        return current

    def _escalate__time_management(self,current:tuple,pct:float)->tuple:
        #current is the (time_allowed,logical_steps) pair; improvements of 10/20/40%
        #allow +30s/+60s/+90s and +1/+2/+3 extra steps, never less
        max_time,max_steps=current
        if pct>=40: return (max_time+180,max_steps+6)
        if pct>=20: return (max_time+120,max_steps+4)
        if pct>=10: return (max_time+60,max_steps+2)
        return current

    
