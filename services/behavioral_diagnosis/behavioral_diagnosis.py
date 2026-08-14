import numpy as np
import pandas as pd
import scipy.stats as stats

#create a dictionary of parameters for every diagnosis
parameters={'reasoning':{'mean':-0.1933,'std':0.0705,'count':2891},
            'language':{'mean':-0.1668,'std':0.0673,'count':2891},
            'attention_span':{'mean':2.679,'std':0.546,'count':2891},
            'flexibility':{'mean':-0.0705,'std':0.0322,'count':2891},
            # (edited by mostafa nashaat reason: adding frustration threshold parameters using 2nd pass logic on train set)
            'frustration':{'mean':-0.2943,'std':0.0432,'count':1736},
            # (edited by mostafa nashaat reason: adding working memory threshold parameters on train set)
            'working_memory':{'mean':-0.2232,'std':0.0479,'count':959},
            'processing_speed':{'mean':-0.1226,'std':0.0782,'count':2885},
            'time_management':{'UTM_to_STM_ratio':1.5},
            'stress':{'stress_ratio':2.0}
        }

def CI(delta,diagnosis_mean,diagnosis_std,n,confidence=0.75):
    se=diagnosis_std/np.sqrt(n)

    z_critical= stats.norm.ppf(1 - (1 - confidence) / 2)

    ci_upper = diagnosis_mean + z_critical * se

    return delta < ci_upper

def test_threshold(metric,threshold):
    if metric > threshold:
        return True
    return False

def stress(df):
    stress_conditions=df[df['stress_triggers']>0]
    no_stress_conditions=df[df['stress_triggers']==0]
    return stress_conditions['error'].mean()/no_stress_conditions['error'].mean() if no_stress_conditions['error'].mean()>0 else np.nan


#functions that calculate the deltas
def processing_speed(df):
    time_pressure=df[df['time_pressure']==1]
    no_time_pressure=df[df['time_pressure']==0]

    delta=time_pressure['error'].mean()-no_time_pressure['error'].mean()

    return delta

def time_management(df):
    under_recommended_time=df[df['response_time']<df['time']]
    delta=abs(under_recommended_time['response_time']-under_recommended_time['time'])
    stm=(delta*under_recommended_time['response']).sum()
    utm=(delta*under_recommended_time['error']).sum()

    return{
        'stm':stm,
        'utm':utm,
        'ratio':utm/stm if stm>0 else np.nan
    }
#reasoning
def reasoning(df):
    q1=df[df['reasoning_level']=='Q1']
    q1_error=q1['error'].mean()

    q4=df[df['reasoning_level']=='Q4']
    q4_error=q4['error'].mean()

    delta= q1_error-q4_error
    return delta

    
#language difficulty
def language(df):
    q1=df[df['language_level']=='Q1']
    q1_error=q1['error'].mean()

    q4=df[df['language_level']=='Q4']
    q4_error=q4['error'].mean()

    delta=q1_error-q4_error
    return delta


#cognitive flexibility
def switch(df):
    df = df.sort_values('timestamps').reset_index(drop=True)
    df['next_unit'] = df['unit'].shift(-1)
    df['next_error'] = df['error'].shift(-1)

    # Check switch (unit changed)
    def check_switch(current_unit, next_unit):
        if current_unit is None or pd.isna(current_unit):
            return np.nan
        if next_unit is None or pd.isna(next_unit):
            return np.nan
        return 1 if current_unit != next_unit else 0

    df['is_switch'] = df.apply(
        lambda row: check_switch(row['unit'], row['next_unit']),
        axis=1
    )

    # Filter to valid transitions
    df_valid = df[df['next_unit'].notna()]
    if len(df_valid) == 0:
        return np.nan

    switch_error = df_valid[df_valid['is_switch'] == 1]['next_error'].mean()
    stay_error = df_valid[df_valid['is_switch'] == 0]['next_error'].mean()

    if pd.isna(switch_error) or pd.isna(stay_error):
        return np.nan

    return stay_error-switch_error

#sustained attention 
def attention(df):
    df['cum_error'] = df['error'].expanding().mean()

    df['error_increased'] = df['cum_error'].transform(
            lambda x: x > x.shift()
        )
    
    df['window_id'] = df['error_increased'].cumsum()
    
    # Count questions per window per student
    window_counts = df.groupby('window_id').size().reset_index(name='window_length')
  
    attn_span= window_counts['window_length'].mean()
    return attn_span

#effort
# (edited by mostafa nashaat reason: function to calculate frustration delta using isolated mistake logic)
def frustration_delta(df):
    errors = df['error'].values
    window_size = 3
    deltas = []
    
    for i in range(len(errors)):
        if errors[i] == 1 and (i == 0 or errors[i-1] == 0):
            start_idx = max(0, i - window_size)
            end_idx = min(len(errors), i + 1 + window_size)

            before_mistake = errors[start_idx : i]
            after_mistake = errors[i + 1 : end_idx]

            if len(before_mistake) > 0 and len(after_mistake) > 0:
                eb = np.mean(before_mistake)
                ea = np.mean(after_mistake)
                deltas.append(eb - ea)

    if not deltas:
        return np.nan
    return np.mean(deltas)

#cognitive load/working memory
# (edited by mostafa nashaat reason: calculate working memory delta)
def working_memory_delta(df):
    if 'cognitive_load' not in df.columns:
        return np.nan
    threshold = df['cognitive_load'].quantile(0.75)
    error_high = df[df['cognitive_load'] > threshold]['error'].mean()
    error_reg = df[df['cognitive_load'] <= threshold]['error'].mean()
    
    if pd.isna(error_high) or pd.isna(error_reg):
        return np.nan
        
    return error_reg - error_high


#diagnosis function: inputs-> student responses  

def diagnosis(df):
    #return the list of diagnoses

    processing_speed_delta=processing_speed(df)
    time_management_ratio=time_management(df)['ratio']
    stress_ratio=stress(df)
    reasoning_delta=reasoning(df)
    language_delta=language(df)
    switch_delta=switch(df)
    attention_delta=attention(df)
    # (edited by mostafa nashaat reason: use frustration_delta and working_memory_delta functions)
    frustration_val=frustration_delta(df)
    working_memory_val=working_memory_delta(df)

    #tests
    processing_speed_test=CI(processing_speed_delta,parameters['processing_speed']['mean'],parameters['processing_speed']['std'],parameters['processing_speed']['count'])
    time_management_test=test_threshold(time_management_ratio,parameters['time_management']['UTM_to_STM_ratio'])
    stress_test=test_threshold(stress_ratio,parameters['stress']['stress_ratio'])
    reasoning_test=CI(reasoning_delta,parameters['reasoning']['mean'],parameters['reasoning']['std'],parameters['reasoning']['count'])
    language_test=CI(language_delta,parameters['language']['mean'],parameters['language']['std'],parameters['language']['count'])
    switch_test=CI(switch_delta,parameters['flexibility']['mean'],parameters['flexibility']['std'],parameters['flexibility']['count'])
    attention_test=CI(attention_delta,parameters['attention_span']['mean'],parameters['attention_span']['std'],parameters['attention_span']['count'])  
    # (edited by mostafa nashaat reason: add frustration and working memory test to diagnosis)
    frustration_test=CI(frustration_val,parameters['frustration']['mean'],parameters['frustration']['std'],parameters['frustration']['count'])
    working_memory_test=CI(working_memory_val,parameters['working_memory']['mean'],parameters['working_memory']['std'],parameters['working_memory']['count'])

    diagnoses=[]
    if reasoning_test:
        diagnoses.append('reasoning')
    if language_test:
        diagnoses.append('language')
    if switch_test:
        diagnoses.append('flexibility')
    if attention_test:
        diagnoses.append('attention_span')
    # (edited by mostafa nashaat reason: add frustration and working memory diagnosis)
    if frustration_test:
        diagnoses.append('frustration')
    if working_memory_test:
        diagnoses.append('working_memory')
    if processing_speed_test:
        diagnoses.append('processing_speed')
    if time_management_test:
        diagnoses.append('time_management')
    if stress_test:
        diagnoses.append('stress')

    # (edited by mostafa nashaat reason: add frustration and working memory results to returned series)
    return pd.Series({ 'processing_speed':processing_speed_delta,'time_management_ratio':time_management_ratio,'stress_ratio':stress_ratio,'reasoning':reasoning_delta,'language':language_delta,'flexibility':switch_delta,'attention':attention_delta,'frustration':frustration_val, 'working_memory':working_memory_val,
                      'processing_speed_diag':processing_speed_test,'time_management_diag':time_management_test,'stress_diag':stress_test,'reasoning_diag':reasoning_test,'language_diag':language_test,'flexibility_diag':switch_test,'attention_diag':attention_test,'frustration_diag':frustration_test, 'working_memory_diag':working_memory_test
                      ,'diagnoses':diagnoses})