import numpy as np
import pandas as pd
import scipy.stats as stats

#create a dictionary of parameters for every diagnosis
parameters={'reasoning':{'mean':-0.1933,'std':0.0705,'count':2891},
            'language':{'mean':-0.1668,'std':0.0673,'count':2891},
            'attention_span':{'mean':2.679,'std':0.546,'count':2891},
            'flexibility':{'mean':-0.0705,'std':0.0322,'count':2891}
        }

def CI(delta,diagnosis_mean,diagnosis_std,n,confidence=0.75):
    se=diagnosis_std/np.sqrt(n)

    z_critical= stats.norm.ppf(1 - (1 - confidence) / 2)

    ci_upper = diagnosis_mean + z_critical * se

    return delta < ci_upper


#functions that calculate the deltas
#reasoning
def reasoning(df):
    q1=df[df['reasoning_quartile']=='Q1']
    q1_error=q1['error'].mean()

    q4=df[df['reasoning_quartile']=='Q4']
    q4_error=q4['error'].mean()

    delta= q1_error-q4_error
    return delta
    

#language difficulty
def language(df):
    q1=df[df['language_difficulty_quartile']=='Q1']
    q1_error=q1['error'].mean()

    q4=df[df['language_difficulty_quartile']=='Q4']
    q4_error=q4['error'].mean()

    delta=q1_error-q4_error
    return delta


#cognitive flexibility
def switch(df):
    df = df.sort_values('timestamps').reset_index(drop=True)
    df['next_supertopics'] = df['super_topic_ids'].shift(-1)
    df['next_error'] = df['error'].shift(-1)
    
    # Check switch (no overlap)
    def check_switch(current_topics, next_topics):
        if not isinstance(current_topics, list) or not isinstance(next_topics, list):
            return np.nan
        if len(current_topics) == 0 or len(next_topics) == 0:
            return np.nan
        return 1 if len(set(current_topics) & set(next_topics)) == 0 else 0
    
    df['is_switch'] = df.apply(
        lambda row: check_switch(row['super_topic_ids'], row['next_supertopics']),
        axis=1
    )
    
    # Filter to valid transitions
    df_valid = df[df['next_supertopics'].notna()]
    if len(df_valid) == 0:
        return np.nan
    
    switch_error = df_valid[df_valid['is_switch'] == 1]['error'].mean()
    stay_error = df_valid[df_valid['is_switch'] == 0]['error'].mean()
    
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
#frustration 
#cognitive load/working memory

#diagnosis function: inputs-> student responses  

def diagnosis(df):
    #return the list of diagnoses

    reasoning_delta=reasoning(df)
    language_delta=language(df)
    switch_delta=switch(df)
    attention_delta=attention(df)

    #tests
    reasoning_test=CI(reasoning_delta,parameters['reasoning']['mean'],parameters['reasoning']['std'],parameters['reasoning']['count'])
    language_test=CI(language_delta,parameters['language']['mean'],parameters['language']['std'],parameters['language']['count'])
    switch_test=CI(switch_delta,parameters['flexibility']['mean'],parameters['flexibility']['std'],parameters['flexibility']['count'])
    attention_test=CI(attention_delta,parameters['attention_span']['mean'],parameters['attention_span']['std'],parameters['attention_span']['count'])  

    diagnoses=[]
    if reasoning_test:
        diagnoses.append('reasoning')
    if language_test:
        diagnoses.append('language')
    if switch_test:
        diagnoses.append('flexibility')
    if attention_test:
        diagnoses.append('attention_span')


    return pd.Series({'reasoning':reasoning_delta,'language':language_delta,'flexibility':switch_delta,'attention':attention_delta,
                      'reasoning_diag':reasoning_test,'language_diag':language_test,'flexibility_diag':switch_test,'attention_diag':attention_test
                      ,'diagnoses':diagnoses})