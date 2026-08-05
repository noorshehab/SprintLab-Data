import numpy as np
import pandas as pd
import scipy.stats as stats

#create a dictionary of parameters for every diagnosis
parameters={'reasoning':{'mean':-0.1933,'std':0.0705,'count':2891},
            'language':{'mean':-0.1668,'std':0.0673,'count':2891},
            'attention_span':{'mean':2.679,'std':0.546,'count':2891},
            'flexibility':{'mean':-0.0705,'std':0.0322,'count':2891},
            #effort index cohort stats, measured on the full XES3G5M behavioural set
            #(notebooks/student_effort_index.ipynb): 14,453 students with >=1 easy question
            'effort':{'mean':0.1260,'std':0.0986,'count':14453}
        }

#the effort index is only defined over "easy" questions: Level-1 questions whose
#worsening_rate is at or below this quantile of the Level-1 worsening_rate distribution.
#0.50 reproduces the experiment (tau=0.0651 -> 507 of 1,014 Level-1 questions).
#prepare_test_set() uses this to build the is_easy flag, so it lives here with the
#other cohort constants to keep the flag consistent with the mean/std above.
LOW_WORSENING_QUANTILE=0.50

#reference threshold from the experiment: the 5th percentile of the effort index
#flags 5% of students, vs ~36% for the CI() rule below. Not used by diagnosis(),
#kept for comparison when tuning the effort cut-off.
EFFORT_P5_THRESHOLD=-0.0612

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
def effort_index(df):
    #df is one student's responses ordered by timestamp.
    #on an easy question the cohort errs at avg_baseline; a student putting in effort
    #should err less than that, so delta = avg_baseline - error is their margin over
    #the cohort. The effort index is that margin averaged over the easy questions.
    easy=df[(df['is_easy']==1) & df['avg_baseline'].notna()]

    #a student can see the same question more than once (one row per KC route);
    #the experiment scored each (student,question) pair once
    easy=easy.drop_duplicates(subset='questions')

    if len(easy)==0:
        return np.nan

    return (easy['avg_baseline']-easy['error']).mean()

#frustration
#cognitive load/working memory

#diagnosis function: inputs-> student responses  

def diagnosis(df):
    #return the list of diagnoses

    reasoning_delta=reasoning(df)
    language_delta=language(df)
    switch_delta=switch(df)
    attention_delta=attention(df)
    effort_delta=effort_index(df)

    #tests
    reasoning_test=CI(reasoning_delta,parameters['reasoning']['mean'],parameters['reasoning']['std'],parameters['reasoning']['count'])
    language_test=CI(language_delta,parameters['language']['mean'],parameters['language']['std'],parameters['language']['count'])
    switch_test=CI(switch_delta,parameters['flexibility']['mean'],parameters['flexibility']['std'],parameters['flexibility']['count'])
    attention_test=CI(attention_delta,parameters['attention_span']['mean'],parameters['attention_span']['std'],parameters['attention_span']['count'])
    #a NaN effort index (student answered no easy questions) compares False, i.e. no diagnosis
    effort_test=CI(effort_delta,parameters['effort']['mean'],parameters['effort']['std'],parameters['effort']['count'])

    diagnoses=[]
    if reasoning_test:
        diagnoses.append('reasoning')
    if language_test:
        diagnoses.append('language')
    if switch_test:
        diagnoses.append('flexibility')
    if attention_test:
        diagnoses.append('attention_span')
    if effort_test:
        diagnoses.append('effort')


    return pd.Series({'reasoning':reasoning_delta,'language':language_delta,'flexibility':switch_delta,'attention':attention_delta,'effort':effort_delta,
                      'reasoning_diag':reasoning_test,'language_diag':language_test,'flexibility_diag':switch_test,'attention_diag':attention_test,'effort_diag':effort_test
                      ,'diagnoses':diagnoses})