import numpy as np
import pandas as pd
import os 
import scipy.stats as stats
from dotenv import load_dotenv
from behavioral_diagnosis_engine import diagnosis
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv()

def prepare_test_set():
    #load the test set
    responses=pd.read_csv(os.getenv('TEST_SET'))
    #load the question metadata
    question_metadata=pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    #calculate the necessary scores and quartiles and add them to q metadata
    # add supertopic

    responses=pd.merge(
    responses,question_metadata[['question_id','super_topic_ids']],
   left_on='questions',right_on='question_id',
   how='left' 
    )

    #add language difficulty and quartile
    question_metadata['ql_z']=stats.zscore(question_metadata['question_length'])
    question_metadata['vr_z']=stats.zscore(question_metadata['vocabulary_richness'])
    question_metadata['nc_z']=stats.zscore(question_metadata['num_clauses'])
    question_metadata['ns_z']=stats.zscore(question_metadata['num_sentences'])

    question_metadata['lang_difficulty_raw'] = (
        question_metadata['ql_z'] + 
        question_metadata['vr_z'] + 
        question_metadata['nc_z'] + 
        question_metadata['ns_z']
    )

    min_val = question_metadata['lang_difficulty_raw'].min()
    max_val = question_metadata['lang_difficulty_raw'].max()
    question_metadata['lang_difficulty'] = (
        (question_metadata['lang_difficulty_raw'] - min_val) / (max_val - min_val)
    )

    question_metadata['language_difficulty_quartile'] = pd.qcut(
        question_metadata['lang_difficulty'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    #add reasoning difficulty and quartile
    question_metadata['scy_z']=stats.zscore(question_metadata['solution_complexity_y'])
    question_metadata['sv_z']=stats.zscore(question_metadata['solution_vocab'])
    question_metadata['ne_z']=stats.zscore(question_metadata['num_equations'])
    question_metadata['sns_z']=stats.zscore(question_metadata['num_steps'])

    question_metadata['reasoning_difficulty_raw'] = (
        question_metadata['scy_z'] + 
        question_metadata['sv_z'] + 
        question_metadata['ne_z'] + 
        question_metadata['sns_z']
    )

    min_val = question_metadata['reasoning_difficulty_raw'].min()
    max_val = question_metadata['reasoning_difficulty_raw'].max()
    question_metadata['reasoning_difficulty'] = (
        (question_metadata['reasoning_difficulty_raw'] - min_val) / (max_val - min_val)
    )

    question_metadata['reasoning_quartile'] = pd.qcut(
        question_metadata['reasoning_difficulty'],
        q=4,
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    #merge all this with the response test set  
    responses=responses.merge(
    question_metadata[['question_id','language_difficulty_quartile','reasoning_quartile']],
    on='question_id',
    how='left'
    )
    responses['error']=1-responses['responses']
    return responses



#prepare dataset
test_set=prepare_test_set()
print(test_set.head())
print(test_set['uid'].nunique())

ground_truth = (
    test_set.groupby('uid')
    .apply(diagnosis)
    .reset_index()
)

first_100 = test_set.sort_values(['uid', 'timestamps']).groupby('uid').head(100).reset_index(drop=True)
f100_diagnosis= (
    first_100.groupby('uid')
    .apply(diagnosis)
    .reset_index())

merged = ground_truth.merge(
    f100_diagnosis,
    on='uid',
    suffixes=('_gt', '_f100')
)

metrics = []
for diag in ['attention', 'reasoning', 'language', 'flexibility']:
    y_true = merged[f'{diag}_diag_gt']
    y_pred = merged[f'{diag}_diag_f100']

    metrics.append({
        'diagnosis': diag,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    })

metrics_df = pd.DataFrame(metrics)
print(metrics_df.round(4))

#confusion matrices
for diag in ['attention', 'reasoning', 'language', 'flexibility']:
    y_true = merged[f'{diag}_diag_gt']
    y_pred = merged[f'{diag}_diag_f100']

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diagnosis', 'Diagnosis'], yticklabels=['No Diagnosis', 'Diagnosis'])
    plt.title(f'Confusion Matrix for {diag.capitalize()} Diagnosis')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    #plt.savefig(f'confusion_matrix_{diag}.png')

#correlation between f_100 delta and ground truth delta
for diag in ['attention', 'reasoning', 'language', 'flexibility']:
    correlation = merged[f'{diag}_gt'].corr(merged[f'{diag}_f100'])
    print(f'Correlation between ground truth and first 100 delta for {diag}: {correlation:.4f}')
 
