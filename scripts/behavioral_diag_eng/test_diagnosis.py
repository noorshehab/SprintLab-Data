import numpy as np
import pandas as pd
import os 
import scipy.stats as stats
from dotenv import load_dotenv
from behavioral_diagnosis_engine import diagnosis
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
import mlflow
import dagshub
import json
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv(override=True)

output_path=os.getenv('EXPERIMENT_OUTPUTS')
dagshub.init(repo_owner='nhatemshehab', repo_name='SprintLab-Data', mlflow=True)

def prepare_test_set():
    #load the test set
    responses=pd.read_csv(os.getenv('TEST_SET'))
    #load the question metadata
    question_metadata=pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    
    # (edited by mostafa nashaat reason: fix crash by renaming the column to what the original script expected)
    if 'super_topic_ids' not in question_metadata.columns and 'kc_ids' in question_metadata.columns:
        question_metadata.rename(columns={'kc_ids': 'super_topic_ids'}, inplace=True)
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

    # working memory
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    features = ['num_variables', 'solution_complexity_x', 'relies_on_image']
    df_clean = question_metadata.dropna(subset=features + ['error_rate']).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    pca = PCA(n_components=1)
    df_clean['pca_wm_score'] = pca.fit_transform(X_scaled).flatten()
    if df_clean['pca_wm_score'].corr(df_clean['error_rate']) < 0:
        df_clean['pca_wm_score'] = -df_clean['pca_wm_score']
    threshold = df_clean['pca_wm_score'].quantile(0.75)
    df_clean['is_high_wm'] = df_clean['pca_wm_score'] > threshold
    
    question_metadata = pd.merge(question_metadata, df_clean[['question_id', 'is_high_wm']], on='question_id', how='left')

    #merge all this with the response test set  
    responses=responses.merge(
    question_metadata[['question_id','language_difficulty_quartile','reasoning_quartile', 'is_high_wm']],
    on='question_id',
    how='left'
    )
    responses['error']=1-responses['responses']
    return responses



#prepare dataset
custom_name = input("Enter a name for the experiment: ")

with mlflow.start_run() as run:
    # (edited by mostafa nashaat reason: fetch dagshub username automatically)
    current_user = run.data.tags.get('mlflow.user', 'unknown_user')
    
    try:
        runs = mlflow.search_runs()
        if not runs.empty and 'tags.mlflow.user' in runs.columns:
            user_runs = runs[runs['tags.mlflow.user'].str.lower() == current_user.lower()]
            run_number = len(user_runs)
        else:
            run_number = 1
    except Exception:
        run_number = 1

    run_name = f"{current_user}#{run_number}_{custom_name}"
    
    # (edited by mostafa nashaat reason: update run name dynamically)
    mlflow.set_tag("mlflow.runName", run_name)

    test_set=prepare_test_set()
    ground_truth = (
        test_set.groupby('uid')
        # (edited by mostafa nashaat reason: include_groups=False to silence pandas warning)
        .apply(diagnosis, include_groups=False)
        .reset_index()
    )

    first_100 = test_set.sort_values(['uid', 'timestamps']).groupby('uid').head(100).reset_index(drop=True)
    f100_diagnosis= (
        first_100.groupby('uid')
        # (edited by mostafa nashaat reason: include_groups=False to silence pandas warning)
        .apply(diagnosis, include_groups=False)
        .reset_index())

    merged = ground_truth.merge(
        f100_diagnosis,
        on='uid',
        suffixes=('_gt', '_f100')
    )

    metrics = []
    # (edited by mostafa nashaat reason: added working_memory to metrics loop)
    for diag in ['attention', 'reasoning', 'language', 'flexibility', 'frustration', 'working_memory']:
        y_true = merged[f'{diag}_diag_gt']
        y_pred = merged[f'{diag}_diag_f100']
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        
        mlflow.log_metric(f'{diag}_accuracy', acc)
        mlflow.log_metric(f'{diag}_precision', prec)
        mlflow.log_metric(f'{diag}_recall', rec)
        
        metrics.append({
            'diagnosis': diag,
            'accuracy': acc,
            'precision': prec,
            'recall': rec
        })

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.round(4))

    #confusion matrices
    # (edited by mostafa nashaat reason: added working_memory to confusion matrix loop)
    for diag in ['attention', 'reasoning', 'language', 'flexibility', 'frustration', 'working_memory']:
        y_true = merged[f'{diag}_diag_gt']
        y_pred = merged[f'{diag}_diag_f100']

        cm = confusion_matrix(y_true, y_pred, labels=[False, True])

        #plot confusion matrix heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diagnosis', 'Diagnosis'], yticklabels=['No Diagnosis', 'Diagnosis'])
        plt.title(f'Confusion Matrix for {diag.capitalize()} Diagnosis')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        fig_path=os.path.join(output_path,f'confusion_matrix_{diag}.png')
        plt.savefig(fig_path)

        #save figures to mlflow as artifacts 
        mlflow.log_artifact(fig_path)

    #correlation between f_100 delta and ground truth delta
    # (edited by mostafa nashaat reason: added working_memory to correlation loop)
    for diag in ['attention', 'reasoning', 'language', 'flexibility', 'frustration', 'working_memory']:
        correlation = merged[f'{diag}_gt'].corr(merged[f'{diag}_f100'])
        mlflow.log_metric(f'{diag}_correlation', correlation)
        
