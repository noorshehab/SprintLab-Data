#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from dotenv import load_dotenv
load_dotenv()

#load files
responses=pd.read_csv(os.getenv('TRAINING_SET'))
question_metadata=pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
results_path=os.getenv("EXPERIMENT_OUTPUTS")


question_metadata['scy_z']=stats.zscore(question_metadata['solution_complexity_y'])
question_metadata['sv_z']=stats.zscore(question_metadata['solution_vocab'])
question_metadata['ne_z']=stats.zscore(question_metadata['num_equations'])
question_metadata['ns_z']=stats.zscore(question_metadata['num_steps'])

#assign every question its language difficulty score 
question_metadata['reasoning_difficulty_raw'] = (
    question_metadata['scy_z'] + 
    question_metadata['sv_z'] + 
    question_metadata['ne_z'] + 
    question_metadata['ns_z']
)

min_val = question_metadata['reasoning_difficulty_raw'].min()
max_val = question_metadata['reasoning_difficulty_raw'].max()
question_metadata['reasoning_difficulty'] = (
    (question_metadata['reasoning_difficulty_raw'] - min_val) / (max_val - min_val)
)

#look at the distribution of this difficulty score 

print(question_metadata['reasoning_difficulty'].describe())

plt.figure(figsize=(12,6))
sns.displot(question_metadata['reasoning_difficulty'],kind='hist')
plt.title('Distribution of Reasoning Difficulty')
fig_path = os.path.join(results_path, 'reasoning_dif_hist.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')

#label the qs with 75th percentile as difficult
question_metadata['reasoning_quartile'] = pd.qcut(
    question_metadata['reasoning_difficulty'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

responses=responses.merge(
    question_metadata[['question_id','reasoning_quartile']],
    left_on='questions',right_on='question_id',
    how='left'
)

def reasoning_penalty(df):
    df = df.sort_values(['uid', 'timestamps']).reset_index(drop=True)
    df['error'] = 1 - df['responses']

    student_reasoning_error = df.groupby(['uid', 'reasoning_quartile']).agg(
        error_rate=('error', 'mean'),
        count=('error', 'count')
    ).reset_index()

    pivot = student_reasoning_error.pivot(
        index='uid',
        columns='reasoning_quartile',
        values='error_rate'
    ).reset_index()

    pivot.columns = ['uid'] + [f'error_{col}' for col in pivot.columns if col != 'uid']
    
    # Make sure Q1 and Q4 exist
    for q in ['Q1', 'Q4']:
        if f'error_{q}' not in pivot.columns:
            pivot[f'error_{q}'] = np.nan
    
    pivot['reasoning_penalty'] = pivot['error_Q1']-pivot['error_Q4'] 
    
    # Count how many questions each student had in Q1 and Q4
    count_pivot = student_reasoning_error.pivot(
        index='uid',
        columns='reasoning_quartile',
        values='count'
    ).reset_index()
    count_pivot.columns = ['uid'] + [f'count_{col}' for col in count_pivot.columns if col != 'uid']
    
    # Merge counts back
    pivot = pivot.merge(count_pivot, on='uid', how='left')
    
    # Fill NaN counts with 0
    for q in ['Q1', 'Q4']:
        if f'count_{q}' not in pivot.columns:
            pivot[f'count_{q}'] = 0
        pivot[f'count_{q}'] = pivot[f'count_{q}'].fillna(0)
    
    return pivot
    
student_reasoning_penalty=reasoning_penalty(responses)

print("Reasoning Penalty Statistics:")
print(f"Mean penalty: {student_reasoning_penalty['reasoning_penalty'].mean():.4f}")
print(f"Median penalty: {student_reasoning_penalty['reasoning_penalty'].median():.4f}")
print(f"Std penalty: {student_reasoning_penalty['reasoning_penalty'].std():.4f}")
print(f"Min penalty: {student_reasoning_penalty['reasoning_penalty'].min():.4f}")
print(f"Max penalty: {student_reasoning_penalty['reasoning_penalty'].max():.4f}")


fig_path = os.path.join(results_path,'reasoning_penalty_distribution.png')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of reasoning penalty
valid_students = student_reasoning_penalty[
    (student_reasoning_penalty['count_Q1'] > 0) & 
    (student_reasoning_penalty['count_Q4'] > 0)
]

ax1 = axes[0]
sns.histplot(valid_students['reasoning_penalty'], bins=30, kde=True, ax=ax1, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero penalty')
ax1.set_xlabel('Reasoning Penalty (Q1 - Q4)', fontsize=12)
ax1.set_ylabel('Number of Students', fontsize=12)
ax1.set_title('Distribution of Reasoning Penalty', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scatter plot: Q1 error vs Q4 error
ax2 = axes[1]
ax2.scatter(valid_students['error_Q1'], valid_students['error_Q4'], alpha=0.3, s=10, color='green')
ax2.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5, label='y=x (same performance)')
ax2.set_xlabel('Error Rate on Q1 (Least Complex)', fontsize=12)
ax2.set_ylabel('Error Rate on Q4 (Most Complex)', fontsize=12)
ax2.set_title('Q1 Error vs Q4 Error', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()


student_reasoning_penalty['r_quartile'] = pd.qcut(
    student_reasoning_penalty['reasoning_penalty'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

#calculate mean and std then save to csv
worst_pen=student_reasoning_penalty[student_reasoning_penalty['r_quartile']=='Q1']
mean=worst_pen['reasoning_penalty'].mean()
std=worst_pen['reasoning_penalty'].std()
median=worst_pen['reasoning_penalty'].median()

statistics={
    "mean": mean,
    "std": std,
    "median": median
}

stats_path = os.path.join(results_path,'reasoning_penalty_statistics.csv')
pd.DataFrame([statistics]).to_csv(stats_path, index=False)

