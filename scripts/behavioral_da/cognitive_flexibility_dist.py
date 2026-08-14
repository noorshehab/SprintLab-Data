#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import ast
load_dotenv()

#load files
responses=pd.read_csv(os.getenv('TRAINING_SET'))
question_metadata=pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))

for col in ['kc_ids', 'super_topic_ids']:
    if col in question_metadata.columns:
        question_metadata[col] = question_metadata[col].apply(ast.literal_eval)

question_metadata=question_metadata.explode('kc_ids')
question_metadata=question_metadata.explode('super_topic_ids')

responses=pd.merge(
   responses,question_metadata[['question_id','super_topic_ids']],
   left_on='concepts',right_on='question_id',
   how='left' 
)

responses = responses.groupby(['uid', 'question_id', 'timestamps']).agg({
        'responses': 'first',  # Response is same for all KCs in a question
        'concepts': list,      # List of all KCs in this question
        'super_topic_ids': lambda x: list(set(x))  # Unique supertopics
    }).reset_index()

print(responses.head())

#set results path
results_path=os.getenv("EXPERIMENT_OUTPUTS")

#calculate deltas
def switch_penalty_dist(df):
    df = df.sort_values(['uid', 'timestamps']).reset_index(drop=True)
    df['error'] = 1 - df['responses']
    
    # Shift
    df['next_supertopics'] = df.groupby('uid')['super_topic_ids'].shift(-1)
    df['next_error'] = df.groupby('uid')['error'].shift(-1)
    
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

    print(df[df['is_switch']==1].head())
    
    # Filter to valid transitions
    df_valid = df[df['next_supertopics'].notna()].copy()
    
    # Per-student aggregation
    student_stats = df_valid.groupby(['uid', 'is_switch']).agg(
        error_rate=('next_error', 'mean'),
        count=('next_error', 'count')
    ).reset_index()
    
    # Pivot
    pivot = student_stats.pivot(
        index='uid',
        columns='is_switch',
        values=['error_rate', 'count']
    ).fillna(0)
    
    pivot.columns = ['_'.join(map(str, col)).strip() for col in pivot.columns.values]
    
    # Rename
    rename_map = {}
    for col in pivot.columns:
        if 'error_rate_0' in col:
            rename_map[col] = 'stay_error_rate'
        elif 'error_rate_1' in col:
            rename_map[col] = 'switch_error_rate'
        elif 'count_0' in col:
            rename_map[col] = 'stay_count'
        elif 'count_1' in col:
            rename_map[col] = 'switch_count'
    
    pivot = pivot.rename(columns=rename_map)
    pivot['penalty'] = pivot['stay_error_rate']-pivot['switch_error_rate']  
    
    return pivot.reset_index()

switch_dist=switch_penalty_dist(responses)

#distribution
print("\nPer-student summary:")
print(f"Average penalty: {switch_dist['penalty'].mean():.4f}")
print(f"Students with positive penalty (switching helps): {(switch_dist['penalty'] > 0).sum()}")
print(f"Students with negative penalty (switching hurts): {(switch_dist['penalty'] < 0).sum()}")

quartiles = switch_dist['penalty'].quantile([0.25, 0.5, 0.75])
print("Quartile boundaries:")
print(f"Q1 (25th percentile): {quartiles[0.25]:.4f}")
print(f"Q2 (50th percentile): {quartiles[0.5]:.4f}")
print(f"Q3 (75th percentile): {quartiles[0.75]:.4f}")

# Assign quartile labels
switch_dist['penalty_quartile'] = pd.qcut(
    switch_dist['penalty'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

quartile_stats = switch_dist.groupby('penalty_quartile')['penalty'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count'),
    ('min', 'min'),
    ('max', 'max'),
    ('median', 'median')
]).round(4)

# Add percentile information
quartile_stats['percentile_range'] = ['0-25%', '25-50%', '50-75%', '75-100%']

print("\n=== Quartile Statistics ===")
print(quartile_stats)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.histplot(switch_dist['penalty'], bins=30, kde=True, ax=ax1, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero penalty')
ax1.axvline(x=quartiles[0.25], color='green', linestyle='--', alpha=0.5, label='Q1')
ax1.axvline(x=quartiles[0.75], color='orange', linestyle='--', alpha=0.5, label='Q3')
ax1.set_xlabel('Switch Penalty', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Switch Penalties', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig1_path = os.path.join(results_path, 'switch_penalty_histogram.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()

order = ['Q1', 'Q2', 'Q3', 'Q4']
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Get data in the correct order
data_by_quartile = []
for q in order:
    data = switch_dist[switch_dist['penalty_quartile'] == q]['penalty'].dropna()
    data_by_quartile.append(data)
    print(f"{q}: {len(data)} students")

bp = ax2.boxplot(data_by_quartile, patch_artist=True)
ax2.set_xticklabels(['Q1\nWorst', 'Q2', 'Q3', 'Q4\nBest'])

colors = ['green', 'yellowgreen', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_ylabel('Switch Penalty', fontsize=12)
ax2.set_xlabel('Quartile', fontsize=12)
ax2.set_title('Switch Penalty by Quartile', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig2_path = os.path.join(results_path, 'switch_penalty_boxplot.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()

#Q1 distribution
q1_data = switch_dist[switch_dist['penalty_quartile'] == 'Q1']
q1_mean = q1_data['penalty'].mean()
q1_std = q1_data['penalty'].std()
q1_median = q1_data['penalty'].median()
q1_min = q1_data['penalty'].min()
q1_max = q1_data['penalty'].max()

print(f"\n=== Q1 Statistics ===")
print(f"Students in Q1: {len(q1_data)}")
print(f"Mean penalty: {q1_mean:.4f}")
print(f"Median penalty: {q1_median:.4f}")
print(f"Std penalty: {q1_std:.4f}")
print(f"Range: [{q1_min:.4f}, {q1_max:.4f}]")
print(f"95% CI: [{q1_mean - 1.96*q1_std:.4f}, {q1_mean + 1.96*q1_std:.4f}]")

stats_path=os.path.join(results_path,"Q1_switch_stats.csv")

# Save quartile statistics
quartile_stats.to_csv(stats_path)

# Save dataframe with quartile labels
penalties_path=os.path.join(results_path,'switch_penalties.csv')
switch_dist.to_csv(penalties_path, index=False)

print("\n✅ Results saved to CSV files")

