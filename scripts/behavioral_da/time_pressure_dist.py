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

#calculate time pressure score as z(solution_length)+z(solution_complexity_x) scaled to 1
question_metadata['sl_z']=stats.zscore(question_metadata['solution_length'])
question_metadata['sc_z']=stats.zscore(question_metadata['solution_complexity_x'])

#assign every question its time pressure score
question_metadata['tp_raw'] = (
    question_metadata['sl_z'] +
    question_metadata['sc_z']
)

min_val = question_metadata['tp_raw'].min()
max_val = question_metadata['tp_raw'].max()
question_metadata['time_pressure_score'] = (
    (question_metadata['tp_raw'] - min_val) / (max_val - min_val)
)

#assign the time pressure tag 
threshold = 0.15
question_metadata['time_pressure'] = (question_metadata['time_pressure_score'] > threshold).astype(int)

#look at the distribution of this score
print(question_metadata['time_pressure_score'].describe())

plt.figure(figsize=(12,6))
sns.histplot(question_metadata['time_pressure_score'], kde=True, color='steelblue')
plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold {threshold}')
plt.title('Distribution of Time Pressure Score')
plt.xlabel('Time Pressure Score')
plt.ylabel('Number of Questions')
plt.legend()
plt.grid(True, alpha=0.3)
fig_path = os.path.join(results_path, 'time_pressure_score_hist.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

#frequency of the time pressure tag at threshold 0.5
tag_counts = question_metadata['time_pressure'].value_counts().reindex([0, 1], fill_value=0)
tagged_fraction = tag_counts[1] / len(question_metadata)
print("\nTime Pressure Tag Frequency @ threshold 0.5:")
print(f"Regular questions (0): {tag_counts[0]}")
print(f"Time pressure questions (1): {tag_counts[1]}")
print(f"Fraction tagged: {tagged_fraction:.4f}")

plt.figure(figsize=(8, 6))
sns.barplot(x=['Regular (0)', 'Time Pressure (1)'], y=tag_counts.values, color='steelblue', width=0.5)
plt.title(f'Frequency of Time Pressure Tag (threshold={threshold})')
plt.ylabel('Number of Questions')
for i, v in enumerate(tag_counts.values):
    plt.text(i, v, f'{v}', ha='center', va='bottom')
plt.grid(True, alpha=0.3, axis='y')
fig_path = os.path.join(results_path, 'time_pressure_tag_frequency.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

#fraction tagged as a function of threshold
sweep_thresholds = np.linspace(0, 1, 101)
sweep_fractions = [
    (question_metadata['time_pressure_score'] > t).mean()
    for t in sweep_thresholds
]

plt.figure(figsize=(12, 6))
plt.plot(sweep_thresholds, sweep_fractions, color='steelblue')
plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold {threshold}')
plt.axhline(y=tagged_fraction, color='red', linestyle=':', alpha=0.5)
plt.title('Fraction of Questions Tagged by Threshold')
plt.xlabel('Threshold')
plt.ylabel('Fraction Tagged')
plt.legend()
plt.grid(True, alpha=0.3)
fig_path = os.path.join(results_path, 'time_pressure_threshold_sweep.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

#merge time pressure tag onto responses
responses=responses.merge(
    question_metadata[['question_id', 'time_pressure']],
    left_on='questions', right_on='question_id',
    how='left'
)

def time_pressure_penalty(df):
    df = df.sort_values(['uid', 'timestamps']).reset_index(drop=True)
    df['error'] = 1 - df['responses']

    student_tp_error = df.groupby(['uid', 'time_pressure']).agg(
        error_rate=('error', 'mean'),
        count=('error', 'count')
    ).reset_index()

    pivot = student_tp_error.pivot(
        index='uid',
        columns='time_pressure',
        values='error_rate'
    ).reset_index()

    pivot.columns = ['uid'] + [f'error_{int(col)}' for col in pivot.columns if col != 'uid']

    #make sure both tags exist
    for tag in [0, 1]:
        if f'error_{tag}' not in pivot.columns:
            pivot[f'error_{tag}'] = np.nan

    pivot['time_pressure_delta'] = pivot['error_1'] - pivot['error_0']

    #count how many questions each student had under each tag
    count_pivot = student_tp_error.pivot(
        index='uid',
        columns='time_pressure',
        values='count'
    ).reset_index()
    count_pivot.columns = ['uid'] + [f'count_{int(col)}' for col in count_pivot.columns if col != 'uid']

    #merge counts back
    pivot = pivot.merge(count_pivot, on='uid', how='left')

    #fill NaN counts with 0
    for tag in [0, 1]:
        if f'count_{tag}' not in pivot.columns:
            pivot[f'count_{tag}'] = 0
        pivot[f'count_{tag}'] = pivot[f'count_{tag}'].fillna(0)

    return pivot

student_time_pressure_penalty = time_pressure_penalty(responses)

valid_students = student_time_pressure_penalty[
    (student_time_pressure_penalty['count_0'] > 0) &
    (student_time_pressure_penalty['count_1'] > 0)
].copy()

print("\nTime Pressure Delta Statistics (error_1 - error_0):")
print(f"Mean delta: {valid_students['time_pressure_delta'].mean():.4f}")
print(f"Median delta: {valid_students['time_pressure_delta'].median():.4f}")
print(f"Std delta: {valid_students['time_pressure_delta'].std():.4f}")
print(f"Min delta: {valid_students['time_pressure_delta'].min():.4f}")
print(f"Max delta: {valid_students['time_pressure_delta'].max():.4f}")

#distribution of the per-student time pressure delta
plt.figure(figsize=(12, 6))
sns.histplot(valid_students['time_pressure_delta'], bins=30, kde=True, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero delta')
plt.title('Distribution of Time Pressure Delta (error_time_pressure - error_no_time_pressure)')
plt.xlabel('Time Pressure Delta')
plt.ylabel('Number of Students')
plt.legend()
plt.grid(True, alpha=0.3)
fig_path = os.path.join(results_path, 'time_pressure_delta_distribution.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

#lowest quartile statistics
valid_students['tp_delta_quartile'] = pd.qcut(
    valid_students['time_pressure_delta'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

worst_delta = valid_students[valid_students['tp_delta_quartile'] == 'Q1']
mean = worst_delta['time_pressure_delta'].mean()
std = worst_delta['time_pressure_delta'].std()
median = worst_delta['time_pressure_delta'].median()

print("\nLowest Quartile (Q1) Time Pressure Delta Statistics:")
print(f"Students in Q1: {len(worst_delta)}")
print(f"Mean delta: {mean:.4f}")
print(f"Std delta: {std:.4f}")
print(f"Median delta: {median:.4f}")

statistics = {
    "mean": mean,
    "std": std,
    "median": median
}

stats_path = os.path.join(results_path, 'time_pressure_delta_statistics.csv')
pd.DataFrame([statistics]).to_csv(stats_path, index=False)

print("\nResults saved.")