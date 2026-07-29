#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

#load files
responses=pd.read_csv(os.getenv('TRAINING_SET'))
results_path=os.getenv("EXPERIMENT_OUTPUTS")

def calculate_sustained_attention(df):
    """
    Calculate sustained attention metrics for each student.
    - error_rate: cumulative error rate up to each question
    - longest_window: longest streak of questions before error rate increases
    """
    
    # Sort by student and timestamp
    df = df.sort_values(['uid', 'timestamps']).reset_index(drop=True)
    
    # Convert to error (1=wrong, 0=correct)
    df['error'] = 1 - df['responses']
    

    df['cum_error'] = df.groupby('uid')['error'].transform(
        lambda x: x.expanding().mean()
    )
    

    df['error_increased'] = df.groupby('uid')['cum_error'].transform(
        lambda x: x > x.shift()
    )

    
    df['window_id'] = df.groupby('uid')['error_increased'].transform(
        lambda x: x.cumsum()
    )
    
    # Count questions per window per student
    window_counts = df.groupby(['uid', 'window_id']).size().reset_index(name='window_length')
    
    # Get the longest window for each student
    longest_window = window_counts.groupby('uid')['window_length'].max().reset_index()
    longest_window.columns = ['uid', 'longest_window_before_increase']
    

    
    avg_window = window_counts.groupby('uid')['window_length'].mean().reset_index()
    avg_window.columns = ['uid', 'avg_window_before_increase']

    attention_breaks = df.groupby('uid')['error_increased'].sum().reset_index()
    attention_breaks.columns = ['uid', 'attention_breaks']

    final_error = df.groupby('uid')['cum_error'].last().reset_index()
    final_error.columns = ['uid', 'final_cum_error']
    
    # Merge everything
    sustained_attention = final_error.merge(longest_window, on='uid', how='outer')
    sustained_attention = sustained_attention.merge(avg_window, on='uid', how='outer')
    sustained_attention = sustained_attention.merge(attention_breaks, on='uid', how='outer')
    
    # Fill NaN values
    sustained_attention['longest_window_before_increase'] = sustained_attention['longest_window_before_increase'].fillna(0)
    sustained_attention['avg_window_before_increase'] = sustained_attention['avg_window_before_increase'].fillna(0)
    sustained_attention['attention_breaks'] = sustained_attention['attention_breaks'].fillna(0)

    total_questions = df.groupby('uid').size().reset_index(name='total_questions')
    sustained_attention = sustained_attention.merge(total_questions, on='uid', how='left')
    
    sustained_attention['pct_in_longest_window'] = (
        sustained_attention['longest_window_before_increase'] / 
        sustained_attention['total_questions']
    )
    
    return sustained_attention, df

# Run the function
sustained_attention, df_with_cummulative = calculate_sustained_attention(responses)


print("\n=== Summary Statistics ===")
print(sustained_attention['attention_breaks'].describe())
print(sustained_attention['longest_window_before_increase'].describe())
print(sustained_attention['avg_window_before_increase'].describe())
print(sustained_attention['final_cum_error'].describe())
print(sustained_attention['pct_in_longest_window'].describe())

# ============================================================
# Visualize sustained attention
# ============================================================


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of longest window
ax1 = axes[0, 0]
sns.histplot(sustained_attention['avg_window_before_increase'], bins=30, kde=True, ax=ax1, color='steelblue')
ax1.set_xlabel('Avg Window Before Error Increase', fontsize=12)
ax1.set_ylabel('Number of Students', fontsize=12)
ax1.set_title('Distribution of Sustained Attention (Average Window)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Plot 2: Longest window vs final error rate
ax2 = axes[0, 1]
ax2.scatter(sustained_attention['final_cum_error'], 
            sustained_attention['longest_window_before_increase'],
            alpha=0.3, s=10, color='green')
ax2.set_xlabel('Final Cumulative Error Rate', fontsize=12)
ax2.set_ylabel('Longest Window Before Error Increase', fontsize=12)
ax2.set_title('Longest Window vs Final Error Rate', fontsize=14)
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of attention breaks
ax3 = axes[1, 0]
sns.histplot(sustained_attention['attention_breaks'], bins=30, kde=True, ax=ax3, color='orange')
ax3.set_xlabel('Number of Attention Breaks (Error Increases)', fontsize=12)
ax3.set_ylabel('Number of Students', fontsize=12)
ax3.set_title('Distribution of Attention Breaks', fontsize=14)
ax3.grid(True, alpha=0.3)

# Plot 4: Average window length vs final error rate
ax4 = axes[1, 1]
ax4.scatter(sustained_attention['final_cum_error'], 
            sustained_attention['avg_window_before_increase'],
            alpha=0.3, s=10, color='purple')
ax4.set_xlabel('Final Cumulative Error Rate', fontsize=12)
ax4.set_ylabel('Average Window Length', fontsize=12)
ax4.set_title('Average Window Length vs Final Error Rate', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path=os.path.join(results_path,'sustained_attention_metrics.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# Quartiles for avg_window
# ============================================================

sustained_attention['a_quartile'] = pd.qcut(
    sustained_attention['avg_window_before_increase'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

#calculate mean and std then save to csv
worst_pen=sustained_attention[sustained_attention['a_quartile']=='Q1']
mean=worst_pen['avg_window_before_increase'].mean()
std=worst_pen['avg_window_before_increase'].std()
median=worst_pen['avg_window_before_increase'].median()
count=len(worst_pen)

statistics={
    "mean": mean,
    "std": std,
    "median": median,
    "count":count
}

stats_path = os.path.join(results_path,'sustained_attention_statistics.csv')
pd.DataFrame([statistics]).to_csv(stats_path, index=False)
