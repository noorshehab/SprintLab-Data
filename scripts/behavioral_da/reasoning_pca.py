#imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
load_dotenv()

#load files
responses=pd.read_csv(os.getenv('TRAINING_SET'))
question_metadata=pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
q_diff=pd.read_csv(os.getenv('differentiation_cumulative'))
results_path=os.getenv("EXPERIMENT_OUTPUTS")
question_metadata=pd.merge(
    question_metadata,q_diff[['questions','avg_delta','worsening_rate','relative_improvement']],
    left_on='question_id',right_on='questions',
    how='left'

)
#pca on linguistic features: question_length,num_sentences,num_clauses,vocabulary_richness
features = [
   'solution_complexity_y',
   'num_equations',
   'num_steps',
   'solution_vocab',
    'avg_delta',
    'worsening_rate',
    'relative_improvement'
]

# Extract features and handle any missing values
pca_data = question_metadata[features].copy()
pca_data = pca_data.dropna()
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)
pca_scaled_df = pd.DataFrame(pca_scaled, columns=features)

pca = PCA()
pca_result = pca.fit_transform(pca_scaled)

# Create DataFrame with PCA components
pca_df = pd.DataFrame(
    pca_result, 
    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
)

print("\n=== Explained Variance ===")
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {ev:.4f} ({ev*100:.2f}%) | Cumulative: {cv:.4f} ({cv*100:.2f}%)")

print(f"\nTotal variance explained by first 2 PCs: {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
print(f"Total variance explained by first 3 PCs: {cumulative_variance[2]:.4f} ({cumulative_variance[2]*100:.2f}%)")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
    index=features
)

print("\n=== Component Loadings (Feature Contributions) ===")
print(loadings.round(4))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Scree plot (explained variance)
ax1 = axes[0, 0]
ax1.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='Individual')
ax1.plot(range(1, len(explained_variance)+1), cumulative_variance, 'ro-', label='Cumulative')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('Scree Plot', fontsize=14)
ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loadings heatmap
ax2 = axes[0, 1]
sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0, 
            ax=ax2, cbar_kws={'label': 'Loading'})
ax2.set_title('Component Loadings', fontsize=14)
ax2.set_xlabel('Principal Component', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)

# Plot 3: First two PCA components (scatter)
ax3 = axes[1, 0]
scatter = ax3.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, s=10, c=pca_data['avg_delta'], cmap='viridis')
ax3.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
ax3.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
ax3.set_title('PCA: Questions by Linguistic Features', fontsize=14)
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Error Rate', fontsize=10)

# Plot 4: Feature contributions to PC1 and PC2
ax4 = axes[1, 1]
loadings_2d = loadings[['PC1', 'PC2']]
for i, feature in enumerate(loadings_2d.index):
    ax4.arrow(0, 0, loadings_2d.iloc[i, 0], loadings_2d.iloc[i, 1], 
              head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
    ax4.text(loadings_2d.iloc[i, 0] * 1.1, loadings_2d.iloc[i, 1] * 1.1, 
             feature, fontsize=10)
ax4.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
ax4.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
ax4.set_title('Feature Contributions to PC1 and PC2', fontsize=14)
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlim(-1.1, 1.1)
ax4.set_ylim(-1.1, 1.1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path=os.path.join(results_path,'pca_reasoning.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

print("\n=== PCA Summary ===")
print(f"Number of questions: {len(pca_data)}")
print(f"Number of features: {len(features)}")
print(f"\nTop 3 features for PC1 (positive):")
top_pc1_pos = loadings['PC1'].nlargest(3)
for feat, val in top_pc1_pos.items():
    print(f"  {feat}: {val:.4f}")

print(f"\nTop 3 features for PC1 (negative):")
top_pc1_neg = loadings['PC1'].nsmallest(3)
for feat, val in top_pc1_neg.items():
    print(f"  {feat}: {val:.4f}")

print(f"\nTop 3 features for PC2 (positive):")
top_pc2_pos = loadings['PC2'].nlargest(3)
for feat, val in top_pc2_pos.items():
    print(f"  {feat}: {val:.4f}")

print(f"\nTop 3 features for PC2 (negative):")
top_pc2_neg = loadings['PC2'].nsmallest(3)
for feat, val in top_pc2_neg.items():
    print(f"  {feat}: {val:.4f}")