import ast
import os
import sys

#make imports work regardless of the directory the script is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from services.knowledge_tracing import knowledge_tracing_engine
from services.knowledge_tracing import data_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
import mlflow
import dagshub
load_dotenv(override=True)

output_path=os.getenv('EXPERIMENT_OUTPUTS')
dagshub.init(repo_owner='nhatemshehab', repo_name='SprintLab-Data', mlflow=True)

#the engine uses the first 10 responses to calibrate the student's priors
CALIBRATION_WINDOW=10

def prepare_engine():
    """Build the knowledge tracing engine: skills, question library and per-item probabilities."""
    #similar kcs from semantic clustering and per-item bkt probabilities
    similar_skills = data_preprocess.label_clusters()
    probabilities = data_preprocess.set_probs()

    #parse the kc_ids column which is stored as a string like "[0]" or "[1, 2]"
    question_metadata = pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    question_metadata['kc_ids'] = question_metadata['kc_ids'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    question_metadata['kc_ids'] = question_metadata['kc_ids'].apply(
        lambda x: x if isinstance(x, list) else [x]
    )

    question_lib = question_metadata[['question_id', 'kc_ids']].merge(
        probabilities[['question_id', 'p_s', 'p_g', 'p_t']],
        on='question_id', how='left'
    )

    eng = knowledge_tracing_engine.knowledge_tracing_engine(calibration_window=CALIBRATION_WINDOW)

    #register every kc as a skill so similar-skill updates never hit a missing key
    skill_map = dict(zip(similar_skills['kc'], similar_skills['similar_kcs']))
    all_kcs = set(skill_map.keys())
    for kc_list in question_lib.dropna(subset=['p_s'])['kc_ids']:
        all_kcs.update(kc_list)
    for kc in sorted(all_kcs):
        eng.add_skill(kc, skill_map.get(kc, []))

    #register the questions that have a full set of bkt parameters
    n_questions = 0
    for row in question_lib.itertuples():
        if pd.isna(row.p_s) or pd.isna(row.p_g) or pd.isna(row.p_t):
            continue
        eng.add_question(row.question_id, row.kc_ids, row.p_t, row.p_s, row.p_g)
        n_questions += 1

    print(f"\nEngine ready: {len(eng.skills)} skills, {n_questions} questions")
    return eng

def run_simulation(engine, responses, max_students=None):
    """Feed responses chronologically and predict each response (after calibration) before updating priors."""
    records = []

    for rank, (uid, group) in enumerate(responses.groupby('uid', sort=False)):
        if max_students is not None and rank >= max_students:
            break

        engine.add_student(uid)
        group = group.sort_values('timestamps')
        q_ids = group['questions'].tolist()
        actuals = group['responses'].tolist()

        for i in range(len(q_ids)):
            if i >= CALIBRATION_WINDOW:
                p_C, pred = engine.predict_response(uid, q_ids[i])
                records.append((uid, i, q_ids[i], p_C, pred, actuals[i]))
            engine.add_student_response(uid, [q_ids[i]], [actuals[i]])

    predictions = pd.DataFrame(
        records, columns=['uid', 'idx', 'question_id', 'p_C', 'pred', 'actual']
    )
    predictions['hit'] = (predictions['pred'] == predictions['actual']).astype(int)
    return predictions

def evaluate_predictions(predictions):
    """Return classification metrics and the per-student accuracy frame."""
    y_true = predictions['actual'].astype(int)
    y_pred = predictions['pred'].astype(int)
    p_C = predictions['p_C'].astype(float)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, p_C),
        'base_rate_correct': y_true.mean(),
    }
    if y_true.nunique() > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, p_C)

    per_student = predictions.groupby('uid').agg(
        accuracy=('hit', 'mean'),
        predictions=('hit', 'count')
    )
    metrics['per_student_acc_mean'] = per_student['accuracy'].mean()
    metrics['per_student_acc_std'] = per_student['accuracy'].std()

    return metrics, per_student

def improvement_metrics(predictions):
    """Accuracy in the early vs late half of each student's prediction window."""
    def phase_by_idx(series):
        median = series.median()
        return np.where(series <= median, 'early', 'late')

    predictions = predictions.copy()
    predictions['phase'] = predictions.groupby('uid')['idx'].transform(phase_by_idx)

    early = predictions[predictions['phase'] == 'early']['hit'].mean()
    late = predictions[predictions['phase'] == 'late']['hit'].mean()
    return early, late

def plot_confusion_matrix(predictions, fig_path):
    cm = confusion_matrix(
        predictions['actual'].astype(int), predictions['pred'].astype(int), labels=[0, 1]
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Incorrect', 'Predicted Correct'],
                yticklabels=['Actual Incorrect', 'Actual Correct'])
    plt.title('Knowledge Tracing Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def plot_learning_curve(predictions, fig_path):
    #accuracy by prediction index bucket shows the improvement as the history grows
    buckets = pd.cut(predictions['idx'], bins=10)
    curve = predictions.groupby(buckets, observed=True)['hit'].agg(['mean', 'count']).reset_index()
    curve['bucket'] = curve['idx'].astype(str)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(curve)), curve['mean'], marker='o', color='steelblue')
    plt.xticks(ticks=range(len(curve)), labels=curve['bucket'], rotation=45)
    plt.title('Prediction Accuracy vs Response Index\n(calibration window = 10)')
    plt.xlabel('Response Index Bucket')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def plot_calibration_curve(predictions, fig_path):
    #mean predicted probability vs actual correctness rate per decile
    predictions['p_C_bin'] = pd.qcut(predictions['p_C'], q=10, duplicates='drop')
    calib = predictions.groupby('p_C_bin', observed=True)['actual'].agg(['mean', 'count']).reset_index()
    calib['mid'] = calib['p_C_bin'].apply(lambda b: b.mid)

    plt.figure(figsize=(12, 6))
    plt.scatter(calib['mid'], calib['mean'], s=calib['count'] / calib['count'].max() * 200, color='steelblue', alpha=0.7, label='Model')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect calibration')
    plt.title('Calibration Curve: Predicted P(Correct) vs Observed')
    plt.xlabel('Mean Predicted P(Correct)')
    plt.ylabel('Observed Correct Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def plot_student_accuracy_hist(per_student, fig_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(per_student['accuracy'], bins=50, kde=True, color='steelblue')
    plt.axvline(x=per_student['accuracy'].mean(), color='red', linestyle='--', alpha=0.7, label=f"Mean: {per_student['accuracy'].mean():.4f}")
    plt.title('Distribution of Per-Student Prediction Accuracy')
    plt.xlabel('Per-Student Accuracy')
    plt.ylabel('Number of Students')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def main():
    custom_name = input("Enter a name for the experiment: ")

    with mlflow.start_run() as run:
        #fetch dagshub username automatically
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
        mlflow.set_tag("mlflow.runName", run_name)

        #prepare the engine and run the sequential prediction simulation
        engine = prepare_engine()
        responses = pd.read_csv(os.getenv('TEST_SET'))
        print(f"Loaded {len(responses)} responses from {responses['uid'].nunique()} students")

        max_students = os.getenv('MAX_STUDENTS')
        max_students = int(max_students) if max_students else None

        predictions = run_simulation(engine, responses, max_students=max_students)
        print(f"\nGenerated {len(predictions)} predictions (after the {CALIBRATION_WINDOW}-response calibration window)")

        metrics, per_student = evaluate_predictions(predictions)

        #early vs late accuracy: improvement with more responses
        early_acc, late_acc = improvement_metrics(predictions)
        metrics['early_half_accuracy'] = early_acc
        metrics['late_half_accuracy'] = late_acc

        metrics_df = pd.DataFrame([metrics]).round(4)
        print("\nKnowledge Tracing Metrics:")
        print(metrics_df.to_string(index=False))

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        #artifacts
        fig_path = os.path.join(output_path, 'kt_confusion_matrix.png')
        plot_confusion_matrix(predictions, fig_path)
        mlflow.log_artifact(fig_path)

        fig_path = os.path.join(output_path, 'kt_learning_curve.png')
        plot_learning_curve(predictions, fig_path)
        mlflow.log_artifact(fig_path)

        fig_path = os.path.join(output_path, 'kt_calibration_curve.png')
        plot_calibration_curve(predictions, fig_path)
        mlflow.log_artifact(fig_path)

        fig_path = os.path.join(output_path, 'kt_student_accuracy_hist.png')
        plot_student_accuracy_hist(per_student, fig_path)
        mlflow.log_artifact(fig_path)

        #correlation between per-student accuracy and number of predictions
        corr = per_student['accuracy'].corr(per_student['predictions'])
        mlflow.log_metric('accuracy_count_correlation', corr)
        print(f"\nCorrelation between per-student accuracy and #predictions: {corr:.4f}")

if __name__ == '__main__':
    main()