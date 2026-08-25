import ast
import os
import sys

#mlflow prints an emoji run-URL banner on stdout; keep it from crashing on
#consoles that default to a non-unicode encoding (e.g. Windows cp1252)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')


#make imports work regardless of the directory the script is run from
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'services'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'knowledge_tracing'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'services', 'behavioral_diagnosis'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mlflow
import dagshub
load_dotenv(override=True)

from services.Data_service import Data_Service
from services.Diagnosis_service import Diagnosis_service
from services.knowledge_tracing.knowledge_tracing_engine import knowledge_tracing_engine
from services.behavioral_diagnosis.behavioral_diagnosis_engine import behavioral_diagnosis_engine
from services.knowledge_tracing import data_preprocess
from services.question_processing import feature_derivation
from services.question_processing import prob_calcs

output_path = os.getenv('EXPERIMENT_OUTPUTS')
dagshub.init(repo_owner='nhatemshehab', repo_name='SprintLab-Data', mlflow=True)

#the KT engine uses the first 10 responses to calibrate the student's priors
CALIBRATION_WINDOW = 10
#behavioral diagnosis is only ever run once a student has 100 responses
MIN_DIAGNOSIS_RESPONSES = 100
#MCQ timing bounds
MIN_SOLVE_TIME = 60
MAX_SOLVE_TIME = 900  #15 minutes


#--------------------------------------------------------------------------------
# timing generation
#--------------------------------------------------------------------------------
def assign_solve_times(question_metadata):
    """Scale solution_length into the [60s, 900s] range. Returns a per-q solve time keyed by question_id."""
    id_col = 'question_id' if 'question_id' in question_metadata.columns else 'Question_ID'
    if 'Time_Allowed' in question_metadata.columns:
        t = pd.to_numeric(question_metadata['Time_Allowed'], errors='coerce').fillna(MIN_SOLVE_TIME).clip(MIN_SOLVE_TIME, MAX_SOLVE_TIME)
        t.index = question_metadata[id_col].values
        return t.astype(float)
    s = question_metadata['solution_length'].astype(float).fillna(
        question_metadata['solution_length'].min() if not question_metadata['solution_length'].isna().all() else 0
    )
    smin, smax = s.min(), s.max()
    if smax > smin:
        solve_time = MIN_SOLVE_TIME + (s - smin) / (smax - smin) * (MAX_SOLVE_TIME - MIN_SOLVE_TIME)
    else:
        solve_time = pd.Series(MIN_SOLVE_TIME, index=s.index)
    solve_time = solve_time.clip(MIN_SOLVE_TIME, MAX_SOLVE_TIME)
    solve_time.index = question_metadata['question_id'].astype(int).values
    return solve_time.astype(float)


def generate_response_timing(solve_time):
    """response_time ~ N(solve_time, 0.5*solve_time), floored at 0. Also draws stress triggers 0-3."""
    response_time = float(np.random.normal(solve_time, 0.5 * solve_time))
    response_time = max(0.0, response_time)
    stress_triggers = int(np.random.randint(0, 4))  # random 0-3
    return response_time, stress_triggers


#--------------------------------------------------------------------------------
# question attribute derivation (mirrors test_diagnosis.prepare_test_set)
#--------------------------------------------------------------------------------
def derive_question_attributes(question_metadata):
    """Derive the question attributes for the generated question-set schema.

    Runs the feature-derivation + prob pipelines so the renamed entity fields
    (language_level, reasoning_level, cognitive_load, num_unknowns, num_operations,
    p_s/p_g/p_t) are produced even though the generated set does not contain them.
    """
    md = feature_derivation.derive_features(
        question_metadata,
        text_col='Question_Text',
        solutions_col='Correct_Answer_Content',
        language_col='Question_Language',
    )
    if 'difficulty_level' not in md.columns:
        md['difficulty_level'] = pd.to_numeric(md.get('Difficulty_Level'))
    if 'time_pressure' not in md.columns and 'Time_Pressure_Flag' in md.columns:
        md['time_pressure'] = md['Time_Pressure_Flag'].astype(str).str.lower().isin(
            ['1', 'true', 'yes', 'y', 't']).astype(int)
    probs = prob_calcs.compute_probs(md.copy())
    md = pd.merge(md, probs[['question_id', 'p_s', 'p_g', 'p_t']], on='question_id', how='left')
    md['p_t'] = md['p_t'].fillna(prob_calcs.DEFAULT_P_T)
    return md


def parse_kc_ids(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return [val]
    if isinstance(val, list):
        return val
    return [val]


def parse_unit(val):
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    if isinstance(val, list):
        return val[0] if len(val) > 0 else None
    return val


#--------------------------------------------------------------------------------
# engine / data service setup
#--------------------------------------------------------------------------------
def prepare_services():
    """Register skills and questions into the singleton data service and wire the mediator.

    Set USE_QUESTION_PROCESSING=1 to run ingestion through the new data-ingestion +
    question-processing services (embedding, vector store, similar-skills map, probs).
    When the metadata uses the generated question-set schema (Question_ID column) the
    processed path is used automatically.
    """
    question_metadata = pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    if 'Question_ID' in question_metadata.columns:
        print('[prepare] detected generated question-set schema -> processed ingestion')
        return prepare_services_processed()

    if os.getenv('USE_QUESTION_PROCESSING', '0').strip() in ('1', 'true', 'True'):
        return prepare_services_processed()
    data_service = Data_Service()

    similar_skills = data_preprocess.label_clusters()
    probabilities = data_preprocess.set_probs()

    question_metadata['kc_ids'] = question_metadata['kc_ids'].apply(parse_kc_ids)
    md = derive_question_attributes(question_metadata)
    solve_times = assign_solve_times(md)
    unit_index = md['question_id'].values
    unit_series = pd.Series(md.get('super_topic_ids', pd.Series(None, index=md.index)).apply(parse_unit).values, index=unit_index)

    question_lib = md.merge(
        probabilities[['question_id', 'p_s', 'p_g', 'p_t']],
        on='question_id', how='left'
    )

    #register every kc as a skill so similar-skill updates never hit a missing key
    skill_map = dict(zip(similar_skills['kc'], similar_skills['similar_kcs']))
    all_kcs = set(skill_map.keys())
    for kc_list in question_lib.dropna(subset=['p_s'])['kc_ids']:
        all_kcs.update(kc_list)
    for kc in sorted(all_kcs):
        data_service.add_skill(kc, skill_map.get(kc, []))

    #register the questions that have a full set of bkt parameters
    n_questions = 0
    for idx, row in question_lib.iterrows():
        if pd.isna(row.p_s) or pd.isna(row.p_g) or pd.isna(row.p_t):
            continue
        q_id = row['question_id']
        data_service.add_question(
            q_id=q_id,
            skill_ids=row['kc_ids'],
            unit_id=unit_series.get(q_id, None),
            question_text=str(q_id),
            time_allowed=solve_times.get(q_id, MIN_SOLVE_TIME),
            time_pressure_flag=int(row.get('time_pressure', 0)),
            difficulty_level=float(row.get('difficulty_level', 0.0)),
            cognitive_load=row.get('wm_score', 0.0),
            variables_count=int(row.get('num_variables', 0)),
            logical_steps=int(row.get('num_steps', 0)),
            language_challenging=int(row.get('language_challenge', 0)),
            language_level=str(row.get('language_level', None)) if row.get('language_level') is not None else None,
            reasoning_level=str(row.get('reasoning_level', None)) if row.get('reasoning_level') is not None else None,
            p_t=row.p_t, p_s=row.p_s, p_g=row.p_g,
        )
        n_questions += 1

    kt = knowledge_tracing_engine(calibration_window=CALIBRATION_WINDOW)
    bd = behavioral_diagnosis_engine()
    mediator = Diagnosis_service(kt, bd, data_service)

    print(f"\nServices ready: {len(data_service.skills)} skills, {n_questions} questions, {len(data_service.students)} students")
    return mediator, data_service


def prepare_services_processed():
    """Same contract as prepare_services() but routed through the data-ingestion and
    question-processing services (embedding, per-language vector stores, similar-skills
    map, derived attributes, BKT probs with p_t fixed at 0.017)."""
    from services.data_ingestion import file_loader
    from services.data_ingestion.data_ingestion_service import data_ingestion_service
    from services.question_processing.embedding_service import embedding_service
    from services.question_processing.question_processing_service import question_processing_service

    data_service = Data_Service()
    kt = knowledge_tracing_engine(calibration_window=CALIBRATION_WINDOW)
    bd = behavioral_diagnosis_engine()
    mediator = Diagnosis_service(kt, bd, data_service)

    ingest = data_ingestion_service()
    ingest.mediator = mediator

    qproc = question_processing_service(
        embedder=embedding_service(),
        text_col=os.getenv('QUESTION_TEXT_FIELD', 'Question_Text'),
        solutions_col=os.getenv('QUESTION_SOLUTION_FIELD', 'Correct_Answer_Content'),
    )
    qproc.mediator = mediator

    question_metadata = pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
    md = qproc.process(question_metadata, questions_json=None)
    records = md.to_dict('records')

    n_questions, _ = ingest.populate_questions(records)
    ingest.populate_skills(qproc.similar_skills_map)

    print(f"\nServices ready (processed): {len(data_service.skills)} skills, {n_questions} questions, {len(data_service.students)} students")
    return mediator, data_service


def _load_raw_questions(cache={}):
    if not cache:
        from services.data_ingestion import file_loader
        cache.update(file_loader.load_records(os.getenv('QUESTIONS_PATH')))
    return cache


def _time_pressure_tag(md, qid):
    row = md.loc[md['question_id'] == qid]
    if row.empty or 'time_pressure' not in row.columns:
        return 0
    return int(row['time_pressure'].iloc[0])
    row = md.loc[md['question_id'] == qid]
    if row.empty or 'time_pressure' not in row.columns:
        return 0
    return int(row['time_pressure'].iloc[0])


#--------------------------------------------------------------------------------
# simulation
#--------------------------------------------------------------------------------
def run_simulation(mediator, responses, solve_time_map, max_students=None):
    """Feed responses chronologically with generated timings. Predicts after calibration."""
    records = []

    for rank, (uid, group) in enumerate(responses.groupby('uid', sort=False)):
        if max_students is not None and rank >= max_students:
            break

        mediator.Data_service.add_student(uid)
        group = group.sort_values('timestamps')
        q_ids = [int(q) for q in group['questions'].tolist()]
        actuals = group['responses'].tolist()

        for i in range(len(q_ids)):
            q_id = q_ids[i]
            actual = actuals[i]

            solve_time = solve_time_map.get(q_id, MIN_SOLVE_TIME)

            response_time, stress_triggers = generate_response_timing(solve_time)

            if i >= CALIBRATION_WINDOW and q_id in mediator.Data_service.questions:
                p_C, pred = mediator.predict_response(uid, q_id)
                records.append((uid, i, q_id, p_C, pred, actual))

            mediator.add_student_response(
                uid, [q_id], [actual], [response_time], [stress_triggers]
            )

    predictions = pd.DataFrame(
        records, columns=['uid', 'idx', 'question_id', 'p_C', 'pred', 'actual']
    )
    if not predictions.empty:
        predictions['hit'] = (predictions['pred'] == predictions['actual']).astype(int)
    return predictions


def build_solve_time_map(question_metadata):
    md = question_metadata.copy()
    if 'kc_ids' in md.columns:
        md['kc_ids'] = md['kc_ids'].apply(parse_kc_ids)
    solve_times = assign_solve_times(md)
    return solve_times.to_dict()


#--------------------------------------------------------------------------------
# knowledge tracing evaluation
#--------------------------------------------------------------------------------
def evaluate_predictions(predictions):
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


#--------------------------------------------------------------------------------
# behavioral diagnosis comparison (f100 triggered at count==100 vs full set)
#--------------------------------------------------------------------------------
DIAGNOSTICS = ['attention', 'reasoning', 'language', 'flexibility', 'frustration', 'working_memory']


def collect_diagnoses(mediator, min_responses=MIN_DIAGNOSIS_RESPONSES):
    """Full-set diagnosis per student. f100 diagnoses were already stored in the mediator."""
    full = {}
    for uid, student in mediator.Data_service.students.items():
        if len(student.get_responses()) >= min_responses:
            res = mediator.diagnose_student(uid)
            if res is not None:
                full[uid] = res
    return full


def compare_diagnoses(mediator):
    f100 = {uid: v for uid, v in mediator.f100_diagnoses.items() if v is not None}
    full = collect_diagnoses(mediator)

    common = sorted(set(f100.keys()) & set(full.keys()))
    print(f"\nStudents with f100 & full diagnoses: {len(common)}")

    f100_df = pd.DataFrame({uid: f100[uid] for uid in common}).T
    full_df = pd.DataFrame({uid: full[uid] for uid in common}).T

    merged = full_df.merge(f100_df, left_index=True, right_index=True, suffixes=('_gt', '_f100'))

    metrics = []
    for diag in DIAGNOSTICS:
        y_true = merged[f'{diag}_diag_gt'].astype(bool)
        y_pred = merged[f'{diag}_diag_f100'].astype(bool)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        corr = merged[f'{diag}_gt'].corr(merged[f'{diag}_f100'])

        mlflow.log_metric(f'{diag}_accuracy', acc)
        mlflow.log_metric(f'{diag}_precision', prec)
        mlflow.log_metric(f'{diag}_recall', rec)
        mlflow.log_metric(f'{diag}_correlation', corr)

        metrics.append({'diagnosis': diag, 'accuracy': acc, 'precision': prec, 'recall': rec, 'correlation': corr})

        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diagnosis', 'Diagnosis'], yticklabels=['No Diagnosis', 'Diagnosis'])
        plt.title(f'Confusion Matrix for {diag.capitalize()} Diagnosis (f100 vs full)')
        plt.xlabel('Predicted (f100)')
        plt.ylabel('Actual (full)')
        fig_path = os.path.join(output_path, f'confusion_matrix_{diag}.png')
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.round(4))
    return metrics_df


#--------------------------------------------------------------------------------
# main
#--------------------------------------------------------------------------------
def main():
    custom_name = input("Enter a name for the experiment: ")

    with mlflow.start_run() as run:
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

        run_name = f"{current_user}#{run_number}_combined_{custom_name}"
        mlflow.set_tag("mlflow.runName", run_name)

        #---------------------------------------------------------------
        # services + data
        #---------------------------------------------------------------
        mediator, data_service = prepare_services()
        responses = pd.read_csv(os.getenv('TEST_SET'))
        print(f"Loaded {len(responses)} responses from {responses['uid'].nunique()} students")

        max_students = os.getenv('MAX_STUDENTS')
        max_students = int(max_students) if max_students else None

        question_metadata = pd.read_csv(os.getenv('QUESTION_METADATA_PATH'))
        solve_time_map = build_solve_time_map(question_metadata)

        #---------------------------------------------------------------
        # knowledge tracing simulation (BD is triggered at 100 responses
        # automatically inside mediator.add_student_response)
        #---------------------------------------------------------------
        predictions = run_simulation(mediator, responses, solve_time_map, max_students=max_students)
        print(f"\nGenerated {len(predictions)} predictions (after the {CALIBRATION_WINDOW}-response calibration window)")

        if not predictions.empty:
            metrics, per_student = evaluate_predictions(predictions)
            early_acc, late_acc = improvement_metrics(predictions)
            metrics['early_half_accuracy'] = early_acc
            metrics['late_half_accuracy'] = late_acc

            metrics_df = pd.DataFrame([metrics]).round(4)
            print("\nKnowledge Tracing Metrics:")
            print(metrics_df.to_string(index=False))

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

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

            corr = per_student['accuracy'].corr(per_student['predictions'])
            mlflow.log_metric('accuracy_count_correlation', corr)
            print(f"\nCorrelation between per-student accuracy and #predictions: {corr:.4f}")

        #---------------------------------------------------------------
        # behavioral diagnosis: f100 (stored during simulation) vs full set
        #---------------------------------------------------------------
        metrics_df = compare_diagnoses(mediator)


if __name__ == '__main__':
    main()