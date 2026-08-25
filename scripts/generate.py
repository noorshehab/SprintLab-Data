import numpy as np
import pandas as pd

# Load questions dataset
df_q = pd.read_csv("E://projects//sprintlabfiles//sprintlab_candidate_science_questions.csv")

# 1. Define 4 Test Students representing different Priors and Behavioral Diagnoses
# 25% distribution across student types for specific content gap / behavioral problems

students = [
    {
        "student_id": "S101",
        "name": "Student A (Low Prior, High Impulsivity & Language Gap)",
        "priors": {"KC-BIO-01": 0.25, "KC-PHYS-01": 0.30, "KC-CHEM-01": 0.35},
        "content_gaps": {"KC-BIO-01": "Gap_Absence", "KC-PHYS-01": "Gap_Prior"},
        "diagnoses": ["language", "impulsive", "processing_speed"],
        "deltas_history": [
            {"language": -0.35, "processing_speed": -0.25, "timestamp": "2026-08-01T10:00:00"},
            {"language": -0.20, "processing_speed": -0.15, "timestamp": "2026-08-15T10:00:00"} # 42.8% lang imp, 40% speed imp
        ],
        "priors_history": [
            {"KC-BIO-01": 0.25, "KC-PHYS-01": 0.30, "timestamp": "2026-08-01T10:00:00"},
            {"KC-BIO-01": 0.40, "KC-PHYS-01": 0.48, "timestamp": "2026-08-15T10:00:00"} # 60% prior imp
        ]
    },
    {
        "student_id": "S102",
        "name": "Student B (Low Prior, Working Memory & Concept Gap)",
        "priors": {"KC-BIO-02": 0.20, "KC-PHYS-02": 0.40, "KC-CHEM-02": 0.45},
        "content_gaps": {"KC-BIO-02": "Gap_Concept", "KC-CHEM-02": "Gap_Misconception"},
        "diagnoses": ["working_memory", "time_management"],
        "deltas_history": [
            {"working_memory": -0.40, "time_management_ratio": 2.5, "timestamp": "2026-08-01T10:00:00"},
            {"working_memory": -0.28, "time_management_ratio": 1.8, "timestamp": "2026-08-15T10:00:00"} # 30% WM imp, 28% TM imp
        ],
        "priors_history": [
            {"KC-BIO-02": 0.20, "KC-CHEM-02": 0.45, "timestamp": "2026-08-01T10:00:00"},
            {"KC-BIO-02": 0.32, "KC-CHEM-02": 0.60, "timestamp": "2026-08-15T10:00:00"}
        ]
    },
    {
        "student_id": "S103",
        "name": "Student C (Moderate Prior, Flexibility & Stress)",
        "priors": {"KC-BIO-01": 0.55, "KC-PHYS-02": 0.60, "KC-CHEM-01": 0.48},
        "content_gaps": {"KC-CHEM-01": "Gap_Absence"},
        "diagnoses": ["flexibility", "stress"],
        "deltas_history": [
            {"flexibility": -0.15, "stress_ratio": 2.8, "timestamp": "2026-08-01T10:00:00"},
            {"flexibility": -0.09, "stress_ratio": 1.9, "timestamp": "2026-08-15T10:00:00"}
        ],
        "priors_history": [
            {"KC-CHEM-01": 0.48, "timestamp": "2026-08-01T10:00:00"},
            {"KC-CHEM-01": 0.65, "timestamp": "2026-08-15T10:00:00"}
        ]
    },
    {
        "student_id": "S104",
        "name": "Student D (High Prior, Attention Span)",
        "priors": {"KC-BIO-01": 0.75, "KC-PHYS-01": 0.82, "KC-CHEM-02": 0.88},
        "content_gaps": {},
        "diagnoses": ["attention_span"],
        "deltas_history": [
            {"attention": 1.8, "timestamp": "2026-08-01T10:00:00"},
            {"attention": 2.5, "timestamp": "2026-08-15T10:00:00"}
        ],
        "priors_history": [
            {"KC-BIO-01": 0.75, "KC-PHYS-01": 0.82, "timestamp": "2026-08-01T10:00:00"},
            {"KC-BIO-01": 0.88, "KC-PHYS-01": 0.92, "timestamp": "2026-08-15T10:00:00"}
        ]
    }
]

print("Test dataset created with 4 student response profiles.")

# Let's write the logic for treatment plan initialization, update scaling, candidate filtering, and MAB optimal selection according to the code snippets provided.

def get_initial_treatment_plan(student):
    # Based on Treatment_Service code (source 5)
    initial_map = {
        'language': {'lang_difficulty': 'Q1', 'Operator': '=='},
        'working_memory': {'max_cognitive_load': 1, 'Operator': '<='},
        'processing_speed': {'time_pressure': False, 'Operator': '=='},
        'time_management': {'max_time': 180, 'max_steps': 2, 'Operator': '<='},
        'Gap_Absence': {'bloom_types': ["Remember", "Understand"], 'Operator': 'in'},
        'Gap_Concept': {'has_image': True, 'Operator': '=='}
    }
    
    treatment_plan = {'general': {}, 'specific': {}}
    for diag in student['diagnoses']:
        if diag in initial_map:
            treatment_plan['general'][diag] = initial_map[diag].copy()
            
    for skill_id, gap_type in student['content_gaps'].items():
        if gap_type in initial_map:
            treatment_plan['specific'][str(skill_id)] = initial_map[gap_type].copy()
            
    return treatment_plan

def update_treatment_plan(student, old_plan):
    # Based on Treatment_Service update_treatment_plan logic (source 5)
    # Calculate delta improvement
    old_deltas = student['deltas_history'][0]
    new_deltas = student['deltas_history'][1]
    
    delta_metric_map = {
        'language': 'language',
        'attention_span': 'attention',
        'flexibility': 'flexibility',
        'working_memory': 'working_memory',
        'processing_speed': 'processing_speed',
        'time_management': 'time_management_ratio',
        'stress': 'stress_ratio'
    }
    
    improvement = {}
    for metric in set([*old_deltas.keys(), *new_deltas.keys()]):
        if metric == 'timestamp': continue
        o, n = old_deltas.get(metric), new_deltas.get(metric)
        if o is not None and n is not None and o != 0:
            improvement[metric] = (abs(o) - abs(n)) / abs(o) * 100.0
            
    updated_plan = {'general': {}, 'specific': {}}
    
    for t_name, params in old_plan.get('general', {}).items():
        metric = delta_metric_map.get(t_name)
        pct = improvement.get(metric, 0.0)
        scaled = params.copy()
        
        if t_name == 'language':
            if pct >= 40: scaled['lang_difficulty'] = 'Q4'
            elif pct >= 20: scaled['lang_difficulty'] = 'Q3'
            elif pct >= 10: scaled['lang_difficulty'] = 'Q2'
        elif t_name == 'working_memory':
            if pct >= 40: scaled['max_cognitive_load'] = 4
            elif pct >= 20: scaled['max_cognitive_load'] = 3
            elif pct >= 10: scaled['max_cognitive_load'] = 2
        elif t_name == 'time_management':
            if pct >= 40: 
                scaled['max_time'] += 180
                scaled['max_steps'] += 6
            elif pct >= 20: 
                scaled['max_time'] += 120
                scaled['max_steps'] += 4
            elif pct >= 10: 
                scaled['max_time'] += 60
                scaled['max_steps'] += 2
                
        updated_plan['general'][t_name] = scaled

    # Specific skills update based on prior improvement
    old_priors = student['priors_history'][0]
    new_priors = student['priors_history'][1]
    prior_imp = {}
    for sk in set([*old_priors.keys(), *new_priors.keys()]):
        if sk == 'timestamp': continue
        o, n = old_priors.get(sk, 0.0), new_priors.get(sk, 0.0)
        if o != 0: prior_imp[sk] = (n - o) / o * 100.0
        
    for sk_id, params in old_plan.get('specific', {}).items():
        pct = prior_imp.get(sk_id, 0.0)
        scaled = {}
        for k, v in params.items():
            if isinstance(v, (int, float)):
                scaled[k] = round(v * (1 + pct / 100.0), 4)
            else:
                scaled[k] = v
        updated_plan['specific'][sk_id] = scaled

    return updated_plan

# Run simulation for all students
sim_results = []
for st in students:
    p1 = get_initial_treatment_plan(st)
    p2 = update_treatment_plan(st, p1)
    sim_results.append({
        'student': st['student_id'],
        'name': st['name'],
        'initial_plan': p1,
        'updated_plan': p2
    })

for res in sim_results:
    print("=== " + res['student'] + ": " + res['name'] + " ===")
    print("Initial Plan:", res['initial_plan'])
    print("Updated Plan:", res['updated_plan'])
    print()

# Let's filter candidates and run Contextual Bandit match simulation for each student

def filter_candidates(df_q, treatment_plan):
    # Unwrap conditions from treatment plan
    df_filtered = df_q.copy()
    
    # Apply language condition if present
    gen = treatment_plan.get('general', {})
    if 'language' in gen:
        lang_diff = gen['language']['lang_difficulty']
        # Map Q1..Q4 to Bloom/Difficulty or Language_Challenging
        if lang_diff == 'Q1':
            df_filtered = df_filtered[df_filtered['Language_Challenging'] == False]
            
    if 'processing_speed' in gen:
        if gen['processing_speed']['time_pressure'] == False:
            df_filtered = df_filtered[df_filtered['Time_Pressure_Flag'] == False]
            
    if 'working_memory' in gen:
        max_cog = gen['working_memory']['max_cognitive_load']
        df_filtered = df_filtered[df_filtered['Cognitive_Load_Index'] <= max_cog]
        
    if 'time_management' in gen:
        max_t = gen['time_management']['max_time']
        max_s = gen['time_management']['max_steps']
        df_filtered = df_filtered[(df_filtered['Time_Allowed'] <= max_t) & (df_filtered['Logical_Steps'] <= max_s)]

    # Specific conditions (e.g. bloom_types, has_image)
    spec = treatment_plan.get('specific', {})
    for sk, params in spec.items():
        if 'bloom_types' in params:
            df_filtered = df_filtered[df_filtered['Bloom_Taxonomy_Level'].isin(params['bloom_types'])]
            
    return df_filtered

# Simulate candidate set vs optimal set selection for Student A and Student B
for st_info in sim_results[:2]:
    st_id = st_info['student']
    st_obj = [s for s in students if s['student_id'] == st_id][0]
    
    # Candidate Set Initial Plan
    cand_init = filter_candidates(df_q, st_info['initial_plan'])
    # Candidate Set Updated Plan
    cand_upd = filter_candidates(df_q, st_info['updated_plan'])
    
    # Calculate MAB optimal set (top 5 by expected reward = (1 - mean_priors) * difficulty * learning)
    priors = st_obj['priors']
    avg_prior = np.mean(list(priors.values()))
    
    cand_init = cand_init.copy()
    cand_init['mab_score'] = (1 - avg_prior) * cand_init['Difficulty_Level'] * (1 - cand_init['Population_Difficulty'])
    optimal_init = cand_init.sort_values('mab_score', ascending=False).head(5)
    
    cand_upd = cand_upd.copy()
    cand_upd['mab_score'] = (1 - avg_prior) * cand_upd['Difficulty_Level'] * (1 - cand_upd['Population_Difficulty'])
    optimal_upd = cand_upd.sort_values('mab_score', ascending=False).head(5)

    print(f"=== MATCH SIMULATION FOR {st_id} ({st_obj['name']}) ===")
    print(f"Candidate Set Size (Initial Plan): {len(cand_init)} / {len(df_q)}")
    print("Optimal Top 3 Questions (Initial Plan):", optimal_init[['Question_ID', 'Bloom_Taxonomy_Level', 'Difficulty_Level', 'mab_score']].to_dict(orient='records')[:3])
    print(f"Candidate Set Size (Updated Plan): {len(cand_upd)} / {len(df_q)}")
    print("Optimal Top 3 Questions (Updated Plan):", optimal_upd[['Question_ID', 'Bloom_Taxonomy_Level', 'Difficulty_Level', 'mab_score']].to_dict(orient='records')[:3])
    print("-" * 60)