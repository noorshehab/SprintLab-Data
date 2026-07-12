"""Behavioural-Diagnosis synthetic-scenario lab.

The proxy dataset is correctness-only (see notebook 00) -- it has no
response time, distractor choice, or cognitive-load field, so none of the
~20 behavioural metrics in the RQ note can be computed from real logs. This
module instead builds a PARAMETRISED SYNTHETIC GENERATOR for the 12
authored student cases, scaffolded on real item difficulty so the synthetic
sessions are at least grounded in a realistic difficulty distribution, then
computes the metrics and tests whether the resulting trait vectors are
separable at all -- an honest feasibility test, not a claim of real-world
validation.
"""
import numpy as np
import pandas as pd

# The fixed treatment-plan vocabulary (backlog B-2 / property test T-2):
# every constraint the Behavioural module may emit must be a member of this set.
TREATMENT_VOCABULARY = {
    "stop_complex_application_items", "serve_tf_and_definitions",
    "insert_scaffolding_prerequisites", "give_rule_reminder_hints",
    "switch_to_visual_item_types", "cognitive_conflict_cue_refutational_feedback",
    "fade_out_background_music", "prompt_pen_and_paper",
    "reduce_time_pressure_items_early", "nudge_accuracy_first",
    "silence_pause_before_distractor_item", "avoid_complex_items_late_game",
    "block_practice_to_interleaving", "advance_warning_rule_change",
    "fading_negation_cue", "internal_pacer_sfx_at_50pct",
    "train_short_fast_items", "slow_music_tempo", "disable_score_multipliers",
    "heartbeat_tempo_critical_moments", "confidence_restoring_direct_item",
}

CASES = [
    "knowledge_absence", "weak_prior_knowledge", "rote_without_understanding", "misconceptions",
    "working_memory", "processing_speed", "attention", "cognitive_flexibility",
    "question_interpretation", "time_management_perfectionism",
    "impulsiveness", "stress_overthinking",
]

CASE_TO_TREATMENTS = {
    "knowledge_absence": ["stop_complex_application_items", "serve_tf_and_definitions"],
    "weak_prior_knowledge": ["insert_scaffolding_prerequisites", "give_rule_reminder_hints"],
    "rote_without_understanding": ["switch_to_visual_item_types"],
    "misconceptions": ["cognitive_conflict_cue_refutational_feedback"],
    "working_memory": ["fade_out_background_music", "prompt_pen_and_paper"],
    "processing_speed": ["reduce_time_pressure_items_early", "nudge_accuracy_first"],
    "attention": ["silence_pause_before_distractor_item", "avoid_complex_items_late_game"],
    "cognitive_flexibility": ["block_practice_to_interleaving", "advance_warning_rule_change"],
    "question_interpretation": ["fading_negation_cue"],
    "time_management_perfectionism": ["internal_pacer_sfx_at_50pct", "train_short_fast_items"],
    "impulsiveness": ["slow_music_tempo", "disable_score_multipliers"],
    "stress_overthinking": ["heartbeat_tempo_critical_moments", "confidence_restoring_direct_item"],
}


def generate_synthetic_session(
    rng: np.random.Generator, case: str, item_difficulty: np.ndarray, n_items: int = 60
) -> pd.DataFrame:
    """Generate one synthetic student's session of `n_items` attempts against
    a real item-difficulty sample. Each case perturbs response-time ratio
    (tau) and/or correctness relative to a difficulty-only baseline, per the
    diagnosis rules authored in the RQ note. Cognitive-load, negation-item,
    and context-switch flags are simulated item metadata (also absent from
    the real logs, generated here so the metrics in `compute_metrics` are
    computable at all)."""
    D = rng.choice(item_difficulty, size=n_items, replace=True)
    cli = rng.uniform(0, 1, n_items)  # cognitive-load index
    is_negation = rng.random(n_items) < 0.2
    is_last_k = np.arange(n_items) >= n_items - 5
    topic = rng.integers(0, 6, n_items)
    context_switch = np.r_[False, topic[1:] != topic[:-1]]
    stress_state = rng.random(n_items) < 0.15  # last-place / rival-threat game states

    base_p_correct = 1 - D  # difficulty-only baseline success probability
    tau = rng.lognormal(mean=0.0, sigma=0.35, size=n_items)  # baseline normalised RT

    if case == "knowledge_absence":
        base_p_correct *= rng.uniform(0.2, 0.4)
    elif case == "weak_prior_knowledge":
        tau *= rng.uniform(1.4, 1.8)
        base_p_correct *= rng.uniform(0.5, 0.7)
    elif case == "rote_without_understanding":
        base_p_correct = np.where(D < 0.4, base_p_correct, base_p_correct * 0.4)
    elif case == "misconceptions":
        base_p_correct *= rng.uniform(0.4, 0.6)
    elif case == "working_memory":
        base_p_correct *= 1 - 0.6 * cli
    elif case == "processing_speed":
        tau = np.clip(tau * rng.uniform(1.3, 1.6), None, 1.0)
        base_p_correct *= np.where(tau >= 0.95, 0.7, 1.0)
    elif case == "attention":
        base_p_correct = np.where(is_last_k, base_p_correct * 0.5, base_p_correct)
    elif case == "cognitive_flexibility":
        tau = np.where(context_switch, tau * rng.uniform(1.5, 2.0), tau)
        base_p_correct = np.where(context_switch, base_p_correct * 0.6, base_p_correct)
    elif case == "question_interpretation":
        tau = np.where(is_negation, tau * 0.5, tau)
        base_p_correct = np.where(is_negation, base_p_correct * 0.4, base_p_correct)
    elif case == "time_management_perfectionism":
        tau = np.where(D < 0.3, tau * rng.uniform(2.5, 4.0), tau)
    elif case == "impulsiveness":
        tau *= rng.uniform(0.15, 0.28)
        base_p_correct *= 0.5
    elif case == "stress_overthinking":
        base_p_correct = np.where(stress_state, base_p_correct * 0.4, base_p_correct)
        tau = np.where(stress_state, tau * 2.0, tau)
    # "typical" (control) case falls through unperturbed if not matched above

    tau = np.clip(tau, 0.02, 3.0)
    correct = rng.random(n_items) < np.clip(base_p_correct, 0.02, 0.98)

    return pd.DataFrame(
        {
            "case": case, "item_idx": np.arange(n_items), "difficulty": D, "tau": tau,
            "correct": correct.astype(int), "cognitive_load": cli, "is_negation": is_negation,
            "is_last_k": is_last_k, "context_switch": context_switch, "stress_state": stress_state,
            "topic": topic,
        }
    )


def compute_metrics(session: pd.DataFrame) -> dict:
    """Compute the RQ note's log-derived metric set for one session,
    difficulty-normalised where noted."""
    correct = session["correct"].astype(bool)
    wrong = ~correct
    tau = session["tau"]

    fast_error_rate = (wrong & (tau < 0.3)).sum() / max(wrong.sum(), 1)
    slow_correct_rate = (correct & (tau >= 1.0)).sum() / max(correct.sum(), 1)
    timeout_rate = (tau >= 1.0).mean()
    time_on_easy_share = tau[session["difficulty"] < 0.3].sum() / max(tau.sum(), 1e-9)

    high_mastery_proxy = session["difficulty"] < 0.25
    carelessness = (wrong & high_mastery_proxy).sum() / max(high_mastery_proxy.sum(), 1)

    neg_err = session.loc[session["is_negation"], "correct"].eq(0).mean() if session["is_negation"].any() else np.nan
    plain_err = session.loc[~session["is_negation"], "correct"].eq(0).mean()
    negation_gap = neg_err - plain_err if not np.isnan(neg_err) else 0.0

    cs = session["context_switch"]
    cs_cost_err = session.loc[cs, "correct"].eq(0).mean() - session.loc[~cs, "correct"].eq(0).mean() if cs.any() else 0.0

    load_slope = np.polyfit(session["cognitive_load"], wrong.astype(float), 1)[0] if session["cognitive_load"].nunique() > 1 else 0.0

    k = 5
    first_k_err = session.iloc[:k]["correct"].eq(0).mean()
    last_k_err = session.iloc[-k:]["correct"].eq(0).mean()
    end_decay = last_k_err - first_k_err

    stressed = session["stress_state"]
    pressure_sensitivity = session.loc[stressed, "correct"].eq(0).mean() - session.loc[~stressed, "correct"].eq(0).mean() if stressed.any() else 0.0

    return {
        "fast_error_rate": fast_error_rate, "slow_correct_rate": slow_correct_rate,
        "timeout_rate": timeout_rate, "time_on_easy_share": time_on_easy_share,
        "carelessness": carelessness, "negation_error_gap": negation_gap,
        "context_switch_cost": cs_cost_err, "cognitive_load_sensitivity": load_slope,
        "end_of_match_decay": end_decay, "pressure_sensitivity": pressure_sensitivity,
        "mean_tau": tau.mean(), "error_rate": wrong.mean(),
    }


def score_to_plan(scores: np.ndarray, thresholds: dict) -> list[str]:
    """A pure, vocabulary-bounded rule layer: emit a treatment for each
    metric whose value exceeds its trigger threshold. `thresholds` maps
    metric name -> (case, cutoff)."""
    plan = []
    for metric, value in scores.items():
        if metric in thresholds and value > thresholds[metric]["cutoff"]:
            case = thresholds[metric]["case"]
            plan.extend(CASE_TO_TREATMENTS.get(case, []))
    return sorted(set(plan))


def plan_smoothness(scores_a: dict, scores_b: dict, plan_a: list[str], plan_b: list[str]) -> dict:
    """RQ2.4 smoothness check: near-identical score vectors should map to
    near-identical plans. Returns the score-vector L2 distance and the plan
    Jaccard distance so the notebook can test the Lipschitz-style relation."""
    keys = sorted(set(scores_a) & set(scores_b))
    a = np.array([scores_a[k] for k in keys])
    b = np.array([scores_b[k] for k in keys])
    score_dist = float(np.linalg.norm(a - b))
    sa, sb = set(plan_a), set(plan_b)
    union = sa | sb
    jaccard_dist = 1 - (len(sa & sb) / len(union) if union else 1.0)
    return {"score_l2_distance": score_dist, "plan_jaccard_distance": jaccard_dist}
