# Component Reference

## Entities (`services/Entities.py`)

- **`student`** — id, priors `{skill: float}`, responses list of
  `(q_id, correct, time, stress_trigger, atag)` tuples, diagnoses, deltas
  history / diagnoses history (`{timestamp: ...}`), priors history,
  treatment plan `{name: (timestamp, params)}`, content gaps.
- **`question`** — ~50 fields; the serving-relevant ones are `skill_ids`,
  `difficulty_level`, `bloom_taxonomy_level`, `language_level`,
  `reasoning_level`, `cognitive_load(_index)`, `logical_steps`,
  `time_pressure_flag`, `visual_dependency`, `multi_concept_flag`,
  `p_t/p_s/p_g`. `get_atts()` returns a dict used by the BD feature frame.
- Canonical attribute names live in `services/question_schema.py`
  (aliases like `lang_difficulty` → `language_level`,
  `has_image` → `visual_dependency` resolve automatically in queries;
  truly unknown names raise).

## Knowledge tracing

**`BKT.update_prior(prior, guess, slip, improvement, response)`** — standard
Bayesian BKT: learning transition `P* = P + P_T(1−P)`, then Bayes' rule on
the response. Guaranteed invariants (contract-tested):
correct raises the prior, wrong lowers it, `P(L|correct) ≥ P(L|wrong)`,
result strictly inside (0, 1). The engine clamps stored values to
**[0.01, 0.99]**.

> History note: the original implementation contradicted its own docstring
> and could *raise* priors on wrong answers (0.05 → 0.949). The contract
> tests in `tests/unit/test_bkt_logic.py` caught it; don't regress.

**`knowledge_tracing_engine`** — `calibrate_priors` (multiplier heuristic over
the first N responses), `update_student_priors` (BKT per skill + proportional
spill-over to similar skills via the skills map), `update_content_gaps`
(error-tag voting for weak skills), `predict_response`.

## Behavioral diagnosis

**Trigger:** first profile at 100 responses, re-profile every 10 after that.
The API's `/complete` and diagnostics pipelines bypass these gates on purpose.

Each metric computes a delta from the per-response feature frame, then tests
against pretrained population parameters — most via a one-sided 75% CI test
(`delta < mean + z·σ/√n`), ratios via plain thresholds:

| Metric | Delta | Diagnose when |
|---|---|---|
| language | error(Q1 language) − error(Q4) | below population CI |
| reasoning | error(Q1 reasoning) − error(Q4) | below CI |
| flexibility | stay-error − switch-error across units | below CI |
| attention_span | mean run-length of rising cumulative error | above CI bound |
| working_memory | regular − high-load (>Q75 cognitive_load) error | below CI |
| frustration | ±3-window error drop around isolated mistakes | below CI |
| processing_speed | error(time pressure) − error(no pressure) | below CI |
| time_management | UTM/STM ratio (under-time errors/successes) | ratio > 1.5 |
| stress | error-with-triggers ÷ error-without | ratio > 2.0 |
| impulsivity | `Distractor_Impulsive`-tagged errors ÷ all errors | rate > 0.33 |

Output: diagnosis list + all deltas, appended to the student's history.
Requires `stress_triggers` and `atag` on answers or those two metrics stay NaN.

## Treatment plans (`Treatment_service.py`)

Initial map (diagnosis/gap → query constraint, canonical field names):

| Diagnosis | Constraint |
|---|---|
| language | `language_level == 'Q1'` |
| working_memory | `cognitive_load <= 1` |
| processing_speed | `time_pressure_flag == False` |
| attention_span | `logical_steps <= 3` |
| flexibility | `multi_concept_flag == True` |
| stress | `time_pressure_flag == False` |
| time_management | `time_allowed <= 180, logical_steps <= 2` |
| Gap_Absence | `bloom_taxonomy_level ∈ [Remember, Understand]` |
| Gap_Concept | `visual_dependency == True` |

Updates scale numeric parameters by % improvement of the matching delta
between the last two windows, with one-way escalations at ≥10/20/40%:
`language_level` Q2→Q4, working-memory load +1 tier, time +60s/+2 steps per
tier. Content-gap constraints scale by per-skill prior improvement.

Unmapped diagnoses (`reasoning`, `frustration`, `impulsive`) produce no
constraints today — known roadmap item.

## Serving chain

1. `Question_Selection_Service.get_candidate_questions` unwraps the plan into
   condition dicts `{'Topic', 'Attribute', 'Operator', 'Threshold'}`
   (general entries keep Topic `'general'`; specific entries carry the skill
   cluster).
2. `Data_service.query()` ANDs them (topic = skill-cluster filter;
   operators `== != > >= < <= in`; aliases resolve; unknown attributes raise).
3. `ContextualBandit.select` scores `(1 − mean(priors_on_skills)) × difficulty
   × p_g`, samples with a confidence bonus over observed rewards, returns top-N
   (10). `update()` absorbs observed rewards per context bucket.
4. `Match_Service.set_match` applies ordering:
   flexibility → unit ascending; attention_span → difficulty descending;
   stress → easy/hard interleave. Applied in that sequence when combined.

## Question processing (`services/question_processing/`)

Pipeline: raw metadata frame → `derive_features` → `compute_probs` → merged
frame → entities (via ingestion or `server/reprocessing.py` write-back).

Derived fields:

| Field | How |
|---|---|
| vocabulary_richness | type–token ratio, EN/Arabic tokenizers |
| language_challenge | negation/exception/hedge word lists (or source flag) |
| num_unknowns | distinct single-letter variables in solution text |
| num_variables | science-entity count from question text via `scibert_ner` (when source lacks `Variables_Count`) |
| num_operations | max(distinct math symbols, vocab-disjoint step count) |
| question_length / sentences / clauses | word/punctuation counts (fallbacks if absent) |
| cognitive_load_index | z(var_count)+z(unknowns), min-max to 1–5 int |
| cognitive_load | same value as float; the WM treatment constraint target |
| language_level / reasoning_level | composite z-scores quartiled Q1–Q4 (**relative** to corpus — recomputed on every batch insert) |
| p_s / p_g / p_t | collective z-score over 8 features; p_s = scaled×0.1, p_g = inverted×0.3, p_t fixed 0.017 |

### SciBERT science-entity NER (`scibert_ner.py`)

Layers: (1) fine-tuned checkpoint via `SCIBERT_NER_MODEL` if set; (2)
unit-anchored quantities (`velocity 6 km/s`; unit family names the concept —
`300 m` is distance) plus a ~25-concept lexicon (physics/chem/bio surface
forms); (3) SciBERT contextual embeddings matching off-lexicon words to
concept prototypes. Falls back layer-by-layer; never blocks ingestion.
Tests run with `USE_SCIBERT_NER=0`.

## Logging (`log_setup.py`)

`sprintlab` logger hierarchy → `logs/sprintlab_log.txt`
(RotatingFileHandler 5 MB × 3). Format:
`timestamp | LEVEL | service | message`. Construction, mediator requests,
subscriptions, bandit runs, prior updates and treatment writes are all
instrumented. WARNING lines ("query matched nothing", "no candidate
questions") historically surfaced real contract bugs — take them seriously.
