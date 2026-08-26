# Testing

103 tests across 12 files. Run from repo root:

```bash
./.venv/Scripts/python.exe -m pytest tests/ -v          # full suite (~3s)
```

> The `.venv` is Windows-layout (`Scripts/`, not `bin/`). From WSL always
> invoke it as `./.venv/Scripts/python.exe`; activating a Linux venv will not
> work. From PowerShell/CMD it behaves like a normal Windows venv.

## Suite map

| File | # | Guards |
|---|---|---|
| `tests/unit/test_bkt_logic.py` | 24 | **The learning math contract**: correct raises priors, wrong lowers them, correct ≥ wrong, monotonic in prior, degenerate inputs finite, engine clamps to [0.01, 0.99] under 200-right/400-wrong hammering, alternating answers track direction step-by-step. *Caught: the original update formula raised low priors on wrong answers and lowered high ones.* |
| `tests/unit/test_schema_contract.py` | 12 | Treatment-plan attributes resolve against `question_schema` and exist on the entity; every treatment condition matches ≥1 question of a seeded corpus (parametrized over all diagnoses/gap types); legacy aliases resolve. *Caught: `lang_difficulty` / `max_cognitive_load` / `has_image` / `bloom_types` matching zero questions silently.* |
| `tests/unit/test_scibert_ner.py` | 15 | Science-entity layers: lexicon, unit-family anchoring, semantic mapping + threshold, fine-tuned pipeline path & its failure fallback, feature-derivation integration (source column wins). Real model never downloaded in CI. |
| `tests/test_integration_pipeline.py` | 6 | Full graph with real services: per-profile treatment+match (S101–S104), notify→match event delivery, response loop through calibration/priors history. History injected via the entity API, not method mocks. |
| `tests/api/test_endpoints.py` | 4 | HTTP lifecycle: batch insert + corpus reprocessing assertions (`language_level` quartiles present), questions query, diagnostics job to completion, match → answers → complete result shapes. *Caught: get_question scalar-vs-list contract, atags/atag column mismatch.* |
| `tests/unit/test_match_ordering.py` | 8 | Diagnosis-driven serving order: flexibility = unit ascending, attention_span = difficulty descending, stress = easy/hard weave, combined sequence, never drops/duplicates. |
| `tests/unit/test_treatment_service.py` | 7 | Plan construction (diagnoses→constraints, gaps), delta-improvement scaling, escalation tiers (language Q4, WM load 4, time +180s/+6 steps), no-history no-op. |
| `tests/unit/test_diagnosis_service.py` | 7 | Calibration fires exactly at window; priors-history written after; BD at 100 responses once; notify dispatch incl. payload remap; request routing; periodic re-check cadence. |
| `tests/unit/test_match_service.py` | 6 | Q_S.mediator wiring, set_match output shape & flexibility sorting, request routing, reward computation from priors windows. |
| `tests/unit/test_question_selection.py` | 6 | Plan→conditions unwrapping, query passthrough, empty-candidate safety, bandit ranking direction, reward absorption. |
| `tests/unit/test_repository.py` | 7 | InMemoryRepository CRUD/query behavior + **swap test** proving any `DataRepository` implementation slots under Data_Service. |
| `tests/test_cicd.py` | 1 | Placeholder smoke test. |

## conftest.py behaviors (apply to every test)

- **Singleton reset** — `SigletonMeta._instances` cleared around each test so
  `Data_Service` never leaks state between tests.
- **RNG seeding** — `random.seed(42)`, `np.random.seed(42)` for deterministic
  bandit sampling.
- **Log cap** — service logs forced to WARNING during tests.
- **NER disabled** — `USE_SCIBERT_NER=0` so no model download ever happens;
  NER-specific tests stub embeddings or re-enable explicitly.

## Testing philosophy that caught the real bugs

1. **Inject fakes through constructors**, don't patch methods on your own
   entities — mocking `student.get_deltas()` hid a notify-payload contract bug
   for weeks; seeding real history via `add_diagnosis_record` exposed it.
2. **Contract tests over integration luck**: the schema-contract suite runs
   every treatment condition against a reference corpus, so "attribute renamed
   but consumers not updated" fails loudly instead of serving zero questions.
3. **Drive the real engines at least once** — the API lifecycle test was the
   first thing to execute actual BKT/BD code paths, surfacing three latent
   bugs immediately.

## Adding tests

Put unit tests in `tests/unit/<area>.py`, HTTP tests in `tests/api/`. Use the
fixtures (`data_service`, `repo`, `client`) as templates. Anything touching
serving must keep the ordering and schema-contract suites green — they are the
system's invariant guards.
