# Architecture

## The shape of the system

Everything is in-process Python objects coordinated by a **Mediator**.
`Diagnosis_service` is the hub: every other service talks to data and to each
other through `request()` dicts (a command bus), and events flow back out
through `notify()`.

```
                       ┌─────────────────────────────┐
   add_student_response│      Diagnosis_service      │ notify('priors_updated')
  ────────────────────►│  (Mediator + Publisher)     │──────────────────┐
                       │  request(): command switch  │                  │
                       └───┬────────┬────────┬───────┘                  │
                           │        │        │                          ▼
             ┌─────────────▼──┐ ┌───▼────┐ ┌─▼──────────────┐  ┌──────────────┐
             │ knowledge_     │ │ behav. │ │ Treatment_     │  │ Match_Service│
             │ tracing_engine │ │ diag.  │ │ Service        │  │ (Subscriber) │
             └───────┬────────┘ │ engine │ └─┬──────────────┘  └───┬──────────┘
                     │          └────────┘   │ treatment plan      │ set_match()
                     ▼                       ▼                     ▼
               ┌─────────────────────────────────────┐  ┌──────────────────────┐
               │ Data_Service (facade, singleton)    │  │ Question_Selection_  │
               │  └─ repository: DataRepository ABC  │  │ Service + Contextual │
               │      (InMemoryRepository today;     │  │ Bandit               │
               │       SqlAlchemyRepository later)   │  └──────────────────────┘
               └─────────────────────────────────────┘
```

### Wiring rules (who sets whose mediator)

- `Diagnosis_service.__init__` sets `.mediator = self` on the KT engine, BD
  engine and Treatment_Service. Construct those bare, then hand them in.
- `Match_Service.__init__` sets `Q_S.mediator = self` so the question
  selection service can send `'query'` requests (only Match_Service handles
  that type).
- Never stomp `.mediator` manually from outside constructors; tests that need
  stubs pass them *into* constructors.

### The full loop

1. **Answer arrives** → `Diagnosis_service.add_student_response(student_id,
   q_ids, responses, timings, stress_triggers)`.
2. Responses append to the student entity. At exactly `calibration_window`
   responses (10 in production) the KT engine calibrates priors; after that
   every response runs the BKT update (`BKT.update_prior`) plus proportional
   updates to similar skills.
3. Priors history is appended every post-calibration response.
4. At **100 responses** the BD engine produces the first behavioral profile;
   Treatment_Service writes the initial plan. Re-diagnosis runs **every 10**
   responses afterwards, updating the plan by delta improvement.
5. `notify('priors_updated')` fans out to subscribers (Match_Service), which
   recomputes bandit rewards from the last two priors-history windows.
6. Next `set_match(student_id)`: treatment constraints filter candidates via
   `Data_service.query()`, the bandit ranks what survives, and active
   behavioral diagnoses order the result (flexibility → unit grouping;
   attention_span → difficulty descending; stress → easy/hard weave).

## Mediator command bus

Request types handled by `Diagnosis_service.request()`:
`get_student`, `get_question`, `get_skill`, `add_diagnosis`, `add_response`,
`update_prior`, `update_content_gaps`, `add_priors_history`,
`update_treatment_plan`, `add_student`, `add_question`, `add_skill`.

`Match_Service.request()` handles: `get_student`, `query`, `get_question`.

Contract notes:
- `get_question` with a scalar id returns a **single entity** (engines expect
  this); with a list it returns a list of entities/None slots.
- Unknown request types are logged as WARNING and dropped — check
  `logs/sprintlab_log.txt` when something silently does nothing.
- Events: only `'priors_updated'` exists; payload keys are remapped
  (`q_ids` → `questions`) for the subscriber's convenience.

## Storage seam

`services/storage/base.py` defines `DataRepository` — 15 abstract methods
covering everything services do to data (students, questions, skills, student
state writes, and `query()`). `Data_Service` is a thin facade over an injected
repository; default is `InMemoryRepository` (the original dict logic).

Why this matters: the future PostgreSQL integration implements the same ABC
(`SqlAlchemyRepository`) and **nothing above the seam changes**. The swap test
(`tests/unit/test_repository.py::test_swap_test_data_service_accepts_any_repository`)
proves any implementation slots under the facade.

Rules that keep the seam honest:
- All writes go through the repository — engines must never mutate loaded
  entities directly without persisting (this was a real bug class; the KT
  engine used to double-write priors).
- Repositories deal in **entities**, not row dicts; conversion belongs inside
  each implementation.
- `query()` lives on the repository because its SQL translation is
  implementation-specific; conditions validate against
  `services/question_schema.py` and raise on unknown attributes.

## Singletons & process model

- `SigletonMeta` makes `Data_Service` a per-process singleton;
  `tests/conftest.py` clears `_instances` around every test.
- The API and demo are therefore **single uvicorn worker** until the storage
  seam gets a real database. Multi-worker would fork divergent universes.
