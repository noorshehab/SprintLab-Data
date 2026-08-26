# SprintLab Handover

SprintLab is an adaptive-learning backend: students answer science questions,
a Bayesian Knowledge Tracing model updates per-skill mastery priors with every
answer, a behavioral diagnosis engine periodically profiles *how* a student
struggles (stress, working memory, attention, ...), and the resulting
treatment plan constrains and orders which questions get served next.

## Repo map

```
services/                  domain layer (no HTTP)
  Entities.py              student / question / skill objects
  Interfaces.py            Mediator, Component, Publisher/Subscriber, singleton meta
  Data_service.py          facade over a DataRepository (singleton)
  Diagnosis_service.py     mediator/command-bus + response pipeline gates
  Match_Service.py         bandit serving + diagnosis-driven ordering
  Treatment_service.py     treatment-plan construction & delta scaling
  question_schema.py       canonical question-field registry + alias resolution
  log_setup.py             rotating-file logging for everything
  storage/
    base.py                DataRepository ABC (the Phase-B seam)
    memory.py              InMemoryRepository (dict-backed reference impl)
    match_store.py         API-side MatchStore / JobStore
  knowledge_tracing/       BKT.py math + knowledge_tracing_engine.py
  behavioral_diagnosis/    10-metric profiling engine
  question_bandit/         MAB.py (UCB-style bandit), Question_Selection_service, context.py
  question_processing/     feature_derivation, prob_calcs, language_tools,
                           embedding_service, scibert_ner (science-entity NER)
  data_ingestion/          file loading + entity population from CSVs
server/                    FastAPI app (the 7-endpoint API + pipelines + stores)
demo/                      single-student interactive frontend demo
tests/                     103 tests (see docs/testing.md)
docs/                      you are here
scripts/generate.py        test student profiles used by tests & demo
```

## Quick start

```bash
# all commands assume repo root; venv is Windows-layout (Scripts/, not bin/)
./.venv/Scripts/python.exe -m pytest tests/ -v        # 103 tests
./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000   # API
./.venv/Scripts/python.exe -m uvicorn demo.app:app --port 8500      # demo UI
```

Open http://localhost:8000/docs for the API's OpenAPI page once running.

## Read in this order

1. [architecture.md](architecture.md) — how the pieces connect
2. [components.md](components.md) — what each module computes and guarantees
3. [api.md](api.md) — the HTTP surface
4. [testing.md](testing.md) — what the 103 tests guard and how to run them
5. [demo.md](demo.md) — the interactive demo
6. [operations.md](operations.md) — env vars, limits, Postgres/vector-DB roadmap
