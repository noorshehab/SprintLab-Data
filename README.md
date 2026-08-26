# SprintLabData

Adaptive-learning backend: students answer science questions, Bayesian
Knowledge Tracing updates per-skill mastery priors with each answer, a
behavioral diagnosis engine profiles *how* a student struggles, and the
resulting treatment plan constrains and orders what gets served next via a
contextual bandit.

## Quick start

```bash
# run all 103 tests (venv is Windows-layout: use Scripts/, not bin/)
./.venv/Scripts/python.exe -m pytest tests/ -v

# HTTP API (OpenAPI docs at /docs)
./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000

# interactive single-student demo
./.venv/Scripts/python.exe -m uvicorn demo.app:app --port 8500
```

## Documentation

Full handover documentation lives in [`docs/`](docs/HANDOVER.md):

| Doc | Contents |
|---|---|
| [HANDOVER](docs/HANDOVER.md) | Start here — overview, repo map, quick start |
| [architecture](docs/architecture.md) | Mediator pattern, service graph, storage seam |
| [components](docs/components.md) | Per-module reference: BKT, behavioral metrics, treatment plans, serving, question processing |
| [api](docs/api.md) | HTTP endpoints and contracts |
| [demo](docs/demo.md) | The interactive demo |
| [testing](docs/testing.md) | What the 103 tests guard and how to run them |
| [operations](docs/operations.md) | Env vars, limits, Postgres/vector-DB roadmap |

## Structure

- `services/` — domain layer (no HTTP): data + repository seam, knowledge
  tracing, behavioral diagnosis, treatment plans, bandit serving, question
  processing
- `server/` — FastAPI app exposing the services
- `demo/` — minimal interactive frontend
- `tests/` — unit / integration / API suites (see docs/testing.md)
- `requirements/` — per-concern requirement files

## Conventions (kept from the original README)

Tests live in the test folder; requirements are split per concern in
`requirements/*.txt` — regenerate with
`pipreqs [foldername]/ --savepath requirements/[foldername].txt`.
Notebooks/Experiments are for EDA and ML experiments; ML holds final model
training. CI workflows run relevant test scripts on path-filtered pushes:

```yml
name: Test [Name of Module/Script]
on:
  push:
    paths: ['[Module subfolder]/**']
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements/[Name of Module/Script].txt
      - run: pytest tests/test_[Name of Module/Script].py
```
