# Operations

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `SPRINTLAB_LOG_LEVEL` | `INFO` | Log level for the `sprintlab` hierarchy (DEBUG shows mediator request traffic) |
| `SPRINTLAB_LOG_CONSOLE` | unset | Set to `1` to mirror logs to stderr |
| `EMBEDDING_FALLBACK` | `1` | `0` forces the real Qwen embedding model; default uses deterministic hash embeddings |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model id |
| `EMBEDDING_CACHE_PATH` | unset | Disk cache file for skill embeddings |
| `USE_SCIBERT_NER` | `1` | `0` = lexicon+unit-only science entity extraction (tests pin this) |
| `SCIBERT_MODEL` | `allenai/scibert_scivocab_uncased` | Base encoder for contextual matching |
| `SCIBERT_NER_MODEL` | unset | Fine-tuned token-classification checkpoint; authoritative when set |
| `SCIBERT_SIM_THRESHOLD` | `0.55` | Semantic-match cutoff for off-lexicon words |
| `DEMO_QUESTIONS_CSV` | the sprintlabfiles CSV | Question source for the demo |

Log output lives at `logs/sprintlab_log.txt` (rotating, 5 MB × 3 backups).

## Running

```bash
# tests
./.venv/Scripts/python.exe -m pytest tests/ -v
# API
./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000
# demo UI
./.venv/Scripts/python.exe -m uvicorn demo.app:app --port 8500
```

**Windows venv from WSL**: always `./.venv/Scripts/python.exe`; there is no
`bin/`. From Windows shells it's a normal venv.

## Hard limits today

- **Single process.** Data_Service, MatchStore and JobStore are in-memory.
  Running more than one uvicorn worker forks divergent state. This is *the*
  blocker for production deployment.
- **No auth** on any route — do not expose publicly as-is.
- **GET /questions takes a JSON body** per the original API spec; unusual but
  intentional.

## Phase B: PostgreSQL (the seam is ready)

Implement `SqlAlchemyRepository(DataRepository)` against
`services/storage/base.py`:

1. Rows ↔ entities convert **inside** the repository (entities stay the
   interface currency).
2. Nested student state → child tables or JSONB:
   responses, priors/deltas/diagnoses histories, treatment plan.
3. `query()` translates condition dicts into WHERE clauses;
   `services/question_schema.py::resolve_attribute` gives canonical column
   names (aliases keep old callers working).
4. `update_question_attributes(q_id, fields)` becomes a targeted UPDATE.
5. Alembic (already installed) for migrations; hand the repository to
   `Data_Service(repository=...)`.
6. Replace MatchStore/JobStore with DB-backed equivalents, then multi-worker
   uvicorn is safe.

The swap test (`test_repository.py`) is the acceptance gate: a repository that
passes it requires zero changes above the seam.

## Roadmap / known gaps

- **Vector DB for embeddings**: skill embeddings currently re-embed per batch
  (disk-cached). `embedding_service.get_cached/embed/save_cache` is exactly
  the seam a vector store replaces.
- **Unmapped diagnoses**: `reasoning`, `frustration`, `impulsive` have BD
  metrics but no entries in `initial_treatment_map` — students diagnosed only
  with those get empty plans (they still affect nothing downstream). Same
  pattern previously hit stress/flexibility/attention_span.
- **Relative quartiles**: language/reasoning levels shift meaning on every
  corpus insert; reprocessing handles it, but stored historical snapshots
  won't be comparable across large corpus changes.
- **Semantic NER is heuristic** without a fine-tuned checkpoint; fine-tune on
  labeled questions and point `SCIBERT_NER_MODEL` at it when label data exists.
- **BD small-sample behavior**: most behavioral metrics are NaN below ~dozens
  of responses; the API bypasses gates deliberately, so early `/complete`
  calls legitimately return few/zero diagnosed scores.
- `reprocessing.py` could move into `question_processing_service` as a method
  if the server-local placement bothers anyone (naming is historical).
