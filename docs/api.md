# HTTP API (`server/`)

FastAPI app: `server/main.py`. Run single-worker (in-memory state):

```bash
./.venv/Scripts/python.exe -m uvicorn server.main:app --port 8000
```

Lifespan wires the full graph: `Data_Service` → KT/BD engines →
`Diagnosis_service` → Treatment_Service; `Match_Service` +
`Question_Selection_Service`; MatchStore and JobStore.

## Endpoints

| Route | Body | Returns | Internals |
|---|---|---|---|
| `POST /diagnostics/run` | `{studentId, Answers:[{questionId, response, responseTimeSeconds, stressTrigger?, answerTag?}]}` | **202** `{jobId, status:"processing"}` | Background task absorbs answers via the mediator, runs KT calibrate+update, content gaps, BD diagnosis, sets/updates treatment plan. Bypasses the 10/100-response gates by design |
| `GET /diagnostics/{jobId}` | — | `{status: processing\|completed\|failed, result: {priorsWritten, treatmentPlanWritten}}` | JobStore lookup; 404 for unknown ids |
| `POST /matches` | `{studentId}` | **201** `{matchId, questions:[{question:{...}, difficulty, unit}]}` | Creates a match record (`status:"in_progress"`), calls `Match_Service.set_match` |
| `POST /matches/{id}/answers` | `{questionId, response, responseTimeSeconds, stressTrigger?, answerTag?}` | 200 `{accepted:true}` | Appends to the stored match; 404 unknown match |
| `POST /matches/{id}/complete` | — | 200 `{deltaMastery, updatedPriors, behavioralScores, updatedTreatmentPlan}` | Progress tracking: feeds accumulated answers through KT update + content gaps + BD + treatment update. See below |
| `GET /questions` | JSON body `{allQs:"True"\|"False", ids:[...]}` | 200 `{Questions:[...]}` | Whitelisted DTOs; 400 when neither filter given |
| `POST /questions/batch` | `{Questions:[{q_id, skill_cluster_id, ...}]}` | **201** `{NumInserted}` | Inserts (duplicates skipped), then **reprocesses the corpus** and rebuilds skills |
| `POST /students` / `GET /students/{id}` | `{studentId}` / — | created / profile+priors+plan | Thin Data_Service calls |
| `GET /health` | — | `{status, runtimeReady}` | |

## `/complete` result shapes

- `deltaMastery`: `{skill_cluster_id: new_prior − old_prior}`
- `updatedPriors`: `{skill_cluster_id: prior}`
- `behavioralScores`: one entry per BD metric —
  `{metric, value (null when NaN), diagnosed}`; metric→diagnosis-key mapping
  lives in `server/pipelines.py::_DIAG_KEYS`
- `updatedTreatmentPlan`: constraint list
  `[{Topic, Attribute, Operator, Threshold}]` from
  `_unwrap_treatment_plan`

## Batch reprocessing contract

Quartiles and BKT parameters are z-score-relative to the corpus, so after
inserting new questions `server/reprocessing.py::reprocess_corpus`:
1. rebuilds the metadata frame from **all** stored questions,
2. re-runs `derive_features` + `compute_probs`,
3. writes derived fields back through
   `Data_service.update_question_attributes`,
4. re-embeds skills (disk cache makes incremental batches cheap) and
   re-registers the similar-skills map.

Synchronous inside the request; graduate it to the 202-job pattern if corpora
grow.

## Stores & limits

- `services/storage/match_store.py`: UUID-keyed in-process `MatchStore` /
  `JobStore` behind the same seam philosophy as the data layer.
- **Single uvicorn worker only** until Phase B (Postgres) replaces the
  in-memory stores.
- `answerTag` matters: without it the impulsivity metric can't fire;
  `stressTrigger` likewise drives stress.
