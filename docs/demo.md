# Demo (`demo/`)

A single-page interactive frontend that makes the adaptive loop visible:
answer science questions, watch priors move after every answer, and watch
behavioral-diagnosis windows get written every few answers.

```bash
./.venv/Scripts/python.exe -m uvicorn demo.app:app --port 8500
# open http://localhost:8500
```

## What it shows

- **Question card** — MCQs from the candidate CSV, options shuffled; clicking
  grades against `Correct_Answer_Content`, feeds the answer through
  `Diagnosis_service.add_student_response`, and flashes the exact prior
  movements per skill.
- **Skill priors** — live bars (BKT updates).
- **Behavioral deltas** — the two seeded history windows from the profile,
  plus a *real* new BD window appended every 5 answers (checkpoint cadence),
  followed by a treatment-plan update.
- **Active serving constraints** — current treatment-plan conditions.

## Scaffolding

- Questions come from
  `E://projects//sprintlabfiles//sprintlab_candidate_science_questions.csv`
  (30 MCQ science questions; override with `DEMO_QUESTIONS_CSV`), ingested via
  `data_ingestion_service` then corpus-reprocessed (quartiles + science-entity
  counts).
- The demo student is **S101** from `scripts/generate.py`
  ("low prior, high impulsivity & language gap"). generate.py keys priors to
  hypothetical clusters (`KC-BIO-01`, ...); the CSV uses real ones (`SC-*`),
  so the profile is remapped 1:1:
  `KC-BIO-01 → SC-CELL-01`, `KC-PHYS-01 → SC-KINEMATICS-01`,
  `KC-CHEM-01 → SC-BOND-01`. Diagnoses and delta windows are verbatim.
- Calibration window is lowered to **3** so priors visibly move within a
  short session. Note the gate exists in two places — engine *and*
  `Diagnosis_service.calibration_window`; both are set by `demo/runtime.py`.
- Serving uses `Match_Service.set_match`; when treatment constraints leave
  fewer than 2 fresh candidates, it falls back to bandit ranking over the
  whole unanswered pool (the UI badge shows which path served each question).

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | the page |
| `GET /api/state` | student state snapshot |
| `GET /api/next` | next question (options shuffled) or `{exhausted:true}` |
| `POST /api/answer` | `{question_id, answer, response_time_seconds}` → correctness, prior changes, checkpoint if due |
| `GET /api/reset` | rebuild the runtime from scratch |

## Demo-specific knobs

- `DEMO_QUESTIONS_CSV` — alternate question file
- `USE_SCIBERT_NER=1` — enable full SciBERT semantic entity matching during
  ingestion reprocessing (default lexicon+units only so startup stays fast)
