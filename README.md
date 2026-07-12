# Noxed / SprintLab — Empirical Study

Noxed is an adaptive, competitive science-quiz game for Egyptian schools, built as an
ensemble of four ML modules: **Diagnostic** (clustering → knowledge priors),
**Behavioural Diagnosis** (response metadata → treatment plan), **Question Bandit**
(LinUCB → next question), and **Progress Tracking** (BKT → mastery). This repository
holds the project's empirical study: the research paper and the 9 executed notebooks
that back its results.

No Egyptian-science behavioural dataset exists yet — the study runs on the **XES3G5M**
dataset (Chinese elementary-math knowledge tracing, ~5.1M interactions) as a
**methodology proxy**, and is explicit throughout about what does and doesn't transfer.
See `paper/noxed_research.pdf` §Threats to Validity.

## Contents

- **`paper/`** — the research paper (`noxed_research.pdf`), extended from the project's
  research-questions note with a Results/Empirical-Outcomes section citing every
  notebook's findings.
- **`notebooks/`** — 9 executed notebooks (00–08), each with real per-cell outputs.
  `notebooks/rendered/` has a standalone HTML render of each, viewable without Jupyter.
- **`outcomes/`** — curated, small model-ready artifacts (calibrated item difficulty,
  KC learning rates, cluster→prior maps, behavioural trait-separability scores, bandit
  reward simulations) plus their figures and an MLflow run summary.
- **`src/noxed/`** — the reusable analysis library every notebook imports (difficulty
  calibration, IRT, a from-scratch BKT implementation, clustering, prerequisite
  discovery, the behavioural synthetic-scenario generator, bandit reward simulation).
- **`tests/`** — Hypothesis property tests encoding the project's core invariants
  (every difficulty/score/mastery value bounded in its valid range, clustering is a
  true partition, treatment plans stay inside the fixed vocabulary).
- **`modal/`** — the Modal.com compute engine: uploads the proxy dataset to a Volume
  and executes all 9 notebooks headlessly via papermill, so every cell's output in
  `notebooks/*.ipynb` is real, reproducible execution evidence, not hand-edited text.

## What each notebook answers

| # | Notebook | Question |
|---|---|---|
| 00 | `data_contract` | What behavioural signals does this proxy actually contain? |
| 01 | `difficulty_irt` | Calibrated item difficulty: empirical, static, IRT, blended |
| 02 | `item_discrimination` | Do the pipeline's 5 differentiation metrics agree, and validate against IRT `a` |
| 03 | `learning_dynamics` | Does practice reduce error — tested (Kruskal-Wallis) + raw-log PFA |
| 04 | `knowledge_tracing_students` | Per-student BKT mastery trajectories + latent learner types |
| 05 | `kc_prerequisites` | Does the authored KC tree encode real prerequisite gating? |
| 06 | `diagnostic_clustering` | Which features make KC clusters meaningful; cluster→prior mapping |
| 07 | `behavioural_synthetic_lab` | Synthetic-scenario feasibility test for the 12 behavioural cases |
| 08 | `bandit_reward_sim` | Does a desirable-difficulty reward keep the bandit in the ZPD? |

## Reproduce

**Compute:** [Modal](https://modal.com). **Tracking:** MLflow, either a remote server
(`MLFLOW_TRACKING_URI`) or a local `./mlruns` file store (the default fallback).

```bash
pip install -r requirements/modal.txt

# one-time: upload the proxy dataset to a Modal Volume
python modal/upload_data.py /path/to/sprintlabfiles

# execute all 9 notebooks headlessly on Modal (per-cell outputs embedded on return)
modal run modal/app.py::run_all

# or run one notebook
modal run modal/app.py --name 01_difficulty_irt
```

**Local run** (no Modal): set `DATA_DIR` to a local copy of the proxy dataset and
execute with [papermill](https://papermill.readthedocs.io/):

```bash
pip install -r requirements/base.txt -r requirements/ml.txt -r requirements/notebooks.txt
export DATA_DIR=/path/to/sprintlabfiles OUTCOMES_DIR=./outcomes CACHE_DIR=./cache
papermill notebooks/00_data_contract.ipynb notebooks/00_data_contract.ipynb --kernel python3
```

**Tests:**

```bash
pip install -r requirements/base.txt
pytest tests/ -q
```

## Conventions (carried over from the original scaffold)

- Tests live in `tests/`; requirements are split per concern in `requirements/` so CI
  installs only what a given check needs.
- To refresh a requirements file from actual imports: `pipreqs <folder>/ --savepath requirements/<name>.txt`.
- GitHub Actions (`.github/workflows/tests.yml`) runs the property-test suite on every
  push/PR.
