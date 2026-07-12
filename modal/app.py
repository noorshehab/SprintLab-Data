"""Modal compute engine for the Noxed empirical study.

Every notebook is executed headlessly with papermill inside this image so
each code cell's real output is embedded in the resulting .ipynb -- the
notebook file itself becomes the evidence artifact, not just its source.

Usage:
    modal run modal/app.py::run_all
    modal run modal/app.py::run_notebook --name 01_difficulty_irt
"""
from pathlib import Path

import modal

DATA_VOLUME = "noxed-data"
ARTIFACTS_VOLUME = "noxed-artifacts"

NOTEBOOK_ORDER = [
    "00_data_contract",
    "01_difficulty_irt",
    "02_item_discrimination",
    "03_learning_dynamics",
    "05_kc_prerequisites",
    "04_knowledge_tracing_students",
    "06_diagnostic_clustering",
    "07_behavioural_synthetic_lab",
    "08_bandit_reward_sim",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements" / "base.txt"))
    .pip_install_from_requirements(str(REPO_ROOT / "requirements" / "ml.txt"))
    .pip_install_from_requirements(str(REPO_ROOT / "requirements" / "notebooks.txt"))
    .pip_install("mlflow==2.18.0")
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
    .add_local_dir(str(REPO_ROOT / "notebooks"), remote_path="/root/notebooks_src")
)

app = modal.App("noxed-study", image=image)
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME, create_if_missing=True)


@app.function(
    volumes={"/data": data_volume, "/artifacts": artifacts_volume},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_notebook(name: str) -> dict:
    import os
    import subprocess
    import sys

    import papermill as pm

    os.environ["DATA_DIR"] = "/data"
    os.environ["OUTCOMES_DIR"] = "/artifacts/outcomes"
    os.environ["CACHE_DIR"] = "/artifacts/cache"  # must live on the persisted Volume, not the
                                                   # ephemeral container fs -- each notebook runs
                                                   # in its own container, so an unset CACHE_DIR
                                                   # would silently defeat cross-notebook reuse of
                                                   # the ~2min canonical-events collapse.
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:/artifacts/mlruns")
    sys.path.insert(0, "/root/src")

    Path("/artifacts/outcomes").mkdir(parents=True, exist_ok=True)
    Path("/artifacts/executed").mkdir(parents=True, exist_ok=True)
    Path("/artifacts/cache").mkdir(parents=True, exist_ok=True)

    src_nb = f"/root/notebooks_src/{name}.ipynb"
    out_nb = f"/artifacts/executed/{name}.ipynb"

    pm.execute_notebook(src_nb, out_nb, kernel_name="python3", progress_bar=False)

    artifacts_volume.commit()
    return {"notebook": name, "status": "completed", "output_path": out_nb}


@app.function(volumes={"/artifacts": artifacts_volume}, timeout=600)
def export_mlflow_summary() -> str:
    import os

    import mlflow
    import pandas as pd

    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:/artifacts/mlruns")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.tracking.MlflowClient()
    rows = []
    for exp in client.search_experiments():
        for run in client.search_runs(exp.experiment_id):
            rows.append(
                {
                    "experiment": exp.name,
                    "run_name": run.info.run_name,
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    **{f"param.{k}": v for k, v in run.data.params.items()},
                    **{f"metric.{k}": v for k, v in run.data.metrics.items()},
                }
            )
    out_path = "/artifacts/outcomes/mlflow_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    artifacts_volume.commit()
    return out_path


@app.local_entrypoint()
def run_all():
    """Execute all 8 notebooks in dependency order, then export the MLflow
    run summary. Notebooks with no cross-notebook dependency (02, 03, 05)
    could run concurrently, but we run sequentially for a predictable,
    debuggable log stream given each notebook is itself compute-heavy."""
    results = []
    for name in NOTEBOOK_ORDER:
        print(f"=== running {name} ===")
        result = run_notebook.remote(name)
        print(result)
        results.append(result)
    summary_path = export_mlflow_summary.remote()
    print(f"MLflow summary written to {summary_path}")
    return results


@app.local_entrypoint()
def main(name: str = ""):
    """`modal run modal/app.py --name 01_difficulty_irt` runs one notebook."""
    if not name:
        raise SystemExit("pass --name <notebook_stem>, or run modal/app.py::run_all")
    print(run_notebook.remote(name))
