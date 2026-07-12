"""MLflow tracking helper with a local-file fallback so every notebook runs
identically on Modal (remote Postgres backend) or on a laptop (no server)."""
import os
from contextlib import contextmanager

import mlflow


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


@contextmanager
def start_run(notebook: str, module: str, params: dict | None = None):
    """Set the tracking URI + experiment, start a run named after the notebook,
    log params up front, and yield the active run for metric/artifact logging."""
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(f"noxed-{module}")
    with mlflow.start_run(run_name=notebook) as run:
        if params:
            mlflow.log_params(params)
        yield run
