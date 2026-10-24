import os
from pathlib import Path

from hotel_cancellation.entity.config_entity import MLflowConfig


def setup_mlflow(config: MLflowConfig) -> None:
    """Point MLflow at the project's tracking store and make sure the experiment exists."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    if config.tracking_uri.startswith("sqlite:///"):
        Path(config.tracking_uri.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(config.tracking_uri)
    if mlflow.get_experiment_by_name(config.experiment_name) is None:
        artifact_root = Path(config.artifact_root).resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(config.experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(config.experiment_name)
