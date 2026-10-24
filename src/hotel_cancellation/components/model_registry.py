"""Model registry backed by the MLflow Model Registry, plus export of a serving bundle.

MLflow is the source of truth: every accepted model becomes a new version of the registered
model, and the live one carries the `production` alias. The API does not talk to MLflow.
Instead, the production version is exported to a small folder (the serving bundle) that the
API and the Docker image load:

    artifacts/serving/
        model.joblib            preprocessing + XGBoost in one sklearn Pipeline
        metadata.json           version, threshold, costs, metrics, params, feature names
        reference_profile.json  training-data distribution for drift checks
        versions.json           every registered version and which one is in production

Rolling back: point the alias at an older version (MLflow UI, or `hc-registry promote 3`),
then export (`hc-registry export`) and reload the API.
"""

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from hotel_cancellation.entity.artifact_entity import (
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelRegistryArtifact,
    ModelTrainerArtifact,
)
from hotel_cancellation.entity.config_entity import ModelRegistryConfig
from hotel_cancellation.exception import ModelError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.pipeline.prediction_pipeline import (
    METADATA_FILE,
    MODEL_FILE,
    PROFILE_FILE,
    VERSIONS_FILE,
)
from hotel_cancellation.utils.common import load_object, read_json, save_object, write_json

logger = get_logger(__name__)

TAGGED_METRICS = ("roc_auc", "pr_auc", "f1", "cost_per_booking")
# MLflow stores sklearn models with skops, which refuses to load any type not listed here
# (safer than pickle, which can run arbitrary code when a model file is loaded).
SKOPS_TRUSTED_TYPES = [
    "hotel_cancellation.components.data_transformation.FeatureEngineer",
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def _requirements() -> list[str]:
    """Pinned runtime packages, given explicitly so MLflow does not have to infer them."""
    from importlib.metadata import version

    packages = ["scikit-learn", "xgboost-cpu", "pandas", "numpy"]
    return [f"{name}=={version(name)}" for name in packages]


class ModelRegistry:
    def __init__(self, config: ModelRegistryConfig) -> None:
        self.name = config.registered_model_name
        self.alias = config.production_alias
        self.serving_dir = Path(config.serving_dir)

    @property
    def client(self):
        from mlflow import MlflowClient

        return MlflowClient()

    def production_version(self):
        """The MLflow ModelVersion carrying the production alias, or None."""
        from mlflow.exceptions import MlflowException

        try:
            return self.client.get_model_version_by_alias(self.name, self.alias)
        except MlflowException:
            return None

    def load_production(self) -> tuple[Pipeline, dict[str, Any]] | None:
        import mlflow

        version = self.production_version()
        if version is None:
            return None
        uri = f"models:/{self.name}/{version.version}"
        model = mlflow.sklearn.load_model(uri)
        metadata = dict(mlflow.models.get_model_info(uri).metadata or {})
        metadata["version"] = str(version.version)
        return model, metadata

    def register(
        self,
        trainer: ModelTrainerArtifact,
        evaluation: ModelEvaluationArtifact,
        validation: DataValidationArtifact,
        input_example: pd.DataFrame,
    ) -> ModelRegistryArtifact:
        """Register the candidate (inside the active MLflow run) and make it production."""
        if not evaluation.is_accepted:
            logger.info("Candidate not accepted, production model unchanged")
            return ModelRegistryArtifact(promoted=False, version=None, serving_dir=None)

        import mlflow
        from mlflow.models import infer_signature

        pipeline = load_object(trainer.model_path)
        metadata = read_json(trainer.metadata_path)
        metadata["test_metrics"] = evaluation.metrics
        metadata["promotion_reason"] = evaluation.reason

        mlflow.log_artifact(str(validation.reference_profile_path))
        # Integer columns can hold missing values at prediction time, so declare them as float.
        int_columns = input_example.select_dtypes("integer").columns
        input_example = input_example.astype({column: "float64" for column in int_columns})
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=self.name,
            signature=infer_signature(input_example, pipeline.predict(input_example)),
            metadata=metadata,
            pip_requirements=_requirements(),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = str(info.registered_model_version)

        client = self.client
        client.set_registered_model_alias(self.name, self.alias, version)
        for name in TAGGED_METRICS:
            client.set_model_version_tag(
                self.name, version, f"test_{name}", f"{evaluation.metrics[name]:.4f}"
            )
        client.set_model_version_tag(self.name, version, "promotion_reason", evaluation.reason)
        logger.info("Registered %s version %s as @%s", self.name, version, self.alias)

        self.export_production()
        return ModelRegistryArtifact(promoted=True, version=version, serving_dir=self.serving_dir)

    def set_production(self, version: str) -> None:
        self.client.get_model_version(self.name, version)  # raises if it does not exist
        self.client.set_registered_model_alias(self.name, self.alias, version)
        logger.info("Moved @%s to version %s", self.alias, version)

    def versions(self) -> list[dict[str, Any]]:
        production = self.production_version()
        live = str(production.version) if production else None
        rows = []
        for mv in self.client.search_model_versions(f"name='{self.name}'"):
            rows.append(
                {
                    "version": str(mv.version),
                    "registered_at": datetime.fromtimestamp(
                        mv.creation_timestamp / 1000, UTC
                    ).isoformat(timespec="seconds"),
                    "run_id": mv.run_id,
                    "test_metrics": {
                        name: float(mv.tags[f"test_{name}"])
                        for name in TAGGED_METRICS
                        if f"test_{name}" in mv.tags
                    },
                    "is_production": str(mv.version) == live,
                }
            )
        return sorted(rows, key=lambda row: int(row["version"]))

    def export_production(self) -> Path:
        """Write the production version to the serving bundle folder (replacing it whole)."""
        import mlflow

        loaded = self.load_production()
        if loaded is None:
            raise ModelError(f"No version of '{self.name}' has the @{self.alias} alias")
        model, metadata = loaded
        run_id = self.production_version().run_id

        staging = self.serving_dir.with_name(self.serving_dir.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        save_object(staging / MODEL_FILE, model)
        write_json(staging / METADATA_FILE, metadata)
        write_json(
            staging / PROFILE_FILE, mlflow.artifacts.load_dict(f"runs:/{run_id}/{PROFILE_FILE}")
        )
        write_json(
            staging / VERSIONS_FILE,
            {"production": metadata["version"], "versions": self.versions()},
        )
        shutil.rmtree(self.serving_dir, ignore_errors=True)
        staging.rename(self.serving_dir)
        logger.info("Exported version %s to %s", metadata["version"], self.serving_dir)
        return self.serving_dir


def main(argv: list[str] | None = None) -> int:
    """CLI: list versions, move the production alias, or re-export the serving bundle."""
    from hotel_cancellation.config.configuration import ConfigurationManager
    from hotel_cancellation.utils.tracking import setup_mlflow

    parser = argparse.ArgumentParser(description="Manage the MLflow model registry.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show registered versions")
    promote = commands.add_parser("promote", help="Point @production at a version, then export")
    promote.add_argument("version")
    commands.add_parser("export", help="Export the production version to the serving bundle")
    args = parser.parse_args(argv)

    config = ConfigurationManager()
    setup_mlflow(config.get_mlflow_config())
    registry = ModelRegistry(config.get_model_registry_config())

    if args.command == "list":
        rows = registry.versions()
        if not rows:
            print("No registered versions yet. Run `uv run hc-train`.")
        for row in rows:
            flag = f"@{registry.alias}" if row["is_production"] else ""
            auc = row["test_metrics"].get("roc_auc", float("nan"))
            print(f"v{row['version']:<4} {row['registered_at']}  ROC-AUC {auc:.4f}  {flag}")
        return 0
    try:
        if args.command == "promote":
            registry.set_production(args.version)
        path = registry.export_production()
    except Exception as error:
        print(f"Failed: {error}", file=sys.stderr)
        return 1
    print(f"Serving bundle written to {path}. Reload the API (POST /model/reload).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
