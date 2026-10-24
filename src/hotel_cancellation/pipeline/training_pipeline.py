import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hotel_cancellation.components.data_ingestion import DataIngestion
from hotel_cancellation.components.data_transformation import DataTransformation
from hotel_cancellation.components.data_validation import DataValidation
from hotel_cancellation.components.model_evaluation import ModelEvaluation
from hotel_cancellation.components.model_registry import ModelRegistry
from hotel_cancellation.components.model_trainer import ModelTrainer
from hotel_cancellation.config.configuration import ConfigurationManager
from hotel_cancellation.entity.config_entity import MLflowConfig
from hotel_cancellation.exception import HotelCancellationError
from hotel_cancellation.logger import get_logger

logger = get_logger(__name__)


class _NoTracker:
    """Stand-in used when MLflow tracking is switched off."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@contextmanager
def experiment_tracker(config: MLflowConfig):
    """Yield the mlflow module inside an active run, or a no-op object if tracking is off."""
    if not config.enabled:
        yield _NoTracker()
        return
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow is not installed (uv sync --group train); tracking is off")
        yield _NoTracker()
        return

    if config.tracking_uri.startswith("sqlite:///"):
        Path(config.tracking_uri.removeprefix("sqlite:///")).parent.mkdir(
            parents=True, exist_ok=True
        )
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run():
        yield mlflow


class TrainingPipeline:
    """Run every stage: ingest -> validate -> transform -> train -> evaluate -> register."""

    def __init__(self, config: ConfigurationManager | None = None) -> None:
        self.config = config or ConfigurationManager()

    def run(self) -> dict[str, Any]:
        config = self.config
        registry = ModelRegistry(config.get_model_registry_config())

        with experiment_tracker(config.get_mlflow_config()) as tracker:
            logger.info("===== Stage 1/6: data ingestion =====")
            ingestion = DataIngestion(config.get_data_ingestion_config()).run()

            logger.info("===== Stage 2/6: data validation =====")
            validation = DataValidation(config.get_data_validation_config()).run(ingestion)

            logger.info("===== Stage 3/6: data transformation =====")
            transformation = DataTransformation(config.get_data_transformation_config()).run(
                ingestion
            )

            logger.info("===== Stage 4/6: model training =====")
            trainer_config = config.get_model_trainer_config()
            trainer = ModelTrainer(trainer_config).run(ingestion, transformation)

            logger.info("===== Stage 5/6: model evaluation =====")
            evaluation = ModelEvaluation(config.get_model_evaluation_config(), registry).run(
                ingestion, trainer
            )

            logger.info("===== Stage 6/6: model registry =====")
            registered = registry.run(trainer, evaluation, validation)

            tracker.log_params(trainer_config.model_params)
            tracker.log_params(
                {
                    "validation_size": trainer_config.validation_size,
                    "early_stopping_rounds": trainer_config.early_stopping_rounds,
                    "n_features": len(transformation.feature_names),
                }
            )
            tracker.log_metrics({f"val_{k}": v for k, v in trainer.validation_metrics.items()})
            tracker.log_metrics({f"test_{k}": v for k, v in evaluation.metrics.items()})
            tracker.set_tags(
                {
                    "accepted": evaluation.is_accepted,
                    "registered_version": registered.version or "none",
                    "train_test_drift": validation.dataset_drift,
                }
            )
            for path in (validation.report_path, trainer.metadata_path, evaluation.metrics_path):
                tracker.log_artifact(str(path))

        summary = {
            "test_metrics": evaluation.metrics,
            "production_metrics": evaluation.production_metrics,
            "accepted": evaluation.is_accepted,
            "reason": evaluation.reason,
            "registered_version": registered.version,
        }
        logger.info("Training pipeline finished: %s", evaluation.reason)
        return summary


def main() -> int:
    try:
        summary = TrainingPipeline().run()
    except HotelCancellationError as error:
        logger.error("Training pipeline failed: %s", error)
        return 1

    metrics = summary["test_metrics"]
    print("\nTest set metrics (candidate):")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")
    if summary["registered_version"]:
        print(f"\nPromoted to production as {summary['registered_version']}")
    else:
        print(f"\nNot promoted: {summary['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
