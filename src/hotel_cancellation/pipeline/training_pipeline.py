import argparse
from typing import Any

import pandas as pd

from hotel_cancellation.components.data_ingestion import DataIngestion
from hotel_cancellation.components.data_transformation import DataTransformation, feature_names
from hotel_cancellation.components.data_validation import DataValidation
from hotel_cancellation.components.model_evaluation import ModelEvaluation
from hotel_cancellation.components.model_registry import ModelRegistry
from hotel_cancellation.components.model_trainer import ModelTrainer
from hotel_cancellation.config.configuration import ConfigurationManager
from hotel_cancellation.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from hotel_cancellation.exception import HotelCancellationError, ModelError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import load_object, read_json, write_json
from hotel_cancellation.utils.tracking import setup_mlflow

logger = get_logger(__name__)

STAGES = ("ingest", "validate", "transform", "train", "evaluate")


class TrainingPipeline:
    """Run the stages: ingest -> validate -> transform -> train -> evaluate (+ register).

    Each stage reads its inputs from the previous stage's files, so a stage can run on its own
    (`hc-train --stage train`), which is how DVC runs them. The train and evaluate stages share
    one MLflow run: train starts it and records its id, evaluate resumes it.
    """

    def __init__(
        self, config: ConfigurationManager | None = None, force_promote: bool = False
    ) -> None:
        self.config = config or ConfigurationManager()
        self.force_promote = force_promote
        setup_mlflow(self.config.get_mlflow_config())

    # ---- inputs rebuilt from disk -------------------------------------------------------

    def _ingestion_artifact(self) -> DataIngestionArtifact:
        cfg = self.config.get_data_ingestion_config()
        if not (cfg.train_path.exists() and cfg.test_path.exists()):
            raise ModelError("Ingested data not found; run the ingest stage first")
        return DataIngestionArtifact(train_path=cfg.train_path, test_path=cfg.test_path)

    def _validation_artifact(self) -> DataValidationArtifact:
        cfg = self.config.get_data_validation_config()
        if not cfg.report_path.exists():
            raise ModelError("Validation report not found; run the validate stage first")
        report = read_json(cfg.report_path)
        return DataValidationArtifact(
            is_valid=report["is_valid"],
            report_path=cfg.report_path,
            reference_profile_path=cfg.reference_profile_path,
            evidently_report_path=cfg.evidently_report_path,
            dataset_drift=report["train_test_drift"]["dataset_drift"],
        )

    def _transformation_artifact(self) -> DataTransformationArtifact:
        cfg = self.config.get_data_transformation_config()
        return DataTransformationArtifact(
            preprocessor_path=cfg.preprocessor_path,
            feature_names=feature_names(load_object(cfg.preprocessor_path)),
        )

    def _trainer_artifact(self) -> ModelTrainerArtifact:
        cfg = self.config.get_model_trainer_config()
        metadata = read_json(cfg.metadata_path)
        return ModelTrainerArtifact(
            model_path=cfg.model_path,
            metadata_path=cfg.metadata_path,
            threshold=metadata["threshold"],
            validation_metrics=metadata["validation_metrics"],
            baseline_paths={
                name: cfg.baselines_dir / f"{name}.joblib" for name in metadata["baselines"]
            },
        )

    # ---- stages ---------------------------------------------------------------------------

    def ingest(self) -> DataIngestionArtifact:
        return DataIngestion(self.config.get_data_ingestion_config()).run()

    def validate(self) -> DataValidationArtifact:
        stage = DataValidation(self.config.get_data_validation_config())
        return stage.run(self._ingestion_artifact())

    def transform(self) -> DataTransformationArtifact:
        stage = DataTransformation(self.config.get_data_transformation_config())
        return stage.run(self._ingestion_artifact())

    def train(self) -> ModelTrainerArtifact:
        import mlflow

        cfg = self.config.get_model_trainer_config()
        with mlflow.start_run() as run:
            artifact = ModelTrainer(cfg).run(
                self._ingestion_artifact(), self._transformation_artifact()
            )
            metadata = read_json(cfg.metadata_path)
            metadata["mlflow_run_id"] = run.info.run_id
            write_json(cfg.metadata_path, metadata)

            mlflow.log_params({f"xgb_{k}": v for k, v in metadata["model_params"].items()})
            mlflow.log_params(
                {
                    "n_trees": metadata["n_trees"],
                    "refit_on_full_train": metadata["refit_on_full_train"],
                    "cost_false_negative": cfg.costs.false_negative,
                    "cost_false_positive": cfg.costs.false_positive,
                    "optuna_trials": cfg.n_trials,
                    "n_features": len(metadata["feature_names"]),
                }
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in metadata["validation_metrics"].items()})
            if metadata["tuning"]:
                best = metadata["tuning"]["best_validation_roc_auc"]
                mlflow.log_metric("optuna_best_val_roc_auc", best)
            for name, baseline in metadata["baselines"].items():
                mlflow.log_metrics(
                    {f"val_{name}_{k}": v for k, v in baseline["validation_metrics"].items()}
                )
        return artifact

    def evaluate(self) -> dict[str, Any]:
        """Evaluate on the test set and, if accepted, register and export the model."""
        import mlflow

        ingestion = self._ingestion_artifact()
        validation = self._validation_artifact()
        trainer = self._trainer_artifact()
        registry = ModelRegistry(self.config.get_model_registry_config())
        run_id = read_json(trainer.metadata_path).get("mlflow_run_id")

        with mlflow.start_run(run_id=run_id):
            evaluation = ModelEvaluation(
                self.config.get_model_evaluation_config(), registry, self.force_promote
            ).run(ingestion, trainer)

            target = self.config.target
            example = pd.read_csv(ingestion.test_path, nrows=5).drop(columns=[target])
            registered = registry.register(trainer, evaluation, validation, example)

            mlflow.log_metrics({f"test_{k}": v for k, v in evaluation.metrics.items()})
            for name, metrics in evaluation.baseline_metrics.items():
                mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})
            mlflow.set_tags(
                {
                    "accepted": evaluation.is_accepted,
                    "registered_version": registered.version or "none",
                    "train_test_drift": validation.dataset_drift,
                }
            )
            for path in (
                validation.report_path,
                validation.evidently_report_path,
                trainer.metadata_path,
                evaluation.metrics_path,
            ):
                mlflow.log_artifact(str(path))

        logger.info("Evaluation finished: %s", evaluation.reason)
        return {
            "test_metrics": evaluation.metrics,
            "baseline_metrics": evaluation.baseline_metrics,
            "production_metrics": evaluation.production_metrics,
            "accepted": evaluation.is_accepted,
            "reason": evaluation.reason,
            "registered_version": registered.version,
        }

    def run(self) -> dict[str, Any]:
        for number, stage in enumerate(STAGES[:-1], start=1):
            logger.info("===== Stage %d/%d: %s =====", number, len(STAGES), stage)
            getattr(self, stage)()
        logger.info("===== Stage %d/%d: evaluate + register =====", len(STAGES), len(STAGES))
        return self.evaluate()


def _print_summary(summary: dict[str, Any]) -> None:
    rows = {"xgboost": summary["test_metrics"], **summary["baseline_metrics"]}
    if summary["production_metrics"]:
        rows["production (before)"] = summary["production_metrics"]
    columns = ["roc_auc", "pr_auc", "f1", "threshold", "cost_per_booking"]
    print("\nTest set (newest 20% of arrivals):")
    print(pd.DataFrame(rows).T[columns].round(4).to_string())
    if summary["registered_version"]:
        print(f"\nRegistered as version {summary['registered_version']} (@production)")
    else:
        print(f"\nNot promoted: {summary['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--stage", choices=STAGES, help="Run only this stage (used by DVC)")
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote even if it does not beat production (e.g. after removing leaky features)",
    )
    args = parser.parse_args(argv)

    try:
        pipeline = TrainingPipeline(force_promote=args.force_promote)
        if args.stage is None:
            _print_summary(pipeline.run())
        elif args.stage == "evaluate":
            _print_summary(pipeline.evaluate())
        else:
            getattr(pipeline, args.stage)()
    except HotelCancellationError as error:
        logger.error("Training pipeline failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
