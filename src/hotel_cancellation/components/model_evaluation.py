from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hotel_cancellation.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from hotel_cancellation.entity.config_entity import ModelEvaluationConfig
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import load_object, read_json, write_json

if TYPE_CHECKING:
    from hotel_cancellation.components.model_registry import ModelRegistry

logger = get_logger(__name__)


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


class ModelEvaluation:
    """Score the candidate on the held-out test set and decide whether it should go live.

    The current production model (if any) is scored on the same test set, so the comparison
    is fair even when the data has changed since it was trained.
    """

    def __init__(self, config: ModelEvaluationConfig, registry: "ModelRegistry") -> None:
        self.config = config
        self.registry = registry

    def run(
        self, ingestion: DataIngestionArtifact, trainer: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        test = pd.read_csv(ingestion.test_path)
        X_test = test.drop(columns=[self.config.target])
        y_test = test[self.config.target].to_numpy()

        candidate = load_object(trainer.model_path)
        threshold = read_json(trainer.metadata_path)["threshold"]
        metrics = classification_metrics(y_test, candidate.predict_proba(X_test)[:, 1], threshold)
        logger.info(
            "Candidate test metrics: ROC-AUC %.4f, PR-AUC %.4f, F1 %.4f",
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
        )

        production_metrics = None
        production = self.registry.load_production()
        if production is not None:
            prod_model, prod_metadata = production
            try:
                prod_proba = prod_model.predict_proba(X_test)[:, 1]
                production_metrics = classification_metrics(
                    y_test, prod_proba, prod_metadata["threshold"]
                )
                logger.info(
                    "Production (%s) test ROC-AUC %.4f",
                    prod_metadata.get("version"),
                    production_metrics["roc_auc"],
                )
            except Exception as error:  # incompatible old model: treat as no baseline
                logger.warning("Could not score the production model: %s", error)

        is_accepted, reason = self._decide(metrics, production_metrics)
        logger.info("Candidate %s: %s", "accepted" if is_accepted else "rejected", reason)

        write_json(
            self.config.metrics_path,
            {
                "candidate": metrics,
                "production": production_metrics,
                "is_accepted": is_accepted,
                "reason": reason,
                "n_test_rows": len(y_test),
            },
        )
        return ModelEvaluationArtifact(
            metrics_path=self.config.metrics_path,
            metrics=metrics,
            production_metrics=production_metrics,
            is_accepted=is_accepted,
            reason=reason,
        )

    def _decide(
        self, metrics: dict[str, float], production: dict[str, float] | None
    ) -> tuple[bool, str]:
        auc = metrics["roc_auc"]
        if auc < self.config.min_roc_auc:
            return False, f"ROC-AUC {auc:.4f} is below the minimum {self.config.min_roc_auc}"
        if production is None:
            return True, f"ROC-AUC {auc:.4f} meets the minimum and no production model exists"
        needed = production["roc_auc"] + self.config.min_improvement
        if auc < needed:
            return False, (
                f"ROC-AUC {auc:.4f} does not beat production {production['roc_auc']:.4f} "
                f"by {self.config.min_improvement}"
            )
        return True, f"ROC-AUC {auc:.4f} beats production {production['roc_auc']:.4f}"
