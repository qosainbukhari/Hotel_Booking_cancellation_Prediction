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
from hotel_cancellation.entity.config_entity import CostConfig, ModelEvaluationConfig
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import load_object, read_json, write_json

if TYPE_CHECKING:
    from hotel_cancellation.components.model_registry import ModelRegistry

logger = get_logger(__name__)

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def expected_cost(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, costs: CostConfig
) -> float:
    """Total cost of the mistakes at this threshold, per booking."""
    flagged = proba >= threshold
    false_negatives = np.sum(~flagged & (y_true == 1))
    false_positives = np.sum(flagged & (y_true == 0))
    total = false_negatives * costs.false_negative + false_positives * costs.false_positive
    return float(total / len(y_true))


def cost_optimal_threshold(y_true: np.ndarray, proba: np.ndarray, costs: CostConfig) -> float:
    """Threshold that minimises expected cost. Ties go to the higher (more cautious) threshold."""
    cost = np.array([expected_cost(y_true, proba, t, costs) for t in THRESHOLD_GRID])
    best = np.flatnonzero(cost == cost.min())[-1]
    return float(THRESHOLD_GRID[best])


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, costs: CostConfig | None = None
) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    if costs is not None:
        # Cost with no model: every cancellation is missed.
        no_model = float(np.mean(y_true) * costs.false_negative)
        cost = expected_cost(y_true, proba, threshold, costs)
        metrics["cost_per_booking"] = cost
        metrics["cost_saving"] = 1 - cost / no_model if no_model else 0.0
    return metrics


class ModelEvaluation:
    """Score the candidate and the baselines on the held-out test set, and decide whether the
    candidate should go live.

    The current production model (if any) is scored on the same test set, so the comparison
    is fair even when the data has changed since it was trained.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        registry: "ModelRegistry",
        force_promote: bool = False,
    ) -> None:
        self.config = config
        self.registry = registry
        self.force_promote = force_promote

    def run(
        self, ingestion: DataIngestionArtifact, trainer: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        test = pd.read_csv(ingestion.test_path)
        X_test = test.drop(columns=[self.config.target])
        y_test = test[self.config.target].to_numpy()
        costs = self.config.costs

        def score(model, threshold: float) -> dict[str, float]:
            return classification_metrics(
                y_test, model.predict_proba(X_test)[:, 1], threshold, costs
            )

        metadata = read_json(trainer.metadata_path)
        metrics = score(load_object(trainer.model_path), metadata["threshold"])
        logger.info(
            "Candidate test metrics: ROC-AUC %.4f, PR-AUC %.4f, F1 %.4f, cost/booking %.2f",
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
            metrics["cost_per_booking"],
        )

        baseline_metrics = {}
        for name, path in trainer.baseline_paths.items():
            baseline_metrics[name] = score(
                load_object(path), metadata["baselines"][name]["threshold"]
            )
            logger.info("Baseline %s test ROC-AUC %.4f", name, baseline_metrics[name]["roc_auc"])

        production_metrics = None
        production = self.registry.load_production()
        if production is not None:
            prod_model, prod_metadata = production
            try:
                production_metrics = score(prod_model, prod_metadata["threshold"])
                logger.info(
                    "Production (v%s) test ROC-AUC %.4f",
                    prod_metadata.get("version"),
                    production_metrics["roc_auc"],
                )
            except Exception as error:  # e.g. trained on a different feature set
                logger.warning("Could not score the production model: %s", error)

        is_accepted, reason = self._decide(metrics, production_metrics)
        logger.info("Candidate %s: %s", "accepted" if is_accepted else "rejected", reason)

        write_json(
            self.config.metrics_path,
            {
                "candidate": metrics,
                "production": production_metrics,
                "baselines": baseline_metrics,
                "is_accepted": is_accepted,
                "reason": reason,
                "n_test_rows": len(y_test),
            },
        )
        return ModelEvaluationArtifact(
            metrics_path=self.config.metrics_path,
            metrics=metrics,
            production_metrics=production_metrics,
            baseline_metrics=baseline_metrics,
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
        if self.force_promote:
            return True, (
                f"Promotion forced: ROC-AUC {auc:.4f} vs production {production['roc_auc']:.4f}"
            )
        needed = production["roc_auc"] + self.config.min_improvement
        if auc < needed:
            return False, (
                f"ROC-AUC {auc:.4f} does not beat production {production['roc_auc']:.4f} "
                f"by {self.config.min_improvement}"
            )
        return True, f"ROC-AUC {auc:.4f} beats production {production['roc_auc']:.4f}"
