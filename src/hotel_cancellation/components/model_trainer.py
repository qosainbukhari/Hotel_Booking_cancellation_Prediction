from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from hotel_cancellation.components.model_evaluation import classification_metrics
from hotel_cancellation.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from hotel_cancellation.entity.config_entity import ModelTrainerConfig
from hotel_cancellation.exception import ModelError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import load_object, save_object, split_tail, write_json

logger = get_logger(__name__)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Probability cut-off that gives the highest F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # The last precision / recall pair has no threshold.
    return float(thresholds[int(np.argmax(f1[:-1]))])


class ModelTrainer:
    """Train XGBoost with early stopping on the most recent slice of the training data.

    The saved model is one sklearn Pipeline (preprocessing + XGBoost) that takes raw booking
    rows, plus a metadata file with the tuned decision threshold.
    """

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def run(
        self, ingestion: DataIngestionArtifact, transformation: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        target = self.config.target
        train = pd.read_csv(ingestion.train_path)
        fit_df, val_df = split_tail(train, self.config.validation_size)
        preprocessor = load_object(transformation.preprocessor_path)

        X_fit = preprocessor.transform(fit_df.drop(columns=[target]))
        X_val = preprocessor.transform(val_df.drop(columns=[target]))
        y_fit, y_val = fit_df[target].to_numpy(), val_df[target].to_numpy()

        model = XGBClassifier(
            **self.config.model_params,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=self.config.early_stopping_rounds,
        )
        logger.info("Training XGBoost on %d rows, validating on %d", len(y_fit), len(y_val))
        try:
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        except Exception as error:
            raise ModelError(f"Model training failed: {error}") from error

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = best_f1_threshold(y_val, val_proba)
        val_metrics = classification_metrics(y_val, val_proba, threshold)
        logger.info(
            "Best iteration %d, validation ROC-AUC %.4f, F1 %.4f at threshold %.3f",
            model.best_iteration,
            val_metrics["roc_auc"],
            val_metrics["f1"],
            threshold,
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        save_object(self.config.model_path, pipeline)
        metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "threshold": threshold,
            "best_iteration": int(model.best_iteration),
            "n_train_rows": len(y_fit),
            "n_validation_rows": len(y_val),
            "validation_metrics": val_metrics,
            "model_params": self.config.model_params,
            "feature_names": transformation.feature_names,
        }
        write_json(self.config.metadata_path, metadata)
        return ModelTrainerArtifact(
            model_path=self.config.model_path,
            metadata_path=self.config.metadata_path,
            threshold=threshold,
            validation_metrics=val_metrics,
        )
