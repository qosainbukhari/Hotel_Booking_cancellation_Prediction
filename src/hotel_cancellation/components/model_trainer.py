from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from hotel_cancellation.components.model_evaluation import (
    classification_metrics,
    cost_optimal_threshold,
)
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

TUNING_MAX_TREES = 1000


def make_xgb(params: dict[str, Any], early_stopping_rounds: int | None = None) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
    )


class ModelTrainer:
    """Train XGBoost (optionally tuned with Optuna) plus two baseline models.

    Every model follows the same recipe so the comparison is fair:

    1. fit on the older part of the training data,
    2. choose the decision threshold on the newest slice (validation) by minimising cost,
    3. refit on all training rows (fit + validation).

    For XGBoost, step 1 also tunes hyper-parameters and picks the number of trees by early
    stopping on the validation slice. Each saved model is one sklearn Pipeline (preprocessing +
    estimator) that takes raw booking rows.
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
        X_all, y_all = np.vstack([X_fit, X_val]), np.concatenate([y_fit, y_val])
        logger.info("Fit rows %d, validation rows %d", len(y_fit), len(y_val))

        params = dict(self.config.model_params)
        tuning = None
        if self.config.n_trials > 0:
            best_params, tuning = self._tune(params, X_fit, y_fit, X_val, y_val)
            params.update(best_params)

        try:
            selector = make_xgb(params, self.config.early_stopping_rounds)
            selector.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        except Exception as error:
            raise ModelError(f"Model training failed: {error}") from error
        n_trees = int(selector.best_iteration) + 1
        val_proba = selector.predict_proba(X_val)[:, 1]
        threshold = cost_optimal_threshold(y_val, val_proba, self.config.costs)
        val_metrics = classification_metrics(y_val, val_proba, threshold, self.config.costs)
        logger.info(
            "XGBoost: %d trees, validation ROC-AUC %.4f, threshold %.3f (cost/booking %.2f)",
            n_trees,
            val_metrics["roc_auc"],
            threshold,
            val_metrics["cost_per_booking"],
        )

        # Final model: a fixed number of trees and no early-stopping state, so prediction and
        # SHAP explanations use exactly the same trees.
        final_params = {**params, "n_estimators": n_trees}
        X_final, y_final = (X_all, y_all) if self.config.refit_on_full_train else (X_fit, y_fit)
        model = make_xgb(final_params).fit(X_final, y_final)
        save_object(
            self.config.model_path, Pipeline([("preprocessor", preprocessor), ("model", model)])
        )

        baselines, baseline_paths = {}, {}
        for name, estimator in self._baselines().items():
            estimator.fit(X_fit, y_fit)
            proba = estimator.predict_proba(X_val)[:, 1]
            b_threshold = cost_optimal_threshold(y_val, proba, self.config.costs)
            baselines[name] = {
                "threshold": b_threshold,
                "validation_metrics": classification_metrics(
                    y_val, proba, b_threshold, self.config.costs
                ),
            }
            final = clone(estimator).fit(X_final, y_final)
            path = self.config.baselines_dir / f"{name}.joblib"
            save_object(path, Pipeline([("preprocessor", preprocessor), ("model", final)]))
            baseline_paths[name] = path
            logger.info(
                "Baseline %s: validation ROC-AUC %.4f",
                name,
                baselines[name]["validation_metrics"]["roc_auc"],
            )

        write_json(
            self.config.metadata_path,
            {
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "threshold": threshold,
                "threshold_rule": "minimum expected cost on the validation slice",
                "costs": {
                    "false_negative": self.config.costs.false_negative,
                    "false_positive": self.config.costs.false_positive,
                },
                "n_trees": n_trees,
                "refit_on_full_train": self.config.refit_on_full_train,
                "n_train_rows": len(y_final),
                "n_validation_rows": len(y_val),
                "validation_metrics": val_metrics,
                "model_params": final_params,
                "tuning": tuning,
                "baselines": baselines,
                "feature_names": transformation.feature_names,
            },
        )
        return ModelTrainerArtifact(
            model_path=self.config.model_path,
            metadata_path=self.config.metadata_path,
            threshold=threshold,
            validation_metrics=val_metrics,
            baseline_paths=baseline_paths,
        )

    def _baselines(self) -> dict[str, Any]:
        params = self.config.baseline_params
        seed = self.config.model_params.get("random_state", 42)
        return {
            "logistic_regression": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(**params["logistic_regression"])),
                ]
            ),
            "random_forest": RandomForestClassifier(
                **params["random_forest"], n_jobs=-1, random_state=seed
            ),
        }

    def _tune(
        self,
        base: dict[str, Any],
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Search XGBoost hyper-parameters with Optuna, maximising validation ROC-AUC."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {
                **base,
                "n_estimators": TUNING_MAX_TREES,
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
            model = make_xgb(params, self.config.tuning_early_stopping_rounds)
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

        sampler = optuna.samplers.TPESampler(seed=base.get("random_state", 42))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        logger.info("Tuning XGBoost with Optuna (%d trials)", self.config.n_trials)
        study.optimize(objective, n_trials=self.config.n_trials)
        logger.info(
            "Best trial %d: validation ROC-AUC %.4f", study.best_trial.number, study.best_value
        )
        summary = {
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_validation_roc_auc": study.best_value,
            "best_params": study.best_params,
        }
        return study.best_params, summary
