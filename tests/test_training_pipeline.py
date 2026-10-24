import numpy as np
import pandas as pd
import pytest

from hotel_cancellation.components.model_evaluation import (
    cost_optimal_threshold,
    expected_cost,
)
from hotel_cancellation.components.model_registry import ModelRegistry
from hotel_cancellation.entity.config_entity import CostConfig
from hotel_cancellation.exception import ModelError
from hotel_cancellation.pipeline.prediction_pipeline import PredictionPipeline
from hotel_cancellation.pipeline.training_pipeline import TrainingPipeline
from hotel_cancellation.utils.common import read_json


def _serving_dir(config_manager):
    return config_manager.get_model_registry_config().serving_dir


def test_pipeline_trains_registers_and_serves(config_manager, bookings):
    summary = TrainingPipeline(config_manager).run()

    assert summary["accepted"]
    assert summary["registered_version"] == "1"
    assert summary["test_metrics"]["roc_auc"] > 0.6
    assert set(summary["baseline_metrics"]) == {"logistic_regression", "random_forest"}

    serving = _serving_dir(config_manager)
    metadata = read_json(serving / "metadata.json")
    assert metadata["version"] == "1"
    assert metadata["tuning"]["n_trials"] == 2
    assert metadata["threshold_rule"].startswith("minimum expected cost")
    assert read_json(serving / "versions.json")["production"] == "1"
    assert (serving / "reference_profile.json").exists()

    predictor = PredictionPipeline(serving)
    rows = bookings.drop(columns=["is_canceled"]).head(5)
    predictions = predictor.predict(rows)
    assert predictions["cancel_probability"].between(0, 1).all()
    expected = predictions["cancel_probability"] >= predictor.threshold
    assert (predictions["will_cancel"] == expected).all()

    explanation = predictor.explain(rows.head(1), top_k=3)[0]
    assert len(explanation) == 3
    assert {"feature", "value", "contribution"} <= explanation[0].keys()


def test_shap_values_add_up_to_the_prediction(config_manager, bookings):
    TrainingPipeline(config_manager).run()
    predictor = PredictionPipeline(_serving_dir(config_manager))
    rows = bookings.drop(columns=["is_canceled"]).head(20)

    log_odds = predictor.shap_values(rows).sum(axis=1) + predictor.base_value(rows)
    proba = predictor.predict(rows)["cancel_probability"]
    np.testing.assert_allclose(1 / (1 + np.exp(-log_odds)), proba, atol=1e-4)


def test_serving_explanations_match_the_shap_package(config_manager, bookings):
    shap = pytest.importorskip("shap")
    TrainingPipeline(config_manager).run()
    predictor = PredictionPipeline(_serving_dir(config_manager))
    rows = bookings.drop(columns=["is_canceled"]).head(50)

    X = predictor.model.named_steps["preprocessor"].transform(rows)
    explainer = shap.TreeExplainer(predictor.model.named_steps["model"])
    np.testing.assert_allclose(
        predictor._contributions(rows)[:, :-1], explainer.shap_values(X), atol=1e-4
    )
    assert predictor.base_value(rows) == pytest.approx(float(explainer.expected_value), abs=1e-4)


def test_retraining_without_improvement_keeps_production(config_manager):
    TrainingPipeline(config_manager).run()
    summary = TrainingPipeline(config_manager).run()

    assert not summary["accepted"]
    assert summary["registered_version"] is None
    registry = ModelRegistry(config_manager.get_model_registry_config())
    assert str(registry.production_version().version) == "1"


def test_force_promote_registers_a_new_version(config_manager):
    TrainingPipeline(config_manager).run()
    summary = TrainingPipeline(config_manager, force_promote=True).run()

    assert summary["registered_version"] == "2"
    versions = read_json(_serving_dir(config_manager) / "versions.json")
    assert versions["production"] == "2"
    assert [v["is_production"] for v in versions["versions"]] == [False, True]


def test_stage_without_its_inputs_fails_clearly(config_manager):
    with pytest.raises(ModelError, match="ingest"):
        TrainingPipeline(config_manager).train()


def test_single_and_batch_predictions_agree(config_manager, bookings):
    TrainingPipeline(config_manager).run()
    predictor = PredictionPipeline(_serving_dir(config_manager))
    rows = bookings.drop(columns=["is_canceled"]).head(10)

    batch = predictor.predict(rows)["cancel_probability"]
    single = pd.concat([predictor.predict(rows.iloc[[i]]) for i in range(len(rows))])
    pd.testing.assert_series_equal(batch, single["cancel_probability"])


def test_cost_optimal_threshold_follows_the_costs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    proba = np.clip(y * 0.3 + rng.random(5000) * 0.7, 0, 1)

    cheap_alarms = cost_optimal_threshold(y, proba, CostConfig(100, 5))
    costly_alarms = cost_optimal_threshold(y, proba, CostConfig(100, 200))
    assert cheap_alarms < costly_alarms

    best = expected_cost(y, proba, cheap_alarms, CostConfig(100, 5))
    for other in (0.1, 0.5, 0.9):
        assert best <= expected_cost(y, proba, other, CostConfig(100, 5))
