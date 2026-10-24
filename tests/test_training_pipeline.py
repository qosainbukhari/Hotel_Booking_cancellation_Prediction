import pandas as pd

from hotel_cancellation.components.model_registry import ModelRegistry
from hotel_cancellation.pipeline.prediction_pipeline import PredictionPipeline
from hotel_cancellation.pipeline.training_pipeline import TrainingPipeline
from hotel_cancellation.utils.common import read_json


def test_pipeline_trains_registers_and_serves(config_manager, bookings):
    summary = TrainingPipeline(config_manager).run()

    assert summary["accepted"]
    assert summary["registered_version"] == "v1"
    assert summary["test_metrics"]["roc_auc"] > 0.6

    registry = ModelRegistry(config_manager.get_model_registry_config())
    model_dir = registry.production_dir()
    assert model_dir.name == "v1"
    assert (model_dir / "reference_profile.json").exists()
    assert read_json(model_dir / "metadata.json")["version"] == "v1"

    predictor = PredictionPipeline(registry)
    rows = bookings.drop(columns=["is_canceled"]).head(5)
    predictions = predictor.predict(rows)
    assert predictions["cancel_probability"].between(0, 1).all()
    assert (
        predictions["will_cancel"] == (predictions["cancel_probability"] >= predictor.threshold)
    ).all()

    explanation = predictor.explain(rows.head(1), top_k=3)[0]
    assert len(explanation) == 3
    assert {"feature", "value", "contribution"} <= explanation[0].keys()


def test_retraining_without_improvement_keeps_production(config_manager):
    TrainingPipeline(config_manager).run()
    summary = TrainingPipeline(config_manager).run()

    assert not summary["accepted"]
    assert summary["registered_version"] is None
    registry = ModelRegistry(config_manager.get_model_registry_config())
    assert registry.production_dir().name == "v1"


def test_single_and_batch_predictions_agree(config_manager, bookings):
    TrainingPipeline(config_manager).run()
    predictor = PredictionPipeline(ModelRegistry(config_manager.get_model_registry_config()))
    rows = bookings.drop(columns=["is_canceled"]).head(10)

    batch = predictor.predict(rows)["cancel_probability"]
    single = pd.concat([predictor.predict(rows.iloc[[i]]) for i in range(len(rows))])
    pd.testing.assert_series_equal(batch, single["cancel_probability"])
