from typing import Any

import numpy as np
import pandas as pd
from xgboost import DMatrix

from hotel_cancellation.components.model_registry import PROFILE_FILE, ModelRegistry
from hotel_cancellation.config.configuration import ConfigurationManager
from hotel_cancellation.exception import ModelError
from hotel_cancellation.utils.common import read_json


class PredictionPipeline:
    """Score raw booking rows with the production model from the registry."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        if registry is None:
            registry = ModelRegistry(ConfigurationManager().get_model_registry_config())
        production = registry.load_production()
        if production is None:
            raise ModelError("No production model in the registry; run the training pipeline")
        self.registry = registry
        self.model, self.metadata = production
        # Training-data distribution of this model version, used for drift checks.
        self.reference_profile = read_json(registry.production_dir() / PROFILE_FILE)
        self.threshold = float(self.metadata["threshold"])
        self._source_columns = self._map_features_to_source()

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(df)[:, 1]
        return pd.DataFrame(
            {"cancel_probability": proba, "will_cancel": proba >= self.threshold},
            index=df.index,
        )

    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Top features pushing each prediction up or down.

        Uses XGBoost's built-in TreeSHAP contributions (log-odds). One-hot columns are summed
        back into their source feature, so "country" is reported once rather than per country.
        """
        preprocessor = self.model.named_steps["preprocessor"]
        booster_model = self.model.named_steps["model"]
        X = preprocessor.transform(df)
        contribs = booster_model.get_booster().predict(
            DMatrix(X),
            pred_contribs=True,
            iteration_range=(0, booster_model.best_iteration + 1),
        )[:, :-1]  # last column is the bias term

        engineered = preprocessor.named_steps["features"].transform(df)
        grouped = (
            pd.DataFrame(contribs, columns=self._source_columns)
            .T.groupby(level=0, sort=False)
            .sum()
            .T
        )
        explanations = []
        for row in range(len(df)):
            values = grouped.iloc[row]
            top = values.reindex(values.abs().sort_values(ascending=False).index[:top_k])
            explanations.append(
                [
                    {
                        "feature": feature,
                        "value": _to_builtin(engineered[feature].iloc[row]),
                        "contribution": round(float(contribution), 4),
                    }
                    for feature, contribution in top.items()
                ]
            )
        return explanations

    def _map_features_to_source(self) -> list[str]:
        columns = self.model.named_steps["preprocessor"].named_steps["columns"]
        sources: list[str] = []
        for name, transformer, input_columns in columns.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if hasattr(transformer, "categories_"):
                for column, names in zip(
                    input_columns,
                    _onehot_groups(transformer, input_columns),
                    strict=True,
                ):
                    sources.extend([column] * names)
            else:
                sources.extend(input_columns)
        return sources


def _onehot_groups(encoder, input_columns: list[str]) -> list[int]:
    """Number of output columns the one-hot encoder makes for each input column."""
    names = encoder.get_feature_names_out(input_columns)
    counts = []
    position = 0
    for column in input_columns:
        count = 0
        while position < len(names) and names[position].startswith(f"{column}_"):
            count += 1
            position += 1
        counts.append(count)
    if position != len(names):
        raise ModelError("Could not map one-hot features back to their source columns")
    return counts


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
