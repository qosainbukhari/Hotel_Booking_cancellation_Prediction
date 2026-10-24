from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import DMatrix

from hotel_cancellation.config.configuration import ConfigurationManager
from hotel_cancellation.exception import ModelError
from hotel_cancellation.utils.common import load_object, read_json

# File names inside the serving bundle (see components/model_registry.py).
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
PROFILE_FILE = "reference_profile.json"
VERSIONS_FILE = "versions.json"


class PredictionPipeline:
    """Score raw booking rows with the production model's serving bundle.

    Only reads files from the bundle folder, so serving needs neither MLflow nor the
    training data.
    """

    def __init__(self, serving_dir: Path | None = None) -> None:
        if serving_dir is None:
            serving_dir = ConfigurationManager().get_model_registry_config().serving_dir
        self.serving_dir = Path(serving_dir)
        if not (self.serving_dir / MODEL_FILE).exists():
            raise ModelError(
                f"No serving bundle in {self.serving_dir}; run `uv run hc-train` first"
            )
        self.model = load_object(self.serving_dir / MODEL_FILE)
        self.metadata = read_json(self.serving_dir / METADATA_FILE)
        self.reference_profile = read_json(self.serving_dir / PROFILE_FILE)
        self.versions = read_json(self.serving_dir / VERSIONS_FILE)
        self.threshold = float(self.metadata["threshold"])
        self._source_columns = self._map_features_to_source()

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.model.predict_proba(df)[:, 1]
        return pd.DataFrame(
            {"cancel_probability": proba, "will_cancel": proba >= self.threshold},
            index=df.index,
        )

    def _contributions(self, df: pd.DataFrame) -> np.ndarray:
        """TreeSHAP values per model column plus the bias (last column), in log-odds.

        XGBoost computes exact TreeSHAP itself, so serving does not need the `shap` package
        (and its ~190 MB of numba/llvmlite). The values equal shap.TreeExplainer's, which a test
        checks.
        """
        X = self.model.named_steps["preprocessor"].transform(df)
        booster = self.model.named_steps["model"].get_booster()
        return booster.predict(DMatrix(X), pred_contribs=True)

    def base_value(self, df: pd.DataFrame) -> float:
        """Log-odds of the average booking (the SHAP base value)."""
        return float(self._contributions(df.head(1))[0, -1])

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds) per source feature, one row per booking.

        One-hot columns are summed back into their source feature, so "deposit_type" appears
        once rather than once per deposit type.
        """
        values = self._contributions(df)[:, :-1]
        return (
            pd.DataFrame(values, columns=self._source_columns, index=df.index)
            .T.groupby(level=0, sort=False)
            .sum()
            .T
        )

    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Top features pushing each prediction up (towards cancelling) or down."""
        grouped = self.shap_values(df)
        engineered = self.model.named_steps["preprocessor"].named_steps["features"].transform(df)
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
            if name == "remainder":
                continue
            if hasattr(transformer, "categories_"):
                for column, count in zip(
                    input_columns, _onehot_groups(transformer, input_columns), strict=True
                ):
                    sources.extend([column] * count)
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
