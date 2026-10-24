from typing import Any

import pandas as pd

from hotel_cancellation.components.data_transformation import FeatureEngineer
from hotel_cancellation.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from hotel_cancellation.entity.config_entity import DataValidationConfig
from hotel_cancellation.exception import DataValidationError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.monitoring.drift import build_reference_profile, detect_drift
from hotel_cancellation.utils.common import write_json

logger = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    columns: dict[str, dict[str, Any]],
    target: str,
    max_invalid_fraction: float,
) -> dict[str, Any]:
    """Check a frame against the schema.

    Structural problems (missing columns, wrong types, bad target) are always errors.
    Row-level rule breaks (ranges, allowed values, unexpected nulls) are warnings while they
    stay under `max_invalid_fraction` of the rows, and errors above it.
    """
    errors: list[str] = []
    warnings: list[str] = []
    n_rows = len(df)

    if n_rows == 0:
        return {"is_valid": False, "n_rows": 0, "errors": ["No rows"], "warnings": []}

    missing = sorted(set(columns) - set(df.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")

    for name, rule in columns.items():
        if name not in df.columns:
            continue
        series = df[name]
        nulls = series.isna()

        if rule["type"] == "numeric" and not pd.api.types.is_numeric_dtype(series):
            errors.append(f"{name}: expected numeric values, got dtype {series.dtype}")
            continue

        invalid = pd.Series(False, index=df.index)
        if not rule.get("nullable", False):
            invalid |= nulls
        values = series[~nulls]
        if "allowed" in rule:
            invalid.loc[values.index] |= ~values.isin(rule["allowed"])
        if "min" in rule:
            invalid.loc[values.index] |= values < rule["min"]
        if "max" in rule:
            invalid.loc[values.index] |= values > rule["max"]

        n_invalid = int(invalid.sum())
        if n_invalid == 0:
            continue
        message = f"{name}: {n_invalid} rows ({n_invalid / n_rows:.2%}) break the schema rules"
        if name == target or n_invalid / n_rows > max_invalid_fraction:
            errors.append(message)
        else:
            warnings.append(message)

    return {"is_valid": not errors, "n_rows": n_rows, "errors": errors, "warnings": warnings}


class DataValidation:
    """Validate train / test data, profile the training distribution and check train-test drift."""

    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def run(self, ingestion: DataIngestionArtifact) -> DataValidationArtifact:
        train = pd.read_csv(ingestion.train_path)
        test = pd.read_csv(ingestion.test_path)

        report: dict[str, Any] = {}
        for split, df in (("train", train), ("test", test)):
            result = validate_dataframe(
                df, self.config.columns, self.config.target, self.config.max_invalid_fraction
            )
            report[split] = result
            for warning in result["warnings"]:
                logger.warning("[%s] %s", split, warning)
            for error in result["errors"]:
                logger.error("[%s] %s", split, error)

        is_valid = report["train"]["is_valid"] and report["test"]["is_valid"]
        if not is_valid:
            write_json(self.config.report_path, {"is_valid": False, **report})
            raise DataValidationError(f"Data validation failed, see {self.config.report_path}")

        features = FeatureEngineer()
        profile = build_reference_profile(
            features.transform(train),
            self.config.numerical_features,
            self.config.categorical_features,
            self.config.n_bins,
        )
        drift = detect_drift(
            profile,
            features.transform(test),
            self.config.psi_threshold,
            self.config.drift_share_threshold,
        )
        report["train_test_drift"] = drift
        if drift["drifted_features"]:
            logger.warning("Train -> test drift in: %s", ", ".join(drift["drifted_features"]))

        write_json(self.config.reference_profile_path, profile)
        write_json(self.config.report_path, {"is_valid": True, **report})
        logger.info("Data validation passed, report saved to %s", self.config.report_path)
        return DataValidationArtifact(
            is_valid=True,
            report_path=self.config.report_path,
            reference_profile_path=self.config.reference_profile_path,
            dataset_drift=drift["dataset_drift"],
        )
