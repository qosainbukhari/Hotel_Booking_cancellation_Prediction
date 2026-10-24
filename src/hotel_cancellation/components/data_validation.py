from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

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


def build_schema(columns: dict[str, dict[str, Any]]) -> pa.DataFrameSchema:
    """Turn the rules in schema.yaml into a Pandera schema (row-level checks only)."""
    pandera_columns = {}
    for name, rule in columns.items():
        checks = []
        if "allowed" in rule:
            checks.append(pa.Check.isin(rule["allowed"]))
        if "min" in rule:
            checks.append(pa.Check.ge(rule["min"]))
        if "max" in rule:
            checks.append(pa.Check.le(rule["max"]))
        pandera_columns[name] = pa.Column(
            nullable=bool(rule.get("nullable", False)), checks=checks, required=True
        )
    return pa.DataFrameSchema(pandera_columns, strict=False)


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
    n_rows = len(df)
    if n_rows == 0:
        return {"is_valid": False, "n_rows": 0, "errors": ["No rows"], "warnings": []}

    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(set(columns) - set(df.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Range checks on text would raise, so type problems are caught before Pandera runs.
    checkable = {}
    for name, rule in columns.items():
        if name not in df.columns:
            continue
        if rule["type"] == "numeric" and not pd.api.types.is_numeric_dtype(df[name]):
            errors.append(f"{name}: expected numeric values, got dtype {df[name].dtype}")
            continue
        checkable[name] = rule

    try:
        build_schema(checkable).validate(df, lazy=True)
    except SchemaErrors as error:
        failures = error.failure_cases
        for column, cases in failures.groupby("column", dropna=False):
            rows = cases["index"].dropna()
            if rows.empty:  # a whole-column failure, e.g. a check that could not run
                errors.append(f"{column}: {', '.join(cases['check'].astype(str).unique())}")
                continue
            n_invalid = rows.nunique()
            checks = ", ".join(sorted(cases["check"].astype(str).unique()))
            message = (
                f"{column}: {n_invalid} rows ({n_invalid / n_rows:.2%}) break the schema "
                f"rules ({checks})"
            )
            if column == target or n_invalid / n_rows > max_invalid_fraction:
                errors.append(message)
            else:
                warnings.append(message)

    return {"is_valid": not errors, "n_rows": n_rows, "errors": errors, "warnings": warnings}


def write_evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, path, sample_size: int, seed: int = 0
) -> None:
    """Save an Evidently data-drift report as HTML (for people; the PSI check is for code)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    def sample(df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(min(sample_size, len(df)), random_state=seed)

    snapshot = Report([DataDriftPreset()]).run(
        current_data=sample(current), reference_data=sample(reference)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))


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

        if not (report["train"]["is_valid"] and report["test"]["is_valid"]):
            write_json(self.config.report_path, {"is_valid": False, **report})
            raise DataValidationError(f"Data validation failed, see {self.config.report_path}")

        features = self.config.numerical_features + self.config.categorical_features
        engineer = FeatureEngineer()
        train_features = engineer.transform(train)[features]
        test_features = engineer.transform(test)[features]

        profile = build_reference_profile(
            train_features,
            self.config.numerical_features,
            self.config.categorical_features,
            self.config.n_bins,
        )
        drift = detect_drift(
            profile, test_features, self.config.psi_threshold, self.config.drift_share_threshold
        )
        report["train_test_drift"] = drift
        if drift["drifted_features"]:
            logger.warning("Train -> test drift in: %s", ", ".join(drift["drifted_features"]))

        write_evidently_report(
            train_features,
            test_features,
            self.config.evidently_report_path,
            self.config.evidently_sample_size,
        )
        write_json(self.config.reference_profile_path, profile)
        write_json(self.config.report_path, {"is_valid": True, **report})
        logger.info("Data validation passed, report saved to %s", self.config.report_path)
        return DataValidationArtifact(
            is_valid=True,
            report_path=self.config.report_path,
            reference_profile_path=self.config.reference_profile_path,
            evidently_report_path=self.config.evidently_report_path,
            dataset_drift=drift["dataset_drift"],
        )
