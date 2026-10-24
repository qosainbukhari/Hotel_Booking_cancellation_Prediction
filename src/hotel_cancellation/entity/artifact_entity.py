from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_path: Path
    test_path: Path


@dataclass(frozen=True)
class DataValidationArtifact:
    is_valid: bool
    report_path: Path
    reference_profile_path: Path
    evidently_report_path: Path
    dataset_drift: bool


@dataclass(frozen=True)
class DataTransformationArtifact:
    preprocessor_path: Path
    feature_names: list[str]


@dataclass(frozen=True)
class ModelTrainerArtifact:
    model_path: Path
    metadata_path: Path
    threshold: float
    validation_metrics: dict[str, float]
    baseline_paths: dict[str, Path]


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    metrics_path: Path
    metrics: dict[str, float]
    production_metrics: dict[str, float] | None
    baseline_metrics: dict[str, dict[str, float]]
    is_accepted: bool
    reason: str


@dataclass(frozen=True)
class ModelRegistryArtifact:
    promoted: bool
    version: str | None
    serving_dir: Path | None
