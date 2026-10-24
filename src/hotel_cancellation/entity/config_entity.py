from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    raw_data_path: Path
    train_path: Path
    test_path: Path
    test_size: float
    leakage_columns: list[str]


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    report_path: Path
    reference_profile_path: Path
    evidently_report_path: Path
    target: str
    columns: dict[str, dict[str, Any]]
    numerical_features: list[str]
    categorical_features: list[str]
    max_invalid_fraction: float
    evidently_sample_size: int
    n_bins: int
    psi_threshold: float
    drift_share_threshold: float


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessor_path: Path
    target: str
    numerical_features: list[str]
    categorical_features: list[str]
    min_category_frequency: int


@dataclass(frozen=True)
class CostConfig:
    """Business cost of each kind of mistake, used to pick the decision threshold."""

    false_negative: float
    false_positive: float


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    metadata_path: Path
    baselines_dir: Path
    target: str
    validation_size: float
    early_stopping_rounds: int
    refit_on_full_train: bool
    n_trials: int
    tuning_early_stopping_rounds: int
    costs: CostConfig
    model_params: dict[str, Any] = field(default_factory=dict)
    baseline_params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_path: Path
    target: str
    min_roc_auc: float
    min_improvement: float
    costs: CostConfig


@dataclass(frozen=True)
class ModelRegistryConfig:
    registered_model_name: str
    production_alias: str
    serving_dir: Path


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    artifact_root: Path
    experiment_name: str
