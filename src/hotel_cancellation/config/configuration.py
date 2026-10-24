from pathlib import Path

from hotel_cancellation.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    MLflowConfig,
    ModelEvaluationConfig,
    ModelRegistryConfig,
    ModelTrainerConfig,
)
from hotel_cancellation.exception import ConfigError
from hotel_cancellation.utils.common import read_yaml

CONFIG_FILE_PATH = Path("configs/config.yaml")
PARAMS_FILE_PATH = Path("configs/params.yaml")
SCHEMA_FILE_PATH = Path("configs/schema.yaml")


class ConfigurationManager:
    """Reads the three YAML files once and hands each pipeline stage its own typed config."""

    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
        schema_path: Path = SCHEMA_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)

    def _section(self, source: dict, *keys: str):
        value = source
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ConfigError(f"Missing config key: {'.'.join(keys)}")
            value = value[key]
        return value

    def _root_dir(self, stage: str) -> Path:
        return Path(self._section(self.config, stage, "root_dir"))

    @property
    def target(self) -> str:
        return self._section(self.schema, "target")

    @property
    def numerical_features(self) -> list[str]:
        return list(self._section(self.schema, "features", "numerical"))

    @property
    def categorical_features(self) -> list[str]:
        return list(self._section(self.schema, "features", "categorical"))

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        root_dir = self._root_dir("data_ingestion")
        return DataIngestionConfig(
            root_dir=root_dir,
            raw_data_path=Path(self._section(self.config, "data_ingestion", "raw_data_path")),
            train_path=root_dir / "train.csv",
            test_path=root_dir / "test.csv",
            test_size=float(self._section(self.params, "data_split", "test_size")),
            leakage_columns=list(self.schema.get("leakage_columns", [])),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        root_dir = self._root_dir("data_validation")
        monitoring = self._section(self.params, "monitoring")
        return DataValidationConfig(
            root_dir=root_dir,
            report_path=root_dir / "validation_report.json",
            reference_profile_path=root_dir / "reference_profile.json",
            target=self.target,
            columns=dict(self._section(self.schema, "columns")),
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            max_invalid_fraction=float(
                self._section(self.params, "data_validation", "max_invalid_fraction")
            ),
            n_bins=int(monitoring["n_bins"]),
            psi_threshold=float(monitoring["psi_threshold"]),
            drift_share_threshold=float(monitoring["drift_share_threshold"]),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        root_dir = self._root_dir("data_transformation")
        return DataTransformationConfig(
            root_dir=root_dir,
            preprocessor_path=root_dir / "preprocessor.joblib",
            target=self.target,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            min_category_frequency=int(
                self._section(self.params, "preprocessing", "min_category_frequency")
            ),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        root_dir = self._root_dir("model_trainer")
        return ModelTrainerConfig(
            root_dir=root_dir,
            model_path=root_dir / "model.joblib",
            metadata_path=root_dir / "metadata.json",
            target=self.target,
            validation_size=float(self._section(self.params, "data_split", "validation_size")),
            early_stopping_rounds=int(
                self._section(self.params, "training", "early_stopping_rounds")
            ),
            model_params=dict(self._section(self.params, "model")),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        root_dir = self._root_dir("model_evaluation")
        evaluation = self._section(self.params, "evaluation")
        return ModelEvaluationConfig(
            root_dir=root_dir,
            metrics_path=root_dir / "metrics.json",
            target=self.target,
            min_roc_auc=float(evaluation["min_roc_auc"]),
            min_improvement=float(evaluation["min_improvement"]),
        )

    def get_model_registry_config(self) -> ModelRegistryConfig:
        return ModelRegistryConfig(root_dir=self._root_dir("model_registry"))

    def get_mlflow_config(self) -> MLflowConfig:
        mlflow = self.params.get("mlflow", {})
        return MLflowConfig(
            enabled=bool(mlflow.get("enabled", False)),
            tracking_uri=str(mlflow.get("tracking_uri", "sqlite:///mlruns/mlflow.db")),
            experiment_name=str(mlflow.get("experiment_name", "hotel-cancellation")),
        )
