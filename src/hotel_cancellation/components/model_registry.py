import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from hotel_cancellation.entity.artifact_entity import (
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelRegistryArtifact,
    ModelTrainerArtifact,
)
from hotel_cancellation.entity.config_entity import ModelRegistryConfig
from hotel_cancellation.exception import ModelError
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import load_object, read_json, write_json

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
PROFILE_FILE = "reference_profile.json"


class ModelRegistry:
    """A small file-based model registry.

    Layout::

        <root_dir>/registry.json          index of versions + which one is in production
        <root_dir>/versions/v1/model.joblib
        <root_dir>/versions/v1/metadata.json
        <root_dir>/versions/v1/reference_profile.json
    """

    def __init__(self, config: ModelRegistryConfig) -> None:
        self.root_dir = Path(config.root_dir)
        self.index_path = self.root_dir / "registry.json"

    def index(self) -> dict[str, Any]:
        """Registry contents: {"production": version or None, "versions": [...]}."""
        if not self.index_path.exists():
            return {"production": None, "versions": []}
        return read_json(self.index_path)

    def production_dir(self) -> Path | None:
        version = self.index()["production"]
        return None if version is None else self.root_dir / "versions" / version

    def load_production(self) -> tuple[Pipeline, dict[str, Any]] | None:
        model_dir = self.production_dir()
        if model_dir is None:
            return None
        if not (model_dir / MODEL_FILE).exists():
            raise ModelError(f"Production model files are missing in {model_dir}")
        return load_object(model_dir / MODEL_FILE), read_json(model_dir / METADATA_FILE)

    def run(
        self,
        trainer: ModelTrainerArtifact,
        evaluation: ModelEvaluationArtifact,
        validation: DataValidationArtifact,
    ) -> ModelRegistryArtifact:
        """Register the candidate as a new version and make it production, if it was accepted."""
        if not evaluation.is_accepted:
            logger.info("Candidate not accepted, production model unchanged")
            return ModelRegistryArtifact(promoted=False, version=None, model_dir=None)

        index = self.index()
        version = f"v{len(index['versions']) + 1}"
        model_dir = self.root_dir / "versions" / version
        model_dir.mkdir(parents=True, exist_ok=False)

        shutil.copy2(trainer.model_path, model_dir / MODEL_FILE)
        shutil.copy2(validation.reference_profile_path, model_dir / PROFILE_FILE)
        metadata = read_json(trainer.metadata_path)
        metadata.update(version=version, test_metrics=evaluation.metrics)
        write_json(model_dir / METADATA_FILE, metadata)

        index["versions"].append(
            {
                "version": version,
                "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "test_metrics": evaluation.metrics,
                "replaced": index["production"],
            }
        )
        index["production"] = version
        write_json(self.index_path, index)
        logger.info("Registered %s and promoted it to production", version)
        return ModelRegistryArtifact(promoted=True, version=version, model_dir=model_dir)
