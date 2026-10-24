import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_cancellation.components.data_ingestion import MONTHS, arrival_dates
from hotel_cancellation.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from hotel_cancellation.entity.config_entity import DataTransformationConfig
from hotel_cancellation.logger import get_logger
from hotel_cancellation.utils.common import save_object

logger = get_logger(__name__)

CATEGORY_COLUMNS = (
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived booking features and clean category columns.

    Works on raw booking rows, so the same object runs in training and in the API.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        children = pd.to_numeric(df["children"], errors="coerce")
        df["children"] = children

        df["arrival_month"] = df["arrival_date_month"].map(MONTHS)
        df["arrival_weekday"] = arrival_dates(df).dt.dayofweek
        df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["total_guests"] = df["adults"] + children.fillna(0) + df["babies"]
        df["adr_per_guest"] = df["adr"] / df["total_guests"].where(df["total_guests"] > 0)
        df["has_company"] = df["company"].notna().astype(int)

        # Agent ids are labels, not quantities: 9.0 -> "9", missing -> "none".
        agent = pd.to_numeric(df["agent"], errors="coerce")
        df["agent"] = agent.map(lambda v: "none" if pd.isna(v) else str(int(v)))
        for column in CATEGORY_COLUMNS:
            df[column] = df[column].astype(object).where(df[column].notna(), "missing").astype(str)
        return df


def build_preprocessor(
    numerical: list[str], categorical: list[str], min_category_frequency: int
) -> Pipeline:
    """Raw booking rows -> dense float matrix for the model."""
    columns = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical,
            ),
        ],
        # Always dense: XGBoost treats sparse zeros as "missing", which would make
        # single-row (dense) and batch (sparse) predictions disagree.
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", FeatureEngineer()), ("columns", columns)])


def feature_names(preprocessor: Pipeline) -> list[str]:
    return [str(name) for name in preprocessor.named_steps["columns"].get_feature_names_out()]


class DataTransformation:
    """Fit the preprocessing pipeline on the training data and save it."""

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def run(self, ingestion: DataIngestionArtifact) -> DataTransformationArtifact:
        train = pd.read_csv(ingestion.train_path)
        X_train = train.drop(columns=[self.config.target])

        preprocessor = build_preprocessor(
            self.config.numerical_features,
            self.config.categorical_features,
            self.config.min_category_frequency,
        )
        preprocessor.fit(X_train)
        names = feature_names(preprocessor)

        save_object(self.config.preprocessor_path, preprocessor)
        logger.info(
            "Fitted preprocessor: %d raw columns -> %d model features",
            X_train.shape[1],
            len(names),
        )
        return DataTransformationArtifact(
            preprocessor_path=self.config.preprocessor_path, feature_names=names
        )
