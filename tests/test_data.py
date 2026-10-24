import numpy as np
import pandas as pd
import pytest

from hotel_cancellation.components.data_ingestion import DataIngestion, arrival_dates
from hotel_cancellation.components.data_transformation import FeatureEngineer, build_preprocessor
from hotel_cancellation.components.data_validation import validate_dataframe
from hotel_cancellation.exception import ConfigError
from hotel_cancellation.utils.common import split_tail


def _validate(df, schema, max_invalid_fraction=0.01):
    return validate_dataframe(df, schema["columns"], schema["target"], max_invalid_fraction)


def test_valid_data_passes(bookings, schema):
    result = _validate(bookings, schema)
    assert result["is_valid"], result["errors"]


def test_missing_column_fails(bookings, schema):
    result = _validate(bookings.drop(columns=["lead_time"]), schema)
    assert not result["is_valid"]
    assert any("lead_time" in error for error in result["errors"])


def test_non_numeric_column_fails(bookings, schema):
    bookings["adr"] = "expensive"
    assert not _validate(bookings, schema)["is_valid"]


def test_few_bad_rows_warn_but_many_fail(bookings, schema):
    bookings.loc[:4, "adr"] = -10  # 5 of 1500 rows -> under 1%
    result = _validate(bookings, schema)
    assert result["is_valid"] and result["warnings"]

    bookings.loc[:99, "adr"] = -10  # 100 rows -> over 1%
    assert not _validate(bookings, schema)["is_valid"]


def test_any_bad_target_fails(bookings, schema):
    bookings.loc[0, "is_canceled"] = 2
    assert not _validate(bookings, schema)["is_valid"]


def test_ingestion_drops_leakage_and_splits_by_time(config_manager):
    artifact = DataIngestion(config_manager.get_data_ingestion_config()).run()
    train, test = pd.read_csv(artifact.train_path), pd.read_csv(artifact.test_path)

    assert "reservation_status" not in train.columns
    assert "assigned_room_type" not in test.columns
    assert len(test) == pytest.approx(0.2 * (len(train) + len(test)), abs=1)
    assert arrival_dates(train).max() <= arrival_dates(test).min()


def test_split_tail_rejects_bad_fraction():
    with pytest.raises(ConfigError):
        split_tail(pd.DataFrame({"a": [1, 2]}), 1.5)


def test_feature_engineer_derives_features(bookings):
    row = bookings.iloc[[0]].copy()
    row[["stays_in_weekend_nights", "stays_in_week_nights"]] = [2, 3]
    row[["adults", "children", "babies", "adr"]] = [2, np.nan, 0, 100.0]
    row[["agent", "company", "country"]] = [9.0, np.nan, None]

    out = FeatureEngineer().transform(row).iloc[0]
    assert out["total_nights"] == 5
    assert out["total_guests"] == 2
    assert out["adr_per_guest"] == 50
    assert out["agent"] == "9"
    assert out["has_company"] == 0
    assert out["country"] == "missing"


def test_preprocessor_handles_unseen_categories(bookings, schema):
    features = schema["features"]
    preprocessor = build_preprocessor(features["numerical"], features["categorical"], 5)
    preprocessor.fit(bookings)

    new = bookings.head(2).copy()
    new["country"] = "ATL"  # never seen in training
    out = preprocessor.transform(new)
    assert out.shape[0] == 2 and not np.isnan(out).any()
